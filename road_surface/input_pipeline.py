"""Shared deterministic preparation for training, validation and inference."""
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from .data import CLASSES, safe_path


def decode_rgb_file(path):
    """Pillow RGB decoding matches application and synthetic-stress evaluation."""
    if hasattr(path,'item'):path=path.item()
    if isinstance(path,bytes):path=path.decode('utf-8')
    with Image.open(path) as image:return np.asarray(image.convert('RGB'),dtype=np.uint8)


def prepare_image(rgb):
    image=tf.cast(tf.convert_to_tensor(rgb),tf.float32)
    tf.debugging.assert_rank(image,3)
    tf.debugging.assert_equal(tf.shape(image)[-1],3)
    return tf.image.resize_with_pad(image,224,224,method='bilinear')


def make_dataset(rows, data_root, batch_size=32, training=False, seed=42):
    if not rows:raise ValueError('Empty image manifest')
    paths=[str(safe_path(data_root,r['local_path'])) for r in rows]
    labels=[CLASSES.index(r['label']) for r in rows]
    dataset=tf.data.Dataset.from_tensor_slices((paths,labels))
    if training:dataset=dataset.shuffle(len(rows),seed=seed,reshuffle_each_iteration=True)
    def decode(path,label):
        image=tf.numpy_function(decode_rgb_file,[path],tf.uint8)
        image.set_shape([None,None,3])
        return prepare_image(image),label
    options=tf.data.Options();options.experimental_deterministic=True
    options.threading.private_threadpool_size=4
    return dataset.map(decode,num_parallel_calls=4).batch(batch_size).with_options(options).prefetch(2)
