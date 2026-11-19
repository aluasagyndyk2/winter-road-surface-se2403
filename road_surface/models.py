"""Three architectures with their preprocessing embedded in serializable models."""
import tensorflow as tf
from .data import CLASSES

MODEL_NAMES=('custom_cnn','mobilenet_v2','efficientnet_b0')


def build_model(name, weights='imagenet', seed=42):
    if name not in MODEL_NAMES:raise ValueError(f'Unknown architecture: {name}')
    tf.keras.utils.set_random_seed(seed)
    layers=tf.keras.layers
    inputs=layers.Input((224,224,3),name='rgb_0_255')
    augmentation=tf.keras.Sequential([
        layers.RandomFlip('horizontal',seed=seed),
        layers.RandomBrightness(0.1,value_range=(0,255),seed=seed+1),
        layers.RandomZoom(0.05,seed=seed+2)],name='training_augmentation')
    x=augmentation(inputs)
    if name=='custom_cnn':
        x=layers.Rescaling(1/255.,name='input_scaling')(x)
        for filters in [24,48,96,128]:
            x=layers.Conv2D(filters,3,padding='same',activation='relu')(x)
            x=layers.MaxPooling2D()(x)
    else:
        if name=='mobilenet_v2':
            x=layers.Rescaling(1/127.5,offset=-1,name='input_scaling')(x)
            base=tf.keras.applications.MobileNetV2(include_top=False,weights=weights,input_shape=(224,224,3))
        else:
            # EfficientNetB0 already rescales its 0..255 inputs internally.
            base=tf.keras.applications.EfficientNetB0(include_top=False,weights=weights,input_shape=(224,224,3))
        backbone=tf.keras.Model(base.input,base.output,name='backbone');backbone.trainable=False
        x=backbone(x,training=False)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dense(128,activation='relu')(x)
    x=layers.Dropout(0.25,seed=seed+3)(x)
    outputs=layers.Dense(len(CLASSES),activation='softmax',dtype='float32',name='class_scores')(x)
    return tf.keras.Model(inputs,outputs,name=name)


def enable_fine_tuning(model):
    base=model.get_layer('backbone');base.trainable=True
    for layer in base.layers:
        layer.trainable=not isinstance(layer,tf.keras.layers.BatchNormalization)


def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
