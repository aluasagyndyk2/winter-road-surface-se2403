"""Inference uses the training preparation and requires an actual run artifact."""
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from .data import CLASSES, safe_path
from .input_pipeline import prepare_image
from .training_state import read_state
from .roboflow_data import DATASET_ID


def load_trained_artifact(run_dir, allow_smoke=False):
    root=Path(run_dir);path=root/'artifact.json'
    if not path.is_file():raise FileNotFoundError('artifact.json is missing; train and export a model first')
    meta=read_state(path)
    if meta.get('classes')!=list(CLASSES):raise ValueError('Unexpected class order')
    if meta.get('dataset_id')!=DATASET_ID:raise ValueError('Unexpected dataset identity')
    if not meta.get('training_completed'):raise ValueError('Model training has not completed')
    if meta.get('smoke_test') and not allow_smoke:raise ValueError('A smoke-test checkpoint is not a trained research model')
    checkpoint=safe_path(root,meta['checkpoint'])
    if not checkpoint.is_file():raise FileNotFoundError(f'Checkpoint missing: {checkpoint.name}')
    model=tf.keras.models.load_model(checkpoint,compile=False)
    if model.output_shape[-1]!=len(CLASSES):raise ValueError('Unexpected model output dimensions')
    return model,meta


def crop_region(image,roi=None):
    if not isinstance(image,Image.Image):raise ValueError('A decoded image is required')
    if roi is None:return image.convert('RGB')
    if len(roi)!=4:raise ValueError('Region requires left, top, right, bottom')
    left,top,right,bottom=roi
    if not all(isinstance(v,int) for v in roi) or not (0<=left<right<=image.width and 0<=top<bottom<=image.height):
        raise ValueError('Region must have positive area and lie inside the image')
    return image.crop(roi).convert('RGB')


def predict_image(image,artifact,roi=None):
    model,meta=artifact
    rgb=crop_region(image,roi)
    scores=np.asarray(model(prepare_image(np.asarray(rgb))[None,...],training=False))[0]
    if scores.shape!=(len(CLASSES),) or not np.isfinite(scores).all():raise ValueError('Invalid model scores')
    label=CLASSES[int(scores.argmax())];threshold=meta.get('score_threshold')
    return {'class':label,'scores':{c:float(scores[i]) for i,c in enumerate(CLASSES)},
            'low_score':bool(meta.get('force_manual_review',False) or (threshold is not None and scores.max()<threshold))}
