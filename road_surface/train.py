"""Validation-only selection with an epoch journal and resumable Keras checkpoints."""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from .data import CLASSES, load_manifest
from .input_pipeline import make_dataset
from .metrics import classification_metrics, select_validation_threshold
from .models import MODEL_NAMES, build_model, compile_model, enable_fine_tuning
from .training_state import advance_state, read_state, save_state, ensure_storage
from .roboflow_data import DATASET_ID, validate_manifest, manifest_digest, validate_files


def run_training(config):
    config=dict(config)
    name=config['model'];seed=int(config.get('seed',42));batch=int(config.get('batch_size',32))
    if name not in MODEL_NAMES:raise ValueError('Unsupported architecture')
    if not 1<=int(config.get('epochs',10))<=10:raise ValueError('Epoch budget must be 1..10')
    if config.get('weights','imagenet')!='imagenet' and not config.get('smoke_test',False):
        raise ValueError('Real transfer-learning runs require ImageNet weights')
    manifest=load_manifest(config['manifest'])
    validate_manifest(manifest)
    train=[r for r in manifest if r['split']=='train'];validation=[r for r in manifest if r['split']=='validation']
    if not train or not validation:raise ValueError('Training and validation manifests are required')
    config['classes']=list(CLASSES)
    config['dataset_id']=DATASET_ID
    config['input_pipeline']='pillow_rgb_pad224_v1'
    config['manifest_sha256']=manifest_digest(manifest)
    config['test_manifest_sha256']=manifest_digest([r for r in manifest if r['split']=='test'])
    if not config.get('smoke_test',False):
        receipt_path=config.get('dataset_receipt',str(Path(config['manifest']).with_name('frozen_dataset.json')))
        receipt=read_state(receipt_path)
        if (receipt.get('status')!='frozen' or receipt.get('dataset_id')!=DATASET_ID or
            receipt.get('manifest_sha256')!=config['manifest_sha256'] or
            not receipt.get('near_duplicate_review_complete')):
            raise ValueError('A frozen, reviewed dataset receipt matching the manifest is required')
        validate_files(manifest,config['data_root'])
    run=Path(config['run_dir']);ensure_storage(run);run.mkdir(parents=True,exist_ok=True)
    old=read_state(run/'config.json')
    # Paths can change between Colab sessions; experiment-defining values cannot.
    keys=['model','seed','batch_size','epochs','manifest_sha256','smoke_test','weights','input_pipeline','dataset_id','classes']
    if old and any(old.get(k)!=config.get(k) for k in keys):raise ValueError('Resume configuration differs; use a new run directory')
    save_state(run/'config.json',config)
    save_state(run/'environment.json',{'python':platform.python_version(),'tensorflow':tf.__version__,
        'platform':platform.platform(),'devices':[str(d) for d in tf.config.list_physical_devices()],
        'gpu_details':[tf.config.experimental.get_device_details(d) for d in tf.config.list_physical_devices('GPU')],
        'mixed_precision':tf.keras.mixed_precision.global_policy().name})
    tf.keras.utils.set_random_seed(seed)
    ds_train=make_dataset(train,config['data_root'],batch,True,seed)
    ds_val=make_dataset(validation,config['data_root'],batch,False,seed)
    truth=np.asarray([CLASSES.index(r['label']) for r in validation])
    state=read_state(run/'state.json')
    if state:
        model=tf.keras.models.load_model(run/state['last_checkpoint'])
    else:
        model=build_model(name,weights=config.get('weights','imagenet'),seed=seed)
        compile_model(model,1e-3)
    history=state.get('history',[])
    for epoch in range(state.get('next_epoch',0),int(config.get('epochs',10))):
        ensure_storage(run)
        if state.get('wait',0)>=3:break
        if name!='custom_cnn' and epoch==2:
            enable_fine_tuning(model);compile_model(model,1e-5)
        started=time.perf_counter()
        fit=model.fit(ds_train,initial_epoch=epoch,epochs=epoch+1,verbose=2)
        scores=model.predict(ds_val,verbose=0)
        metrics=classification_metrics(truth,scores.argmax(axis=1))
        state=advance_state(state,epoch,metrics['macro_f1'])
        record={'epoch':epoch+1,'phase':'fine_tune' if name!='custom_cnn' and epoch>=2 else 'initial',
                'train_loss':float(fit.history['loss'][-1]),'train_accuracy':float(fit.history['accuracy'][-1]),
                'validation_macro_f1':metrics['macro_f1'],'validation_accuracy':metrics['accuracy'],
                'elapsed_seconds':time.perf_counter()-started}
        history=history+[record]
        # Write checkpoint first, then atomically commit the journal referencing it.
        checkpoint=f'epoch_{epoch+1:02d}.keras'
        ensure_storage(run);model.save(run/checkpoint)
        state.update(last_checkpoint=checkpoint,history=history)
        if state['improved']:state['best_checkpoint']=checkpoint
        save_state(run/'state.json',state)
        print(json.dumps(record),flush=True)
    if not state:raise RuntimeError('No completed epoch')
    best=tf.keras.models.load_model(run/state['best_checkpoint'],compile=False)
    threshold=select_validation_threshold(truth,best.predict(ds_val,verbose=0))
    save_state(run/'artifact.json',{'classes':list(CLASSES),'model':name,'dataset_id':DATASET_ID,
        'checkpoint':state['best_checkpoint'],'training_completed':True,
        'smoke_test':bool(config.get('smoke_test',False)),
        'best_validation_macro_f1':state['best_macro_f1'],'best_epoch':state['best_epoch']+1,
        **threshold,'threshold_note':'Threshold is a validation-only heuristic. Scores are not calibrated probabilities; accepted validation accuracy is not a future accuracy guarantee.',
        'manifest_sha256':config['manifest_sha256'],'test_manifest_sha256':config['test_manifest_sha256']})
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    args=p.parse_args();run_training(json.loads(Path(args.config).read_text(encoding='utf-8')))


if __name__=='__main__':main()
