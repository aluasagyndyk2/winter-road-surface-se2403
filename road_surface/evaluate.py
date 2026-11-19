"""Common held-out evaluation; this module never chooses the model checkpoint."""
import argparse
import csv
import hashlib
import io
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from .data import CLASSES, load_manifest, safe_path
from .inference import load_trained_artifact
from .input_pipeline import make_dataset, prepare_image
from .metrics import classification_metrics, metrics_by_source_split
from .training_state import save_state
from .roboflow_data import manifest_digest, validate_files


STRESS_PROTOCOL={'brightness':[0.8,0.6,0.4],'gaussian_blur':[0.5,1.0,2.0],
                 'gaussian_noise':[5.0,15.0,30.0],'jpeg':[70,40,20]}


def stress_image(image,condition,severity=1,seed=42):
    if condition not in STRESS_PROTOCOL or severity not in (1,2,3):raise ValueError('Unknown predeclared stress condition or severity')
    amount=STRESS_PROTOCOL[condition][severity-1]
    if condition=='brightness':return ImageEnhance.Brightness(image).enhance(amount)
    if condition=='gaussian_blur':return image.filter(ImageFilter.GaussianBlur(amount))
    if condition=='gaussian_noise':
        values=np.asarray(image,dtype=np.float32)
        noise=np.random.default_rng(seed).normal(0,amount,values.shape)
        return Image.fromarray(np.clip(np.rint(values+noise),0,255).astype(np.uint8))
    if condition=='jpeg':
        buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=amount);buffer.seek(0)
        with Image.open(buffer) as decoded:return decoded.convert('RGB')
    raise ValueError('Unknown predeclared stress condition')


def evaluate_run(run_dir,rows,data_root,output_dir,batch_size=32,stress=True,allow_smoke=False):
    if not rows or any(r['split']!='test' for r in rows):raise ValueError('Only a nonempty final test manifest is accepted')
    model,meta=load_trained_artifact(run_dir,allow_smoke=allow_smoke)
    if manifest_digest(rows)!=meta.get('test_manifest_sha256'):
        raise ValueError('Final test manifest differs from the one fixed before training')
    if not meta.get('smoke_test'): validate_files(rows,data_root)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    truth=np.asarray([CLASSES.index(r['label']) for r in rows])
    scores=model.predict(make_dataset(rows,data_root,batch_size),verbose=0)
    result=classification_metrics(truth,scores.argmax(axis=1))
    result['per_source_split']=metrics_by_source_split(rows,scores)
    result['grouping_note']='Source filename families are not verified independent scenes. No capture-day metadata is available.'
    result.update(model=meta['model'],checkpoint=meta['checkpoint'],smoke_test=meta['smoke_test'],
                  dataset_id=meta['dataset_id'],test_count=len(rows),test_source_groups=len({r['source_group'] for r in rows}))
    with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['local_path','source_group','source_split','truth','prediction',*CLASSES])
        for row,score in zip(rows,scores):writer.writerow([row['local_path'],row['source_group'],row['source_split'],row['label'],CLASSES[int(score.argmax())],*score.tolist()])
    sample=prepare_image(np.asarray(Image.open(safe_path(data_root,rows[0]['local_path'])).convert('RGB')))[None,...]
    for _ in range(10):model(sample,training=False).numpy()
    timings=[]
    for _ in range(50):
        start=time.perf_counter();model(sample,training=False).numpy();timings.append((time.perf_counter()-start)*1000)
    result['latency']={'batch_size':1,'warmup':10,'measurements':50,'median_ms':float(np.median(timings)),
                       'p95_ms':float(np.percentile(timings,95)),'includes_preprocessing':False}
    result['checkpoint_bytes']=(Path(run_dir)/meta['checkpoint']).stat().st_size
    result['parameters']=int(model.count_params())
    if stress:
        result['synthetic_stress']={}
        result['stress_protocol']={'parameters':STRESS_PROTOCOL,'levels':[1,2,3],
            'noise_units':'standard deviation in raw 0..255 RGB values',
            'blur_units':'Pillow GaussianBlur radius in original-image pixels',
            'application':'before resizing/padding; one distortion at a time; deterministic per-image noise'}
        for family in STRESS_PROTOCOL:
            for severity in [1,2,3]:
                predictions=[]
                for start in range(0,len(rows),batch_size):
                    images=[]
                    for row in rows[start:start+batch_size]:
                        seed=int.from_bytes(hashlib.sha256(row['local_path'].encode()).digest()[:4],'little')
                        with Image.open(safe_path(data_root,row['local_path'])) as image:
                            images.append(prepare_image(np.asarray(stress_image(image.convert('RGB'),family,severity,seed))))
                    predictions.extend(np.asarray(model(np.stack(images),training=False)).argmax(axis=1).tolist())
                result['synthetic_stress'][f'{family}_level_{severity}']=classification_metrics(truth,predictions)
    save_state(out/'metrics.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,ax=plt.subplots(figsize=(9,8))
    ConfusionMatrixDisplay(np.asarray(result['confusion_matrix']),display_labels=CLASSES).plot(ax=ax,cmap='Blues',xticks_rotation=40,colorbar=False)
    ax.set_title(meta['model']+' — held-out test');fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=180);plt.close(fig)
    if stress:
        with (out/'robustness.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['family','severity','parameter','accuracy','macro_f1','accuracy_drop','macro_f1_drop'])
            for family,parameters in STRESS_PROTOCOL.items():
                for severity,parameter in enumerate(parameters,1):
                    m=result['synthetic_stress'][f'{family}_level_{severity}']
                    writer.writerow([family,severity,parameter,m['accuracy'],m['macro_f1'],result['accuracy']-m['accuracy'],result['macro_f1']-m['macro_f1']])
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,(family,parameters) in zip(axes.flat,STRESS_PROTOCOL.items()):
            for metric in ['accuracy','macro_f1']:
                values=[result[metric]]+[result['synthetic_stress'][f'{family}_level_{s}'][metric] for s in [1,2,3]]
                ax.plot(range(4),values,'o-',label=metric)
            ax.set_title(f'{family}: {parameters}');ax.set_xticks(range(4),['clean','1','2','3']);ax.set_ylim(0,1);ax.legend();ax.grid(alpha=.2)
        fig.suptitle(meta['model']+' — synthetic robustness');fig.tight_layout();fig.savefig(out/'robustness_curves.png',dpi=180);plt.close(fig)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--no-stress',action='store_true')
    a=p.parse_args();rows=[r for r in load_manifest(a.manifest) if r['split']=='test']
    evaluate_run(a.run_dir,rows,a.data_root,a.output_dir,stress=not a.no_stress)


if __name__=='__main__':main()
