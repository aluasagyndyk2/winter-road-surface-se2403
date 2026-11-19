"""Fixed five-class metrics: absent classes remain in the macro average."""
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
from .data import CLASSES


def classification_metrics(truth, predicted):
    labels=list(range(len(CLASSES)))
    precision,recall,f1,support=precision_recall_fscore_support(truth,predicted,labels=labels,zero_division=0)
    return {'accuracy':float(accuracy_score(truth,predicted)),
            'macro_f1':float(f1.mean()),
            'per_class':{c:{'precision':float(precision[i]),'recall':float(recall[i]),
                            'f1':float(f1[i]),'support':int(support[i])} for i,c in enumerate(CLASSES)},
            'confusion_matrix':confusion_matrix(truth,predicted,labels=labels).tolist()}


def select_validation_threshold(truth,scores,target_accuracy=.9,min_support=30):
    """Maximize validation coverage under a predeclared empirical accuracy target.

    This is a selective-classification heuristic, not probability calibration or
    a deployment accuracy guarantee. If no supported threshold qualifies, flag all.
    """
    scores=np.asarray(scores);truth=np.asarray(truth)
    if scores.shape!=(len(truth),len(CLASSES)) or not len(truth) or not np.isfinite(scores).all():
        raise ValueError('Invalid validation scores')
    confidence=scores.max(axis=1);correct=scores.argmax(axis=1)==truth
    rule={'score_threshold':1.0,'force_manual_review':True,'validation_coverage':0.0,
          'accepted_validation_accuracy':None,'validation_support':int(len(truth)),
          'accepted_validation_support':0,'target_accuracy':target_accuracy,'min_support':min_support,
          'threshold_rule':'Maximum validation coverage with empirical accepted accuracy >= target; no calibration guarantee'}
    for threshold in np.unique(confidence):
        accepted=confidence>=threshold;n=int(accepted.sum())
        if n>=min_support and correct[accepted].mean()>=target_accuracy:
            rule.update(score_threshold=float(threshold),force_manual_review=False,
                        validation_coverage=float(n/len(truth)),accepted_validation_accuracy=float(correct[accepted].mean()),
                        accepted_validation_support=n)
            break
    return rule


def metrics_by_source_split(rows,scores):
    scores=np.asarray(scores)
    return {split:classification_metrics([CLASSES.index(r['label']) for r in rows if r['source_split']==split],
            scores[[i for i,r in enumerate(rows) if r['source_split']==split]].argmax(axis=1))
            for split in sorted({r['source_split'] for r in rows})}
