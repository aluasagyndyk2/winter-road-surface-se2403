"""Small atomic journal shared by training and recovery."""
import json
import math
import os
from pathlib import Path


def ensure_storage(path):
    value=Path(path).as_posix()
    if (value=='/content/drive' or value.startswith('/content/drive/')) and not os.path.ismount('/content/drive'):
        raise RuntimeError('Google Drive is disconnected. Remount it before saving or training; local fallback is forbidden.')


def read_state(path):
    path=Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def save_state(path,state):
    ensure_storage(path)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def advance_state(state,epoch,score):
    if not math.isfinite(score):raise ValueError('Non-finite validation macro-F1')
    improved=score>state.get('best_macro_f1',-1)
    return {**state,'next_epoch':epoch+1,'improved':improved,
            'best_macro_f1':score if improved else state['best_macro_f1'],
            'best_epoch':epoch if improved else state['best_epoch'],
            'wait':0 if improved else state.get('wait',0)+1}
