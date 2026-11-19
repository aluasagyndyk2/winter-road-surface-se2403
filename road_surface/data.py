"""Five-class schema, safe local paths and portable JSONL manifests."""
import gzip
import json
from pathlib import Path, PureWindowsPath
from .training_state import ensure_storage

CLASSES = ('dry','wet','snow','ice','slush')


def safe_path(root, relative):
    root=Path(root).resolve();relative=str(relative).replace('\\','/')
    if Path(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError('Absolute archive paths are not allowed')
    result=(root/relative).resolve()
    if not result.is_relative_to(root): raise ValueError('Path escapes the dataset root')
    return result


def load_manifest(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_manifest(path, rows):
    ensure_storage(path)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt',encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
