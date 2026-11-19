"""Auditable local Roboflow Folder Structure import; never reads credentials.

Filename families are conservative duplicate controls, not known capture scenes.
The audit must be reviewed and frozen separately before research training.
"""
import argparse
import collections
import csv
import hashlib
import html
import io
import json
import re
import stat
import zipfile
import numpy as np
from pathlib import Path, PurePosixPath
from PIL import Image, UnidentifiedImageError
from .data import CLASSES, safe_path, write_manifest, load_manifest

DATASET_ID = 'roboflow:roads-pjwmf/roads-q0nh0:2'
SPLITS = {'train':'train', 'valid':'validation', 'validation':'validation', 'test':'test'}
PRIORITY = {'test':0, 'validation':1, 'train':2}


def validate_manifest(rows, require_all_classes=True):
    if not rows: raise ValueError('Empty dataset manifest')
    groups = collections.defaultdict(set)
    pixels = collections.defaultdict(set)
    paths = set()
    for row in rows:
        if row.get('dataset_id') != DATASET_ID: raise ValueError('Unexpected dataset identity')
        if row.get('label') not in CLASSES or row.get('split') not in PRIORITY:
            raise ValueError('Unknown label or split')
        for field in ['local_path','source_group','decoded_sha256']:
            if not isinstance(row.get(field),str) or not row[field]:
                raise ValueError(f'Missing manifest field: {field}')
        safe_path(Path.cwd(),row['local_path'])
        if row['local_path'] in paths: raise ValueError('Repeated manifest path')
        paths.add(row['local_path'])
        groups[row['source_group']].add(row['split'])
        pixels[row['decoded_sha256']].add(row['split'])
    if any(len(s)>1 for s in groups.values()): raise ValueError('Source-family leakage between splits')
    if any(len(s)>1 for s in pixels.values()): raise ValueError('Decoded-pixel leakage between splits')
    if require_all_classes:
        for split in PRIORITY:
            if {r['label'] for r in rows if r['split']==split} != set(CLASSES):
                raise ValueError(f'All five classes are required in {split}')


def source_family(filename):
    # Roboflow's hash identifies the exported variant. The prefix may identify
    # an original, but this must be checked against the actual archive.
    prefix = re.split(r'\.rf\.',filename,flags=re.IGNORECASE)[0]
    if prefix == filename: prefix = Path(filename).stem
    if not prefix: raise ValueError('Empty source filename prefix')
    return prefix.casefold()


def import_archive(archive_path, output_root):
    root=Path(output_root).resolve()
    if root.exists() and any(root.iterdir()): raise ValueError('Import destination must be empty')
    root.mkdir(parents=True,exist_ok=True)
    rows=[];metadata={};seen=set();total=0
    with zipfile.ZipFile(archive_path) as archive:
        members=archive.infolist()
        if len(members)>100_000: raise ValueError('Too many ZIP members')
        for info in sorted(members,key=lambda m:m.filename):
            name=info.filename.replace('\\','/')
            safe_path(root,name)
            if '..' in PurePosixPath(name).parts: raise ValueError('Parent traversal in archive')
            if stat.S_ISLNK(info.external_attr>>16): raise ValueError('ZIP links are not allowed')
            if info.is_dir(): continue
            if name.casefold() in seen: raise ValueError('Duplicate archive path')
            seen.add(name.casefold());total+=info.file_size
            if info.file_size>40*1024*1024 or total>4*1024**3:
                raise ValueError('Archive exceeds expected size limits')
            if info.flag_bits & 1: raise ValueError('Encrypted archive not supported')
            parts=PurePosixPath(name).parts
            if len(parts)==1 and (name.lower().startswith('readme') or name.lower()=='classes.txt'):
                if info.file_size>1_000_000: raise ValueError('Oversized metadata')
                metadata[name]=archive.read(info).decode('utf-8',errors='replace')
                continue
            if len(parts)!=3 or parts[0].lower() not in SPLITS:
                raise ValueError(f'Unsupported Folder Structure path: {name}')
            split=SPLITS[parts[0].lower()];label=parts[1].lower()
            if label not in CLASSES: raise ValueError(f'Unexpected class: {parts[1]}')
            if Path(parts[2]).suffix.lower() not in ['.jpg','.jpeg','.png','.webp']:
                raise ValueError(f'Unexpected image file: {name}')
            raw=archive.read(info)  # zipfile verifies CRC while reading.
            try:
                with Image.open(io.BytesIO(raw)) as opened:
                    if opened.width*opened.height>30_000_000: raise ValueError('Oversized image')
                    rgb=opened.convert('RGB');rgb.load()
                    digest=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                    gray=list(rgb.convert('L').resize((9,8)).tobytes())
                    dhash=sum(int(gray[y*9+x]>gray[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))
                    width,height=rgb.size
            except (OSError,UnidentifiedImageError,Image.DecompressionBombError) as exc:
                raise ValueError(f'Image decoding failed: {name}') from exc
            destination=safe_path(root,'images/'+name)
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
            rows.append(dict(local_path=destination.relative_to(root).as_posix(),label=label,
                split=split,source_split=parts[0],export_filename=parts[2],
                source_group=label+'/'+source_family(parts[2]),
                source_group_basis='export_filename_prefix_not_verified_scene',
                decoded_sha256=digest,sha256=hashlib.sha256(raw).hexdigest(),
                difference_hash=f'{dhash:016x}',width=width,height=height,dataset_id=DATASET_ID))
    if not rows: raise ValueError('No supported images in archive')
    # Connected components join filename families and exact decoded duplicates.
    # Quarantine entire conflicting components, including their augmentations.
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(a,b): parent[find(b)]=find(a)
    for field in ['source_group','decoded_sha256']:
        first={}
        for i,row in enumerate(rows):
            if row[field] in first: join(first[row[field]],i)
            else: first[row[field]]=i
    components=collections.defaultdict(list)
    for i in range(len(rows)): components[find(i)].append(rows[i])
    kept=[];excluded=[]
    for component in components.values():
        prefixes=sorted({r['source_group'] for r in component})
        component_id='family:'+hashlib.sha256(json.dumps(prefixes).encode()).hexdigest()[:24]
        for row in component:
            row['source_filename_prefix']=row['source_group']
            row['source_group']=component_id
        if len({r['label'] for r in component})>1:
            excluded.extend(dict(r,reason='conflicting_labels_in_connected_family') for r in component)
            continue
        chosen=min((r['split'] for r in component),key=PRIORITY.get)
        pixels=set()
        for row in sorted(component,key=lambda r:r['local_path']):
            if row['split']!=chosen:
                excluded.append(dict(row,reason='family_or_pixels_cross_split'))
            elif row['decoded_sha256'] in pixels:
                excluded.append(dict(row,reason='within_split_decoded_duplicate'))
            else: kept.append(row);pixels.add(row['decoded_sha256'])
    kept.sort(key=lambda r:r['local_path'])
    if kept: validate_manifest(kept,require_all_classes=False)
    audit=dict(dataset_id=DATASET_ID,status='manual_review_required',
        archive_sha256=hash_file(archive_path),exported_images=len(rows),retained_images=len(kept),
        exported_counts=dict(collections.Counter(f"{r['split']}/{r['label']}" for r in rows)),
        retained_counts=dict(collections.Counter(f"{r['split']}/{r['label']}" for r in kept)),
        excluded_counts=dict(collections.Counter(r['reason'] for r in excluded)),
        metadata=metadata,near_duplicate_review_complete=False,
        source_scene_independence_verified=False,
        augmentation_note='Version 2 declares three training outputs per example. Original-versus-variant identity is not inferred from the hash.',
        grouping_note='Filename prefixes plus decoded pixel equality; no acquisition dates or scene independence asserted.')
    write_manifest(root/'candidate_manifest.jsonl',kept)
    (root/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (root/'exclusions.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    return dict(rows=kept,audit=audit,exclusions=excluded)


def hash_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def manifest_digest(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_files(rows, data_root):
    """Bind the reviewed metadata to actual bytes before freeze/train/test."""
    for row in rows:
        expected=row.get('sha256','')
        if not re.fullmatch('[0-9a-f]{64}',expected): raise ValueError('Missing valid image hash')
        path=safe_path(data_root,row['local_path'])
        if not path.is_file() or hash_file(path)!=expected:
            raise ValueError(f"Image hash differs from the audited source: {row['local_path']}")


def find_near_pairs(rows, max_distance=6, max_pairs=50_000):
    """Exhaustive cross-split 64-bit dHash candidates; not semantic deduplication."""
    if not 0<=max_distance<=64: raise ValueError('Invalid hash distance')
    pairs=[];ordered=sorted(rows,key=lambda r:r['local_path'])
    values=[int(r['difference_hash'],16) for r in ordered]
    for i,left in enumerate(ordered):
        for j in range(i+1,len(ordered)):
            right=ordered[j]
            if left['split']==right['split']: continue
            distance=(values[i]^values[j]).bit_count()
            if distance<=max_distance:
                pair=dict(left=left['local_path'],right=right['local_path'],distance=distance)
                pair['pair_id']=hashlib.sha256((pair['left']+'\n'+pair['right']).encode()).hexdigest()[:20]
                pairs.append(pair)
                if len(pairs)>max_pairs:
                    raise ValueError('Near-duplicate candidate limit exceeded; review the audit protocol before training')
    return pairs


def freeze_rows(rows, pairs, decisions):
    expected={p['pair_id']:p for p in pairs}
    submitted={p.get('pair_id'):p for p in decisions}
    if len(submitted)!=len(decisions) or set(submitted)!=set(expected):
        raise ValueError('Complete near-duplicate review is required')
    parent={r['source_group']:r['source_group'] for r in rows}
    by_path={r['local_path']:r for r in rows};quarantine=set()
    def find(key):
        while parent[key]!=key: parent[key]=parent[parent[key]];key=parent[key]
        return key
    def join(a,b): parent[find(b)]=find(a)
    for key,pair in expected.items():
        decision=submitted[key]
        if any(str(decision.get(k))!=str(pair[k]) for k in ['left','right','distance']):
            raise ValueError('Review entries do not match the candidate list')
        value=decision.get('decision')
        if value not in ['same_source','different','uncertain']:
            raise ValueError('Every candidate requires a review decision')
        a=by_path[pair['left']]['source_group'];b=by_path[pair['right']]['source_group']
        if value!='different': join(a,b)
        if value=='uncertain': quarantine.update([a,b])
    groups=collections.defaultdict(list)
    for row in rows: groups[find(row['source_group'])].append(row)
    quarantine={find(g) for g in quarantine};kept=[];excluded=[]
    for key,group in groups.items():
        conflict=len({r['label'] for r in group})>1
        chosen=min((r['split'] for r in group),key=PRIORITY.get)
        for row in group:
            reason=('uncertain_near_source' if key in quarantine else
                    'conflicting_near_source_labels' if conflict else
                    'reviewed_near_source_cross_split' if row['split']!=chosen else None)
            if reason: excluded.append(dict(row,reason=reason))
            else: kept.append(row)
    return sorted(kept,key=lambda r:r['local_path']),excluded


def write_review(root):
    root=Path(root);rows=load_manifest(root/'candidate_manifest.jsonl')
    pairs=find_near_pairs(rows)
    with (root/'near_review.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['pair_id','left','right','distance','decision'])
        writer.writeheader();writer.writerows(dict(p,decision='') for p in pairs)
    by_path={r['local_path']:r for r in rows};cards=[]
    for pair in pairs:
        cells=[]
        for side in ['left','right']:
            row=by_path[pair[side]]
            cells.append('<figure><img loading="lazy" src="'+html.escape(row['local_path'],quote=True)+'"><figcaption>'+
                         html.escape(row['split']+' / '+row['label']+' / '+row['export_filename'])+'</figcaption></figure>')
        cards.append('<section><p>'+pair['pair_id']+' · dHash '+str(pair['distance'])+'</p><div>'+''.join(cells)+'</div></section>')
    page='<!doctype html><meta charset="utf-8"><title>Roads duplicate review</title><style>body{font:16px Arial;margin:32px;background:#eef3f5}section{background:white;padding:16px;margin:16px 0}section div{display:flex}figure{width:48%;margin:8px}img{width:100%;max-width:480px}figcaption{overflow-wrap:anywhere}</style><h1>Cross-split visual review</h1><p>Record same_source, different or uncertain in near_review.csv. Filename families are not verified capture scenes.</p>'+''.join(cards)
    (root/'near_review.html').write_text(page,encoding='utf-8')
    return pairs


def freeze_dataset(root):
    root=Path(root);rows=load_manifest(root/'candidate_manifest.jsonl');pairs=find_near_pairs(rows)
    validate_files(rows,root)
    with (root/'near_review.csv').open(newline='',encoding='utf-8') as stream:
        decisions=list(csv.DictReader(stream))
    kept,excluded=freeze_rows(rows,pairs,decisions);validate_manifest(kept)
    manifest=root/'manifest.jsonl';write_manifest(manifest,kept)
    receipt=dict(dataset_id=DATASET_ID,status='frozen',manifest_sha256=manifest_digest(kept),
        near_duplicate_review_complete=True,near_review_protocol='cross-split 64-bit dHash Hamming distance <= 6, visually reviewed',
        reviewed_candidates=len(pairs),review_csv_sha256=hash_file(root/'near_review.csv'),
        source_scene_independence_verified=False,final_counts=dict(collections.Counter(f"{r['split']}/{r['label']}" for r in kept)))
    (root/'frozen_dataset.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    (root/'near_exclusions.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    return receipt


def quarantine_rows(rows, max_distance=6):
    """Exclude every family implicated by any cross-split hash candidate.

    This deliberately trades data coverage for a deterministic conservative
    uncertainty policy. It does not label all candidate pairs as duplicates.
    """
    groups={split:[r for r in rows if r['split']==split] for split in PRIORITY}
    affected=set();histogram=np.zeros(max_distance+1,dtype=np.int64)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        left=groups[a];hashes=np.asarray([int(r['difference_hash'],16) for r in left],dtype=np.uint64)
        for right in groups[b]:
            values=hashes ^ np.uint64(int(right['difference_hash'],16))
            if hasattr(np,'bitwise_count'): distances=np.bitwise_count(values)
            else:
                # Portable vector popcount for older NumPy in Colab images.
                values=values-((values>>1)&np.uint64(0x5555555555555555))
                values=(values&np.uint64(0x3333333333333333))+((values>>2)&np.uint64(0x3333333333333333))
                distances=(((values+(values>>4))&np.uint64(0x0f0f0f0f0f0f0f0f))*np.uint64(0x0101010101010101))>>56
            hits=np.flatnonzero(distances<=max_distance)
            histogram+=np.bincount(distances[hits].astype(np.int64),minlength=max_distance+1)
            if len(hits):
                affected.add(right['source_group']);affected.update(left[int(i)]['source_group'] for i in hits)
    kept=[r for r in rows if r['source_group'] not in affected]
    excluded=[dict(r,reason='conservative_cross_split_near_family_quarantine') for r in rows if r['source_group'] in affected]
    return kept,excluded,dict(candidate_pairs=int(histogram.sum()),distance_histogram=histogram.tolist(),
                              affected_families=len(affected),excluded_images=len(excluded))


def quarantine_and_freeze(root):
    root=Path(root);rows=load_manifest(root/'candidate_manifest.jsonl');validate_files(rows,root)
    kept,excluded,summary=quarantine_rows(rows);validate_manifest(kept)
    receipt=dict(dataset_id=DATASET_ID,status='frozen',manifest_sha256=manifest_digest(kept),
        near_duplicate_review_complete=True,near_duplicate_check_complete=True,all_pairs_visually_reviewed=False,
        near_review_protocol='All cross-split 64-bit dHash pairs at Hamming distance <=6; quarantine both sides and every filename/pixel family sibling',
        policy_note='Conservative automated uncertainty exclusion, not a claim that all candidates are true duplicates. Changes the evaluated sample distribution.',
        source_scene_independence_verified=False,**summary,
        final_counts=dict(collections.Counter(f"{r['split']}/{r['label']}" for r in kept)))
    write_manifest(root/'manifest.jsonl',kept)
    (root/'near_exclusions.json').write_text(json.dumps(excluded,indent=2),encoding='utf-8')
    (root/'frozen_dataset.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive')
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--review',action='store_true');parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--quarantine-near',action='store_true');args=parser.parse_args()
    if args.quarantine_near:
        print(json.dumps(quarantine_and_freeze(args.output_root),indent=2));return
    if args.review:
        print('Near candidates:',len(write_review(args.output_root)));return
    if args.freeze:
        print(json.dumps(freeze_dataset(args.output_root),indent=2));return
    if not args.archive: parser.error('--archive is required for import')
    result=import_archive(args.archive,args.output_root)
    print(json.dumps(result['audit'],indent=2))
    print('Candidate manifests only: review real images, filename groups and near duplicates before freezing.')


if __name__=='__main__': main()
