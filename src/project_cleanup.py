"""Non-destructive local project inventory and safe GitHub staging (Windows/Unix).

    python scripts/project_cleanup.py audit
    python scripts/project_cleanup.py stage --destination outputs/release_candidate

Never deletes or modifies source files. Stage copies only source/docs/config/tests and
small non-sensitive text files. Models, videos, datasets, outputs, and secrets stay local.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git','.venv','venv','__pycache__','.pytest_cache','.mypy_cache',
             '.ruff_cache','.vscode','.idea','node_modules','build','dist'}
LOCAL_DIRS = {'data','models','outputs'}
KEEP_DIRS = {'src','tests','docs','scripts','config','.github'}
KEEP_ROOT = {'.gitignore','.gitattributes','README.md','CONTRIBUTING.md',
             'SECURITY.md','RELEASE_NOTES.md','requirements.txt','compose.yaml',
             'compose.rtsp.yaml','pyproject.toml','LICENSE','LICENSE.md'}
TEXT_SUFFIXES = {'.py','.md','.rst','.txt','.html','.css','.js','.json','.yaml',
                 '.yml','.toml','.ini','.cfg','.svg','.sh','.bat','.ps1'}
BLOCKED_SUFFIXES = {'.pt','.pth','.onnx','.safetensors','.ckpt','.weights',
  '.mp4','.avi','.mov','.mkv','.jpg','.jpeg','.png','.webp','.npy','.npz',
  '.pem','.key','.p12','.pfx','.sqlite','.db','.zip','.7z','.rar'}
SENSITIVE_NAMES = ('secret','password','credential','token','private_key')
MAX_FILE = 5_000_000


def discover(root):
    root = Path(root).resolve()
    for p in root.rglob('*'):
        if not p.is_file() or p.is_symlink():
            continue
        parts = p.relative_to(root).parts
        if any(x in SKIP_DIRS for x in parts):
            continue
        yield p


def classify(rel, size):
    parts = rel.parts
    name = rel.name.lower()
    if name.startswith('.env') or any(word in name for word in SENSITIVE_NAMES):
        return 'SECRET_OR_REVIEW',False
    if parts[0] == 'models' or rel.suffix.lower() in {'.pt','.pth','.onnx','.safetensors','.ckpt','.weights'}:
        return 'MODEL_LOCAL',False
    if parts[0] == 'data':
        return 'DATA_LOCAL',False
    if parts[0] == 'outputs':
        return 'OUTPUT_LOCAL',False
    if rel.suffix.lower() in BLOCKED_SUFFIXES:
        return 'MEDIA_OR_BINARY_LOCAL',False
    if size > MAX_FILE:
        return 'LARGE_REVIEW',False
    if len(parts) == 1:
        return ('SOURCE',True) if rel.name in KEEP_ROOT else ('ROOT_REVIEW',False)
    if parts[0] in KEEP_DIRS and rel.suffix.lower() in TEXT_SUFFIXES:
        return 'SOURCE',True
    return 'REVIEW',False


def audit(root, report):
    root=Path(root).resolve()
    rows=[]
    for p in discover(root):
        rel=p.relative_to(root)
        typ,stage=classify(rel,p.stat().st_size)
        rows.append({'path':rel.as_posix(),'category':typ,'bytes':p.stat().st_size,
                     'stage_safe_source':stage})
    report=Path(report)
    if not report.is_absolute(): report=root/report
    report.parent.mkdir(parents=True,exist_ok=True)
    if report.exists():
        raise FileExistsError(f'Report already exists: {report}. Supply a new --report path.')
    report.write_text(json.dumps({'root':str(root), 'count':len(rows),
         'category_counts':dict(Counter(r['category'] for r in rows)),
         'entries':rows},indent=2)+'\n',encoding='utf-8')
    print('AUDIT only; no files deleted or moved.')
    print('Categories:',dict(Counter(r['category'] for r in rows)))
    print('Report:',report)
    return rows


def filehash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def stage(root, destination):
    root=Path(root).resolve()
    destination=Path(destination)
    if not destination.is_absolute():destination=root/destination
    destination=destination.resolve()
    if destination==root or root in destination.parents and destination.name in {'models','data'}:
        raise ValueError('Unsafe staging destination')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Destination must be empty: {destination}')
    copied=[]
    # Stable list before destination is created to prevent recursive inclusion.
    paths=list(discover(root))
    for p in paths:
        rel=p.relative_to(root)
        category,eligible=classify(rel,p.stat().st_size)
        if not eligible:continue
        dest=destination/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
        copied.append({'path':rel.as_posix(),'size':p.stat().st_size,'sha256':filehash(dest)})
    manifest=destination/'release_manifest.json'
    manifest.write_text(json.dumps({'source_root_not_published':True,'file_count':len(copied),
        'files':copied},indent=2)+'\n',encoding='utf-8')
    print(f'Staged {len(copied)} safe source files to {destination}')
    print('No source files were modified, deleted or moved.')
    print('Review staged files before committing to GitHub.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['audit','stage'])
    ap.add_argument('--root',default=str(ROOT))
    ap.add_argument('--report',default='outputs/release_audit.json')
    ap.add_argument('--destination',default='outputs/release_candidate')
    a=ap.parse_args()
    if a.command=='audit':audit(a.root,a.report)
    else:stage(a.root,a.destination)
if __name__=='__main__':main()
