#!/usr/bin/env python3
"""Check the public source tree for syntax, private paths and local artifacts."""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_DIRS={'.github','compare','configs','datasets','docs','models','scripts','tests','tools','utils'}
SOURCE_SUFFIXES={'.py','.cpp','.cu','.cuh','.h','.md','.json','.xml','.txt','.yml','.yaml','.toml','.cfg','.ini','.npz','.svg','.png','.jpg'}
EXCLUDED_DIRS={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','build','dist','node_modules'}
PRIVATE_DIRS={'local','data','runs','outputs','results','checkpoints','cache','logs','backup'}
OUTPUT_NAMES={'result.json','aggregate.json','training_summary.json','progress.json','config_snapshot.json',
              'protocol.json','study_plan.json','plan.json','audit_report.json','reuse_manifest.json',
              'cleanup_manifest.json','source_import.json'}
PRIVATE_DOCS={'docs/EXPERIMENT_PROTOCOL.md','docs/EXPERIMENT_CATALOG.md','docs/FAIR_COMPARISONS.md'}
PRIVATE_PATH=re.compile(r'/(?:home|Users|media)/[^/\s"\x27]+/')
TOKEN=re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def public_path(rel):
    rel=Path(rel)
    if rel.as_posix() in PRIVATE_DOCS or rel.name in OUTPUT_NAMES:return False
    if any(p in EXCLUDED_DIRS or p.endswith('.egg-info') for p in rel.parts):return False
    if any(p in PRIVATE_DIRS for p in rel.parts):
        if not rel.as_posix().startswith('compare/official/wavehfg/pytorch_wavelets/dtcwt/data/'):return False
    if len(rel.parts)==1:
        return rel.name in {'.gitignore','.gitattributes','README.md','requirements.txt','requirements-baselines.txt','requirements-optional.txt'} or rel.name.startswith(('LICENSE','NOTICE'))
    if rel.parts[0] not in SOURCE_DIRS:return False
    if rel.suffix=='.npz':return rel.as_posix().startswith('compare/official/wavehfg/pytorch_wavelets/dtcwt/data/')
    return rel.suffix in SOURCE_SUFFIXES or rel.name.startswith(('LICENSE','NOTICE'))


def source_files(root):
    root=Path(root).resolve();files=[]
    candidates=list(root.iterdir())
    for name in sorted(SOURCE_DIRS):
        folder=root/name
        if folder.is_symlink():raise ValueError(f'Source symlink is not exported: {name}')
        if folder.is_dir():candidates.extend(folder.rglob('*'))
    for path in candidates:
        rel=path.relative_to(root)
        if not public_path(rel):continue
        if path.is_symlink():
            if rel.parts[0] in SOURCE_DIRS:raise ValueError(f'Source symlink is not exported: {rel}')
            continue
        if not path.is_file():continue
        files.append(rel)
    return sorted(files)


def validate_git_candidates(root, files):
    """Inspect tracked files as well as unignored additions; ignore rules are not retroactive."""
    root=Path(root).resolve()
    result=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    if result.returncode or Path(result.stdout.strip()).resolve()!=root:return None
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    candidates={Path(n) for n in names if n and ((root/n).is_file() or (root/n).is_symlink())}
    unexpected=candidates-set(files)
    if unexpected:raise ValueError('Non-source files in Git candidates: '+', '.join(p.as_posix() for p in sorted(unexpected)))
    return len(candidates)


def validate_source(root,files):
    root=Path(root);errors=[];total=0
    for rel in files:
        path=root/rel;total+=path.stat().st_size
        if path.stat().st_size>=100*1024**2:errors.append(f'File exceeds 100 MiB: {rel}')
        if path.suffix in {'.npz','.png','.jpg'}:continue
        text=path.read_text(encoding='utf-8')
        if PRIVATE_PATH.search(text):errors.append(f'Workstation path in public source: {rel}')
        if TOKEN.search(text):errors.append(f'Credential-shaped content in public source: {rel}')
        try:
            if path.suffix=='.py':ast.parse(text,filename=str(rel))
            elif path.suffix=='.json':json.loads(text)
            elif path.suffix=='.xml':ET.fromstring(text)
        except (SyntaxError,ValueError,ET.ParseError) as exc:errors.append(f'Invalid source format: {rel}: {exc}')
    if errors:raise ValueError('\n'.join(errors))
    return {'files':len(files),'source_bytes':total}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args();root=Path(__file__).resolve().parents[1]
    files=source_files(root);summary=validate_source(root,files)
    summary['git_candidates']=validate_git_candidates(root,files)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
