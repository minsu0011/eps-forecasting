"""Recover code-only upstream snapshots from checkout MAX_PATH failures."""
from __future__ import annotations
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import EXTERNAL,RUN,save_json,sha

for name,paths in {
    'epspredict':['modules','run_scripts','README.md','app.py','requirements.txt'],
    'granite_tsfm':['tsfm_public','pyproject.toml','README.md','LICENSE'],
}.items():
    original=EXTERNAL/name
    target=EXTERNAL/(name+'_code')
    raw=subprocess.run(['git','-C',str(original),'archive','--format=zip','HEAD',*paths],capture_output=True,check=True).stdout
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            dest=target/info.filename
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as stream: stream.write(z.read(info))
    commit=subprocess.run(['git','-C',str(original),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    prior=json.loads((RUN/'source_receipts'/f'{name}.json').read_text(encoding='utf-8'))
    save_json(RUN/'source_receipts'/f'{name}_initial_checkout_failure.json',prior)
    prior.update(status='SOURCE_ARCHIVED_SPARSE_NOT_EXECUTED',commit=commit,path=str(target),
                 recovery='Code-only git archive at identical commit; original partial checkout preserved',
                 failure_reason=None,modifications='NONE; excluded upstream result/data/demo trees')
    save_json(RUN/'source_receipts'/f'{name}.json',prior)
    print(name,commit,'recovered',len(list(target.rglob('*'))),'entries',flush=True)
