"""Official checkpoint snapshots, pinned revisions, hashes, no remote-code execution."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import shutil
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
os.environ['HF_HOME']=str(PROJECT.parent/'.cache/eps_models')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
from huggingface_hub import HfApi,snapshot_download
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha

IDS=['amazon/chronos-2','amazon/chronos-bolt-mini','google/timesfm-2.5-200m-pytorch','Datadog/Toto-2.0-4m','Datadog/Toto-2.0-22m','NX-AI/TiRex']

def one(repo,local=False):
    r={'weight_id':repo,'status':'DOWNLOADING','source':'https://huggingface.co/'+repo}
    receipt=RUN/'weight_receipts'/(repo.replace('/','__')+'.json')
    try:
        if shutil.disk_usage(PROJECT).free<50*2**30: raise RuntimeError('Disk guard below 50 GiB')
        info=HfApi().model_info(repo,files_metadata=True)
        r.update(revision=info.sha,license=(info.card_data.to_dict() if info.card_data else {}).get('license','UNKNOWN'),
                 model_card=info.card_data.to_dict() if info.card_data else {},created_at=str(info.created_at),
                 pretraining_pit_status='UNKNOWN_HISTORICAL_OVERLAP_NOT_STRICT_PIT_CERTIFIED')
        size=sum(s.size or 0 for s in info.siblings)
        if size>6*2**30: raise RuntimeError('Small-checkpoint intake cap 6 GiB')
        save_json(receipt,r)
        # Short deterministic directory avoids Windows MAX_PATH in HF temporary names.
        kwargs={'local_dir':str(PROJECT.parent/'.cache/eps_models/direct'/hashlib.sha256(repo.encode()).hexdigest()[:12]/info.sha[:12])} if local else {}
        folder=Path(snapshot_download(repo,revision=info.sha,allow_patterns=['*.json','*.yaml','*.yml','*.safetensors','*.bin','*.pt','*.ckpt','README.md','LICENSE*','*.txt'],max_workers=3,**kwargs))
        files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in folder.rglob('*') if p.is_file()]
        r.update(status='WEIGHTS_CACHED_NOT_EXECUTED',path=str(folder),files=files,bytes=sum(f['bytes'] for f in files),downloaded_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc: r.update(status='WEIGHT_BLOCKED',failure_reason=f'{type(exc).__name__}: {exc}')
    save_json(receipt,r);return r

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('repos',nargs='*');parser.add_argument('--local-dir',action='store_true');args=parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(one,r,args.local_dir) for r in (args.repos or IDS)]):
            r=future.result();print(r['weight_id'],r['status'],flush=True)
