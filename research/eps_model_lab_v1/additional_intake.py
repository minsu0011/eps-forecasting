"""Pinned, non-executing Sundial and TabPFN-v2 public artifact intake."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
os.environ['HF_HOME']=str(PROJECT.parent/'.cache/eps_models')
from huggingface_hub import HfApi,snapshot_download
from research.eps_model_lab_v1.bootstrap import RUN,EXTERNAL,save_json,sha

SPECS=[('sundial','https://github.com/thuml/Sundial','thuml/sundial-base-128m',
 '3212e42564493f520593e5414af4367fc4b49226',['*.json','*.py','*.safetensors','README.md'],'Apache-2.0'),
 ('tabpfn','https://github.com/PriorLabs/TabPFN','Prior-Labs/TabPFN-v2-reg',
 '4972a65a1b30806315c6f92499959ffbfc69a673',['*.json','README.md','LICENSE.txt','tabpfn-v2-regressor-v2_default.ckpt'],'PriorLabs-1.1')]

if __name__=='__main__':
    for key,repository,repo,revision,patterns,license_name in SPECS:
        if shutil.disk_usage(PROJECT).free<50*2**30:raise RuntimeError('Disk guard below50GiB')
        receipt=RUN/'weight_receipts'/(repo.replace('/','__')+'.json')
        if receipt.exists():raise RuntimeError('Explicit additional intake already exists')
        source=EXTERNAL/key
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
        save_json(RUN/'source_receipts'/f'{key}.json',{'repository':repository,'commit':commit,'path':str(source),
           'license':license_name,'status':'OFFICIAL_REFERENCE_CLONED','downloaded_utc':datetime.now(timezone.utc).isoformat(),
           'license_sha256':sha(source/'LICENSE'),'source_type':'official_repository','modifications':'None; EPS wrappers separate'})
        info=HfApi().model_info(repo,revision=revision,files_metadata=True)
        if info.gated:raise RuntimeError('No unattended acceptance of gated terms')
        folder=Path(snapshot_download(repo,revision=revision,allow_patterns=patterns,max_workers=3))
        files=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in folder.rglob('*') if p.is_file()]
        save_json(receipt,{'weight_id':repo,'revision':revision,'path':str(folder),'status':'WEIGHTS_CACHED_NOT_EXECUTED',
          'source':'https://huggingface.co/'+repo,'license':license_name,'files':files,'bytes':sum(x['bytes'] for x in files),
          'created_at':str(info.created_at),'model_card':info.card_data.to_dict() if info.card_data else {},
          'pretraining_pit_status':'UNKNOWN_HISTORICAL_OVERLAP_NOT_STRICT_PIT_CERTIFIED',
          'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),'gated':False,
          'remote_code_execution':'NONE_DURING_DOWNLOAD; Sundial pinned Python requires inspection before load'})
        print(key,'PINNED_ARTIFACTS_CACHED',len(files),flush=True)
