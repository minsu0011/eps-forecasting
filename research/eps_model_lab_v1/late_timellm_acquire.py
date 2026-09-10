"""Pinned public GPT-2 weights for bounded TimeLLM architecture diagnostic."""
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from huggingface_hub import HfApi,snapshot_download
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha


if __name__=='__main__':
    root=RUN/'late_frontier/TimeLLM';receipt=root/'WEIGHTS.json'
    if receipt.exists():raise RuntimeError('No implicit revision update')
    info=HfApi().model_info('openai-community/gpt2',token=False)
    wanted=['config.json','generation_config.json','merges.txt','vocab.json','tokenizer.json','tokenizer_config.json','model.safetensors','README.md']
    path=Path(snapshot_download('openai-community/gpt2',revision=info.sha,allow_patterns=wanted,
        cache_dir=PROJECT.parent/'.cache/eps_models',token=False,max_workers=4))
    records=[{'name':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in path.iterdir() if p.is_file()]
    assert (path/'model.safetensors').is_file()
    save_json(receipt,{'created_utc':datetime.now(timezone.utc).isoformat(),'repo_id':'openai-community/gpt2','revision':info.sha,
        'path':str(path),'files':records,'license':'MIT according to public model card','remote_code_executed':False,
        'authenticated_request':False,'training_data_overlap_unresolved':True,'weights_in_compact_zip':False,
        'source':'https://huggingface.co/openai-community/gpt2'})
    print('GPT2_PINNED_CACHE_READY',info.sha,len(records),flush=True)
