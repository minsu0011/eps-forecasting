"""Initialize persistent 10-hour EPS run and intake official source snapshots."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import urllib.request

PROJECT = Path(__file__).resolve().parents[2]
LAB = PROJECT/'research/eps_model_lab_v1'
ORIGINAL_RUN = PROJECT/'outputs/eps_model_lab_v1_20260908T012326'
RUN = Path(os.environ.get('EPS_LAB_RUN_DIR',str(PROJECT/'outputs/eps_model_lab_v1_20260908T012326_pit_r2'))).resolve()
if RUN.parent != (PROJECT/'outputs').resolve() or not RUN.name.startswith('eps_model_lab_v1_'):
    raise RuntimeError('EPS run path must be a specifically named EPS output directory')
EXTERNAL = PROJECT/'external/eps_model_zoo'
SOURCES = {
    'afden': 'leonard67/AFDEN-Python-Workshop',
    'epspredict': 'seferlab/epspredict',
    'statsforecast': 'Nixtla/statsforecast',
    'mlforecast': 'Nixtla/mlforecast',
    'neuralforecast': 'Nixtla/neuralforecast',
    'autogluon': 'autogluon/autogluon',
    'darts': 'unit8co/darts',
    'chronos': 'amazon-science/chronos-forecasting',
    'timesfm': 'google-research/timesfm',
    'moirai': 'SalesforceAIResearch/uni2ts',
    'lag_llama': 'time-series-foundation-models/lag-llama',
    'granite_tsfm': 'ibm-granite/granite-tsfm',
    'moment': 'moment-timeseries-foundation-model/moment',
    'toto': 'DataDog/toto',
    'time_moe': 'Time-MoE/Time-MoE',
    'tirex': 'NX-AI/tirex',
}


def sha(path):
    # Keep checksum memory bounded even for multi-GB trained checkpoints.
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_replace(source,target):
    """Windows readers/AV may briefly deny replace; retain atomicity and retry."""
    for attempt in range(12):
        try:
            os.replace(source,target)
            return
        except PermissionError:
            if attempt==11: raise
            time.sleep(min(.025*(2**attempt),.5))


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n',encoding='utf-8')
    atomic_replace(tmp,path)


def get_json(url):
    request=urllib.request.Request(url,headers={'User-Agent':'EPSModelLab research client','Accept':'application/json'})
    with urllib.request.urlopen(request,timeout=45) as stream:
        return json.load(stream)


def initialize():
    RUN.mkdir(parents=True,exist_ok=True)
    for name in ('EPS_SYSTEM_CONTRACT_V1.json','EPS_TARGET_CONTRACT_V1.md'):
        target=RUN/name
        if not target.exists(): target.write_bytes((LAB/name).read_bytes())
    pe=PROJECT/'outputs/c4_r2_final_freeze_20260907_run01'
    rows=[]
    for line in (pe/'CHECKSUMS.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if sha(pe/name)!=digest: raise RuntimeError('PE frozen evidence drift: '+name)
        rows.append({'path':name,'sha256':digest})
    save_json(RUN/'PE_READONLY_BASELINE.json',{'root':str(pe),'files':rows,'checksum_sha256':sha(pe/'CHECKSUMS.sha256')})
    save_json(RUN/'RUN_STATE.json',{'start_utc':'2026-09-07T16:23:26Z','deadline_utc':'2026-09-08T02:23:26Z',
        'finalize_after_utc':'2026-09-08T01:38:26Z','duration_hours':10,'status':'INTAKE_AND_DATA_AUDIT',
        'updated_utc':datetime.now(timezone.utc).isoformat(),'completed_models':[],
        'next_actions':['SEC vintage data acquisition','common target/split geometry lock','baseline scoring',
                        'isolated TS/GPU environments','official-source adapter audit']})
    save_json(RUN/'EPS_MODEL_REGISTRY_V1.json',{'status':'INITIALIZED_BEFORE_SCORES','evidence_class':'RESEARCH_ONLY','models':[]})
    return RUN


def intake(item):
    name,repo=item
    record={'source':name,'repository':'https://github.com/'+repo,'commit':None,'license':'UNKNOWN',
            'environment':None,'adapter':None,'weight_id':None,'weight_hash':None,'status':'INTAKE_STARTED'}
    try:
        meta=get_json('https://api.github.com/repos/'+repo)
        record['license']=(meta.get('license') or {}).get('spdx_id') or 'UNKNOWN'
        record['size_kib']=meta.get('size')
        target=EXTERNAL/name
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            result=subprocess.run(['git','clone','--depth','1','--filter=blob:none',
                                   'https://github.com/'+repo+'.git',str(target)],capture_output=True,text=True,
                                  encoding='utf-8',errors='replace',timeout=300)
            (RUN/'intake_logs').mkdir(exist_ok=True)
            (RUN/'intake_logs'/f'{name}.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
            if result.returncode: raise RuntimeError(result.stderr[-1000:])
        commit=subprocess.run(['git','-C',str(target),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        record.update(commit=commit,path=str(target),status='SOURCE_CLONED_NOT_EXECUTED',
                      downloaded_at_utc=datetime.now(timezone.utc).isoformat(),modifications='NONE')
        licenses=[p for p in target.iterdir() if p.is_file() and ('license' in p.name.lower() or p.name.lower()=='copying')]
        record['license_files']=[{'name':p.name,'sha256':sha(p)} for p in licenses]
        if record['license'] in ('UNKNOWN','NOASSERTION'):
            record['license_status']='RESEARCH_ONLY_LICENSE_UNCLEAR'
    except Exception as exc:
        record.update(status='INTAKE_BLOCKED',failure_reason=f'{type(exc).__name__}: {exc}')
    save_json(RUN/'source_receipts'/f'{name}.json',record)
    return record


if __name__=='__main__':
    initialize()
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(intake,item) for item in SOURCES.items()]):
            record=future.result(); records.append(record)
            print(record['source'],record['status'],record.get('commit'),flush=True)
            save_json(RUN/'SOURCE_INTAKE_STATE.json',records)
    fields=['source','repository','commit','license','environment','adapter','weight_id','weight_hash','status']
    with (RUN/'EPS_EXTERNAL_SOURCE_MANIFEST.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
