"""Independent CPU/GPU replay queues; wait for existing resource owner."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha

if __name__=='__main__':
    lane=sys.argv[1]
    steps=[('gpu','primary_gpu'),('ts','toto'),('granite','granite')] if lane=='gpu' else [('moirai','moirai'),('moirai','lag'),('granite','tirex2')]
    dependency=RUN/('GPU_POST_QUEUE.json' if lane=='gpu' else 'RERUN_QUEUE_legacy_cpu.json')
    path=RUN/f'FOUNDATION_REPLAY_QUEUE_{lane}.json'
    if path.exists():raise RuntimeError('Replay queue exists; inspect/resume explicitly')
    save_json(path,{'status':'WAITING_FOR_RESOURCE_OWNER','dependency':dependency.name,'steps':[]})
    while not dependency.exists() or json.loads(dependency.read_text(encoding='utf-8'))['status']!='COMPLETE':
        if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-08T01:38:26+00:00'):raise RuntimeError('Deadline while waiting')
        time.sleep(15)
    records=[]
    for index,(environment,group) in enumerate(steps):
        command=[str(PROJECT.parent/f'.venv_eps_{environment}_py312/Scripts/python.exe'),'-B',str(PROJECT/'research/eps_model_lab_v1/foundation_replay.py'),group]
        logfile=RUN/'rerun_logs'/f'foundation_replay_{lane}_{index}_{group}.log'
        record={'group':group,'command':command,'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat(),'log':str(logfile.relative_to(RUN))};records.append(record)
        with logfile.open('w',encoding='utf-8') as stream:
            p=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,PYTHONUTF8='1',EPS_LAB_RUN_DIR=str(RUN)),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            record['pid']=p.pid;save_json(path,{'status':'RUNNING','steps':records});code=p.wait()
        record.update(exit_code=code,status='PROCESS_FINISHED' if code==0 else 'NONZERO_EXIT_REQUIRES_AUDIT',
            end_utc=datetime.now(timezone.utc).isoformat(),log_sha256=sha(logfile))
        save_json(path,{'status':'RUNNING','steps':records});print('FOUNDATION_REPLAY_QUEUE',lane,group,'EXIT',code,flush=True)
    save_json(path,{'status':'COMPLETE','steps':records,'end_utc':datetime.now(timezone.utc).isoformat()})
