"""One explicit API serialization recovery; never retry a statistical recipe."""
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
    state=RUN/'TABPFN_API_RECOVERY_QUEUE.json';dependency=RUN/'ADDITIONAL_FOUNDATION_QUEUE.json'
    if state.exists():raise RuntimeError('Recovery queue already exists')
    save_json(state,{'status':'WAITING_FOR_RESOURCE_OWNER','dependency':dependency.name,'steps':[]})
    while json.loads(dependency.read_text(encoding='utf-8'))['status']!='COMPLETE':time.sleep(15)
    records=[]
    for action in ['retry-tabpfn-pathstring','replay-tabpfn']:
        command=[str(PROJECT.parent/'.venv_eps_extra_py312/Scripts/python.exe'),'-B',str(PROJECT/'research/eps_model_lab_v1/additional_foundation_wave.py'),action]
        log=RUN/'rerun_logs'/f'tabpfn_api_{action}.log';r={'action':action,'status':'RUNNING','command':command,'start_utc':datetime.now(timezone.utc).isoformat()};records.append(r)
        with log.open('w',encoding='utf-8') as stream:
            p=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,PYTHONUTF8='1'),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            r['pid']=p.pid;save_json(state,{'status':'RUNNING','steps':records});code=p.wait()
        r.update(exit_code=code,status='PROCESS_FINISHED' if code==0 else 'NONZERO_EXIT_REQUIRES_REVIEW',log=str(log.relative_to(RUN)),log_sha256=sha(log))
        save_json(state,{'status':'RUNNING','steps':records})
    save_json(state,{'status':'COMPLETE','steps':records,'end_utc':datetime.now(timezone.utc).isoformat()})
