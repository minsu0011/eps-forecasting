"""Persistent CPU probabilistic score/replay sequence, no GPU contention."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha

if __name__=='__main__':
    state=RUN/'PROBABILISTIC_QUEUE.json';records=[]
    if state.exists():raise RuntimeError('Queue exists; explicit resume required')
    for action in ['score','replay']:
        command=[str(PROJECT.parent/'.venv_eps_prob_py312/Scripts/python.exe'),'-B',str(PROJECT/'research/eps_model_lab_v1/probabilistic_wave.py'),action]
        log=RUN/'rerun_logs'/f'probabilistic_{action}.log';r={'action':action,'command':command,'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat()};records.append(r)
        with log.open('w',encoding='utf-8') as stream:
            p=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,PYTHONUTF8='1'),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            r['pid']=p.pid;save_json(state,{'status':'RUNNING','steps':records});code=p.wait()
        r.update(exit_code=code,status='PROCESS_FINISHED' if code==0 else 'NONZERO_EXIT_REQUIRES_REVIEW',log=str(log.relative_to(RUN)),log_sha256=sha(log),end_utc=datetime.now(timezone.utc).isoformat())
        save_json(state,{'status':'RUNNING','steps':records})
        if code!=0:save_json(state,{'status':'FAILED','steps':records});raise SystemExit(code)
    save_json(state,{'status':'COMPLETE','steps':records,'end_utc':datetime.now(timezone.utc).isoformat()})
