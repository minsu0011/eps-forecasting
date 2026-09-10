"""Wait for single-GPU base queue, then extra breadth and independent audits."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha

STEPS=[('gpu','extra_architecture_wave.py',[]),
 ('gpu','batch_causality_audit.py',['neuralforecast','native','multivariate']),
 ('gpu','reload_neural_audit.py',['neuralforecast','native','multivariate']),
 ('ts','reload_autogluon_audit.py',[]),('ts','batch_causality_audit.py',['autogluon'])]

if __name__=='__main__':
    path=RUN/'GPU_POST_QUEUE.json'
    if path.exists():raise RuntimeError('Post queue exists; inspect/resume explicitly')
    save_json(path,{'status':'WAITING_FOR_BASE_GPU_QUEUE','steps':[]})
    while json.loads((RUN/'RERUN_QUEUE_gpu.json').read_text(encoding='utf-8'))['status']!='COMPLETE':
        if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-08T01:38:26+00:00'):raise RuntimeError('Finalization deadline')
        time.sleep(15)
    records=[]
    for index,(env,script,args) in enumerate(STEPS):
        command=[str(PROJECT.parent/f'.venv_eps_{env}_py312/Scripts/python.exe'),'-B',str(PROJECT/'research/eps_model_lab_v1'/script),*args]
        logfile=RUN/'rerun_logs'/f'post_gpu_{index:02d}_{Path(script).stem}.log'
        record={'index':index,'command':command,'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat(),'log':str(logfile.relative_to(RUN))}
        records.append(record);save_json(path,{'status':'RUNNING','steps':records})
        with logfile.open('w',encoding='utf-8') as out:
            p=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,PYTHONUTF8='1',EPS_LAB_RUN_DIR=str(RUN)),
               stdout=out,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            record['pid']=p.pid;save_json(path,{'status':'RUNNING','steps':records});code=p.wait()
        record.update(status='PROCESS_FINISHED' if code==0 else 'NONZERO_EXIT_REQUIRES_AUDIT',exit_code=code,
           end_utc=datetime.now(timezone.utc).isoformat(),log_sha256=sha(logfile))
        save_json(path,{'status':'RUNNING','steps':records});print('POST_GPU',script,'EXIT',code,flush=True)
    save_json(path,{'status':'COMPLETE','steps':records,'end_utc':datetime.now(timezone.utc).isoformat()})
