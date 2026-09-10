"""One GPU process at a time, after preceding base/replay/maintenance owners."""
from datetime import datetime,timezone
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha

if __name__=='__main__':
    state=RUN/'ADDITIONAL_FOUNDATION_QUEUE.json';dependency=RUN/'GPU_MAINTENANCE_QUEUE.json'
    if state.exists():
        if '--resume-preimport-failure' not in sys.argv:raise RuntimeError('Queue already exists; explicit resume only')
        if (RUN/'ADDITIONAL_FOUNDATION_PRESCORE_FREEZE.json').exists():raise RuntimeError('Not a pre-import failure')
        old=json.loads(state.read_text(encoding='utf-8'))
        if not all(r.get('exit_code')==1 and 'SyntaxError' in (RUN/r['log']).read_text(encoding='utf-8') for r in old['steps']):raise RuntimeError('Different failure needs separate review')
        archive=RUN/'audit_corrections/additional_preimport_syntax_failure';archive.mkdir(parents=True,exist_ok=False)
        shutil.copy2(state,archive/state.name)
        for r in old['steps']:shutil.copy2(RUN/r['log'],archive/Path(r['log']).name)
    save_json(state,{'status':'WAITING_FOR_RESOURCE_OWNER','dependency':dependency.name,'steps':[]})
    while not dependency.exists() or json.loads(dependency.read_text(encoding='utf-8'))['status']!='COMPLETE':
        if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-08T01:38:26+00:00'):raise RuntimeError('Finalization deadline')
        time.sleep(15)
    records=[]
    for action in ['score','replay']:
        command=[str(PROJECT.parent/'.venv_eps_extra_py312/Scripts/python.exe'),'-B',str(PROJECT/'research/eps_model_lab_v1/additional_foundation_wave.py'),action]
        log=RUN/'rerun_logs'/f'additional_foundation_{action}.log';r={'action':action,'command':command,'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat()};records.append(r)
        with log.open('w',encoding='utf-8') as stream:
            p=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,PYTHONUTF8='1'),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            r['pid']=p.pid;save_json(state,{'status':'RUNNING','steps':records});code=p.wait()
        r.update(exit_code=code,status='PROCESS_FINISHED' if code==0 else 'NONZERO_EXIT_REQUIRES_REVIEW',log=str(log.relative_to(RUN)),log_sha256=sha(log),end_utc=datetime.now(timezone.utc).isoformat())
        save_json(state,{'status':'RUNNING','steps':records})
    save_json(state,{'status':'COMPLETE','steps':records,'end_utc':datetime.now(timezone.utc).isoformat()})
