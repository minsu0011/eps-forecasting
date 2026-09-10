"""Three bounded CPU lanes, four threads each; fresh replay follows each score."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import os
import json
import shutil
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha


def one(base):
    records=[];state=RUN/'origin_isolation_queues'/f'{base}.json'
    if state.exists():
        if '--resume-preimport-failure' not in sys.argv:raise RuntimeError('Isolation queue already exists')
        if (RUN/'origin_isolation_specs'/f'{base}_origin_isolated.json').exists():raise RuntimeError('Not a pre-import failure')
        old=json.loads(state.read_text(encoding='utf-8'))
        if not all(r.get('exit_code')==1 and 'SyntaxError' in (RUN/r['log']).read_text(encoding='utf-8') for r in old['steps']):raise RuntimeError('Different failure needs review')
        archive=RUN/'audit_corrections/origin_preimport_syntax_failure'/base;archive.mkdir(parents=True,exist_ok=False)
        shutil.copy2(state,archive/state.name)
        for r in old['steps']:shutil.copy2(RUN/r['log'],archive/Path(r['log']).name)
    for action in ['score','replay']:
        command=[str(PROJECT.parent/'.venv_eps_moirai_py312/Scripts/python.exe'),'-B',str(PROJECT/'research/eps_model_lab_v1/origin_isolated_foundations.py'),action,base]
        log=RUN/'rerun_logs'/f'origin_isolated_{base}_{action}.log';r={'action':action,'command':command,'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat()};records.append(r)
        with log.open('w',encoding='utf-8') as stream:
            p=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,PYTHONUTF8='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            r['pid']=p.pid;save_json(state,{'status':'RUNNING','steps':records});code=p.wait()
        r.update(status='PROCESS_FINISHED' if code==0 else 'NONZERO_EXIT',exit_code=code,log=str(log.relative_to(RUN)),log_sha256=sha(log),end_utc=datetime.now(timezone.utc).isoformat())
        save_json(state,{'status':'RUNNING' if code==0 else 'FAILED','steps':records})
        if code!=0:return base,'FAILED'
    save_json(state,{'status':'COMPLETE','steps':records});return base,'COMPLETE'


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,b) for b in ['lag_llama_zero_shot','moirai1p1_small','moirai_moe_small']]):print('ORIGIN_ISOLATION_QUEUE',*f.result(),flush=True)
