"""Add an isolated xLSTM-capable environment; existing locked environments stay intact."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha


def run():
    envdir=(PROJECT.parent/'.venv_eps_frontier_py312').resolve()
    if envdir.exists() or envdir.parent!=PROJECT.parent.resolve():raise RuntimeError('Expected specifically named new environment')
    if shutil.disk_usage(PROJECT).free<40*1024**3:raise RuntimeError('Insufficient disk headroom')
    root=RUN/'late_frontier/environment';root.mkdir(parents=True,exist_ok=True)
    lock=RUN/'environments/gpu_requirements_lock.txt'
    uv=shutil.which('uv');python=envdir/'Scripts/python.exe'
    state={'status':'INSTALLING','created_utc':datetime.now(timezone.utc).isoformat(),
        'base_lock_sha256':sha(lock),'existing_environments_modified':False,'environment':str(envdir),'steps':[]}
    receipt=root/'INSTALLATION.json';save_json(receipt,state)
    commands=[[uv,'venv',str(envdir),'--python',sys.executable],
        [uv,'pip','sync','--python',str(python),str(lock),'--extra-index-url','https://download.pytorch.org/whl/cu128','--index-strategy','unsafe-best-match'],
        [uv,'pip','install','--python',str(python),'xlstm==2.0.6','mlstm-kernels==2.0.0','--constraint',str(lock)],
        [uv,'pip','check','--python',str(python)],
        [str(python),'-B','-c',"import torch;import neuralforecast.models.xlstm as m;assert m.IS_XLSTM_INSTALLED;print(torch.__version__,torch.cuda.is_available(),m.IS_XLSTM_INSTALLED)"]]
    for index,command in enumerate(commands):
        log=root/f'install_{index}.log';start=datetime.now(timezone.utc).isoformat()
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,PYTHONUTF8='1'),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            state['running_pid']=process.pid;save_json(receipt,state)
            try:code=process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                process.terminate();code=process.wait(timeout=30);state['bounded_timeout']=True
        state['steps'].append({'command':command,'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'exit_code':code,'log':str(log.relative_to(RUN)),'sha256':sha(log)})
        save_json(receipt,state)
        if code or state.get('bounded_timeout'):
            state['status']='FAILED_REVIEW_REQUIRED';save_json(receipt,state);raise RuntimeError('Environment step failed '+str(index))
    frozen=subprocess.run([uv,'pip','freeze','--python',str(python)],capture_output=True,text=True,check=True).stdout
    (root/'requirements_lock.txt').write_text(frozen,encoding='utf-8')
    expected=set(lock.read_text(encoding='utf-8').lower().splitlines());observed=set(frozen.lower().splitlines())
    if expected-observed:raise RuntimeError('Base pinned package changed')
    state.update(status='PASS_ISOLATED_PINNED_ENVIRONMENT',running_pid=None,base_unchanged=True,
        extra_packages=sorted(observed-expected),lock_sha256=sha(root/'requirements_lock.txt'))
    save_json(receipt,state);print(state['status'],state['extra_packages'],flush=True)


if __name__=='__main__':run()
