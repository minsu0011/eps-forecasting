"""Bounded one-GPU TimeLLM fit and actual fresh saved-artifact replay."""
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha


if __name__=='__main__':
    root=RUN/'late_frontier/TimeLLM';receipt=root/'QUEUE.json'
    if receipt.exists():raise RuntimeError('Queue already exists; review without overwriting')
    python=PROJECT.parent/'.venv_eps_gpu_py312/Scripts/python.exe'
    script=PROJECT/'research/eps_model_lab_v1/late_timellm.py'
    state={'status':'RUNNING','created_utc':datetime.now(timezone.utc).isoformat(),'steps':[]}
    save_json(receipt,state)
    for stage,extra,timeout in [('fit',[],2400),('replay',['--replay'],900)]:
        command=[str(python),'-B',str(script),*extra];log=root/f'{stage}.log'
        with log.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,PYTHONUTF8='1'),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            state['running_pid']=process.pid;save_json(receipt,state)
            try:code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate();code=process.wait(timeout=30);state['bounded_timeout']=True
        state['steps'].append({'stage':stage,'command':command,'exit_code':code,'log_sha256':sha(log),
            'completed_utc':datetime.now(timezone.utc).isoformat()})
        save_json(receipt,state)
        if code or state.get('bounded_timeout'):
            state.update(status='FAILED_REVIEW_REQUIRED',running_pid=None);save_json(receipt,state)
            raise RuntimeError('Bounded TimeLLM '+stage+' failed; inspect preserved log')
    state.update(status='FIT_AND_FRESH_REPLAY_COMPLETE',running_pid=None);save_json(receipt,state)
    print('TIMELLM_BOUNDED_QUEUE_COMPLETE',flush=True)
