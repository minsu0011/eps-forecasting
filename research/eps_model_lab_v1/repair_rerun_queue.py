"""Resource-separated fixed-recipe rerun after the independently discovered PIT repair.

This is ordinary local batch execution, not agent delegation. One GPU queue;
one CPU pool queue; one small-thread legacy foundation queue. All subprocess
output is retained in this new run, with exact commands and exit codes.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha

NF=['NHITS','NBEATSx','TFT','DLinear','PatchTST','GRU','TCN']
QUEUES={
 'cpu':[
  ('core','core_models.py',['--baselines']),('core','core_models.py',['--smoke']),('core','core_models.py',[]),
  ('core','mlforecast_models.py',['--smoke']),('core','mlforecast_models.py',[]),
  ('core','statistical_models.py',['--smoke']),('core','statistical_models.py',[]),
  ('core','annual_accounting.py',[]),('ts','darts_models.py',['FFT','FourTheta','KalmanForecaster'])],
 'gpu':[
  ('gpu','foundation_models.py',['chronos2_zero_shot','chronos2_past_covariates','chronos_bolt_mini','timesfm_2p5','tirex_zero_shot']),
  ('ts','foundation_models.py',['toto2_4m','toto2_22m']),
  ('gpu','neural_models.py',[*NF,'--smoke']),('gpu','neural_models.py',NF),('gpu','neural_breadth_wave.py',[]),
  ('gpu','eps_native_models.py',['LSTM','CNNLSTM','MultiFreqLSTM','MultiFreqCNNLSTM','--smoke']),
  ('gpu','eps_native_models.py',['LSTM','CNNLSTM','MultiFreqLSTM','MultiFreqCNNLSTM']),
  ('gpu','multivariate_models.py',[]),('gpu','chronos_finetune.py',[]),
  ('granite','additional_foundation.py',['time_moe_50m','moment_embedding_ridge']),('granite','ibm_patchtst_model.py',[]),
  ('ts','autogluon_models.py',['--smoke']),('ts','autogluon_models.py',[])],
 'legacy_cpu':[
  ('moirai','lag_llama_model.py',[]),('granite','tirex2_models.py',[]),
  ('moirai','moirai_models.py',['moirai2_small','moirai1p1_small','moirai_moe_small'])],
}


if __name__=='__main__':
    lane=sys.argv[1];steps=QUEUES[lane]
    audit=json.loads((RUN/'SEC_FUTURE_SNAPSHOT_TRUNCATION_AUDIT.json').read_text(encoding='utf-8'))
    if not audit['all_pass'] or audit['completed_tickers']!=audit['expected_tickers']:
        raise RuntimeError('Full-cohort prefix audit must pass before any repaired-data score')
    if audit['samples_sha256_before']!=sha(RUN/'data/samples.parquet') or audit['samples_sha256_after']!=audit['samples_sha256_before']:
        raise RuntimeError('Prefix audit does not bind the current dataset hash')
    if not (RUN/'EPS_DATA_GEOMETRY_LOCK_V1_3.json').exists(): raise RuntimeError('Native-context calendar lock missing')
    root=RUN/'rerun_logs';root.mkdir(exist_ok=True);records=[]
    statepath=RUN/f'RERUN_QUEUE_{lane}.json'
    resume=int(sys.argv[sys.argv.index('--resume-from')+1]) if '--resume-from' in sys.argv else 0
    if statepath.exists():
        if '--resume-from' not in sys.argv: raise RuntimeError('Queue exists; explicit inspected --resume-from required')
        prior=json.loads(statepath.read_text(encoding='utf-8'))
        save_json(root/f'{lane}_queue_before_resume_{datetime.now(timezone.utc).strftime("%H%M%S")}.json',prior)
        records=prior['steps']
        if lane=='cpu' and resume==3 and not (RUN/'CORE_SAVED_HEAD_RECOVERY.json').exists():
            raise RuntimeError('Core saved-head recovery required before skipping failed collector')
    for index,(environment,script,args) in enumerate(steps):
        if index<resume: continue
        if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-08T01:38:26+00:00'):
            records.append({'step':index,'script':script,'status':'NOT_STARTED_FINALIZATION_DEADLINE'});break
        # AutoGluon's internal CPU models should not compete with the 20-worker CPU queue.
        if script=='autogluon_models.py':
            other=RUN/'RERUN_QUEUE_cpu.json'
            while not other.exists() or json.loads(other.read_text(encoding='utf-8')).get('status')!='COMPLETE':
                if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-08T01:38:26+00:00'): raise RuntimeError('CPU queue wait reached finalization deadline')
                time.sleep(15)
        interpreter=PROJECT.parent/f'.venv_eps_{environment}_py312/Scripts/python.exe'
        command=[str(interpreter),'-B',str(PROJECT/'research/eps_model_lab_v1'/script),*args]
        logfile=root/f'{lane}_{index:02d}_{Path(script).stem}.log'
        env=dict(os.environ,EPS_LAB_RUN_DIR=str(RUN),PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        record={'step':index,'environment':environment,'script':script,'arguments':args,'command':command,
                 'start_utc':datetime.now(timezone.utc).isoformat(),'status':'RUNNING','log':str(logfile.relative_to(RUN))}
        records.append(record);save_json(statepath,{'lane':lane,'status':'RUNNING','steps':records,'total_steps':len(steps)})
        began=time.perf_counter();print('RERUN_START',lane,index,script,' '.join(args),flush=True)
        with logfile.open('w',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=PROJECT,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            record['pid']=process.pid;save_json(statepath,{'lane':lane,'status':'RUNNING','steps':records,'total_steps':len(steps)})
            code=process.wait()
        record.update(exit_code=code,status='PROCESS_FINISHED' if code==0 else 'NONZERO_EXIT_REQUIRES_ARTIFACT_AUDIT',
           seconds=time.perf_counter()-began,end_utc=datetime.now(timezone.utc).isoformat(),log_sha256=sha(logfile))
        save_json(statepath,{'lane':lane,'status':'RUNNING','steps':records,'total_steps':len(steps)})
        print('RERUN_END',lane,index,script,'EXIT',code,round(record['seconds'],1),flush=True)
    save_json(statepath,{'lane':lane,'status':'COMPLETE','steps':records,'total_steps':len(steps),
       'process_failures':[r for r in records if r.get('exit_code',0)!=0],'finished_utc':datetime.now(timezone.utc).isoformat()})
