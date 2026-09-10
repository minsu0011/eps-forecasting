"""Bounded, explicit failed-smoke retry after the current GPU resource owner."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha


def gpu_work():
    import torch
    from research.eps_model_lab_v1 import multivariate_models as m
    from research.eps_model_lab_v1.extra_architecture_specs import EXTRA_MULTIVARIATE
    from research.eps_model_lab_v1.common import dataset
    frozen=json.loads((RUN/'EXTRA_ARCHITECTURE_PRESCORE_FREEZE.json').read_text(encoding='utf-8'))
    if frozen['multivariate']!=EXTRA_MULTIVARIATE:raise RuntimeError('Extra specification drift')
    old=RUN/'model_receipts/nf_multivar_StemGNN.json'
    if json.loads(old.read_text(encoding='utf-8'))['status']!='BROKEN':raise RuntimeError('Not an explicit failed candidate retry')
    if (RUN/'predictions/nf_multivar_StemGNN.parquet').exists():raise RuntimeError('Never overwrite scored prediction')
    archive=RUN/'audit_corrections/stemgnn_initial_stride_failure'
    archive.mkdir(parents=True,exist_ok=False);shutil.copy2(old,archive/old.name)
    initial=RUN/'multivariate_smoke_fitted/nf_multivar_StemGNN/2019.pt'
    if initial.exists():shutil.copy2(initial,archive/'initial_smoke_2019.pt')
    torch.set_num_threads(4);m.SPECS={**m.SPECS,**EXTRA_MULTIVARIATE}
    frame=dataset();data=m.to_device(m.tensors(frame));began=time.perf_counter()
    m.run('StemGNN',frame,data,True);m.run('StemGNN',frame,data,False)
    save_json(RUN/'STEMGNN_STRIDE_RECOVERY.json',{'status':'RECOVERED_WITH_UNCHANGED_PRESCORE_RECIPE',
      'seconds':time.perf_counter()-began,'old_failure_archive':str(archive.relative_to(RUN)),
      'change':'np.array(copy=True,order=C) owns a length-one reversed index; ascontiguousarray may retain its negative stride',
      'adapter_sha256':sha(PROJECT/'research/eps_model_lab_v1/multivariate_models.py'),
      'test_scores_consulted':False,'first_full_fit_for_this_candidate':True})


if __name__=='__main__':
    if '--gpu-work' in sys.argv:gpu_work();raise SystemExit(0)
    state=RUN/'GPU_MAINTENANCE_QUEUE.json'
    if state.exists():raise RuntimeError('Maintenance queue exists; no implicit duplicate')
    dependency=RUN/'FOUNDATION_REPLAY_QUEUE_gpu.json'
    save_json(state,{'status':'WAITING_FOR_RESOURCE_OWNER','dependency':dependency.name,'steps':[]})
    while not dependency.exists() or json.loads(dependency.read_text(encoding='utf-8'))['status']!='COMPLETE':
        if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-08T01:38:26+00:00'):raise RuntimeError('Finalization deadline')
        time.sleep(15)
    steps=[('gpu_maintenance_followup.py',['--gpu-work']),('batch_numerical_review.py',[]),
       ('batch_causality_audit.py',['multivariate']),('reload_neural_audit.py',['multivariate'])]
    archive=RUN/'audit_corrections/before_stemgnn_recovery';archive.mkdir(parents=True,exist_ok=False)
    for directory in ['batch_causality_audits','reload_audits']:
        path=RUN/directory/'multivariate.json'
        if path.exists():shutil.copy2(path,archive/(directory+'_multivariate.json'))
    records=[]
    for i,(script,args) in enumerate(steps):
        command=[str(PROJECT.parent/'.venv_eps_gpu_py312/Scripts/python.exe'),'-B',str(PROJECT/'research/eps_model_lab_v1'/script),*args]
        log=RUN/'rerun_logs'/f'gpu_maintenance_{i}_{Path(script).stem}.log'
        record={'script':script,'command':command,'status':'RUNNING','start_utc':datetime.now(timezone.utc).isoformat()};records.append(record)
        with log.open('w',encoding='utf-8') as stream:
            p=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,PYTHONUTF8='1'),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            record['pid']=p.pid;save_json(state,{'status':'RUNNING','steps':records});code=p.wait()
        record.update(exit_code=code,status='PROCESS_FINISHED' if code==0 else 'NONZERO_EXIT_REQUIRES_REVIEW',log=str(log.relative_to(RUN)),log_sha256=sha(log))
        save_json(state,{'status':'RUNNING','steps':records})
    save_json(state,{'status':'COMPLETE','steps':records,'end_utc':datetime.now(timezone.utc).isoformat()})
