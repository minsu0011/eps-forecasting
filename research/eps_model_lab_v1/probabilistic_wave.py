"""Prescore-frozen native probabilistic CPU breadth, then full saved-model replay."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset,split_train,prediction_frame,save_predictions,evaluate_all,features
from research.eps_model_lab_v1.probabilistic_models import ProbabilisticAdapter,SPECS


def worker(task):
    name,year,target,replay=task;began=time.perf_counter();threadpool_limits(1)
    frame=dataset();train,valid=split_train(frame,year,target);path=RUN/'probabilistic_fitted'/name/f'{year}_{target}.pkl'
    if replay:a=ProbabilisticAdapter.load(path)
    else:
        if path.exists():raise RuntimeError('No fitted model overwrite')
        a=ProbabilisticAdapter(name).fit(train,target);a.save(path)
    p,q=a.distribution(valid)
    if not np.isfinite(p).all() or not np.isfinite(q).all():raise RuntimeError('Nonfinite native predictive distribution')
    if (q[:,0]>q[:,1]).any() or (q[:,1]>q[:,2]).any():raise RuntimeError('Native quantile crossing')
    if replay:
        old=pd.read_parquet(RUN/'predictions'/f'{name}.parquet');ref=old[old.target_key==target].set_index('sample_id').loc[valid.sample_id]
        np.testing.assert_allclose(p,ref.predicted_eps,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(q,ref[['p10_eps','p50_eps','p90_eps']],rtol=1e-12,atol=1e-12)
        out=None
    else:out=prediction_frame(valid,target,name,p,time.perf_counter()-began,q,a.metadata)
    record={'model_id':name,'year':year,'target':target,'status':'PASS','training_rows':len(train),'prediction_rows':len(valid),
       'max_training_label_asof':str(train['label_asof_'+target].max()),'cutoff':f'{year}-01-01',
       'seconds':time.perf_counter()-began,'weight_sha256':sha(path),'fresh_process_replay':replay}
    return name,out,record


def smoke(name):
    frame=dataset();train,valid=split_train(frame,2019,'h1');train=train.iloc[:128];valid=valid.iloc[:8]
    a=ProbabilisticAdapter(name,smoke=True).fit(train,'h1');p,q=a.distribution(valid)
    assert np.isfinite(p).all() and np.isfinite(q).all() and q.shape==(8,3)
    path=RUN/'probabilistic_smoke'/f'{name}.pkl';a.save(path);b=ProbabilisticAdapter.load(path)
    pp,qq=b.distribution(valid);np.testing.assert_allclose(p,pp,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(q,qq,rtol=1e-12,atol=1e-12)
    changed=valid.copy()
    for c in changed:
        if c.startswith(('y_','label_')):changed[c]=1e12
    cols=features(changed).columns;changed.loc[changed.index[1:],cols]=changed.loc[changed.index[1:],cols]*3+7
    pp,qq=b.distribution(changed);np.testing.assert_allclose(p[0],pp[0],rtol=1e-12,atol=1e-12);np.testing.assert_allclose(q[0],qq[0],rtol=1e-12,atol=1e-12)
    save_json(RUN/'smoke_receipts'/f'{name}.json',{'status':'PASS','rows':8,'reload':'PASS','other_origin_and_future_label_mutation':'PASS','spec':SPECS[name]})
    print(name,'SMOKE_PASS',flush=True)


def run(action):
    replay=action=='replay';lock=RUN/'PROBABILISTIC_PRESCORE_FREEZE.json'
    if not replay:
        if lock.exists():raise RuntimeError('Probabilistic wave already frozen; explicit resume required')
        save_json(lock,{'specifications':SPECS,'frozen_utc':datetime.now(timezone.utc).isoformat(),'dataset_sha256':sha(RUN/'data/samples.parquet'),
          'test_scores_consumed':False,'selection':'Native probabilistic local-data models underrepresented in first wave, no test-result selection',
          'precision':'FP64_CPU','outer_workers':20,'inner_threads':1,'random_split':False,'early_stopping':False,
          'source_sha256':sha(__file__),'adapter_sha256':sha(PROJECT/'research/eps_model_lab_v1/probabilistic_models.py')})
    names=[]
    for name in SPECS:
        try:
            if replay:
                if not (RUN/'predictions'/f'{name}.parquet').exists():continue
            else:smoke(name)
            names.append(name)
        except Exception as exc:
            save_json(RUN/'model_receipts'/f'{name}.json',{'model_id':name,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
    outputs={n:[] for n in names};records={n:[] for n in names};failures={n:[] for n in names};began=time.perf_counter()
    with ProcessPoolExecutor(max_workers=20) as pool:
        futures={pool.submit(worker,(n,y,t,replay)):(n,y,t) for n in names for y in range(2019,2027) for t in ['h1','h2','h3','h4','ttm']}
        for future in as_completed(futures):
            name,year,target=futures[future]
            try:
                name,out,record=future.result();records[name].append(record)
                if out is not None:outputs[name].append(out)
            except Exception as exc:failures[name].append({'year':year,'target':target,'reason':str(exc),'traceback':traceback.format_exc()})
            save_json(RUN/('probabilistic_replay' if replay else 'probabilistic_audits')/f'{name}.json',
               {'expected':40,'completed':len(records[name])+len(failures[name]),'all_pass':not failures[name],
                'records':records[name],'failures':failures[name],'dataset_sha256':sha(RUN/'data/samples.parquet')})
            print('PROBABILISTIC',action,name,year,target,'COMPLETE',len(records[name])+len(failures[name]),'FAILED',len(failures[name]),flush=True)
    if not replay:
        for name in names:
            if failures[name]:
                save_json(RUN/'model_receipts'/f'{name}.json',{'model_id':name,'status':'BROKEN','failure_reason':'Some full annual heads failed; no partial rows presented as successful full model','failures':failures[name]});continue
            a=ProbabilisticAdapter(name)
            save_predictions(name,outputs[name],{'family':a.metadata['family'],'metadata':a.metadata,'spec':SPECS[name],
              'runtime_seconds':sum(x['seconds'] for x in records[name]),'runtime_scope':'Sum of CPU fold-job walltimes; concurrent throughput differs',
              'wave_walltime_seconds':time.perf_counter()-began,'smoke_status':'PASS','audit':records[name]})
        evaluate_all()


if __name__=='__main__':run(sys.argv[1])
