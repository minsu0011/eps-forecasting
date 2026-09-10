"""Recover all forecasts from completed heads after collector JSON lock failure.

No retraining and no copied pre-repair checkpoints. Only already-complete files
inside the new data-repair run are accepted. Missing or inconsistent heads fail.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
import time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,split_train,prediction_frame,save_predictions,evaluate_all,TARGETS
from research.eps_model_lab_v1.core_models import CORE_SPECS


def recover(task):
    model,year,target=task;start=time.perf_counter()
    path=RUN/'fitted_models'/model/f'{year}_{target}.pkl'
    if not path.exists():raise RuntimeError('Required completed head missing '+str(path))
    train,valid=split_train(dataset(),year,target);a=EPSAdapter.load(path)
    if a.model_id!=model or a.target!=target or a.spec!=CORE_SPECS[model]:raise RuntimeError('Saved head identity mismatch')
    pred=a.predict(valid,target)
    np.testing.assert_array_equal(pred,EPSAdapter.load(path).predict(valid,target))
    if not np.isfinite(pred).all():raise RuntimeError('Nonfinite restored forecast')
    receipt={'model_id':model,'year':year,'target':target,'train_rows':len(train),'prediction_rows':len(valid),
       'train_max_origin':str(train.asof_date.max()),'train_max_label_asof':str(train['label_asof_'+target].max()),
       'fit_cutoff':f'{year}-01-01T00:00:00Z','status':'RECOVERED_FROM_COMPLETED_NEW_RUN_HEAD',
       'head_sha256':sha(path),'prediction_reload_seconds':time.perf_counter()-start,'training_seconds_unavailable_collector_interrupted':True,
       'actual_double_reload':'EXACT_PASS','no_refit':True}
    return prediction_frame(valid,target,model,pred),receipt


if __name__=='__main__':
    start=time.perf_counter();results={m:[] for m in CORE_SPECS};records={m:[] for m in CORE_SPECS}
    tasks=[(m,y,t) for m in CORE_SPECS for y in range(2019,2027) for t in TARGETS]
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(recover,t) for t in tasks]
        for future in as_completed(jobs):
            frame,receipt=future.result();model=receipt['model_id'];results[model].append(frame);records[model].append(receipt)
    for model in CORE_SPECS:
        if len(records[model])!=40:raise RuntimeError('Incomplete head set')
        save_predictions(model,results[model],{'family':'TABULAR_ACCOUNTING','spec':CORE_SPECS[model],
           'smoke_status':'PASS','folds':records[model],'runtime_seconds':None,
           'recovery':'Predictions generated from 40 completed new-run heads after collector atomic JSON replace failed; no refit',
           'original_process_log':'rerun_logs/cpu_02_core_models.log'})
        print('CORE_HEAD_RECOVERY',model,40,'PASS',flush=True)
    save_json(RUN/'CORE_SAVED_HEAD_RECOVERY.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
       'status':'PASS','heads':440,'models':records,'seconds':time.perf_counter()-start,
       'dataset_sha256':sha(RUN/'data/samples.parquet'),'new_training_runs':0})
    evaluate_all()
