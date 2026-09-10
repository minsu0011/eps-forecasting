"""Reload all trained neural weights and compare every original forecast row."""
import argparse
from datetime import datetime,timezone
import gc
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OMP_NUM_THREADS']='4'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset


def run(family):
    frame=dataset();records=[];torch.set_num_threads(4)
    if family=='neuralforecast':
        from research.eps_model_lab_v1.neural_models import NeuralAdapter
        tasks=[(p.parent.name,int(p.name),p) for p in sorted((RUN/'neural_fitted').glob('*/*')) if p.is_dir()]
    elif family=='native':
        from research.eps_model_lab_v1.eps_native_models import NativeAdapter
        tasks=[('epspredict_'+p.parent.name,int(p.stem),p) for p in sorted((RUN/'native_fitted').glob('*/*.pt'))]
    else:
        from research.eps_model_lab_v1.multivariate_models import MultivariateAdapter,tensors,to_device
        data=to_device(tensors(frame))
        tasks=[(p.parent.name,int(p.stem),p) for p in sorted((RUN/'multivariate_fitted').glob('*/*.pt'))]
    for model,year,path in tasks:
        start=time.perf_counter()
        try:
            val=frame[frame.asof_date.dt.year==year];old=pd.read_parquet(RUN/'predictions'/f'{model}.parquet')
            if family=='neuralforecast': a=NeuralAdapter.load(path);p=a.path(val);ttm=p.sum(axis=1)
            elif family=='native': a=NativeAdapter.load(path);p=a.path(val);ttm=p[:,4]
            else:
                a=MultivariateAdapter.load(path);indices=np.flatnonzero((frame.asof_date.dt.year==year).to_numpy())
                full=a.path(data,indices);p=full[:,:,0];ttm=full[:,3,1]
            a.get_metadata()
            diffs=[];exact=True
            for target,j in [('h1',0),('h2',1),('h3',2),('h4',3),('ttm',4)]:
                actual=ttm if target=='ttm' else p[:,j]
                expected=old[old.target_key==target].set_index('sample_id').loc[val.sample_id].predicted_eps.to_numpy()
                # Fixed FP32 numerical gate; exact equality is additionally reported.
                np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
                exact=exact and np.array_equal(actual,expected);diffs.append(float(np.max(np.abs(actual-expected))))
            files=[path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
            record={'model_id':model,'year':year,'status':'PASS','exact_all_targets':exact,'maximum_absolute_difference':max(diffs),
               'rows_per_target':len(val),'metadata_api':'PASS','weights':{str(p.relative_to(RUN)):sha(p) for p in files},'seconds':time.perf_counter()-start}
            del a
        except Exception as exc: record={'model_id':model,'year':year,'status':'FAIL','reason':str(exc),'traceback':traceback.format_exc()}
        records.append(record);save_json(RUN/'reload_audits'/f'{family}.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
            'expected':len(tasks),'completed':len(records),'all_pass':all(r['status']=='PASS' for r in records),'records':records,
            'rtol':2e-5,'atol':2e-5,'scope':'Actual locally trained weights, all intended prediction origins; no refit'})
        print('NEURAL_RELOAD',family,model,year,record['status'],len(records),'OF',len(tasks),flush=True)
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('families',nargs='*');args=parser.parse_args()
    for family in args.families or ['neuralforecast','native','multivariate']: run(family)
