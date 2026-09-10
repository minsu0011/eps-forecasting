"""Actual saved-recipe API replay for local models; local fits use only past context."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,TARGETS


def task(name,ticker,kind):
    try:
        a=EPSAdapter.load(RUN/'recipe_adapters'/f'{name}.pkl');a.get_metadata()
        f=dataset();f=f[(f.ticker==ticker)&(f.asof_date.dt.year>=2019)]
        expected=pd.read_parquet(RUN/'predictions'/f'{name}.parquet');maximum=0.;exact=True
        path=a.forecast(f)[0] if kind!='baselines' else None
        for target in TARGETS:
            point=a.predict(f,target) if kind=='baselines' else (path.sum(axis=1) if target=='ttm' else path[:,int(target[1:])-1])
            old=expected[expected.target_key==target].set_index('sample_id').loc[f.sample_id].predicted_eps.to_numpy()
            np.testing.assert_allclose(point,old,rtol=1e-12,atol=1e-12,equal_nan=True)
            exact &= np.array_equal(point,old,equal_nan=True)
            ok=np.isfinite(point)&np.isfinite(old)
            if ok.any():maximum=max(maximum,float(np.max(np.abs(point[ok]-old[ok]))))
        # Actual common predict() C-lane entry point, not only a helper.
        probe=f.iloc[:1];api=a.predict(probe,'ttm')
        old=expected[expected.target_key=='ttm'].set_index('sample_id').loc[probe.sample_id].predicted_eps.to_numpy()
        np.testing.assert_allclose(api,old,rtol=1e-12,atol=1e-12,equal_nan=True)
        return {'model_id':name,'ticker':ticker,'status':'PASS','rows_per_target':len(f),
           'exact':bool(exact),'maximum_difference':maximum,'common_predict_TTM_API':'PASS'}
    except Exception as exc:return {'model_id':name,'ticker':ticker,'status':'FAIL','reason':str(exc),'traceback':traceback.format_exc()}


def run(kind):
    if kind=='stats':
        from research.eps_model_lab_v1.statistical_models import StatisticalAdapter,SF_NAMES,ACCOUNTING
        names=['sf_'+n for n in SF_NAMES]+list(ACCOUNTING)
        adapters=[StatisticalAdapter(n).fit() for n in names]
    elif kind=='baselines':
        from research.eps_model_lab_v1.core_models import BaselineAdapter,BASELINES
        names=BASELINES;adapters=[BaselineAdapter(n).fit(None,None) for n in names]
    else:
        from research.eps_model_lab_v1.darts_models import DartsAdapter,SPEC
        names=['darts_'+n for n in SPEC];adapters=[DartsAdapter(n).fit() for n in SPEC]
    for a in adapters:
        path=RUN/'recipe_adapters'/f'{a.model_id}.pkl'
        if not path.exists():a.save(path)
    records=[];tickers=sorted(dataset().ticker.unique());jobspec=[(name,ticker,kind) for name in names for ticker in tickers]
    with ProcessPoolExecutor(max_workers=2 if kind=='darts' else 4) as pool:
        jobs=[pool.submit(task,*t) for t in jobspec]
        for future in as_completed(jobs):
            records.append(future.result())
            if len(records)%25==0 or records[-1]['status']=='FAIL':
                save_json(RUN/'classical_replay_audits'/f'{kind}.json',{'completed':len(records),'expected':len(jobspec),
                  'all_pass':all(r['status']=='PASS' for r in records),'records':records})
                print('CLASSICAL_REPLAY',kind,len(records),'OF',len(jobspec),'FAIL',sum(r['status']=='FAIL' for r in records),flush=True)
    save_json(RUN/'classical_replay_audits'/f'{kind}.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
       'completed':len(records),'expected':len(jobspec),'all_pass':all(r['status']=='PASS' for r in records),
       'records':records,'recipe_hashes':{n:sha(RUN/'recipe_adapters'/f'{n}.pkl') for n in names},
       'scope':'Saved local forecast recipe reload and full native-origin replay. Local statistical model fit is computed from each past context at predict time; this is not a stored global trained head.',
       'rtol':1e-12,'atol':1e-12})


if __name__=='__main__':
    for kind in sys.argv[1:]:run(kind)
