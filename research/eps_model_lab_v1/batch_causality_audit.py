"""Perturb OTHER forecast origins; a fixed origin's forecast must not change."""
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import sys
import time
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset


def perturb(frame):
    result=frame.copy()
    for c in result:
        if c.startswith(('eps_lag_','filled_lag_','ttm_lag_','account_')):
            result.loc[result.index[1:],c]=result.loc[result.index[1:],c]*3+7
    return result


def audit(family):
    frame=dataset();records=[];torch.set_num_threads(4);years=[2019,2022,2026]
    if family=='neuralforecast':
        from research.eps_model_lab_v1.neural_models import NeuralAdapter
        tasks=[(p.parent.name,int(p.name),p) for p in sorted((RUN/'neural_fitted').glob('*/*')) if p.is_dir() and int(p.name) in years]
    elif family=='native':
        from research.eps_model_lab_v1.eps_native_models import NativeAdapter
        tasks=[('epspredict_'+p.parent.name,int(p.stem),p) for p in sorted((RUN/'native_fitted').glob('*/*.pt')) if int(p.stem) in years]
    elif family=='multivariate':
        from research.eps_model_lab_v1.multivariate_models import MultivariateAdapter,tensors,to_device
        data=to_device(tensors(frame))
        tasks=[(p.parent.name,int(p.stem),p) for p in sorted((RUN/'multivariate_fitted').glob('*/*.pt')) if int(p.stem) in years]
    elif family=='autogluon':
        from research.eps_model_lab_v1.autogluon_models import AutoGluonAdapter,contexts
        tasks=[('AUTOGLUON_ALL',year,RUN/'autogluon_fitted'/str(year)) for year in years]
    else:raise ValueError(family)
    for name,year,path in tasks:
        began=time.perf_counter();val=frame[frame.asof_date.dt.year==year].iloc[:8]
        try:
            singleton=None
            if family=='neuralforecast':
                a=NeuralAdapter.load(path);before=a.path(val)[0];after=a.path(perturb(val))[0];singleton=a.path(val.iloc[:1])[0]
            elif family=='native':
                a=NativeAdapter.load(path);before=a.path(val)[0];after=a.path(perturb(val))[0];singleton=a.path(val.iloc[:1])[0]
            elif family=='multivariate':
                a=MultivariateAdapter.load(path);idx=val.index.to_numpy();changed=dict(data);changed['x']=data['x'].clone()
                changed['x'][idx[1:]]=changed['x'][idx[1:]]*3+7
                before=a.path(data,idx)[0];after=a.path(changed,idx)[0];singleton=a.path(data,idx[:1])[0]
            else:
                a=AutoGluonAdapter.load(path)
                for model in a.predictor.model_names():
                    def first(f):
                        z=a.predictor.predict(contexts(f),model=model,use_cache=False,random_seed=123)
                        return z.loc[val.iloc[0].sample_id,['mean','0.1','0.5','0.9']].to_numpy()
                    before=first(val);after=first(perturb(val))
                    np.testing.assert_allclose(before,after,rtol=2e-5,atol=2e-5)
                    records.append({'model_id':'ag_'+model,'year':year,'status':'PASS_OTHER_ORIGIN_MUTATION',
                      'maximum_difference':float(np.max(np.abs(before-after))),'stochastic_seed':123,'prediction_cache':False})
            if family!='autogluon':
                np.testing.assert_allclose(before,after,rtol=2e-5,atol=2e-5)
                np.testing.assert_allclose(before,singleton,rtol=2e-5,atol=2e-5)
                records.append({'model_id':name,'year':year,'status':'PASS_OTHER_ORIGIN_AND_SINGLETON',
                  'maximum_mutation_difference':float(np.max(np.abs(before-after))),
                  'maximum_singleton_difference':float(np.max(np.abs(before-singleton))),'seconds':time.perf_counter()-began})
            del a
        except Exception as exc:
            records.append({'model_id':name,'year':year,'status':'FAIL_REQUIRES_CAUSALITY_REVIEW','reason':str(exc),'traceback':traceback.format_exc()})
        save_json(RUN/'batch_causality_audits'/f'{family}.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
           'dataset_sha256':sha(RUN/'data/samples.parquet'),'all_pass':all(r['status'].startswith('PASS') for r in records),
           'tested_fold_years':years,'records':records,'rtol':2e-5,'atol':2e-5,
           'scope':'Other-origin history perturbation with first-origin inputs unchanged. Stronger than permutation-only checks; selected annual folds, not all possible inputs.'})
        print('BATCH_CAUSALITY',name,year,records[-1]['status'],flush=True)
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    for family in sys.argv[1:]:audit(family)
