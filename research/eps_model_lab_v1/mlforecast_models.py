"""MLForecast native lag transforms + fit_models, restricted to true origin training rows."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean,RollingStd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json
from research.eps_model_lab_v1.common import EPSAdapter,dataset,split_train,prediction_frame,save_predictions,evaluate_all
from research.eps_model_lab_v1.core_models import estimator

MODELS={'ridge':'ridge','lightgbm':'lgb','xgboost':'xgb','catboost':'cat'}

class MLForecastAdapter(EPSAdapter):
    def __init__(self,name):
        super().__init__('mlforecast_'+name,{'family':'MLFORECAST','version':importlib.metadata.version('mlforecast'),
             'source':'https://github.com/Nixtla/mlforecast','causal':True,'PIT_safe':True,
             'training_policy':'Only true-origin rows selected from native lag preprocessing; not all intermediate pseudo-origin rows'})
        pipe=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),estimator(MODELS[name]))
        self.forecaster=MLForecast(models={'model':pipe},freq=1,lags=[1,2,3,4,5,8,12,16],
                                  lag_transforms={1:[RollingMean(4),RollingStd(4)],4:[RollingMean(4)]},num_threads=1)
    def prepare_data(self,frame,target='h1'):
        if target=='ttm':
            y=frame[[f'ttm_lag_{j}' for j in range(7,-1,-1)]].to_numpy(dtype=float).copy()
            for j in range(y.shape[1]):
                replacement=y[:,j-4] if j>=4 else (y[:,j-1] if j else np.zeros(len(frame)))
                y[:,j]=np.where(np.isfinite(y[:,j]),y[:,j],replacement)
        else: y=frame[[f'filled_lag_{j}' for j in range(31,-1,-1)]].to_numpy(dtype=float)
        y=np.column_stack([y/frame.scale.to_numpy()[:,None],np.zeros(len(frame))])
        long=pd.DataFrame({'unique_id':np.repeat(frame.sample_id.to_numpy(),y.shape[1]),'ds':np.tile(np.arange(y.shape[1]),len(frame)),'y':y.ravel()})
        transformed=self.forecaster.preprocess(long,static_features=[],dropna=False)
        last=transformed[transformed.ds==y.shape[1]-1].set_index('unique_id').loc[frame.sample_id]
        return last.drop(columns=['ds','y']).reset_index(drop=True).replace([np.inf,-np.inf],np.nan)
    def fit(self,frame,target):
        self.target=target
        self.forecaster.fit_models(self.prepare_data(frame,target),frame['y_'+target].to_numpy()/frame.scale.to_numpy())
        self.fitted=True;return self
    def predict(self,frame,target):
        return self.forecaster.models_['model'].predict(self.prepare_data(frame,target))*frame.scale.to_numpy()

def task(name,year,target,smoke):
    started=time.perf_counter();frame=dataset();train,valid=split_train(frame,year,target)
    if smoke: train=train.iloc[:128];valid=valid.iloc[:8]
    a=MLForecastAdapter(name).fit(train,target);pred=a.predict(valid,target)
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite MLForecast output')
    if smoke:
        changed=valid.copy();changed['y_'+target]=1e9
        np.testing.assert_array_equal(pred,a.predict(changed,target))
    else: a.save(RUN/'mlforecast_fitted'/name/f'{year}_{target}.pkl')
    elapsed=time.perf_counter()-started
    return prediction_frame(valid,target,a.model_id,pred,elapsed,metadata=a.get_metadata()),{'year':year,'target':target,'seconds':elapsed,'train_rows':len(train),'status':'PASS'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    results={m:[] for m in MODELS};audit={m:[] for m in MODELS}
    tasks=[(m,y,t,args.smoke) for m in MODELS for y in ([2019] if args.smoke else range(2019,2027)) for t in (['h1','ttm'] if args.smoke else ['h1','h2','h3','h4','ttm'])]
    with ProcessPoolExecutor(max_workers=4 if args.smoke else 20) as pool:
        pending={pool.submit(task,*a):a for a in tasks}
        for future in as_completed(pending):
            name,year,target,_=pending[future]
            try:
                p,r=future.result();results[name].append(p);audit[name].append(r)
            except Exception as exc: audit[name].append({'year':year,'target':target,'status':'BROKEN','reason':str(exc),'traceback':traceback.format_exc()})
            save_json(RUN/('mlforecast_smoke' if args.smoke else 'mlforecast_audit')/f'{name}.json',audit[name])
            print(name,year,target,audit[name][-1]['status'],flush=True)
    if not args.smoke:
        for name in MODELS:
            if results[name]: save_predictions('mlforecast_'+name,results[name],{'family':'MLFORECAST','runtime_seconds':sum(r.get('seconds',0) for r in audit[name]),'audit':audit[name],'smoke_status':'PASS'})
        evaluate_all()
