"""AutoGluon with observed-only training segments and train-internal ensemble validation."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
os.environ['HF_HOME']=str(PROJECT.parent/'.cache/eps_models')
import numpy as np
import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame,TimeSeriesPredictor
from research.eps_model_lab_v1.bootstrap import RUN,save_json
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all
from research.eps_model_lab_v1.dataset import normalize_eps

def virtual_dates(first_quarter,n):
    period=pd.Period(year=2000,quarter=int(first_quarter),freq='Q')
    return pd.period_range(period,periods=n,freq='Q').start_time

def training_segments(year):
    cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC');basis=cutoff.tz_localize(None)
    panel=pd.read_parquet(RUN/'data/fiscal_panel.parquet');panel=panel[panel['asof']<cutoff]
    chunks=[];audit=[]
    for ticker,g in panel.groupby('ticker'):
        g=g.sort_values('fiscal_index').copy()
        prices=RUN/'data/raw/prices'/f'{ticker}.parquet'
        p=pd.read_parquet(prices);p.index=p.index.tz_localize(None).normalize();splits=p.loc[p['Stock Splits']!=0,'Stock Splits']
        present=g.eps.notna();groups=(~present|g.fiscal_index.diff().ne(1)).cumsum()
        for number,segment in g.loc[present].groupby(groups[present]):
            if len(segment)<16: continue
            values=np.array([normalize_eps(r.eps,r.basis_date,basis,splits) for r in segment.itertuples()])
            assert np.isfinite(values).all()
            item=f'{ticker}_segment_{number}'
            chunks.append(pd.DataFrame({'item_id':item,'timestamp':virtual_dates(segment.fiscal_quarter.iloc[0],len(segment)),'target':values}))
            audit.append({'item_id':item,'observations':len(segment),'latest_availability':str(segment['asof'].max()),'latest_period_end':segment.end.iloc[-1]})
    if not chunks: raise RuntimeError('No fully observed contiguous training segments')
    return TimeSeriesDataFrame.from_data_frame(pd.concat(chunks,ignore_index=True),id_column='item_id',timestamp_column='timestamp'),audit

def contexts(frame):
    rows=[]
    for r in frame.itertuples():
        values=[getattr(r,f'filled_lag_{j}') for j in range(31,-1,-1)]
        first=(int(r.fiscal_quarter)-32)%4+1
        rows.append(pd.DataFrame({'item_id':r.sample_id,'timestamp':virtual_dates(first,32),'target':values}))
    return TimeSeriesDataFrame.from_data_frame(pd.concat(rows,ignore_index=True),id_column='item_id',timestamp_column='timestamp')

class AutoGluonAdapter(EPSAdapter):
    def __init__(self,year,smoke=False):
        super().__init__('AutoGluon_bundle',{'family':'AUTOGLUON','version':importlib.metadata.version('autogluon.timeseries'),
             'causal':True,'PIT_safe':True,'training_surface':'Only contiguous directly observed native quarter segments >=16; zero training-label imputation',
             'ensemble_fit_surface':'AutoGluon internal last-four validation windows wholly before annual fit cutoff',
             'source':'https://github.com/autogluon/autogluon'})
        self.year=year;self.smoke=smoke
    def prepare_data(self,frame): return contexts(frame)
    def fit(self,frame=None,target=None):
        data,audit=training_segments(self.year)
        path=RUN/('autogluon_smoke_models' if self.smoke else 'autogluon_fitted')/str(self.year)
        if path.exists(): raise RuntimeError('Preserve existing AutoGluon path; do not silently overwrite')
        self.predictor=TimeSeriesPredictor(path=str(path),prediction_length=4,target='target',freq='QS-JAN',
                              eval_metric='MAE',quantile_levels=[.1,.5,.9],verbosity=2)
        epochs=1 if self.smoke else 20
        hyper={'Naive':{'n_jobs':20},'SeasonalNaive':{'n_jobs':20},'AutoETS':{'n_jobs':20},'DirectTabular':{},'RecursiveTabular':{},
            'DeepAR':{'max_epochs':epochs,'num_batches_per_epoch':20,'batch_size':64,'predict_batch_size':256,
                      'trainer_kwargs':{'logger':False,'enable_progress_bar':False,'precision':'32-true'}},
            'TemporalFusionTransformer':{'max_epochs':epochs,'num_batches_per_epoch':20,'batch_size':64,'predict_batch_size':256,
                      'trainer_kwargs':{'logger':False,'enable_progress_bar':False,'precision':'32-true'}}}
        self.predictor.fit(train_data=data,time_limit=60 if self.smoke else 180,hyperparameters=hyper,
               num_val_windows=1,enable_ensemble=True,random_seed=1729)
        self.train_audit=audit;self.fitted=True
        return self
    def predict(self,frame,target):
        pred=self.predictor.predict(contexts(frame),model=target,use_cache=False)
        return pred
    def save(self,path): self.predictor.save()
    @classmethod
    def load(cls,path):
        a=cls(int(Path(path).name),smoke='smoke' in str(path))
        a.predictor=TimeSeriesPredictor.load(str(path));a.fitted=True;return a

def run(smoke=False):
    frame=dataset();outputs={};audits={};torch.set_num_threads(4)
    for year in ([2019] if smoke else range(2019,2027)):
        start=time.perf_counter();a=AutoGluonAdapter(year,smoke).fit();valid=frame[frame.asof_date.dt.year==year]
        if smoke: valid=valid.iloc[:8]
        models=a.predictor.model_names();leaderboard=a.predictor.leaderboard()
        save_json(RUN/('autogluon_smoke' if smoke else 'autogluon_audit')/f'{year}.json',{
               'year':year,'train_segments':a.train_audit,'models':models,'seconds':time.perf_counter()-start,'selection':'Only pre-cutoff internal validation'})
        leaderboard.to_csv(RUN/('autogluon_smoke' if smoke else 'autogluon_audit')/f'{year}_internal_leaderboard.csv',index=False)
        for model in models:
            began=time.perf_counter();p=a.predict(valid,model)
            grouped=p.reset_index().groupby('item_id')
            point=np.array([grouped.get_group(i).sort_values('timestamp')['mean'].to_numpy() for i in valid.sample_id])
            q=np.array([grouped.get_group(i).sort_values('timestamp')[['0.1','0.5','0.9']].to_numpy() for i in valid.sample_id])
            if point.shape!=(len(valid),4) or not np.isfinite(point).all(): raise ValueError('AutoGluon prediction contract failure: '+model)
            model_id='ag_'+model
            if not smoke:
                out=[prediction_frame(valid,f'h{j+1}',model_id,point[:,j],time.perf_counter()-began,q[:,j,:],a.get_metadata()) for j in range(4)]
                ttm=prediction_frame(valid,'ttm',model_id,point.sum(axis=1),time.perf_counter()-began,metadata=a.get_metadata())
                ttm['ttm_prediction_method']='SUM_FORECAST_QUARTERS_WEIGHTED_SHARE_APPROXIMATION';out.append(ttm)
                outputs.setdefault(model_id,[]).extend(out);audits.setdefault(model_id,[]).append({'year':year,'fit_seconds_bundle':time.perf_counter()-start,'prediction_rows':len(valid)})
            print(model_id,year,'SMOKE_PASS' if smoke else 'FOLD_PASS',flush=True)
        del a;gc.collect();torch.cuda.empty_cache()
    if not smoke:
        for model_id,p in outputs.items():
            save_predictions(model_id,p,{'family':'AUTOGLUON','audit':audits[model_id],'smoke_status':'PASS','training':'Only directly observed contiguous segments; no imputed training labels',
                  'ensemble_weights':'AutoGluon train-internal validation, never research test'})
        evaluate_all()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    try: run(a.smoke)
    except Exception as exc:
        save_json(RUN/('autogluon_smoke_failure.json' if a.smoke else 'autogluon_failure.json'),{'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
        print('AUTOGLUON_BROKEN',str(exc),flush=True)
