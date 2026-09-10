"""Actual Darts local forecasters on the unchanged fiscal input grid."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import FFT,FourTheta,KalmanForecaster
from darts.utils.utils import SeasonalityMode,ModelMode,TrendMode
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all

SPEC={'FFT':{'nr_freqs_to_keep':5,'required_matches':[],'trend':'linear'},
      'FourTheta':{'theta':2,'seasonality_period':4,'season_mode':'ADDITIVE','model_mode':'ADDITIVE','trend_mode':'LINEAR','normalization':False},
      'KalmanForecaster':{'dim_x':2,'random_state':1729}}


class DartsAdapter(EPSAdapter):
    def __init__(self,name):
        super().__init__('darts_'+name,{'family':'DARTS_LOCAL','version':importlib.metadata.version('darts'),
           'source':'https://github.com/unit8co/darts','pit_valid':True,'causal':True,
           'input_policy':'Causally filled historical grid as statistical-model input; never imputed future truth',
           'license':'Apache-2.0','zero_shot':False})
        self.name=name
    def prepare_data(self,frame):
        return frame[[f'filled_lag_{j}' for j in range(31,-1,-1)]].to_numpy(dtype=float)
    def fit(self,frame=None,target=None): self.fitted=True;return self
    def predict(self,frame,target):
        point,_,_=self.forecast(frame)
        return point.sum(axis=1) if target=='ttm' else point[:,int(target[1:])-1]
    def forecast(self,frame,smoke=False):
        points=[];errors=[];reload=[]
        for r,values in zip(frame.itertuples(),self.prepare_data(frame)):
            try:
                series=TimeSeries.from_values(values)
                if self.name=='FFT': m=FFT(nr_freqs_to_keep=5,required_matches=set(),trend='linear')
                elif self.name=='FourTheta': m=FourTheta(theta=2,seasonality_period=4,season_mode=SeasonalityMode.ADDITIVE,
                     model_mode=ModelMode.ADDITIVE,trend_mode=TrendMode.LINEAR,normalization=False)
                else: m=KalmanForecaster(dim_x=2,random_state=1729)
                with warnings.catch_warnings(record=True):
                    m.fit(series);p=m.predict(4,num_samples=1).values(copy=True).ravel()
                if p.shape!=(4,) or not np.isfinite(p).all(): raise RuntimeError('Nonfinite or misaligned forecast')
                if smoke:
                    path=RUN/'darts_smoke_fitted'/self.name/f'{r.sample_id}.pkl';path.parent.mkdir(parents=True,exist_ok=True)
                    m.save(path);loaded=type(m).load(path)
                    repeated=loaded.predict(4,num_samples=1).values(copy=True).ravel()
                    # Kalman matrix factorization can differ at one float64 ULP
                    # after pickle reload; this is not a model parameter change.
                    np.testing.assert_allclose(p,repeated,rtol=1e-12,atol=1e-12)
                    reload.append({'sample_id':r.sample_id,'sha256':sha(path),'status':'PASS_FLOAT64_TOLERANCE',
                                   'maximum_absolute_difference':float(np.max(np.abs(p-repeated))),'rtol':1e-12,'atol':1e-12})
            except Exception as exc:
                p=np.full(4,np.nan);errors.append({'sample_id':r.sample_id,'reason':str(exc),'type':type(exc).__name__})
            points.append(p)
        return np.asarray(points),errors,reload


def task(name,ticker,smoke=False):
    frame=dataset();frame=frame[(frame.ticker==ticker)&(frame.asof_date.dt.year>=2019)]
    if smoke: frame=frame.iloc[:2]
    start=time.perf_counter();a=DartsAdapter(name).fit();p,errors,reload=a.forecast(frame,smoke)
    out=[prediction_frame(frame,f'h{h+1}',a.model_id,p[:,h],time.perf_counter()-start,metadata=a.get_metadata()) for h in range(4)]
    c=prediction_frame(frame,'ttm',a.model_id,p.sum(axis=1),time.perf_counter()-start,metadata=a.get_metadata())
    c['ttm_prediction_method']='SUM_FORECAST_QUARTERS_WEIGHTED_SHARE_APPROXIMATION';out.append(c)
    return out,{'ticker':ticker,'errors':errors,'reload':reload,'seconds':time.perf_counter()-start,'rows':len(frame)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='*',choices=list(SPEC));parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    lock=RUN/'DARTS_LOCAL_SPEC_V1.json'
    if not lock.exists(): save_json(lock,{'spec':SPEC,'source_sha256':sha(__file__),'fixed_before_candidate_score':True})
    else:
        if json.loads(lock.read_text(encoding='utf-8'))['spec']!=SPEC: raise RuntimeError('Darts specification drift')
    eligible=[]
    for name in args.names or list(SPEC):
        _,audit=task(name,'AAPL',True)
        save_json(RUN/'darts_smoke'/f'{name}.json',audit)
        if audit['errors']: print(name,'SMOKE_FAILED',audit['errors'],flush=True)
        else: eligible.append(name);print(name,'SMOKE_PASS',flush=True)
    if not args.smoke:
        results={m:[] for m in eligible};audits={m:[] for m in eligible}
        tickers=sorted(dataset().ticker.unique())
        with ProcessPoolExecutor(max_workers=8) as pool:
            jobs={pool.submit(task,m,t):(m,t) for m in eligible for t in tickers}
            for future in as_completed(jobs):
                m,t=jobs[future]
                try: output,audit=future.result();results[m].extend(output);audits[m].append(audit)
                except Exception as exc: audits[m].append({'ticker':t,'failure':str(exc),'traceback':traceback.format_exc()})
                if len(audits[m])%10==0: print('DARTS',m,'TICKERS',len(audits[m]),flush=True)
                save_json(RUN/'darts_audit'/f'{m}.json',audits[m])
        for m in eligible:
            save_predictions('darts_'+m,results[m],{'family':'DARTS_LOCAL','metadata':DartsAdapter(m).get_metadata(),
               'audit':audits[m],'smoke_status':'PASS','spec':SPEC[m],'runtime_seconds':sum(r.get('seconds',0) for r in audits[m])})
        evaluate_all()
