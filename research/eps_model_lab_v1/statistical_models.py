"""Official StatsForecast plus literature-specified accounting SARIMA baselines."""
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
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[key]='1'
import numpy as np
import pandas as pd
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all

SF_NAMES=['Naive','SeasonalNaive','RandomWalkWithDrift','HistoricAverage','AutoARIMA','AutoETS','AutoTheta','AutoCES']
ACCOUNTING={'foster':((1,0,0),(0,1,0,4),'c'),'brown_rozeff':((1,0,0),(0,1,1,4),'n'),'griffin_watts':((0,1,1),(0,1,1,4),'n')}

class StatisticalAdapter(EPSAdapter):
    def fit(self,frame=None,target=None): self.fitted=True;return self
    def predict(self,frame,target):
        path=self.forecast(frame)[0]
        return path.sum(axis=1) if target=='ttm' else path[:,int(target[1:])-1]
    def forecast(self,frame):
        predictions=[];failures=[];convergence=[]
        for row in frame.itertuples():
            values=np.array([getattr(row,f'filled_lag_{j}') for j in range(31,-1,-1)],dtype=float)
            try:
                if self.model_id in ACCOUNTING:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    order,seasonal,trend=ACCOUNTING[self.model_id]
                    # Missing direct observations handled by Kalman filtering, not invented truth.
                    raw=np.array([getattr(row,f'eps_lag_{j}') for j in range(31,-1,-1)],dtype=float)
                    with warnings.catch_warnings(record=True) as caught:
                        model=SARIMAX(raw,order=order,seasonal_order=seasonal,trend=trend,
                                      enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=100)
                    converged=bool(model.mle_retvals.get('converged',False));convergence.append(converged)
                    if not converged: raise RuntimeError('SARIMAX did not converge within fixed 100 iterations')
                    pred=np.asarray(model.forecast(4))
                else:
                    from statsforecast import models
                    name=self.model_id.removeprefix('sf_')
                    kwargs={'season_length':4} if name in ['SeasonalNaive','AutoARIMA','AutoETS','AutoTheta','AutoCES'] else {}
                    if name=='AutoARIMA': kwargs.update(max_p=2,max_q=2,max_P=1,max_Q=1,max_order=4,nmodels=20,approximation=True,stepwise=True)
                    model=getattr(models,name)(**kwargs)
                    with warnings.catch_warnings(record=True): result=model.forecast(y=values,h=4)
                    pred=np.asarray(result['mean'])
                if pred.shape!=(4,) or not np.isfinite(pred).all(): raise ValueError('Nonfinite/incorrect output')
            except Exception as exc:
                pred=np.full(4,np.nan);failures.append({'sample_id':row.sample_id,'reason':f'{type(exc).__name__}: {exc}'})
            predictions.append(pred)
        return np.asarray(predictions),failures,convergence

def task(model,ticker,smoke):
    f=dataset();f=f[(f.asof_date.dt.year>=2019)&(f.ticker==ticker)]
    if smoke: f=f.iloc[:2]
    start=time.perf_counter();a=StatisticalAdapter(model).fit();p,errors,convergence=a.forecast(f)
    elapsed=time.perf_counter()-start
    results=[prediction_frame(f,f'h{j+1}',model,p[:,j],elapsed) for j in range(4)]
    # Sum of quarterly forecasts is an explicit approximate TTM estimator, not a sum of quantiles.
    ttm=prediction_frame(f,'ttm',model,p.sum(axis=1),elapsed)
    ttm['ttm_prediction_method']='SUM_FORECAST_QUARTERS_WEIGHTED_SHARE_APPROXIMATION';results.append(ttm)
    return results,{'ticker':ticker,'seconds':elapsed,'failures':errors,'converged':sum(convergence),'sarimax_fits':len(convergence)}

def run(smoke):
    from statsforecast import models as sf
    discovered=[n for n in SF_NAMES if hasattr(sf,n)]
    save_json(RUN/'ecosystem_discovery/statsforecast.json',{'version':importlib.metadata.version('statsforecast'),'models':discovered,
            'all_exported_classes':[n for n in dir(sf) if isinstance(getattr(sf,n),type)]})
    allmodels=['sf_'+n for n in discovered]+list(ACCOUNTING)
    if not smoke:
        eligible=[]
        for model in allmodels:
            path=RUN/'statistical_smoke'/f'{model}.json'
            checks=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
            passed=bool(checks) and all(not r.get('failures') and not r.get('failure_reason') for r in checks)
            if passed:eligible.append(model)
            else:save_json(RUN/'statistical_gate_failures'/f'{model}.json',{'model_id':model,'status':'SMOKE_FAILED_NOT_ELIGIBLE_FOR_FULL_RUN',
              'reason':'Any smoke row convergence/API/finite failure blocks full candidate advancement','evidence':str(path.relative_to(RUN))})
        allmodels=eligible
    tickers=['AAPL','JPM'] if smoke else sorted(dataset().ticker.unique())
    results={m:[] for m in allmodels};audit={m:[] for m in allmodels}
    workers=4 if smoke else json.loads((RUN/'CPU_THROUGHPUT_BENCHMARK.json').read_text(encoding='utf-8'))['selected_workers']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(task,m,t,smoke):(m,t) for m in allmodels for t in tickers}
        for future in as_completed(pending):
            m,t=pending[future]
            try:
                p,r=future.result();results[m].extend(p);audit[m].append(r)
            except Exception as exc: audit[m].append({'ticker':t,'failure_reason':str(exc),'traceback':traceback.format_exc()})
            save_json(RUN/('statistical_smoke' if smoke else 'statistical_audit')/f'{m}.json',audit[m])
            if len(audit[m])==len(tickers):
                if not smoke:
                    save_predictions(m,results[m],{'family':'ACCOUNTING_SARIMA' if m in ACCOUNTING else 'STATSFORECAST',
                        'runtime_seconds':sum(x.get('seconds',0) for x in audit[m]),'audit':audit[m],'smoke_status':'PASS',
                        'equation':ACCOUNTING.get(m),'source':'https://estudiosdeadministracion.uchile.cl/index.php/EDA/article/view/56413/59757' if m in ACCOUNTING else 'https://github.com/Nixtla/statsforecast'})
                print(m,'SMOKE_DONE' if smoke else 'FULL_SCORE_DONE',flush=True)
    if not smoke: evaluate_all()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();run(args.smoke)
