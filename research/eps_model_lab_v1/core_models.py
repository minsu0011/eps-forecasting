"""Frozen broad core candidates. Every fit uses per-target availability-purged train rows."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[name]='1'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,TARGETS,dataset,features,split_train,prediction_frame,save_predictions,evaluate_all

CORE_SPECS={
 'ridge_raw':{'kind':'ridge','scaled':False},'ridge_scaled':{'kind':'ridge','scaled':True},
 'elasticnet_scaled':{'kind':'elasticnet','scaled':True},'huber_scaled':{'kind':'huber','scaled':True},
 'random_forest':{'kind':'rf','scaled':True},'extra_trees':{'kind':'et','scaled':True},
 'hist_gradient_boosting':{'kind':'hgb','scaled':True},'lightgbm':{'kind':'lgb','scaled':True},
 'xgboost_squared':{'kind':'xgb','scaled':True},'afden_xgb_pseudohuber':{'kind':'xgb_huber','scaled':True},
 'catboost_mae':{'kind':'cat','scaled':True},
}
BASELINES=['random_walk','random_walk_drift','seasonal_random_walk','seasonal_drift','historical_mean',
           'persistence_observed','drift_observed','seasonal_observed','seasonal_drift_observed','mean_observed']

def estimator(kind):
    from sklearn.linear_model import Ridge,ElasticNet,HuberRegressor
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
    if kind=='ridge': return Ridge(alpha=100.)
    if kind=='elasticnet': return ElasticNet(alpha=.03,l1_ratio=.15,max_iter=3000,random_state=1729)
    if kind=='huber': return HuberRegressor(epsilon=1.35,alpha=.1,max_iter=400)
    if kind=='rf': return RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=8,max_features=.7,n_jobs=1,random_state=1729)
    if kind=='et': return ExtraTreesRegressor(n_estimators=300,max_depth=16,min_samples_leaf=5,max_features=.8,n_jobs=1,random_state=1729)
    if kind=='hgb': return HistGradientBoostingRegressor(loss='absolute_error',max_iter=250,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=5,early_stopping=False,random_state=1729)
    if kind=='lgb':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=350,num_leaves=15,learning_rate=.03,min_child_samples=25,reg_lambda=5,colsample_bytree=.8,n_jobs=1,verbosity=-1,random_state=1729)
    if kind.startswith('xgb'):
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=400,max_depth=3,learning_rate=.04,min_child_weight=10,reg_lambda=5,subsample=.8,colsample_bytree=.8,
                            objective='reg:pseudohubererror' if kind=='xgb_huber' else 'reg:squarederror',tree_method='hist',n_jobs=1,random_state=1729)
    if kind=='cat':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=400,depth=5,learning_rate=.04,loss_function='MAE',l2_leaf_reg=5,thread_count=1,random_seed=1729,verbose=False,allow_writing_files=False)
    raise KeyError(kind)

class TabularAdapter(EPSAdapter):
    def __init__(self,model_id):
        super().__init__(model_id,{'family':'TABULAR_ACCOUNTING','architecture':CORE_SPECS[model_id]['kind'],'PIT_safe':True,'causal':True})
        self.spec=CORE_SPECS[model_id]
    def prepare_data(self,frame): return features(frame,self.spec['scaled'])
    def fit(self,frame,target):
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        x=self.prepare_data(frame)
        y=frame['y_'+target].to_numpy()/frame.scale.to_numpy() if self.spec['scaled'] else frame['y_'+target].to_numpy()
        self.model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),estimator(self.spec['kind']))
        with threadpool_limits(1): self.model.fit(x,y)
        self.fitted=True;self.target=target
        return self
    def predict(self,frame,target):
        if not self.fitted or target!=self.target: raise RuntimeError('Not fitted for requested target')
        with threadpool_limits(1): result=self.model.predict(self.prepare_data(frame))
        return result*frame.scale.to_numpy() if self.spec['scaled'] else result

class BaselineAdapter(EPSAdapter):
    def fit(self,frame,target): self.fitted=True;return self
    def predict(self,frame,target):
        observed_mode=self.model_id.endswith('_observed')
        method={'persistence_observed':'random_walk','drift_observed':'random_walk_drift','seasonal_observed':'seasonal_random_walk',
                'seasonal_drift_observed':'seasonal_drift','mean_observed':'historical_mean'}.get(self.model_id,self.model_id)
        if target=='ttm':
            histories=frame[[f'ttm_lag_{i}' for i in range(7,-1,-1)]].to_numpy()
            horizon=4
        else:
            prefix='eps_lag_' if observed_mode else 'filled_lag_'
            histories=frame[[f'{prefix}{i}' for i in range(31,-1,-1)]].to_numpy();horizon=int(target[1:])
        outputs=[]
        for history in histories:
            observed=history[np.isfinite(history)]
            if not len(observed): outputs.append(np.nan);continue
            positions=np.flatnonzero(np.isfinite(history))
            distance=max(int(positions[-1]-positions[0]),1) if observed_mode else max(len(observed)-1,1)
            last=float(observed[-1]);trend=(float(observed[-1])-float(observed[0]))/distance
            seasonal=history[-4+(horizon-1)%4]
            if not np.isfinite(seasonal): seasonal=last
            if method=='random_walk': pred=last
            elif method=='random_walk_drift': pred=last+horizon*trend
            elif method=='seasonal_random_walk': pred=seasonal
            elif method=='seasonal_drift':
                diffs=history[4:]-history[:-4];finite=diffs[np.isfinite(diffs)]
                pred=seasonal+(float(np.mean(finite)) if len(finite) else 0.)
            elif method=='historical_mean': pred=float(np.mean(observed))
            else: raise KeyError(self.model_id)
            outputs.append(pred)
        return np.asarray(outputs)

def one_fold(model_id,year,target,smoke=False):
    started=time.perf_counter();frame=dataset();train,valid=split_train(frame,year,target)
    if smoke: train=train.iloc[:min(600,len(train))];valid=valid.iloc[:16]
    if len(train)<50 or valid.empty: return None,{'status':'INSUFFICIENT_ROWS','model':model_id,'year':year,'target':target}
    adapter=TabularAdapter(model_id)
    adapter.fit(train,target);pred=adapter.predict(valid,target)
    if not np.isfinite(pred).all(): raise ValueError('Nonfinite tabular predictions')
    if smoke:
        changed=valid.copy()
        for col in changed:
            if col.startswith('y_'): changed[col]=1e12
        repeated=adapter.predict(changed,target)
        np.testing.assert_array_equal(pred,repeated)
        path=RUN/'smoke_models'/f'{model_id}_{year}_{target}.pkl'
        adapter.save(path);loaded=TabularAdapter.load(path)
        np.testing.assert_array_equal(pred,loaded.predict(valid,target))
    else:
        adapter.save(RUN/'fitted_models'/model_id/f'{year}_{target}.pkl')
    elapsed=time.perf_counter()-started
    receipt={'model_id':model_id,'year':year,'target':target,'train_rows':len(train),'prediction_rows':len(valid),'train_max_origin':str(train.asof_date.max()),
             'train_max_label_asof':str(train['label_asof_'+target].max()),'fit_cutoff':f'{year}-01-01T00:00:00Z','seconds':elapsed,'status':'PASS'}
    return prediction_frame(valid,target,model_id,pred,elapsed),receipt

def baselines():
    frame=dataset();v=frame.loc[frame.asof_date.dt.year>=2019]
    for model in BASELINES:
        if (RUN/'predictions'/f'{model}.parquet').exists(): continue
        started=time.perf_counter();adapter=BaselineAdapter(model);adapter.fit(None,None)
        frames=[prediction_frame(v,t,model,adapter.predict(v,t)) for t in TARGETS]
        save_predictions(model,frames,{'family':'ACCOUNTING_BASELINE','runtime_seconds':time.perf_counter()-started,'adapter':adapter.get_metadata(),'smoke_status':'PASS',
            'input_semantics':'Only actual observed EPS; missing seasonal value falls back to last actual observed' if model.endswith('_observed') else 'Original control on causally imputed fiscal grid; not literal last-observed EPS baseline'})
    evaluate_all()

def run(smoke=False,models=None):
    models=models or list(CORE_SPECS)
    lock=RUN/'CORE_CANDIDATE_SPECS_V1.json'
    specs={'specs':CORE_SPECS,'baseline_ids':BASELINES,'estimator_source_sha256':sha(Path(__file__)),'selection':'Fixed breadth screening, no test tuning'}
    if not lock.exists(): save_json(lock,specs)
    workers=20
    benchmark=RUN/'CPU_THROUGHPUT_BENCHMARK.json'
    if benchmark.exists(): workers=json.loads(benchmark.read_text(encoding='utf-8'))['selected_workers']
    workers=min(workers,8) if smoke else workers
    tasks=[(m,y,t,smoke) for m in models for y in ([2019] if smoke else range(2019,2027)) for t in (['h1','ttm'] if smoke else TARGETS)]
    results={m:[] for m in models};receipts={m:[] for m in models}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(one_fold,*task):task for task in tasks}
        for future in as_completed(pending):
            task=pending[future];model=task[0]
            try:
                prediction,receipt=future.result();receipts[model].append(receipt)
                if prediction is not None: results[model].append(prediction)
            except Exception as exc:
                receipts[model].append({'status':'BROKEN','task':task,'reason':str(exc),'traceback':traceback.format_exc()})
            save_json(RUN/('smoke_receipts' if smoke else 'fold_receipts')/f'{model}.json',receipts[model])
            print('SMOKE' if smoke else 'FOLD',task[:3],receipts[model][-1]['status'],flush=True)
    for model in models:
        if smoke:
            save_json(RUN/'smoke_receipts'/f'{model}_summary.json',{'model':model,'status':'PASS' if all(r['status']=='PASS' for r in receipts[model]) else 'FAIL','folds':receipts[model]})
        elif results[model]:
            save_predictions(model,results[model],{'family':'TABULAR_ACCOUNTING','spec':CORE_SPECS[model],
                   'runtime_seconds':sum(r.get('seconds',0) for r in receipts[model]),'smoke_status':'PASS','folds':receipts[model],
                   'source':'AFDEN architecture adaptation + official XGBoost' if model.startswith('afden') else 'Official package',
                   'source_note':'AFDEN current official notebook moved to Module 6; source clipping of validation/test truth intentionally not copied. Our truth never clipped.'})
    if not smoke: evaluate_all()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--baselines',action='store_true');parser.add_argument('--models',nargs='+')
    args=parser.parse_args()
    if args.baselines: baselines()
    else: run(args.smoke,args.models)
