"""Bounded V2 native tabular retraining, single-thread native models / outer CPU lane."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import os
import pickle
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from research.eps_model_lab_v2.common import RUN,read_json,save_json,sha,utcnow,check_stop
from research.eps_model_lab_v2.model_common import dataset,native_features,baseline,training_rows,predictions,TARGETS,point_metrics,surface


class TabularModel:
    def __init__(self,recipe,target):self.recipe=recipe;self.target=target
    def fit(self,train,weights):
        settings=read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['tabular_parameters'];family=self.recipe['family']
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        self.scaler=StandardScaler()
        x=self.scaler.fit_transform(self.imputer.fit_transform(native_features(train)))
        raw=train['y_'+self.target].to_numpy();scale=train.scale.to_numpy()
        representation=self.recipe['representation']
        self.transform_scale=max(float(np.median(np.abs(raw))),.1)
        if representation=='asinh':y=np.arcsinh(raw/self.transform_scale)
        else:y=(raw-(baseline(train,self.target) if representation=='residual' else 0))/scale
        if family=='Ridge':
            from sklearn.linear_model import Ridge
            self.model=Ridge(**settings[family])
        elif family=='HistGB':
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.model=HistGradientBoostingRegressor(**settings[family],random_state=1729)
        elif family=='CatBoost':
            from catboost import CatBoostRegressor
            self.model=CatBoostRegressor(**settings[family],random_seed=1729,verbose=False,allow_writing_files=False)
        elif family=='NGBoostLaplace':
            from ngboost import NGBRegressor
            from ngboost.distns import Laplace
            from sklearn.tree import DecisionTreeRegressor
            self.model=NGBRegressor(Dist=Laplace,Base=DecisionTreeRegressor(max_depth=3,min_samples_leaf=5,random_state=1729),
                n_estimators=500,learning_rate=.01,natural_gradient=True,random_state=1729,verbose=False,early_stopping_rounds=None)
        else:raise KeyError(family)
        self.model.fit(x,y,sample_weight=weights)
        return self
    def predict(self,frame):
        x=self.scaler.transform(self.imputer.transform(native_features(frame)))
        p=self.model.predict(x);q=None
        if self.recipe['family']=='NGBoostLaplace':
            d=self.model.pred_dist(x);q=np.column_stack([d.ppf(v) for v in [.1,.5,.9]])
        if self.recipe['representation']=='asinh':
            # Exact inverse; no epsilon clipping or test-tail cap. Overflow fails.
            p=np.sinh(p)*self.transform_scale
            if q is not None:q=np.sinh(q)*self.transform_scale
        else:
            p=p*frame.scale.to_numpy()
            if q is not None:q=q*frame.scale.to_numpy()[:,None]
            if self.recipe['representation']=='residual':
                b=baseline(frame,self.target);p=p+b
                if q is not None:q=q+b[:,None]
        return p,q


def one(recipe,year,target,phase,replicate='main'):
    check_stop();start=time.perf_counter();frame=dataset();name=recipe['model_id']
    train,weights,cutoff=training_rows(frame,year,target,recipe['chronological_mode'])
    valid=frame[frame.asof_date.dt.year==year]
    if len(train)<50:raise RuntimeError(f'Insufficient purged training support {name}/{year}/{target}: {len(train)}')
    path=RUN/'models'/phase/name/replicate/f'{year}_{target}.pkl'
    if path.exists():raise FileExistsError('Existing trained identity; replay it, never silently refit')
    with threadpool_limits(limits=1):
        model=TabularModel(recipe,target).fit(train,weights);point,q=model.predict(valid)
        changed=valid.copy()
        for column in changed:
            if column.startswith(('y_','label_','target_','account_','quality_account_')):changed[column]=1e12
        np.testing.assert_array_equal(point,model.predict(changed)[0])
        np.testing.assert_allclose(point,model.predict(valid.iloc[::-1])[0][::-1],rtol=1e-12,atol=1e-12)
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:pickle.dump(model.__dict__,stream,protocol=5)
        loaded=TabularModel(recipe,target)
        with path.open('rb') as stream:loaded.__dict__.update(pickle.load(stream))
        np.testing.assert_array_equal(point,loaded.predict(valid)[0])
    output=predictions(valid,target,name,point,q)
    predpath=RUN/'predictions'/phase/name/replicate/f'{year}_{target}.parquet'
    predpath.parent.mkdir(parents=True,exist_ok=True)
    if predpath.exists():raise FileExistsError(predpath)
    output.to_parquet(predpath,index=False)
    receipt={'model_id':name,'year':year,'target':target,'phase':phase,'replicate':replicate,
        'train_rows':len(train),'predict_rows':len(valid),'cutoff':str(cutoff),
        'train_max_origin':str(train.asof_date.max()),'train_max_label':str(train['label_asof_'+target].max()),
        'model_path':str(path),'model_sha256':sha(path),'prediction_path':str(predpath),'prediction_sha256':sha(predpath),
        'same_process_save_load_exact':True,'future_labels_and_accounting_mutation_exact':True,
        'seconds':time.perf_counter()-start,'status':'PASS','fresh_process_replay_pending':True}
    save_json(path.with_suffix('.receipt.json'),receipt,immutable=True)
    return receipt


def baseline_run():
    frame=dataset();recipes=read_json(RUN/'EPS_V2_MODEL_RECIPES.json')
    for name in recipes['baseline_names']:
        path=RUN/'predictions/baselines'/f'{name}.parquet';path.parent.mkdir(exist_ok=True)
        if path.exists():raise FileExistsError(path)
        out=pd.concat([predictions(frame[frame.asof_date.dt.year>=2015],t,name,baseline(frame[frame.asof_date.dt.year>=2015],t,name)) for t in TARGETS],ignore_index=True)
        out.to_parquet(path,index=False)
    save_json(RUN/'audit/BASELINES_REGENERATED_V2.json',{'created_utc':utcnow(),'names':recipes['baseline_names'],
        'V1_scores_copied':False,'samples_sha256':sha(RUN/'data/samples_v2.parquet'),
        'aliases_not_diversity':{'random_walk':'persistence_observed','seasonal_random_walk':'seasonal_observed'}},immutable=True)
    print('V2_FIVE_BASELINES_REGENERATED',flush=True)


def benchmark_worker(_):
    from sklearn.ensemble import HistGradientBoostingRegressor
    frame=dataset();train,_,_=training_rows(frame,2015,'h1','expanding')
    with threadpool_limits(limits=1):
        x=np.nan_to_num(native_features(train));y=train.y_h1.to_numpy()/train.scale.to_numpy()
        m=HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=15,early_stopping=False,random_state=1729).fit(x,y)
        return float(m.predict(x[:8]).sum())


def benchmark():
    rows=[]
    for workers in [16,20,24,28]:
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:values=list(pool.map(benchmark_worker,range(56)))
        np.testing.assert_array_equal(values,np.repeat(values[0],len(values)))
        rows.append({'workers':workers,'inner_threads':1,'actual_native_HistGB_fits':56,'seconds':time.perf_counter()-start,
            'prediction_digest_scalar':values[0]})
        print('V2_CPU_BENCHMARK',workers,rows[-1]['seconds'],flush=True)
    save_json(RUN/'audit/CPU_THROUGHPUT_BENCHMARK_V2.json',{'created_utc':utcnow(),'measurements':rows,
        'selected_workers':min(rows,key=lambda r:r['seconds'])['workers'],
        'scope':'Fixed V2 pre-2019 development geometry throughput; predictions not scored or registered as model variants'},immutable=True)


def run(phase='development',family=None):
    recipes=read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['recipes']
    requested_families=['Ridge','HistGB','CatBoost','NGBoostLaplace'] if family is None else (['Ridge','HistGB','NGBoostLaplace'] if family=='prob_lane' else [family])
    recipes=[r for r in recipes if r['family'] in requested_families and r.get('status') is None]
    years=[2015,2016,2017,2018]
    if phase!='development':
        lock=read_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')
        recipes=[r for r in recipes if r['model_id'] in lock['selected_model_ids']]
        years=[2019,2020,2021] if phase=='confirmation' else [2022,2023,2024,2025,2026]
    workers=read_json(RUN/'audit/CPU_THROUGHPUT_BENCHMARK_V2.json')['selected_workers']
    receipts=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(one,r,y,t,phase):(r['model_id'],y,t) for r in recipes for y in years for t in TARGETS}
        for future in as_completed(jobs):
            task=jobs[future]
            try:
                receipt=future.result();receipts.append(receipt)
                print('V2_TABULAR_PASS',*task,round(receipt['seconds'],2),flush=True)
            except Exception as exc:
                failure={'task':task,'error':str(exc),'traceback':traceback.format_exc(),'created_utc':utcnow()};failures.append(failure)
                save_json(RUN/'failures'/f'tabular_{phase}_{task[0]}_{task[1]}_{task[2]}_{os.getpid()}.json',failure,immutable=True)
                print('V2_TABULAR_FAIL',*task,str(exc),flush=True)
            save_json(RUN/'audit'/f'TABULAR_{phase}_{family or "all"}_PROGRESS.json',{'complete':len(receipts),'failed':len(failures),'planned':len(jobs),'receipts':receipts,'failures':failures})
    if failures:raise RuntimeError(f'{len(failures)} tabular fits failed; retained for diagnosis')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baselines',action='store_true');p.add_argument('--benchmark',action='store_true');p.add_argument('--phase',default='development');p.add_argument('--family');a=p.parse_args()
    if a.baselines:baseline_run()
    elif a.benchmark:benchmark()
    else:run(a.phase,a.family)
