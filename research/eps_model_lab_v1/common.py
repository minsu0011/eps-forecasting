"""Shared adapter protocol, feature allowlist and prediction/evaluation contract."""
from __future__ import annotations
from abc import ABC, abstractmethod
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import importlib
import sys
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha,atomic_replace

TARGETS={'h1':('NEXT_Q_EPS',1),'h2':('NEXT_4Q_PATH',2),'h3':('NEXT_4Q_PATH',3),'h4':('NEXT_4Q_PATH',4),'ttm':('FUTURE_TTM_EPS',4)}
EXTRA_FEATURES=['scale','eps_growth','eps_acceleration','eps_volatility','eps_negative','fiscal_sin','fiscal_cos',
                'accruals_to_assets','leverage','operating_margin','roe','cash_to_assets']

def dataset():
    path=RUN/'data/samples.parquet'
    manifest=json.loads((RUN/'EPS_DATASET_MANIFEST_V1.json').read_text(encoding='utf-8'))
    if sha(path)!=manifest['samples_sha256']: raise RuntimeError('Dataset hash drift')
    return pd.read_parquet(path)

def feature_names(frame):
    return [c for c in frame if c.startswith(('filled_lag_','observed_lag_','ttm_lag_','account_')) or c in EXTRA_FEATURES]

def features(frame,scaled=True):
    names=feature_names(frame)
    x=frame[names].astype(float).copy()
    if scaled:
        for c in names:
            if c.startswith(('filled_lag_','ttm_lag_')) or c in ['eps_growth','eps_acceleration','eps_volatility']:
                x[c]=x[c]/frame['scale']
    for c in names:
        if c.startswith('account_'): x[c]=x[c]/1e9
    return x.replace([np.inf,-np.inf],np.nan)

def split_train(frame,year,target):
    cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC')
    mask=(frame.asof_date<cutoff)&(frame['label_asof_'+target]<cutoff)&frame['y_'+target].notna()
    train=frame.loc[mask]
    predict=frame.loc[frame.asof_date.dt.year==year]
    if not (train['label_asof_'+target]<cutoff).all(): raise AssertionError('Future training labels')
    return train,predict

class EPSAdapter(ABC):
    def __init__(self,model_id,metadata=None):
        self.model_id=model_id;self.metadata=metadata or {};self.fitted=False
    def prepare_data(self,frame): return features(frame)
    @abstractmethod
    def fit(self,frame,target): ...
    @abstractmethod
    def predict(self,frame,target): ...
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as stream: pickle.dump(self,stream,protocol=5)
    @classmethod
    def load(cls,path):
        # Only locally generated, hash-verified artifacts; never untrusted downloaded pickle.
        # Windows spawn saved early local adapters under __mp_main__. Resolve
        # only our explicitly named classes; no global sys.modules aliasing and
        # no rewrite of original weights or forecasts. This is NOT a safe loader
        # for untrusted pickle files (ordinary pickle globals remain enabled).
        class LocalAdapterUnpickler(pickle.Unpickler):
            def find_class(self,module,name):
                legacy={'TabularAdapter':'core_models','BaselineAdapter':'core_models','MLForecastAdapter':'mlforecast_models',
                        'HVZAdapter':'annual_accounting'}
                if module in ['__main__','__mp_main__'] and name in legacy:
                    return getattr(importlib.import_module('research.eps_model_lab_v1.'+legacy[name]),name)
                return super().find_class(module,name)
        with Path(path).open('rb') as stream: return LocalAdapterUnpickler(stream).load()
    def get_metadata(self):
        return {'model_id':self.model_id,'python_env':sys.executable,'deployable':False,'evidence_class':'RESEARCH_ONLY',**self.metadata}

def prediction_frame(frame,target,model_id,pred,runtime=0.,quantiles=None,metadata=None):
    metadata=metadata or {}
    cols=['sample_id','ticker','asof_date','origin_period_end','origin_accession','currency','eps_definition','share_basis',
          'origin_price','last_observed_eps','current_ttm','ttm_target_method','ttm_target_approximate']
    result=frame[cols].copy().reset_index(drop=True)
    result['forecast_target'],result['forecast_horizon']=TARGETS[target]
    result['target_key']=target;result['eps_model_id']=model_id
    result['actual_eps']=frame['y_'+target].to_numpy()
    result['label_asof']=frame['label_asof_'+target].to_numpy()
    result['predicted_eps']=np.asarray(pred,dtype=float).reshape(-1)
    result['prediction_valid']=np.isfinite(result.predicted_eps)
    result['pit_valid']=bool(metadata.get('pit_valid',True))
    result['inference_causal']=True
    result['coverage_flag']=np.where(result.prediction_valid,'VALID','MODEL_FAILURE_OR_INSUFFICIENT_CONTEXT')
    result['runtime_ms']=runtime*1000/max(len(frame),1)
    result['model_version']=str(metadata.get('version','V1'))
    result['dataset_run_identity']=RUN.name
    result['source_commit']=str(metadata.get('source_commit','LOCAL_EPS_RESEARCH_V1'))
    result['evidence_class']=metadata.get('evidence_class','RESEARCH_ONLY_NOT_FORMAL_CERTIFICATION')
    result['split']=np.where(result.asof_date.dt.year<=2021,'VALIDATION_OOF','RESEARCH_TEST')
    for i,name in enumerate(['p10_eps','p50_eps','p90_eps']):
        result[name]=np.asarray(quantiles)[:,i] if quantiles is not None else np.nan
    return result

def require_exact_frozen_frame(frame):
    """Bind cached auxiliary tensors to actual values, not merely matching IDs."""
    frozen = dataset()
    try:
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), frozen.reset_index(drop=True),
                                      check_exact=True, check_like=False)
    except AssertionError as exc:
        raise RuntimeError('Cached joint training requires the exact frozen frame, values and column order') from exc
    return frozen


def metric_row(g):
    truth=g.actual_eps.notna()&np.isfinite(g.actual_eps)
    ok=truth&g.prediction_valid&np.isfinite(g.predicted_eps)
    valid=g.loc[ok]; err=valid.predicted_eps-valid.actual_eps; ae=err.abs()
    out={'truth_rows':int(truth.sum()),'predicted_rows':int(ok.sum()),'coverage':float(ok.sum()/truth.sum()) if truth.sum() else 0.}
    if not len(valid): return out
    previous=valid.current_ttm if valid.target_key.iloc[0]=='ttm' else valid.last_observed_eps
    out.update(MAE=float(ae.mean()),RMSE=float(np.sqrt(np.mean(err**2))),MedianAE=float(ae.median()),bias=float(err.mean()),
               price_scaled_MAE=float((ae/valid.origin_price.where(valid.origin_price>0)).mean()),
               direction_accuracy=float((np.sign(valid.predicted_eps-previous)==np.sign(valid.actual_eps-previous)).loc[previous.notna()].mean()))
    subsets={'negative':valid.actual_eps<0,'near_zero':valid.actual_eps.abs()<=.1,'positive':valid.actual_eps>0,
             'high_growth':(valid.actual_eps-previous)>previous.abs().clip(lower=.1),
             'loss_to_profit':(previous<0)&(valid.actual_eps>0),'profit_to_loss':(previous>0)&(valid.actual_eps<0)}
    for name,mask in subsets.items():
        out[name+'_n']=int(mask.sum());out[name+'_MAE']=float(ae.loc[mask].mean()) if mask.any() else None
    qok=np.isfinite(valid[['p10_eps','p50_eps','p90_eps']]).all(axis=1)
    if qok.any():
        q=valid.loc[qok];loss=[]
        for tau,col in zip([.1,.5,.9],['p10_eps','p50_eps','p90_eps']):
            e=q.actual_eps-q[col];loss.append(np.maximum(tau*e,(tau-1)*e).mean())
        out.update(quantile_rows=len(q),pinball_loss=float(np.mean(loss)),interval80_coverage=float(((q.actual_eps>=q.p10_eps)&(q.actual_eps<=q.p90_eps)).mean()),
                   interval80_width=float((q.p90_eps-q.p10_eps).mean()),quantile_crossing_rows=int(((q.p10_eps>q.p50_eps)|(q.p50_eps>q.p90_eps)).sum()))
        for tau,col in zip([.1,.5,.9],['p10_eps','p50_eps','p90_eps']):
            out[col+'_empirical_cdf']=float((q.actual_eps<=q[col]).mean())
    return out

def atomic_csv(frame,path):
    path=Path(path);tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    frame.to_csv(tmp,index=False);atomic_replace(tmp,path)

def evaluate_all():
    files=sorted((RUN/'predictions').glob('*.parquet'))
    if not files: return
    pred=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
    receipts={p.stem:json.loads((RUN/'model_receipts'/f'{p.stem}.json').read_text(encoding='utf-8')) for p in files}
    qualified_bases={name for name,r in receipts.items() if r.get('status')=='FULL_RESEARCH_SCORED' and not name.startswith('ens_')}
    keys=['eps_model_id','target_key','split']
    scores=[]
    for key,g in pred.groupby(keys):
        scores.append(dict(zip(keys,key),mask='COMPLETE_COVERAGE',**metric_row(g)))
        if key[1]=='ttm':
            for approx,name in [(False,'TTM_NONAPPROXIMATE_SUBSET'),(True,'TTM_APPROXIMATE_SUBSET')]:
                scores.append(dict(zip(keys,key),mask=name,**metric_row(g.loc[g.ttm_target_approximate==approx])))
    for (target,split),g in pred.groupby(['target_key','split']):
        truth=g[g.actual_eps.notna()]
        matrix=truth.pivot(index='sample_id',columns='eps_model_id',values='predicted_eps')
        common=matrix.index[np.isfinite(matrix).all(axis=1)]
        for model,m in g.groupby('eps_model_id'):
            scores.append({'eps_model_id':model,'target_key':target,'split':split,'mask':'COMMON_MASK_ALL_SCORED',**metric_row(m[m.sample_id.isin(common)])})
        # Narrow-scope specialists must not shrink the primary broad-coverage ranking.
        broad=matrix.columns[np.isfinite(matrix).all(axis=0)]
        for model in broad:
            m=g[g.eps_model_id==model]
            scores.append({'eps_model_id':model,'target_key':target,'split':split,'mask':'COMMON_MASK_FULL_COVERAGE_MODELS',**metric_row(m)})
        # Preserve the all-output diagnostic masks, but do not let failed-smoke
        # artifacts or unavailable meta-fit predictions define the qualified pool.
        qualified=matrix.loc[:,matrix.columns.isin(qualified_bases)]
        if len(qualified.columns):
            common_qualified=qualified.index[np.isfinite(qualified).all(axis=1)]
            for model in qualified:
                m=g[g.eps_model_id==model]
                scores.append({'eps_model_id':model,'target_key':target,'split':split,'mask':'COMMON_MASK_QUALIFIED_BASE_MODELS',**metric_row(m[m.sample_id.isin(common_qualified)])})
            for model in qualified.columns[np.isfinite(qualified).all(axis=0)]:
                scores.append({'eps_model_id':model,'target_key':target,'split':split,'mask':'COMMON_MASK_QUALIFIED_FULL_COVERAGE_MODELS',**metric_row(g[g.eps_model_id==model])})
    scores=pd.DataFrame(scores)
    # Receipt runtime semantics differ (parallel summed fold times vs process
    # wall time). Keep measured values and missingness, never infer GPU time.
    runtime_records=[]
    for model in scores.eps_model_id.unique():
        receiptpath=RUN/'model_receipts'/f'{model}.json'
        receipt=json.loads(receiptpath.read_text(encoding='utf-8')) if receiptpath.exists() else {}
        runtime_records.append({'eps_model_id':model,'runtime_seconds':receipt.get('runtime_seconds'),
            'GPU_time_seconds':receipt.get('GPU_time_seconds_including_load'),
            'peak_gpu_allocated_gib':receipt.get('peak_gpu_allocated_gib'),'model_status':receipt.get('status','RECEIPT_PENDING'),
            'portfolio_eligible':receipt.get('status')=='FULL_RESEARCH_SCORED',
            'runtime_scope':'See receipt: aggregate fold/runtime metric, not a uniformly measured dedicated-process benchmark',
            'run_identity':RUN.name,'formal_certified':False})
    scores=scores.merge(pd.DataFrame(runtime_records),on='eps_model_id',validate='many_to_one')
    probability=['quantile_rows','pinball_loss','interval80_coverage','interval80_width','quantile_crossing_rows',
                 'p10_eps_empirical_cdf','p50_eps_empirical_cdf','p90_eps_empirical_cdf']
    present=[c for c in probability if c in scores]
    atomic_csv(scores.drop(columns=present),RUN/'EPS_MODEL_ZOO_LEADERBOARD_V1.csv')
    if 'quantile_rows' in scores:
        atomic_csv(scores.loc[scores.quantile_rows.gt(0),['eps_model_id','target_key','split','mask','truth_rows','predicted_rows','coverage','model_status','portfolio_eligible','run_identity','formal_certified',*present]],
                   RUN/'EPS_PROBABILISTIC_LEADERBOARD_V1.csv')

def save_predictions(model_id,predictions,receipt):
    (RUN/'predictions').mkdir(exist_ok=True)
    frame=pd.concat(predictions,ignore_index=True) if isinstance(predictions,list) else predictions
    if frame.duplicated(['sample_id','target_key']).any(): raise AssertionError('Duplicate model predictions')
    path=RUN/'predictions'/f'{model_id}.parquet'
    temporary=path.with_suffix(f'.{os.getpid()}.tmp')
    frame.to_parquet(temporary,index=False);atomic_replace(temporary,path)
    expected=set(dataset().loc[lambda d:d.asof_date.dt.year>=2019,'sample_id'])
    intended_complete=all(set(g.sample_id)==expected for _,g in frame.groupby('target_key'))
    status='FULL_RESEARCH_SCORED' if intended_complete else ('SMOKE_SCORED' if len(frame)<=100 else 'PARTIAL_RESEARCH_SCORED')
    save_json(RUN/'model_receipts'/f'{model_id}.json',{**receipt,'model_id':model_id,'prediction_path':str(path.relative_to(RUN)),
              'dataset_sha256':sha(RUN/'data/samples.parquet'),'dataset_run_identity':RUN.name,
              'prediction_sha256':sha(path),'rows':len(frame),'status':status,'intended_origin_coverage_complete':intended_complete,
              'target_keys':sorted(frame.target_key.unique()),'valid_prediction_rows':int(frame.prediction_valid.sum())})
