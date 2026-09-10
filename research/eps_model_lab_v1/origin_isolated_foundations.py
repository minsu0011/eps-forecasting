"""Corrective inference identity: no other-origin values share the RNG call."""
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all

BASES=['lag_llama_zero_shot','moirai1p1_small','moirai_moe_small']
SPEC={'inference':'Exactly one company-origin per upstream forecast call',
  'seed':'Upstream constant1729 reset for each independent origin; common random numbers across origins but no other-origin input dependency',
  'sample_count':100,'point':'Native predictive median','quantiles':'Native sample P10/P50/P90',
  'weights_and_context':'Identical original checkpoint/context/missing masks; no hyperparameter tuning or refit',
  'cpu_threads':4,'strict_pretraining_certified':False,'basis':'Corrective inference contract, not an independent architecture discovery'}


class OriginIsolatedAdapter(EPSAdapter):
    def __init__(self,base):
        if base not in BASES:raise ValueError(base)
        self.base_id=base
        if base.startswith('lag'):
            from research.eps_model_lab_v1.lag_llama_model import LagAdapter
            self.base=LagAdapter()
        else:
            from research.eps_model_lab_v1.moirai_models import MoiraiAdapter
            self.base=MoiraiAdapter(base)
        meta=dict(self.base.metadata);meta.update(origin_isolation=SPEC,corrected_from=base)
        super().__init__(base+'_origin_isolated',meta)
    def fit(self,frame=None,target=None):
        torch.set_num_threads(SPEC['cpu_threads'])
        self.base.fit();self.fitted=True;return self
    def prepare_data(self,frame):return self.base.prepare_data(frame)
    def forecast(self,frame,lane,progress=False):
        ps=[];qs=[];began=time.perf_counter()
        for i in range(len(frame)):
            p,q=self.base.forecast(frame.iloc[[i]],lane);ps.append(p[0]);qs.append(q[0])
            if progress and ((i+1)%100==0 or i+1==len(frame)):
                record={'model_id':self.model_id,'lane':lane,'completed':i+1,'total':len(frame),'seconds':time.perf_counter()-began,
                    'updated_utc':datetime.now(timezone.utc).isoformat()}
                save_json(RUN/'origin_isolation_progress'/f'{self.model_id}.json',record)
                print('ISOLATED_PROGRESS',self.model_id,lane,i+1,len(frame),round(record['seconds'],1),flush=True)
        return np.asarray(ps),np.asarray(qs)
    def predict(self,frame,target):
        p,_=self.forecast(frame,'ttm' if target=='ttm' else 'quarter');return p[:,3 if target=='ttm' else int(target[1:])-1]
    def save(self,path):save_json(path,{'base_id':self.base_id,'metadata':self.get_metadata(),'spec':SPEC})
    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        if d['spec']!=SPEC:raise RuntimeError('Isolation recipe drift')
        return cls(d['base_id']).fit()


def mark_original_diagnostic(base):
    p=RUN/'model_receipts'/f'{base}.json';r=json.loads(p.read_text(encoding='utf-8'))
    archive=RUN/'audit_corrections/batch_rng_original_receipts'/p.name
    if not archive.exists():archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,archive)
    r.update(status='FULL_DIAGNOSTIC_SCORED_BATCH_RNG_DEPENDENCE',portfolio_eligible=False,
      failure_reason='Fixed first-origin point forecast changes when OTHER origins change within a stochastic batch. Marginal distribution independence is not disproven, but deterministic causal-input contract is not met.',
      corrective_candidate=base+'_origin_isolated',predictions_unchanged=True)
    save_json(p,r)


def score(base):
    name=base+'_origin_isolated';lock=RUN/'origin_isolation_specs'/f'{name}.json'
    if lock.exists() or (RUN/'predictions'/f'{name}.parquet').exists():raise RuntimeError('No implicit repeat/overwrite')
    save_json(lock,{'model_id':name,'spec':SPEC,'samples_sha256':sha(RUN/'data/samples.parquet'),
       'source_sha256':sha(__file__),'frozen_utc':datetime.now(timezone.utc).isoformat(),'test_scores_consumed':False})
    mark_original_diagnostic(base)
    frame=dataset();valid=frame[frame.asof_date.dt.year>=2019];probe=valid.iloc[:8]
    a=OriginIsolatedAdapter(base).fit();saved=RUN/'foundation_adapters'/f'{name}.json';a.save(saved)
    smoke_start=time.perf_counter();p,q=a.forecast(probe,'quarter');seconds=time.perf_counter()-smoke_start
    if not np.isfinite(p).all() or not np.isfinite(q).all():raise RuntimeError('Nonfinite smoke')
    changed=probe.copy()
    for c in changed:
        if c.startswith(('eps_lag_','filled_lag_','ttm_lag_')):changed.loc[changed.index[1:],c]=changed.loc[changed.index[1:],c]*3+7
        if c.startswith(('y_','label_')):changed[c]=1e12
    np.testing.assert_array_equal(p[0],a.forecast(changed,'quarter')[0][0])
    np.testing.assert_array_equal(p,a.forecast(probe.iloc[::-1],'quarter')[0][::-1])
    b=OriginIsolatedAdapter.load(saved);pp,qq=b.forecast(probe,'quarter')
    np.testing.assert_array_equal(p,pp);np.testing.assert_array_equal(q,qq);del b
    estimate=seconds/len(probe)*len(valid)*4
    remaining=(datetime.fromisoformat('2026-09-08T01:38:26+00:00')-datetime.now(timezone.utc)).total_seconds()
    if estimate*1.5>remaining:raise RuntimeError(f'Throughput guard: full plus replay estimate {estimate:.0f}s exceeds available {remaining:.0f}s with headroom')
    save_json(RUN/'smoke_receipts'/f'{name}.json',{'status':'PASS','rows':len(probe),'other_origin_mutation':'EXACT_PASS',
       'reverse_order':'EXACT_PASS','fresh_adapter_reload':'EXACT_PASS','full_and_replay_estimated_seconds':estimate,'spec':SPEC})
    outputs=[];began=time.perf_counter()
    for lane in ['quarter','ttm']:
        start=time.perf_counter();p,q=a.forecast(valid,lane,True);elapsed=time.perf_counter()-start
        if not np.isfinite(p).all() or not np.isfinite(q).all():raise RuntimeError('Nonfinite full prediction')
        for target in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
            j=3 if target=='ttm' else int(target[1:])-1
            outputs.append(prediction_frame(valid,target,name,p[:,j],elapsed,q[:,j],a.get_metadata()))
    save_predictions(name,outputs,{'family':a.metadata['family'],'metadata':a.get_metadata(),'spec':SPEC,
       'smoke_status':'PASS','runtime_seconds':time.perf_counter()-began,'corrected_from':base,'not_an_independent_architecture':True})
    evaluate_all();print(name,'FULL_RESEARCH_SCORED',flush=True)


def replay(base):
    name=base+'_origin_isolated';a=OriginIsolatedAdapter.load(RUN/'foundation_adapters'/f'{name}.json')
    frame=dataset();valid=frame[frame.asof_date.dt.year>=2019];saved=pd.read_parquet(RUN/'predictions'/f'{name}.parquet');records=[]
    for lane in ['quarter','ttm']:
        p,q=a.forecast(valid,lane,True)
        for target in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
            j=3 if target=='ttm' else int(target[1:])-1;old=saved[saved.target_key==target].set_index('sample_id').loc[valid.sample_id]
            np.testing.assert_array_equal(p[:,j],old.predicted_eps)
            np.testing.assert_array_equal(q[:,j],old[['p10_eps','p50_eps','p90_eps']])
            records.append({'target':target,'status':'EXACT_PASS','rows':len(valid)})
        save_json(RUN/'origin_isolation_replay'/f'{name}.json',{'all_pass':True,'records':records,'fresh_process':True,'refit':False,'dataset_sha256':sha(RUN/'data/samples.parquet')})
    print(name,'FRESH_FULL_REPLAY_PASS',flush=True)


if __name__=='__main__':
    action,base=sys.argv[1:];torch.set_num_threads(4)
    try:score(base) if action=='score' else replay(base)
    except Exception as exc:
        name=base+'_origin_isolated';record={'model_id':name,'status':'BROKEN','action':action,'failure_reason':str(exc),'traceback':traceback.format_exc()}
        save_json(RUN/('model_receipts' if action=='score' else 'origin_isolation_replay')/f'{name}.json',record)
        print('ORIGIN_ISOLATION_FAILED',base,action,str(exc),flush=True);raise
