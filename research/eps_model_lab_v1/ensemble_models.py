"""Score-blind-to-test portfolio selection, immutable weights, actual ensembles."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
os.environ['OMP_NUM_THREADS']='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy import sparse
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset,metric_row,prediction_frame,save_predictions,evaluate_all,atomic_csv

CUTOFF=pd.Timestamp('2022-01-01',tz='UTC')
EXCLUDED={'random_walk','random_walk_drift','seasonal_random_walk','seasonal_drift','historical_mean'}
TARGETS=['h1','h2','h3','h4','ttm']
TRACKS=['local_causal','mixed_retrospective']


def family(name):
    if name.startswith('ngboost'):return 'NGBOOST_DISTRIBUTIONAL'
    if name.startswith('gaussian_process_'):return 'GAUSSIAN_PROCESS_KERNEL'
    if name.startswith('nf_'): return 'NEURALFORECAST'
    if name.startswith('epspredict_'): return 'EPS_NATIVE'
    if name.startswith('mlforecast_'): return 'MLFORECAST'
    if name.startswith('ag_'): return 'AUTOGLUON'
    if name.startswith(('sf_','darts_')) or name in ['foster','brown_rozeff','griffin_watts']: return 'CLASSICAL_STATISTICAL'
    if name.endswith('_observed'): return 'OBSERVED_BASELINE'
    for prefix in ['chronos2','chronos_bolt','timesfm','tirex2','tirex_','toto2','moirai','lag_llama','ibm_','moment_','time_moe','sundial','tabpfn']:
        if name.startswith(prefix): return 'CHECKPOINT_'+prefix
    return 'TABULAR_ACCOUNTING'


def load_pool():
    frames={}
    for path in sorted((RUN/'predictions').glob('*.parquet')):
        if path.stem.startswith('ens_') or path.stem in EXCLUDED: continue
        receipt=json.loads((RUN/'model_receipts'/f'{path.stem}.json').read_text(encoding='utf-8'))
        if receipt.get('status')!='FULL_RESEARCH_SCORED': continue
        if sha(path)!=receipt['prediction_sha256']: raise RuntimeError('Prediction hash mismatch '+path.stem)
        frames[path.stem]=pd.read_parquet(path)
    return frames


def convex_l1(x,y):
    n,k=x.shape
    # x*w-y <= t and y-x*w <= t, w simplex; no fitted intercept or negative weight.
    a=sparse.vstack([sparse.hstack([sparse.csr_matrix(x),-sparse.eye(n)]),
                     sparse.hstack([-sparse.csr_matrix(x),-sparse.eye(n)])]).tocsr()
    result=linprog(np.r_[np.zeros(k),np.ones(n)/n],A_ub=a,b_ub=np.r_[y,-y],
       A_eq=sparse.csr_matrix(np.r_[np.ones(k),np.zeros(n)][None,:]),b_eq=[1.],bounds=(0,None),method='highs')
    if not result.success: raise RuntimeError('Convex OOF stack solver failed: '+result.message)
    w=result.x[:k]
    if (w < -1e-9).any() or abs(w.sum()-1)>1e-8: raise RuntimeError('Convex weights failed gate')
    w=np.maximum(w,0);w=w/w.sum()
    return w,{'objective_train_only_MAE':float(np.mean(np.abs(x@w-y))),'solver':'scipy.linprog.Highs',
              'status':int(result.status),'success':bool(result.success),'weight_sum':float(w.sum()),'minimum_weight':float(w.min())}


def select(pool,source,target,track,cutoff):
    rows=source[(source.asof_date>=pd.Timestamp('2019-01-01',tz='UTC'))&(source.asof_date<cutoff)
      &(source['label_asof_'+target]<cutoff)&source['y_'+target].notna()].set_index('sample_id')
    if len(rows)<100: raise RuntimeError('Insufficient eligible OOF selection labels')
    candidates={};scores=[]
    for name,p in pool.items():
        if track=='local_causal' and not p.pit_valid.all(): continue
        g=p[p.target_key==target].set_index('sample_id').reindex(rows.index)
        if not np.isfinite(g.predicted_eps).all(): continue
        if not (g.asof_date<cutoff).all() or not (g.label_asof<cutoff).all(): raise RuntimeError('Ineligible meta training timestamp')
        if not np.allclose(g.actual_eps,rows['y_'+target],rtol=0,atol=0): raise RuntimeError('OOF truth mismatch')
        score=metric_row(g)
        if not all(np.isfinite(score.get(k,np.nan)) for k in ['MAE','MedianAE','price_scaled_MAE']): continue
        candidates[name]=g.predicted_eps.to_numpy()
        scores.append({'model_id':name,'family':family(name),**score})
    rank=pd.DataFrame(scores).set_index('model_id')
    rank['rank_score']=rank[['MAE','MedianAE','price_scaled_MAE']].rank(pct=True).mean(axis=1)
    y=rows['y_'+target].to_numpy();chosen=[];trace=[]
    for step in range(5):
        options=[]
        for name in rank.index:
            if name in chosen: continue
            f=family(name);max_family=1 if f.startswith('CHECKPOINT_') else 2
            if sum(family(c)==f for c in chosen)>=max_family: continue
            if any(np.array_equal(candidates[name],candidates[c]) for c in chosen): continue
            correlations=[]
            for c in chosen:
                corr=np.corrcoef(candidates[name]-y,candidates[c]-y)[0,1]
                correlations.append(abs(float(corr)) if np.isfinite(corr) else 1.)
            corr=max(correlations,default=0.)
            options.append((float(rank.loc[name,'rank_score'])+.15*corr,name,corr))
        if not options: raise RuntimeError('Cannot construct five distinct eligible members')
        value,name,corr=min(options);chosen.append(name)
        trace.append({'member':name,'selection_objective':value,'rank_score':float(rank.loc[name,'rank_score']),
                      'maximum_absolute_residual_corr_prior':corr,'family':family(name)})
    x=np.column_stack([candidates[n] for n in chosen]);w,solver=convex_l1(x,y)
    return {'target':target,'track':track,'cutoff':str(cutoff),'member_ids':chosen,'convex_l1_weights':w.tolist(),
      'selection_rows':len(rows),'selection_sample_ids':rows.index.tolist(),
      'maximum_origin':str(rows.asof_date.max()),'maximum_label_availability':str(rows['label_asof_'+target].max()),
      'selection_trace':trace,'stack_fit':solver,'candidate_count':len(rank)},rank.reset_index()


def freeze():
    path=RUN/'ENSEMBLE_FREEZE_V1.json'
    if path.exists(): raise RuntimeError('Ensemble already frozen; new identity required for any new membership/weights')
    pool=load_pool();source=dataset();plans=[];ranks=[]
    for track in TRACKS:
        for target in TARGETS:
            plan,rank=select(pool,source,target,track,CUTOFF);plans.append(plan)
            ranks.append(rank.assign(track=track,target_key=target))
    save_json(path,{'frozen_utc':datetime.now(timezone.utc).isoformat(),'test_scores_consumed':False,
       'cutoff':str(CUTOFF),'contract_sha256':sha(PROJECT/'research/eps_model_lab_v1/ENSEMBLE_SELECTION_CONTRACT.md'),
       'source_sha256':sha(__file__),'samples_sha256':sha(RUN/'data/samples.parquet'),
       'candidate_prediction_hashes':{n:sha(RUN/'predictions'/f'{n}.parquet') for n in pool},'plans':plans,
       'formal_certification':False,'validation_prediction_policy':'Unavailable: selection/fitting surface, not OOS ensemble performance'})
    atomic_csv(pd.concat(ranks,ignore_index=True),RUN/'ENSEMBLE_SELECTION_CANDIDATE_RANKS.csv')
    print('ENSEMBLE_MEMBERS_AND_WEIGHTS_FROZEN',len(plans),'plans',flush=True)


def aggregate(x,w,method):
    if x.shape[1]!=5: raise RuntimeError('Exactly five actual members required')
    valid=np.isfinite(x).all(axis=1);result=np.full(len(x),np.nan);v=x[valid]
    if method=='mean': result[valid]=v.mean(axis=1)
    elif method=='median': result[valid]=np.median(v,axis=1)
    elif method=='trimmed_mean': result[valid]=np.sort(v,axis=1)[:,1:-1].mean(axis=1)
    elif method=='convex_l1': result[valid]=v@np.asarray(w)
    else: raise ValueError(method)
    return result


def predict():
    lock=json.loads((RUN/'ENSEMBLE_FREEZE_V1.json').read_text(encoding='utf-8'))
    if sha(RUN/'data/samples.parquet')!=lock['samples_sha256']: raise RuntimeError('Frozen samples drift')
    pool={}
    for n,h in lock['candidate_prediction_hashes'].items():
        path=RUN/'predictions'/f'{n}.parquet'
        if sha(path)!=h: raise RuntimeError('Frozen base prediction drift '+n)
        pool[n]=pd.read_parquet(path)
    frame=dataset().loc[lambda f:f.asof_date.dt.year>=2019]
    test=frame.asof_date>=CUTOFF
    for track in TRACKS:
        for method in ['mean','median','trimmed_mean','convex_l1']:
            name=f'ens_{track}_{method}';outputs=[]
            for plan in lock['plans']:
                if plan['track']!=track: continue
                target=plan['target'];columns=[]
                for member in plan['member_ids']:
                    p=pool[member];g=p[p.target_key==target].set_index('sample_id')
                    columns.append(g.reindex(frame.sample_id).predicted_eps.to_numpy())
                x=np.column_stack(columns);values=aggregate(x,plan['convex_l1_weights'],method)
                values[~test.to_numpy()]=np.nan
                result=prediction_frame(frame,target,name,values,metadata={'pit_valid':track=='local_causal','version':'FROZEN_2022_META_V1'})
                result.loc[~test.to_numpy(),'coverage_flag']='META_SELECTION_FIT_SURFACE_NOT_OOS'
                result['ensemble_member_count']=5;result['ensemble_members']='|'.join(plan['member_ids'])
                result['ensemble_weight_fit_cutoff']=str(CUTOFF)
                if target=='ttm': result['ttm_prediction_method']='ENSEMBLE_OF_NATIVE_TTM_TARGET_FORECASTS_MEMBER_METHODS_RETAINED'
                outputs.append(result)
            save_predictions(name,outputs,{'family':'ENSEMBLE','metadata':{'pit_valid':track=='local_causal','zero_shot':False},
                'ensemble_freeze_sha256':sha(RUN/'ENSEMBLE_FREEZE_V1.json'),'track':track,'method':method,
                'smoke_status':'FIVE_MEMBERS_CONVEX_WEIGHTS_AND_AVAILABILITY_GATES','validation_unavailable_by_design':True})
            print(name,'RESEARCH_TEST_PREDICTED',flush=True)
    evaluate_all()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','predict']);args=parser.parse_args()
    freeze() if args.action=='freeze' else predict()
