"""Two genuinely new fits, isolated identities; never select by repeated scores."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v2.common import RUN, read_json, save_json, sha, utcnow
from research.eps_model_lab_v2.model_common import TARGETS, dataset
from research.eps_model_lab_v2.evaluation import gate


def plan():
    lock=read_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')
    confirmation=pd.read_csv(RUN/'EPS_V2_LOCKED_OOF_RESULTS.csv')
    records=[]
    for r in lock['development_gate_passing_representatives']:
        row=confirmation[(confirmation.model_id==r['model_id'])&(confirmation.target==r['target'])]
        if len(row)!=1:raise RuntimeError('Incomplete confirmation')
        reasons=gate(row.iloc[0].to_dict())
        records.append({**r,'confirmation_gate_pass':not reasons,'confirmation_failure_reasons':reasons})
    singles={r['model_id'] for r in records if r['confirmation_gate_pass']}
    # Frozen ensembles keep original members even when a member individually
    # fails confirmation. Such members are verified as components, not rescued.
    ens=pd.read_csv(RUN/'EPS_V2_ENSEMBLE_CONFIRMATION_RESULTS.csv')
    plans=read_json(RUN/'EPS_V2_ENSEMBLE_PLAN.json')['plans'];ensembles=[];components=set()
    for row in ens.to_dict('records'):
        reasons=gate(row)
        record={'model_id':row['model_id'],'target':row['target'],'confirmation_gate_pass':not reasons,'failure_reasons':reasons}
        if not reasons:
            matches=[p for p in plans if row['model_id'].startswith('ENS_'+p['track']+'_'+p['target']+'_')]
            if len(matches)!=1:raise RuntimeError('Ambiguous frozen ensemble')
            record['member_ids']=matches[0]['member_ids'];components.update(record['member_ids'])
        ensembles.append(record)
    save_json(RUN/'prescore/RETRAINING_VERIFICATION_PLAN.json',{
        'created_utc':utcnow(),'individual_gates':records,'ensemble_gates':ensembles,
        'model_ids':sorted(singles|components),'individual_survivor_candidate_ids':sorted(singles),
        'ensemble_component_ids':sorted(components),'new_fit_replicates':['repeat1','repeat2'],
        'years_by_phase':{'confirmation':[2019,2020,2021],'monitor':[2022,2023,2024,2025,2026]},
        'main_fit_plus_two_new_fits':True,'monitor_used_for_selection':False,
        'same_seed':1729,'same_machine_software_only':True,
        'tolerance':'Prediction arrays must match exactly; byte checkpoint hashes separately disclosed',
        'no_weight_or_recipe_updates':True},immutable=True)
    print('V2_RETRAINING_PLAN',sorted(singles|components),flush=True)


def compare(recipe,year,phase,replicate,target=None):
    name=recipe['model_id'];stem=str(year)+(('_'+target) if target else '')
    base=RUN/'predictions'/phase/name/'main'/f'{stem}.parquet'
    repeated=RUN/'predictions'/phase/name/replicate/f'{stem}.parquet'
    a=pd.read_parquet(base).sort_values(['target','sample_id']).reset_index(drop=True)
    b=pd.read_parquet(repeated).sort_values(['target','sample_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_exact=True)
    root=RUN/'models'/phase/name
    if target:
        left=root/'main'/f'{stem}.pkl';right=root/replicate/f'{stem}.pkl'
    elif recipe['family'] in ['Chronos2','Chronos2ZeroShot']:
        left=root/'main'/stem/'final/model.safetensors';right=root/replicate/stem/'final/model.safetensors'
    else:
        left=root/'main'/f'{stem}.pt';right=root/replicate/f'{stem}.pt'
    zero=recipe['family']=='Chronos2ZeroShot'
    byte_equal=sha(left)==sha(right) if not zero else None
    state_exact=None
    if not target and not zero and recipe['family']!='Chronos2':
        import torch
        state_a=torch.load(left,map_location='cpu',weights_only=True)['state_dict']
        state_b=torch.load(right,map_location='cpu',weights_only=True)['state_dict']
        state_exact=state_a.keys()==state_b.keys() and all(torch.equal(state_a[k],state_b[k]) for k in state_a)
        if not state_exact:raise AssertionError('Fresh learned state mismatch')
    # Files may include nonlearned serialization metadata; retain both outcomes.
    return {'model_id':name,'family':recipe['family'],'phase':phase,'year':year,'target':target,
        'replicate':replicate,'status':'PASS','predictions_exact':True,'prediction_rows':len(a),
        'prediction_file_bytes_equal':sha(base)==sha(repeated),
        'checkpoint_file_bytes_equal':byte_equal,'checkpoint_tensor_state_exact':state_exact,
        'main_prediction_sha256':sha(base),'repeat_prediction_sha256':sha(repeated),
        'actual_new_fit':not zero,'zero_shot_inference_only':zero}


def cpu_one(recipe,year,phase,replicate,target):
    from research.eps_model_lab_v2.tabular import one
    one(recipe,year,target,phase,replicate=replicate)
    return compare(recipe,year,phase,replicate,target)


def run(kind,replicate):
    spec=read_json(RUN/'prescore/RETRAINING_VERIFICATION_PLAN.json')
    if replicate not in spec['new_fit_replicates']:raise ValueError('Unplanned repeat')
    families={'prob':['Ridge','HistGB','NGBoostLaplace'],'CatBoost':['CatBoost'],
        'neural':['TimeXer','SOFTS','XLinear','DLinear','LSTM'],'chronos':['Chronos2','Chronos2ZeroShot']}[kind]
    recipes=[r for r in read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['recipes'] if r['model_id'] in spec['model_ids'] and r['family'] in families]
    jobs=[(r,y,phase,replicate) for r in recipes for phase,years in spec['years_by_phase'].items() for y in years]
    results=[];failures=[]
    if kind in ['prob','CatBoost']:
        with ProcessPoolExecutor(max_workers=16) as pool:
            futures={pool.submit(cpu_one,*j,t):(j[0]['model_id'],j[1],j[2],t) for j in jobs for t in TARGETS}
            for future in as_completed(futures):
                try:results.append(future.result())
                except Exception as exc:failures.append({'task':futures[future],'error':str(exc),'traceback':traceback.format_exc()})
    else:
        from research.eps_model_lab_v2.sequence import frozen_tensors
        from research.eps_model_lab_v2 import sequence,chronos_specialization
        frame=dataset();data=frozen_tensors(frame)
        for recipe,year,phase,rep in jobs:
            try:
                (sequence if kind=='neural' else chronos_specialization).one(recipe,year,phase,frame,data,replicate=rep)
                results.append(compare(recipe,year,phase,rep))
            except Exception as exc:
                failures.append({'task':[recipe['model_id'],year,phase],'error':str(exc),'traceback':traceback.format_exc()})
                break
    expected=len(jobs)*(5 if kind in ['prob','CatBoost'] else 1)
    save_json(RUN/'audit'/f'FRESH_RETRAIN_{kind}_{replicate}.json',{
        'created_utc':utcnow(),'status':'PASS' if not failures and len(results)==expected else 'FAIL',
        'expected':expected,'completed':len(results),'results':results,'failures':failures,
        'plan_sha256':sha(RUN/'prescore/RETRAINING_VERIFICATION_PLAN.json')},immutable=True)
    print('V2_FRESH_RETRAIN',kind,replicate,len(results),'/',expected,'failures',len(failures),flush=True)
    if failures:raise RuntimeError('Fresh retraining mismatch retained; no tolerance relaxation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--kind');p.add_argument('--replicate');a=p.parse_args()
    if a.plan:plan()
    else:run(a.kind,a.replicate)
