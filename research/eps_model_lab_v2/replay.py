"""Actual new-process saved model inference and input-independence auditing."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import pickle
import sys
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v2.common import RUN,read_json,save_json,sha,utcnow,check_stop
from research.eps_model_lab_v2.model_common import dataset,TARGETS


def replay_tabular(path_string):
    from threadpoolctl import threadpool_limits
    from research.eps_model_lab_v2.tabular import TabularModel
    path=Path(path_string);receipt=read_json(path.with_suffix('.receipt.json'))
    assert sha(path)==receipt['model_sha256']
    truth_path=Path(receipt['prediction_path']);assert sha(truth_path)==receipt['prediction_sha256']
    frame=dataset();valid=frame[frame.asof_date.dt.year==receipt['year']]
    with path.open('rb') as stream:state=pickle.load(stream)
    model=TabularModel(state['recipe'],state['target']);model.__dict__.update(state)
    expected=pd.read_parquet(truth_path).set_index('sample_id').loc[valid.sample_id]
    with threadpool_limits(limits=1):
        p,q=model.predict(valid);np.testing.assert_array_equal(p,expected.forecast_eps.to_numpy())
        if q is not None:np.testing.assert_array_equal(q,expected[['q10','q50','q90']].to_numpy())
        probe=valid.iloc[:8].copy();point=model.predict(probe)[0]
        mutated=probe.copy()
        for c in mutated:
            if c.startswith(('y_','label_','target_','account_')):mutated[c]=1e12
        np.testing.assert_array_equal(point,model.predict(mutated)[0])
        for c in mutated:
            if c.startswith(('eps_lag_','ttm_lag_','observed_lag_')):mutated.loc[mutated.index[1:],c]=12345.
        np.testing.assert_allclose(point[:1],model.predict(mutated)[0][:1],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(point[:1],model.predict(probe.iloc[:1])[0],rtol=1e-12,atol=1e-12)
    return {'path':str(path),'model_id':receipt['model_id'],'year':receipt['year'],'target':receipt['target'],
        'status':'PASS','actual_new_process_saved_inference':True,'all_rows_exact':len(p),'quantiles_exact':q is not None,
        'future_labels_other_accounting_mutation':True,'other_origin_and_singleton_tolerance':1e-12,
        'model_sha256':sha(path),'prediction_sha256':sha(truth_path)}


def tabular(phase,family=None):
    recipes={r['model_id']:r for r in read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['recipes']}
    paths=[p for p in sorted((RUN/'models'/phase).glob('*/main/*.pkl'))
        if (recipes[p.parents[1].name]['family']==family if family else recipes[p.parents[1].name]['family']!='CatBoost')]
    results=[];failures=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        jobs={pool.submit(replay_tabular,str(p)):str(p) for p in paths}
        for future in as_completed(jobs):
            try:results.append(future.result())
            except Exception as exc:failures.append({'path':jobs[future],'error':str(exc),'traceback':traceback.format_exc()})
    save_json(RUN/'audit'/f'FRESH_PROCESS_TABULAR_REPLAY_{phase}_{family or "prob"}.json',
        {'created_utc':utcnow(),'status':'PASS' if not failures and len(results)==len(paths) else 'FAIL',
         'expected':len(paths),'completed':len(results),'results':results,'failures':failures},immutable=True)
    print('V2_FRESH_TABULAR_REPLAY',phase,family,len(results),'/',len(paths),'failures',len(failures),flush=True)


def neural(phase):
    from research.eps_model_lab_v2.sequence import deterministic,frozen_tensors,SequenceModel
    import torch
    import gc
    deterministic();frame=dataset();data=frozen_tensors(frame);results=[]
    for path in sorted((RUN/'models'/phase).glob('*/main/*.pt')):
        check_stop();receipt=read_json(path.with_suffix('.receipt.json'));assert sha(path)==receipt['model_sha256']
        model=SequenceModel.load(path);year=receipt['year'];indices=np.flatnonzero((frame.asof_date.dt.year==year).to_numpy())
        point=model.predict(data,indices);source=RUN/'predictions'/phase/receipt['model_id']/'main'/f'{year}.parquet'
        assert sha(source)==receipt['prediction_sha256'];stored=pd.read_parquet(source)
        for t in TARGETS:
            expected=stored[stored.target==t].set_index('sample_id').loc[frame.iloc[indices].sample_id,'forecast_eps'].to_numpy()
            np.testing.assert_array_equal(point[:,3,1] if t=='ttm' else point[:,int(t[1:])-1,0],expected)
        probe=indices[:8];first=model.predict(data,probe)[:1]
        mutated={k:v.copy() for k,v in data.items()};mutated['y'][:]=1e12;mutated['availability'][:]=0
        np.testing.assert_array_equal(first,model.predict(mutated,probe)[:1])
        for k in ['raw','mask','age','calendar','scale','base']:
            mutated[k][probe[1:]]=123.
        np.testing.assert_allclose(first,model.predict(mutated,probe)[:1],rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(first,model.predict(data,probe[:1]),rtol=1e-5,atol=1e-5)
        results.append({'path':str(path),'model_id':receipt['model_id'],'year':year,'status':'PASS',
            'saved_full_point_replay_exact':True,'targets':5,'rows':len(indices),
            'future_labels_and_other_origins_probed':True,'model_sha256':sha(path)})
        del model;gc.collect();torch.cuda.empty_cache()
    save_json(RUN/'audit'/f'FRESH_PROCESS_NEURAL_REPLAY_{phase}.json',{'created_utc':utcnow(),'status':'PASS','annual_models':len(results),'results':results},immutable=True)
    print('V2_FRESH_NEURAL_REPLAY',phase,len(results),flush=True)


def chronos(phase):
    from research.eps_model_lab_v2.chronos_specialization import forecast
    from research.eps_model_lab_v2.sequence import deterministic,frozen_tensors
    from chronos import Chronos2Pipeline
    from research.eps_model_lab_v2.common import V1
    import torch
    import gc
    deterministic();frame=dataset();data=frozen_tensors(frame);results=[]
    recipes={r['model_id']:r for r in read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['recipes']}
    for receipt_path in sorted((RUN/'models'/phase).glob('Chronos*/main/*/RECEIPT.json')):
        check_stop();r=read_json(receipt_path);root=receipt_path.parent;recipe=recipes[r['model_id']]
        for rel,digest in r['artifact_hashes'].items():assert sha(root/rel)==digest
        path=(read_json(V1/'weight_receipts/amazon__chronos-2.json')['path'] if recipe['family']=='Chronos2ZeroShot' else root/'final')
        pipeline=Chronos2Pipeline.from_pretrained(path,device_map='cuda',dtype=torch.float32,local_files_only=True)
        indices=np.flatnonzero((frame.asof_date.dt.year==r['year']).to_numpy());p,q=forecast(pipeline,data,indices,recipe)
        stored=pd.read_parquet(RUN/'predictions'/phase/r['model_id']/'main'/f"{r['year']}.parquet")
        for t in TARGETS:
            h,c=(3,1) if t=='ttm' else (int(t[1:])-1,0)
            expected=stored[stored.target==t].set_index('sample_id').loc[frame.iloc[indices].sample_id]
            np.testing.assert_array_equal(p[:,h,c],expected.forecast_eps.to_numpy())
            np.testing.assert_array_equal(q[:,h,c,:],expected[['q10','q50','q90']].to_numpy())
        probe=indices[:8];before=forecast(pipeline,data,probe,recipe)
        mutated={k:v.copy() for k,v in data.items()};mutated['y'][:]=1e12;mutated['availability'][:]=0
        for a,b in zip(before,forecast(pipeline,mutated,probe,recipe)):np.testing.assert_array_equal(a,b)
        for k in ['raw','base']:mutated[k][probe[1:]]=12345.
        for a,b in zip(before,forecast(pipeline,mutated,probe,recipe)):np.testing.assert_array_equal(a[:1],b[:1])
        for a,b in zip(before,forecast(pipeline,data,probe[:1],recipe)):np.testing.assert_allclose(a[:1],b,rtol=1e-5,atol=1e-5)
        results.append({'model_id':r['model_id'],'year':r['year'],'status':'PASS','point_quantile_targets':5,'rows':len(indices),'future_other_origin_probes':True})
        del pipeline;gc.collect();torch.cuda.empty_cache()
    save_json(RUN/'audit'/f'FRESH_PROCESS_CHRONOS_REPLAY_{phase}.json',{'created_utc':utcnow(),'status':'PASS','annual_models':len(results),'results':results},immutable=True)
    print('V2_FRESH_CHRONOS_REPLAY',phase,len(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['tabular','neural','chronos']);p.add_argument('--phase',default='development');p.add_argument('--family');a=p.parse_args()
    try:
        if a.kind=='tabular':tabular(a.phase,a.family)
        elif a.kind=='neural':neural(a.phase)
        else:chronos(a.phase)
    except Exception as exc:
        save_json(RUN/'failures'/f'replay_{a.kind}_{a.phase}_{os.getpid()}.json',{'created_utc':utcnow(),'error':str(exc),'traceback':traceback.format_exc()},immutable=True)
        raise
