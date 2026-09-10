"""Five prespecified Chronos adaptations; all data windows are availability purged."""
from pathlib import Path
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import gc
import sys
import time
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from transformers import TrainerCallback
from research.eps_model_lab_v2.common import RUN,V1,read_json,save_json,sha,utcnow,check_stop
from research.eps_model_lab_v2.model_common import dataset,TARGETS,predictions
from research.eps_model_lab_v2.sequence import deterministic,frozen_tensors


class StopGuard(TrainerCallback):
    def on_step_begin(self,args,state,control,**kwargs):check_stop()


def task_inputs(frame,data,recipe,year):
    cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC');mask=(frame.asof_date<cutoff).to_numpy(copy=True)
    if recipe['chronological_mode']=='rolling':mask &= (frame.asof_date>=cutoff-pd.DateOffset(years=5)).to_numpy()
    out=[];dates=[];residual=recipe['representation']=='residual'
    for i in np.flatnonzero(mask):
        for c in range(2):
            history=data['raw'][i,:,c].copy()
            future=np.where(data['availability'][i,:,c]<cutoff.value,data['y'][i,:,c],np.nan)
            row=history if recipe['representation']=='native_history_only' else np.r_[history,future]
            if residual:row=row-data['base'][i,c]
            valid=np.flatnonzero(np.isfinite(row))
            if not len(valid):continue
            row=row[:valid[-1]+1]
            if len(row)>=12 and np.isfinite(row).sum()>=8:
                out.append(row.astype('float32'));dates.append(frame.iloc[i].asof_date)
    if not out:raise RuntimeError('No Chronos train tasks')
    initial=len(out)
    if recipe['chronological_mode']=='recency_weighted':
        ages=np.array([(cutoff-d).total_seconds() for d in dates]);p=np.exp2(-ages/(2*365.2425*86400));p/=p.sum()
        indices=np.random.default_rng(1729).choice(len(out),size=len(out),replace=True,p=p)
        out=[out[j] for j in indices]
    return out,{'train_origins':int(mask.sum()),'native_tasks_before_weighted_resample':initial,
        'actual_train_tasks':len(out),'cutoff':str(cutoff),
        'recency_policy':'Fixed-seed probability-proportional resampling of tasks with two-year half-life' if recipe['chronological_mode']=='recency_weighted' else None,
        'joint_meaning':'Shared weights over quarterly and native-TTM tasks, individually masked future labels; not fictitious cross-company multivariate series'}


def forecast(pipeline,data,indices,recipe):
    pipeline.model.eval();raw=data['raw'][indices].copy();residual=recipe['representation']=='residual'
    if residual:raw-=data['base'][indices,None,:]
    points=[];quantiles=[]
    for c in range(2):
        pp=[];qq=[]
        for start in range(0,len(indices),64):
            x=torch.as_tensor(raw[start:start+64,:,c].copy())[:,None,:]
            q,p=pipeline.predict_quantiles(x,prediction_length=4,quantile_levels=[.1,.5,.9],batch_size=64,cross_learning=False)
            pp.append(torch.stack(p).squeeze(1).cpu().numpy());qq.append(torch.stack(q).squeeze(1).cpu().numpy())
        points.append(np.concatenate(pp));quantiles.append(np.concatenate(qq))
    p=np.stack(points,axis=-1);q=np.stack(quantiles,axis=2)
    if residual:p+=data['base'][indices,None,:];q+=data['base'][indices,None,:,None]
    return p,q


def one(recipe,year,phase,frame,data,replicate='main',steps=None):
    deterministic();check_stop();start=time.perf_counter();name=recipe['model_id']
    root=RUN/'models'/phase/name/replicate/str(year)
    if root.exists():raise FileExistsError('Existing Chronos attempt: inspect/replay, never overwrite')
    weight=read_json(V1/'weight_receipts/amazon__chronos-2.json')
    for file in weight['files']:
        if sha(file['path'])!=file['sha256']:raise RuntimeError('Frozen original foundation weight changed')
    pipeline=Chronos2Pipeline.from_pretrained(weight['path'],device_map='cuda',dtype=torch.float32,local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    if recipe['family']=='Chronos2ZeroShot':
        audit={'train_origins':0,'actual_train_tasks':0,'zero_shot':True};root.mkdir(parents=True)
        save_json(root/'WEIGHT_REFERENCE.json',weight,immutable=True)
    else:
        inputs,audit=task_inputs(frame,data,recipe,year)
        pipeline=pipeline.fit(inputs,prediction_length=4,finetune_mode='full',context_length=32,min_past=8,
            learning_rate=1e-6,num_steps=steps if steps is not None else 300,batch_size=64,output_dir=root,
            finetuned_ckpt_name='final',validation_inputs=None,bf16=False,tf32=False,fp16=False,
            seed=1729,data_seed=1729,full_determinism=False,disable_tqdm=True,logging_steps=100,
            report_to='none',remove_printer_callback=True,optim='adamw_torch_fused',dataloader_num_workers=0,
            callbacks=[StopGuard()])
    pipeline.model.eval();indices=np.flatnonzero((frame.asof_date.dt.year==year).to_numpy())
    p,q=forecast(pipeline,data,indices,recipe)
    probe=indices[:8]
    before=forecast(pipeline,data,probe,recipe)
    reversed_out=forecast(pipeline,data,probe[::-1],recipe)
    for a,b in zip(before,reversed_out):np.testing.assert_array_equal(a,b[::-1])
    del pipeline;gc.collect();torch.cuda.empty_cache()
    saved=weight['path'] if recipe['family']=='Chronos2ZeroShot' else root/'final'
    loaded=Chronos2Pipeline.from_pretrained(saved,device_map='cuda',dtype=torch.float32,local_files_only=True)
    repeated=forecast(loaded,data,indices,recipe)
    np.testing.assert_array_equal(p,repeated[0]);np.testing.assert_array_equal(q,repeated[1])
    del loaded;gc.collect();torch.cuda.empty_cache()
    rows=[];valid=frame.iloc[indices]
    for target in TARGETS:
        h,c=(3,1) if target=='ttm' else (int(target[1:])-1,0)
        rows.append(predictions(valid,target,name,p[:,h,c],q[:,h,c,:],track='RETROSPECTIVE_FOUNDATION'))
    path=RUN/'predictions'/phase/name/replicate/f'{year}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    pd.concat(rows,ignore_index=True).to_parquet(path,index=False)
    files=[p for p in root.rglob('*') if p.is_file()]
    audit.update(created_utc=utcnow(),status='PASS',model_id=name,year=year,phase=phase,
        seconds=time.perf_counter()-start,prediction_sha256=sha(path),
        artifact_hashes={str(p.relative_to(root)):sha(p) for p in files},
        same_process_saved_point_quantile_replay_exact=True,fresh_process_replay_pending=True,
        original_weight_revision=weight['revision'],pretraining_historical_overlap_certified=False,
        actual_CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    save_json(root/'RECEIPT.json',audit,immutable=True)
    print('V2_CHRONOS_PASS',name,year,round(audit['seconds'],2),flush=True)
    return audit


def run(phase='development',smoke=False):
    deterministic();frame=dataset();data=frozen_tensors(frame)
    recipes=[r for r in read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['recipes'] if r['family'] in ['Chronos2','Chronos2ZeroShot'] and not r.get('status')]
    years=[2015,2016,2017,2018]
    if phase!='development':
        selected=read_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')['selected_model_ids'];recipes=[r for r in recipes if r['model_id'] in selected]
        years=[2019,2020,2021] if phase=='confirmation' else [2022,2023,2024,2025,2026]
    if smoke:recipes=recipes[:1];years=[2015];phase='chronos_smoke'
    for recipe in recipes:
        for year in years:
            try:one(recipe,year,phase,frame,data,steps=3 if smoke else None)
            except Exception as exc:
                save_json(RUN/'failures'/f'chronos_{phase}_{recipe["model_id"]}_{year}_{os.getpid()}.json',
                    {'created_utc':utcnow(),'error':str(exc),'traceback':traceback.format_exc()},immutable=True)
                raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='development');p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.phase,a.smoke)
