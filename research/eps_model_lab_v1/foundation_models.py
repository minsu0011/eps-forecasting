"""Official foundation APIs, independent origin tasks, no cross-origin attention."""
from __future__ import annotations
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
os.environ['HF_HOME']=str(PROJECT.parent/'.cache/eps_models')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all

SPECS={
 'chronos2_zero_shot':('chronos','amazon/chronos-2','chronos-forecasting'),
 'chronos2_past_covariates':('chronos_cov','amazon/chronos-2','chronos-forecasting'),
 'chronos_bolt_mini':('bolt','amazon/chronos-bolt-mini','chronos-forecasting'),
 'timesfm_2p5':('timesfm','google/timesfm-2.5-200m-pytorch','timesfm'),
 'tirex_zero_shot':('tirex','NX-AI/TiRex','tirex-ts'),
 'toto2_4m':('toto','Datadog/Toto-2.0-4m','toto-2'),
 'toto2_22m':('toto','Datadog/Toto-2.0-22m','toto-2'),
}

def histories(frame,lane):
    if lane=='quarter':
        raw=frame[[f'eps_lag_{i}' for i in range(31,-1,-1)]].to_numpy(dtype=np.float32)
        filled=frame[[f'filled_lag_{i}' for i in range(31,-1,-1)]].to_numpy(dtype=np.float32)
    else:
        raw=frame[[f'ttm_lag_{i}' for i in range(7,-1,-1)]].to_numpy(dtype=np.float32)
        filled=raw.copy()
        for j in range(filled.shape[1]):
            replace=filled[:,j-4] if j>=4 else (filled[:,j-1] if j else np.zeros(len(frame)))
            filled[:,j]=np.where(np.isfinite(filled[:,j]),filled[:,j],replace)
    return raw,filled

class FoundationAdapter(EPSAdapter):
    def __init__(self,model_id):
        self.kind,self.repo,self.package=SPECS[model_id]
        self.receipt=json.loads((RUN/'weight_receipts'/(self.repo.replace('/','__')+'.json')).read_text(encoding='utf-8'))
        super().__init__(model_id,{'family':'FOUNDATION_ZERO_SHOT','version':importlib.metadata.version(self.package),
            'source_commit':self.receipt['revision'],'pit_valid':False,'causal':True,'zero_shot':True,
            'evidence_class':'RETROSPECTIVE_RESEARCH_PRETRAINING_OVERLAP_UNRESOLVED','license':self.receipt['license'],
            'native_scaling':'Official causal per-context scaler; no cross-origin or future input',
            'cross_learning':False,'weight_path':self.receipt['path']})
    def prepare_data(self,frame): return histories(frame,'quarter')
    def fit(self,frame=None,target=None):
        path=self.receipt['path'];kind=self.kind
        if kind.startswith('chronos') or kind=='bolt':
            from chronos import BaseChronosPipeline
            self.model=BaseChronosPipeline.from_pretrained(path,device_map='cuda',dtype=torch.float32)
        elif kind=='timesfm':
            import timesfm
            self.model=timesfm.TimesFM_2p5_200M_torch.from_pretrained(path,torch_compile=False)
            self.model.compile(timesfm.ForecastConfig(max_context=64,max_horizon=16,per_core_batch_size=64,
                 normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,
                 infer_is_positive=False,fix_quantile_crossing=True))
        elif kind=='tirex':
            from tirex import load_model
            self.model=load_model(self.repo,device='cuda:0',backend='torch',compile=False,
                                  hf_kwargs={'revision':self.receipt['revision'],'local_files_only':True})
        elif kind=='toto':
            from toto2 import Toto2Model
            self.model=Toto2Model.from_pretrained(path).to('cuda').eval()
        self.fitted=True;return self
    def predict(self,frame,target): return self.forecast(frame,'ttm' if target=='ttm' else 'quarter')[0][:,3 if target=='ttm' else int(target[1:])-1]
    def forecast(self,frame,lane):
        raw,filled=histories(frame,lane);points=[];quantiles=[]
        for start in range(0,len(frame),64):
            r=raw[start:start+64];f=filled[start:start+64]
            with torch.inference_mode():
                if self.kind.startswith('chronos'):
                    inputs=torch.tensor(r)[:,None,:]
                    if self.kind=='chronos_cov':
                        inputs=[{'target':a,'past_covariates':{'observed':np.isfinite(a).astype(np.float32)}} for a in r]
                    q,p=self.model.predict_quantiles(inputs,prediction_length=4,quantile_levels=[.1,.5,.9],batch_size=64,cross_learning=False)
                    q=torch.stack(q).squeeze(1).cpu().numpy();p=torch.stack(p).squeeze(1).cpu().numpy()
                elif self.kind=='bolt':
                    q,p=self.model.predict_quantiles(torch.tensor(r),prediction_length=4,quantile_levels=[.1,.5,.9])
                    q=q.cpu().numpy();p=p.cpu().numpy()
                elif self.kind=='timesfm':
                    p,allq=self.model.forecast(horizon=4,inputs=[a for a in f])
                    q=allq[:,:,[1,5,9]]
                elif self.kind=='tirex':
                    allq,p=self.model.forecast(context=torch.tensor(f),prediction_length=4,batch_size=64,output_type='numpy')
                    q=allq[:,:,[0,4,8]]
                elif self.kind=='toto':
                    # Official Toto patch geometry requires a multiple of patch_size.
                    # Missing left padding is masked, never invented as observed history.
                    pad=(-r.shape[1])%self.model.config.patch_size
                    if pad: r=np.pad(r,((0,0),(pad,0)),constant_values=np.nan)
                    tensor=torch.tensor(np.nan_to_num(r),device='cuda')[:,None,:]
                    mask=torch.tensor(np.isfinite(r),device='cuda')[:,None,:]
                    batch={'target':tensor,'target_mask':mask,'series_ids':torch.arange(len(r),device='cuda')[:,None]}
                    allq=self.model.forecast(batch,horizon=4,decode_block_size=None,has_missing_values=True)
                    q=allq[[0,4,8],:,0,:].permute(1,2,0).cpu().numpy();p=q[:,:,1]
            points.append(np.asarray(p));quantiles.append(np.asarray(q))
        return np.concatenate(points),np.concatenate(quantiles)
    def save(self,path):
        save_json(path,{'model_id':self.model_id,'metadata':self.get_metadata(),'weights':self.receipt})
    @classmethod
    def load(cls,path):
        record=json.loads(Path(path).read_text(encoding='utf-8'));return cls(record['model_id']).fit()

def run(model_id):
    torch.set_num_threads(4);torch.manual_seed(1729)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    frame=dataset();valid=frame.loc[frame.asof_date.dt.year>=2019].reset_index(drop=True)
    adapter=FoundationAdapter(model_id).fit()
    smoke=valid.iloc[:8];p,q=adapter.forecast(smoke,'quarter')
    assert p.shape==(8,4) and q.shape==(8,4,3)
    assert np.isfinite(p).all() and np.isfinite(q).all()
    # Batch-context independence: perturb later origins, earlier item prediction must stay stable.
    changed=smoke.copy()
    for c in changed:
        if c.startswith(('eps_lag_','filled_lag_')): changed.loc[changed.index[1:],c]=777.
    check,_=adapter.forecast(changed,'quarter')
    np.testing.assert_allclose(p[0],check[0],rtol=2e-5,atol=2e-5)
    adapter.save(RUN/'foundation_adapters'/f'{model_id}.json')
    save_json(RUN/'smoke_receipts'/f'{model_id}_summary.json',{'status':'PASS','shape':list(p.shape),'finite':True,
         'cross_origin_mutation_test':'PASS','precision':'FP32','negative_outputs_permitted':True})
    output=[]
    for lane in ['quarter','ttm']:
        began=time.perf_counter();p,q=adapter.forecast(valid,lane);duration=time.perf_counter()-began
        for t in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
            j=3 if t=='ttm' else int(t[1:])-1
            output.append(prediction_frame(valid,t,model_id,p[:,j],duration,q[:,j,:],adapter.get_metadata()))
    save_predictions(model_id,output,{'family':'FOUNDATION_ZERO_SHOT','metadata':adapter.get_metadata(),'weight_receipt':self_path(adapter.repo),
        'runtime_seconds':time.perf_counter()-start,'GPU_time_seconds_including_load':time.perf_counter()-start,
        'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'smoke_status':'PASS'})
    evaluate_all()
    print(model_id,'FULL_SCORE_COMPLETE',round(time.perf_counter()-start,2),flush=True)

def self_path(repo): return str((RUN/'weight_receipts'/(repo.replace('/','__')+'.json')).relative_to(RUN))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('models',nargs='+',choices=list(SPECS));args=parser.parse_args()
    for model_id in args.models:
        try: run(model_id)
        except Exception as exc:
            save_json(RUN/'model_receipts'/f'{model_id}.json',{'model_id':model_id,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
            print(model_id,'BROKEN',str(exc),flush=True)
        gc.collect();torch.cuda.empty_cache()
