"""Official IBM PatchTST-FM-r1; FP32 smoke then audited native BF16 execution."""
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
os.environ['HF_HOME']=str(PROJECT.parent/'.cache/eps_models')
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all
from research.eps_model_lab_v1.foundation_models import histories


class IBMAdapter(EPSAdapter):
    def __init__(self):
        self.receipt=json.loads((RUN/'weight_receipts/ibm-granite__granite-timeseries-patchtst-fm-r1.json').read_text(encoding='utf-8'))
        super().__init__('ibm_patchtst_fm',{'family':'IBM_PATCHTST_FOUNDATION','source_commit':self.receipt['revision'],
             'license':self.receipt['license'],'pit_valid':False,'causal':True,'zero_shot':True,
             'evidence_class':'RETROSPECTIVE_RESEARCH_PRETRAINING_OVERLAP_UNRESOLVED',
             'precision':'FP32 smoke; official native BF16 autocast for full score',
             'input_geometry':'32 quarterly / 8 TTM contexts, official masked left padding to8192; missing raw fiscal quarters preserved',
             'point_estimator':'Official approximate distribution mean from 99-quantile integration',
             'source':'https://github.com/ibm-granite/granite-tsfm'})
    def fit(self,frame=None,target=None):
        from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction
        self.model=PatchTSTFMForPrediction.from_pretrained(self.receipt['path']).to('cuda').eval();self.fitted=True;return self
    def prepare_data(self,frame,lane='quarter'): return histories(frame,lane)[0]
    def forecast(self,frame,lane,batch=16):
        raw=self.prepare_data(frame,lane);points=[];quantiles=[]
        with torch.inference_mode():
            for start in range(0,len(raw),batch):
                x=torch.tensor(raw[start:start+batch],device='cuda')[:,:,None]
                out=self.model(past_values=x,prediction_length=4,quantile_levels=[.1,.5,.9])
                points.append(out.prediction_outputs[:,:,0].float().cpu().numpy())
                quantiles.append(out.quantile_outputs[:,:,:,0].permute(0,2,1).float().cpu().numpy())
                if start%256==0: print('ibm_patchtst_fm',lane,start,'/',len(raw),flush=True)
        return np.concatenate(points),np.concatenate(quantiles)
    def predict(self,frame,target):
        p,_=self.forecast(frame,'ttm' if target=='ttm' else 'quarter');return p[:,3 if target=='ttm' else int(target[1:])-1]
    def save(self,path): save_json(path,{'model_id':self.model_id,'metadata':self.get_metadata(),'weights':self.receipt})
    @classmethod
    def load(cls,path): return cls().fit()


def run():
    import tsfm_public.models.patchtst_fm.modeling_patchtst_fm as native
    torch.set_num_threads(4);torch.manual_seed(1729);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    frame=dataset();valid=frame[frame.asof_date.dt.year>=2019];a=IBMAdapter().fit();smoke=valid.iloc[:4]
    # Precision override is process-local and restored; no upstream source/config/weights change.
    original=native.get_autocast_context
    try:
        native.get_autocast_context=lambda device: contextlib.nullcontext()
        p32,q32=a.forecast(smoke,'quarter',batch=4)
    finally: native.get_autocast_context=original
    p,q=a.forecast(smoke,'quarter',batch=4)
    assert p.shape==(4,4) and np.isfinite(p32).all() and np.isfinite(p).all() and np.isfinite(q).all()
    drift=float(np.max(np.abs(p-p32)/smoke.scale.to_numpy()[:,None]))
    # Predeclared broad numerical sanity bound, not score selection or a bit-equivalence claim.
    if drift>.05: raise RuntimeError(f'BF16 smoke origin-scale drift exceeds .05: {drift}')
    changed=smoke.copy()
    for c in changed:
        if c.startswith(('eps_lag_','ttm_lag_')): changed.loc[changed.index[1:],c]=777.
    np.testing.assert_allclose(p[0],a.forecast(changed,'quarter',batch=4)[0][0],rtol=1e-5,atol=1e-5)
    path=RUN/'foundation_adapters/ibm_patchtst_fm.json';a.save(path)
    save_json(RUN/'smoke_receipts/ibm_patchtst_fm_summary.json',{'status':'PASS','FP32_finite':True,'BF16_finite':True,
         'max_point_origin_scale_precision_drift':drift,'cross_origin_mutation':'PASS','source_module_sha256':sha(native.__file__)})
    outputs=[]
    for lane in ['quarter','ttm']:
        began=time.perf_counter();p,q=a.forecast(valid,lane);duration=time.perf_counter()-began
        for t in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
            j=3 if t=='ttm' else int(t[1:])-1
            outputs.append(prediction_frame(valid,t,a.model_id,p[:,j],duration,q[:,j,:],a.get_metadata()))
    save_predictions(a.model_id,outputs,{'family':'IBM_PATCHTST_FOUNDATION','metadata':a.get_metadata(),'runtime_seconds':time.perf_counter()-started,
             'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'smoke_status':'PASS'})
    evaluate_all();print(a.model_id,'FULL_SCORE_COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':
    try: run()
    except Exception as exc:
        save_json(RUN/'model_receipts/ibm_patchtst_fm.json',{'model_id':'ibm_patchtst_fm','status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
        print('IBM_PATCHTST_BROKEN',str(exc),flush=True)
