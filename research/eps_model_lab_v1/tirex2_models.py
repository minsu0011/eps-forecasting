"""Official TiRex-2 CPU/native path, no CUDA toolkit or compiler installation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TORCHDYNAMO_DISABLE']='1'
import gc
import json
from pathlib import Path
import sys
import time
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from tirex2 import load_model,TimeseriesType
from research.eps_model_lab_v1.bootstrap import RUN,save_json
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all
from research.eps_model_lab_v1.foundation_models import histories


class Tirex2Adapter(EPSAdapter):
    def __init__(self,covariates=False):
        self.covariates=covariates
        self.receipt=json.loads((RUN/'weight_receipts/NX-AI__TiRex-2.json').read_text(encoding='utf-8'))
        super().__init__('tirex2_covariates' if covariates else 'tirex2',{'family':'TIREX2_FOUNDATION',
          'source_commit':self.receipt['revision'],'license':self.receipt['license'],'pit_valid':False,'causal':True,'zero_shot':True,
          'evidence_class':'RETROSPECTIVE_RESEARCH_PRETRAINING_OVERLAP_UNRESOLVED',
          'device':'CPU native kernels, CUDA_VISIBLE_DEVICES=-1, TORCHDYNAMO_DISABLE=1, use_flex_attention=False',
          'covariates':'Original observed mask plus deterministic fiscal sin/cos known over horizon' if covariates else None,
          'input_missingness':'Causal seasonal/last input fill; no training-label fill',
          'point_estimator':'Native predictive median','tta':'Checkpoint defaults, no score-driven overrides'})
    def fit(self,frame=None,target=None):
        self.model=load_model(self.receipt['path'],device='cpu',use_flex_attention=False);self.fitted=True;return self
    def prepare_data(self,frame,lane='quarter'):
        raw,filled=histories(frame,lane);items=[];n=filled.shape[1]
        for j,(q,y) in enumerate(zip(frame.fiscal_quarter,filled)):
            phase=(int(q)+np.arange(-n+1,5))*np.pi/2
            calendar=torch.tensor(np.stack([np.sin(phase),np.cos(phase)]),dtype=torch.float32)
            items.append(TimeseriesType(target=torch.tensor(y)[None,:],
                  past_covariates=torch.tensor(np.isfinite(raw[j]).astype(np.float32))[None,:] if self.covariates else None,
                  future_covariates=calendar if self.covariates else None))
        return items
    def forecast(self,frame,lane):
        with torch.inference_mode():
            result=self.model.forecast(self.prepare_data(frame,lane),prediction_length=4,batch_size=64,output_type='numpy')
        q=np.stack(result)[:,0,[0,4,8],:].transpose(0,2,1)
        return q[:,:,1],q
    def predict(self,frame,target):
        p,_=self.forecast(frame,'ttm' if target=='ttm' else 'quarter');return p[:,3 if target=='ttm' else int(target[1:])-1]
    def save(self,path): save_json(path,{'model_id':self.model_id,'covariates':self.covariates,'metadata':self.get_metadata()})
    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text(encoding='utf-8'))['covariates']).fit()


def run(covariates):
    started=time.perf_counter();torch.set_num_threads(4);frame=dataset();valid=frame[frame.asof_date.dt.year>=2019]
    a=Tirex2Adapter(covariates).fit();smoke=valid.iloc[:8];p,q=a.forecast(smoke,'quarter')
    assert p.shape==(8,4) and q.shape==(8,4,3) and np.isfinite(p).all() and np.isfinite(q).all()
    changed=smoke.copy()
    for c in changed:
        if c.startswith(('eps_lag_','filled_lag_')): changed.loc[changed.index[1:],c]=777.
    np.testing.assert_allclose(p[0],a.forecast(changed,'quarter')[0][0],rtol=2e-5,atol=2e-5)
    path=RUN/'foundation_adapters'/f'{a.model_id}.json';a.save(path);loaded=Tirex2Adapter.load(path)
    np.testing.assert_allclose(p,loaded.forecast(smoke,'quarter')[0],rtol=0,atol=0);del loaded
    save_json(RUN/'smoke_receipts'/f'{a.model_id}_summary.json',{'status':'PASS','cross_origin_mutation':'PASS','save_load':'EXACT_PASS'})
    outputs=[]
    for lane in ['quarter','ttm']:
        began=time.perf_counter();p,q=a.forecast(valid,lane);duration=time.perf_counter()-began
        for t in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
            j=3 if t=='ttm' else int(t[1:])-1
            outputs.append(prediction_frame(valid,t,a.model_id,p[:,j],duration,q[:,j,:],a.get_metadata()))
        print(a.model_id,lane,'FORECAST_DONE',duration,flush=True)
    save_predictions(a.model_id,outputs,{'family':'TIREX2_FOUNDATION','metadata':a.get_metadata(),'runtime_seconds':time.perf_counter()-started,'smoke_status':'PASS'})
    evaluate_all();print(a.model_id,'FULL_SCORE_COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':
    for covariates in [False,True]:
        try: run(covariates)
        except Exception as exc:
            model='tirex2_covariates' if covariates else 'tirex2'
            save_json(RUN/'model_receipts'/f'{model}.json',{'model_id':model,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
            print(model,'BROKEN',str(exc),flush=True)
        gc.collect()
