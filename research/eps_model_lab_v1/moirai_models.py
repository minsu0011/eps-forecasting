"""Official Uni2TS small checkpoints on an isolated CPU-compatible legacy environment."""
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
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from research.eps_model_lab_v1.bootstrap import RUN,save_json
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all

SPECS={'moirai2_small':'Salesforce/moirai-2.0-R-small','moirai1p1_small':'Salesforce/moirai-1.1-R-small','moirai_moe_small':'Salesforce/moirai-moe-1.0-R-small'}

class MoiraiAdapter(EPSAdapter):
    def __init__(self,model_id):
        repo=SPECS[model_id];receipt=json.loads((RUN/'weight_receipts'/(repo.replace('/','__')+'.json')).read_text(encoding='utf-8'))
        super().__init__(model_id,{'family':'MOIRAI_FOUNDATION','version':importlib.metadata.version('uni2ts'),'source_commit':receipt['revision'],
           'license':receipt['license'],'pit_valid':False,'causal':True,'evidence_class':'RETROSPECTIVE_RESEARCH_PRETRAINING_OVERLAP_UNRESOLVED',
           'device':'CPU; official uni2ts torch<2.5 dependency incompatible with 5080 CUDA build','point_estimator':'predictive median',
           'native_missing_mask':True,'zero_shot':True})
        self.receipt=receipt
    def fit(self,frame=None,target=None):
        path=self.receipt['path']
        if self.model_id=='moirai2_small':
            from uni2ts.model.moirai2 import Moirai2Forecast,Moirai2Module
            self.module=Moirai2Module.from_pretrained(path);self.klass=Moirai2Forecast
        elif self.model_id=='moirai1p1_small':
            from uni2ts.model.moirai import MoiraiForecast,MoiraiModule
            self.module=MoiraiModule.from_pretrained(path);self.klass=MoiraiForecast
        else:
            from uni2ts.model.moirai_moe import MoiraiMoEForecast,MoiraiMoEModule
            self.module=MoiraiMoEModule.from_pretrained(path);self.klass=MoiraiMoEForecast
        self.module.eval();self.fitted=True;return self
    def prepare_data(self,frame,lane='quarter'):
        prefix,n=('eps_lag_',32) if lane=='quarter' else ('ttm_lag_',8)
        return frame[[f'{prefix}{j}' for j in range(n-1,-1,-1)]].to_numpy(dtype=np.float32)
    def forecast(self,frame,lane):
        y=self.prepare_data(frame,lane);torch.manual_seed(1729)
        kwargs={'prediction_length':4,'context_length':y.shape[1],'target_dim':1,'feat_dynamic_real_dim':0,'past_feat_dynamic_real_dim':0,'module':self.module}
        if self.model_id!='moirai2_small': kwargs.update(patch_size=8 if self.model_id=='moirai1p1_small' else 16,num_samples=100)
        model=self.klass(**kwargs).eval();predictor=model.create_predictor(batch_size=32,device='cpu')
        data=ListDataset([{'item_id':str(i),'start':pd.Period('2000Q1',freq='Q'),'target':row} for i,row in enumerate(y)],freq='Q')
        with torch.inference_mode(): forecasts=list(predictor.predict(data))
        q=np.array([[forecast.quantile(t) for t in [.1,.5,.9]] for forecast in forecasts]).transpose(0,2,1)
        return q[:,:,1],q
    def predict(self,frame,target):
        p,_=self.forecast(frame,'ttm' if target=='ttm' else 'quarter');return p[:,3 if target=='ttm' else int(target[1:])-1]
    def save(self,path): save_json(path,{'model_id':self.model_id,'metadata':self.get_metadata(),'weight_receipt':self.receipt})
    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text(encoding='utf-8'))['model_id']).fit()

def run(model_id):
    start=time.perf_counter();torch.set_num_threads(8);frame=dataset();valid=frame[frame.asof_date.dt.year>=2019]
    a=MoiraiAdapter(model_id).fit();smoke=valid.iloc[:8];p,q=a.forecast(smoke,'quarter')
    assert p.shape==(8,4) and q.shape==(8,4,3) and np.isfinite(p).all()
    changed=smoke.copy()
    for c in changed:
        if c.startswith('y_'): changed[c]=1e9
    check,_=a.forecast(changed,'quarter');np.testing.assert_allclose(p,check,rtol=1e-5,atol=1e-5)
    save_json(RUN/'smoke_receipts'/f'{model_id}_summary.json',{'status':'PASS','shape':[8,4],'future_label_mutation':'PASS','CPU_threads':8})
    outputs=[]
    for lane in ['quarter','ttm']:
        began=time.perf_counter();p,q=a.forecast(valid,lane);elapsed=time.perf_counter()-began
        for t in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
            j=3 if t=='ttm' else int(t[1:])-1
            outputs.append(prediction_frame(valid,t,model_id,p[:,j],elapsed,q[:,j,:],a.get_metadata()))
    a.save(RUN/'foundation_adapters'/f'{model_id}.json')
    save_predictions(model_id,outputs,{'family':'MOIRAI_FOUNDATION','metadata':a.get_metadata(),'runtime_seconds':time.perf_counter()-start,'smoke_status':'PASS','device':'CPU'})
    evaluate_all();print(model_id,'FULL_SCORE_COMPLETE',round(time.perf_counter()-start,2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('models',nargs='+',choices=list(SPECS));args=parser.parse_args()
    for model in args.models:
        try: run(model)
        except Exception as exc:
            save_json(RUN/'model_receipts'/f'{model}.json',{'model_id':model,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
            print(model,'BROKEN',str(exc),flush=True)
        gc.collect()
