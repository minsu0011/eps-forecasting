"""Official Lag-Llama with bounded safe checkpoint deserialization and native masks."""
import json
from pathlib import Path
import sys
import time
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,EXTERNAL,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all
sys.path.insert(0,str(EXTERNAL/'lag_llama'))
from gluonts.dataset.common import ListDataset
from gluonts.torch.distributions.studentT import StudentTOutput
from gluonts.torch.modules.loss import NegativeLogLikelihood
from lag_llama.gluon.estimator import LagLlamaEstimator


class LagAdapter(EPSAdapter):
    def __init__(self):
        self.receipt=json.loads((RUN/'weight_receipts/time-series-foundation-models__Lag-Llama.json').read_text(encoding='utf-8'))
        super().__init__('lag_llama_zero_shot',{'family':'LAG_LLAMA_FOUNDATION','source_commit':self.receipt['revision'],
          'license':self.receipt['license'],'pit_valid':False,'causal':True,'zero_shot':True,
          'evidence_class':'RETROSPECTIVE_RESEARCH_PRETRAINING_OVERLAP_UNRESOLVED',
          'context_length':32,'num_parallel_samples':100,'point_estimator':'predictive median',
          'missingness':'Official observed-values masks and InstanceSplitter left zero padding; no label imputation',
          'time_features':'Official seven time features derived from deterministic virtual fiscal dates; no future observed covariates',
          'device':'CPU; existing isolated gluonts0.14/torch2.4 compatible environment'})
    def fit(self,frame=None,target=None):
        path=next(Path(self.receipt['path']).glob('*.ckpt'))
        expected=next(f['sha256'] for f in self.receipt['files'] if Path(f['path']).name==path.name)
        if sha(path)!=expected: raise RuntimeError('Checkpoint hash mismatch')
        # pickle GLOBAL inventory inspected first; only known official distribution/loss and builtin set allowed.
        torch.serialization.add_safe_globals([StudentTOutput,NegativeLogLikelihood,set])
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        m=ckpt['hyper_parameters']['model_kwargs']
        self.estimator=LagLlamaEstimator(prediction_length=4,context_length=32,input_size=m['input_size'],
              n_layer=m['n_layer'],n_embd_per_head=m['n_embd_per_head'],n_head=m['n_head'],
              max_context_length=m['max_context_length'],scaling=m['scaling'],time_feat=m['time_feat'],
              num_parallel_samples=100,batch_size=16,nonnegative_pred_samples=False,device=torch.device('cpu'),ckpt_path=None)
        if self.estimator.lags_seq!=m['lags_seq']: raise RuntimeError('Native checkpoint lag geometry mismatch')
        module=self.estimator.create_lightning_module(use_kv_cache=False)
        module.load_state_dict(ckpt['state_dict'],strict=True);module.eval()
        self.predictor=self.estimator.create_predictor(self.estimator.create_transformation(),module)
        self.metadata['checkpoint_safe_load']='weights_only=True, allowlist StudentTOutput/NegativeLogLikelihood/set; strict state_dict load'
        self.fitted=True;return self
    def prepare_data(self,frame,lane='quarter'):
        prefix,n=('eps_lag_',32) if lane=='quarter' else ('ttm_lag_',8)
        rows=frame[[f'{prefix}{j}' for j in range(n-1,-1,-1)]].to_numpy(dtype=np.float32)
        return ListDataset([{'item_id':str(i),'start':pd.Period(year=2000,quarter=(int(q)-n)%4+1,freq='Q'),'target':row}
                             for i,(q,row) in enumerate(zip(frame.fiscal_quarter,rows))],freq='Q')
    def forecast(self,frame,lane):
        torch.manual_seed(1729)
        with torch.inference_mode(): forecasts=list(self.predictor.predict(self.prepare_data(frame,lane)))
        q=np.array([[f.quantile(v) for v in [.1,.5,.9]] for f in forecasts]).transpose(0,2,1)
        return q[:,:,1],q
    def predict(self,frame,target):
        p,_=self.forecast(frame,'ttm' if target=='ttm' else 'quarter');return p[:,3 if target=='ttm' else int(target[1:])-1]
    def save(self,path): save_json(path,{'model_id':self.model_id,'metadata':self.get_metadata(),'weight_receipt':self.receipt})
    @classmethod
    def load(cls,path): return cls().fit()


def run():
    start=time.perf_counter();torch.set_num_threads(8);torch.manual_seed(1729)
    frame=dataset();valid=frame[frame.asof_date.dt.year>=2019];a=LagAdapter().fit();smoke=valid.iloc[:8]
    p,q=a.forecast(smoke,'quarter');assert p.shape==(8,4) and np.isfinite(p).all() and np.isfinite(q).all()
    changed=smoke.copy()
    for c in changed:
        if c.startswith('y_'): changed[c]=1e9
    np.testing.assert_allclose(p,a.forecast(changed,'quarter')[0],rtol=0,atol=0)
    path=RUN/'foundation_adapters/lag_llama_zero_shot.json';a.save(path);loaded=LagAdapter.load(path)
    np.testing.assert_allclose(p,loaded.forecast(smoke,'quarter')[0],rtol=0,atol=0);del loaded
    save_json(RUN/'smoke_receipts/lag_llama_zero_shot_summary.json',{'status':'PASS','shape':[8,4],'future_label_mutation':'EXACT_PASS','save_load':'EXACT_PASS'})
    print('lag_llama_zero_shot SMOKE_PASS',flush=True);outputs=[]
    for lane in ['quarter','ttm']:
        began=time.perf_counter();p,q=a.forecast(valid,lane);duration=time.perf_counter()-began
        for t in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
            j=3 if t=='ttm' else int(t[1:])-1
            outputs.append(prediction_frame(valid,t,a.model_id,p[:,j],duration,q[:,j,:],a.get_metadata()))
        print('lag_llama_zero_shot',lane,'FORECAST_DONE',duration,flush=True)
    save_predictions(a.model_id,outputs,{'family':'LAG_LLAMA_FOUNDATION','metadata':a.get_metadata(),'runtime_seconds':time.perf_counter()-start,'smoke_status':'PASS'})
    evaluate_all();print(a.model_id,'FULL_SCORE_COMPLETE',time.perf_counter()-start,flush=True)


if __name__=='__main__':
    try: run()
    except Exception as exc:
        save_json(RUN/'model_receipts/lag_llama_zero_shot.json',{'model_id':'lag_llama_zero_shot','status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
        print('LAG_LLAMA_BROKEN',str(exc),flush=True)
