"""Audited seferlab architectures, stateless batches and causal EPS multi-head supervision."""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OMP_NUM_THREADS']='4'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,EXTERNAL,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,TARGETS,dataset,prediction_frame,save_predictions,evaluate_all
SOURCE=EXTERNAL/'epspredict_code/modules/Models.py'
spec=importlib.util.spec_from_file_location('eps_upstream_models',SOURCE);upstream=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=upstream;spec.loader.exec_module(upstream)

class ZeroInitMultiFrequency(upstream.MultiFreqLSTMModel):
    def init_hidden(self,batch_size):
        shape=(self.layer_size,batch_size,self.hidden_size)
        return torch.zeros(shape,device=self.device),torch.zeros(shape,device=self.device)

NAMES=['LSTM','CNNLSTM','MultiFreqLSTM','MultiFreqCNNLSTM']

class NativeAdapter(EPSAdapter):
    def __init__(self,name,epochs=80):
        receipt=json.loads((RUN/'source_receipts/epspredict.json').read_text(encoding='utf-8'))
        super().__init__('epspredict_'+name,{'family':'EPS_NATIVE_SEQUENCE','architecture':name,'source_commit':receipt['commit'],
            'license':'UNKNOWN_RESEARCH_ONLY','causal':True,'PIT_safe':True,'source_file_sha256':sha(SOURCE),
            'adaptations':['Architecture-only reuse; original data/tuning not reused','Reset hidden state every batch/forecast','MultiFreqLSTM random hidden initialization replaced by deterministic zeros',
              'Four quarterly + one TTM heads with availability-masked SmoothL1 loss','Origin-only EPS scale; observed masks/fiscal sin/cos',
              'Multi-frequency branches both length 8: last 8 quarters and last 8 same-season annual observations'],
            'epochs':epochs,'hidden_size':64,'layers':2,'dropout':.1,'optimizer':'AdamW lr=.001 weight_decay=.001'})
        self.name=name;self.epochs=epochs
        base=dict(input_dim=4,hidden_size=64,layer_size=2,output_dim=5,dropout_prob=.1,device='cuda')
        conv=dict(conv_channels=8,kernel_size=3,pool_size=2,stride=1)
        if name=='LSTM': model=upstream.LSTMModel(**base)
        elif name=='CNNLSTM': model=upstream.TwoDimCNNLSTMModel(**base,**conv)
        elif name=='MultiFreqLSTM': model=ZeroInitMultiFrequency(**base)
        else: model=upstream.MultiFreqCNNLSTMModel(**base,**conv)
        self.model=model.to('cuda')
    def prepare_data(self,frame):
        eps=frame[[f'filled_lag_{j}' for j in range(31,-1,-1)]].to_numpy(dtype=np.float32)/frame.scale.to_numpy(dtype=np.float32)[:,None]
        observed=frame[[f'observed_lag_{j}' for j in range(31,-1,-1)]].to_numpy(dtype=np.float32)
        quarter=frame.fiscal_quarter.to_numpy()[:,None]-np.arange(31,-1,-1)[None,:]
        arr=np.stack([eps,observed,np.sin(quarter*np.pi/2),np.cos(quarter*np.pi/2)],axis=-1).astype(np.float32)
        return torch.tensor(arr,device='cuda')
    def forward(self,x):
        self.model.reset_hidden()
        if self.name.startswith('MultiFreq'): return self.model(x[:,-8:,:],x[:,3::4,:])
        return self.model(x)
    def fit(self,frame,target):
        cutoff=target;x=self.prepare_data(frame)
        y=np.column_stack([frame['y_'+t].to_numpy(dtype=float) for t in TARGETS])/frame.scale.to_numpy()[:,None]
        mask=np.column_stack([(frame['label_asof_'+t]<cutoff).to_numpy() for t in TARGETS])&np.isfinite(y)
        yt=torch.tensor(np.nan_to_num(y),device='cuda',dtype=torch.float32);mt=torch.tensor(mask,device='cuda')
        opt=torch.optim.AdamW(self.model.parameters(),lr=.001,weight_decay=.001)
        self.model.train();losses=[]
        for epoch in range(self.epochs):
            order=torch.randperm(len(frame),device='cuda');total=0.
            for indexes in order.split(256):
                opt.zero_grad(set_to_none=True);p=self.forward(x[indexes])
                losses_element=torch.nn.functional.smooth_l1_loss(p,yt[indexes],reduction='none')
                loss=(losses_element*mt[indexes]).sum()/mt[indexes].sum().clamp_min(1)
                if not torch.isfinite(loss): raise ValueError('Nonfinite masked training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.);opt.step();total+=float(loss.detach())
            losses.append(total/max((len(frame)+255)//256,1))
        self.training_losses=losses;self.fitted=True;self.model.eval();return self
    def path(self,frame):
        x=self.prepare_data(frame);out=[]
        with torch.inference_mode():
            for b in x.split(256): out.append(self.forward(b).cpu().numpy())
        return np.concatenate(out)*frame.scale.to_numpy()[:,None]
    def predict(self,frame,target): return self.path(frame)[:,list(TARGETS).index(target)]
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':self.model.state_dict(),'name':self.name,'metadata':self.get_metadata()},path)
    @classmethod
    def load(cls,path):
        stored=torch.load(path,map_location='cpu',weights_only=True);a=cls(stored['name']);a.model.load_state_dict(stored['state_dict']);a.model.eval();a.fitted=True;return a

def run(name,smoke=False):
    frame=dataset();out=[];audit=[];started=time.perf_counter();torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    for year in ([2019] if smoke else range(2019,2027)):
        torch.manual_seed(1729);np.random.seed(1729)
        cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC');train=frame[frame.asof_date<cutoff];valid=frame[frame.asof_date.dt.year==year]
        known=np.column_stack([(train['label_asof_'+t]<cutoff)&train['y_'+t].notna() for t in TARGETS]).any(axis=1);train=train.loc[known]
        if smoke: train=train.iloc[:128];valid=valid.iloc[:8]
        start=time.perf_counter();a=NativeAdapter(name,epochs=2 if smoke else 80).fit(train,cutoff);p=a.path(valid)
        assert p.shape==(len(valid),5) and np.isfinite(p).all()
        if smoke:
            np.testing.assert_allclose(p,a.path(valid),rtol=0,atol=0)
            shuffled=valid.iloc[::-1];reverse=a.path(shuffled)[::-1]
            np.testing.assert_allclose(p,reverse,rtol=1e-5,atol=1e-5)
            path=RUN/'native_smoke_weights'/f'{name}.pt';a.save(path);loaded=NativeAdapter.load(path)
            np.testing.assert_allclose(p,loaded.path(valid),rtol=0,atol=0);del loaded
        else:
            a.save(RUN/'native_fitted'/name/f'{year}.pt')
            out.extend(prediction_frame(valid,t,a.model_id,p[:,j],time.perf_counter()-start,metadata=a.get_metadata()) for j,t in enumerate(TARGETS))
        audit.append({'year':year,'train_rows':len(train),'prediction_rows':len(valid),'seconds':time.perf_counter()-start,
                      'final_train_loss':a.training_losses[-1],'training_loss_curve':a.training_losses,'status':'PASS'})
        save_json(RUN/('native_smoke' if smoke else 'native_audit')/f'{name}.json',audit)
        print(a.model_id,year,'SMOKE_PASS' if smoke else 'FOLD_PASS',round(audit[-1]['seconds'],1),flush=True)
        del a;gc.collect();torch.cuda.empty_cache()
    if not smoke:
        save_predictions('epspredict_'+name,out,{'family':'EPS_NATIVE_SEQUENCE','runtime_seconds':time.perf_counter()-started,
            'audit':audit,'smoke_status':'PASS','peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'source_sha256':sha(SOURCE),'license':'UNKNOWN_RESEARCH_ONLY'})
        evaluate_all()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+',choices=NAMES);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    for name in a.names:
        try: run(name,a.smoke)
        except Exception as exc:
            save_json(RUN/('native_smoke_failures' if a.smoke else 'model_receipts')/f'epspredict_{name}.json',{'model_id':'epspredict_'+name,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
            print(name,'BROKEN',str(exc),flush=True)
        gc.collect();torch.cuda.empty_cache()
