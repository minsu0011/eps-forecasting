"""Official NeuralForecast global models trained on availability-purged origin snapshots."""
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
os.environ['OMP_NUM_THREADS']='4'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast,models
from neuralforecast.losses.pytorch import MAE
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all

SPECS={
 'NHITS':{'mlp_units':[[128,128]]*3},'NBEATSx':{'mlp_units':[[128,128]]*3},
 'TFT':{'hidden_size':64,'n_head':4},'DLinear':{'moving_avg_window':5},
 'PatchTST':{'hidden_size':64,'n_heads':4,'encoder_layers':2,'linear_hidden_size':128,'patch_len':4,'stride':2},
 'GRU':{'encoder_hidden_size':64,'encoder_n_layers':2,'decoder_hidden_size':64},
 'TCN':{'encoder_hidden_size':64,'decoder_hidden_size':64,'dilations':[1,2,4,8]},
 'NBEATS':{'mlp_units':[[128,128]]*3},
 'TiDE':{'hidden_size':128,'decoder_output_dim':16,'temporal_decoder_dim':64},
 'MLP':{'hidden_size':128,'num_layers':2},
 'RNN':{'encoder_hidden_size':64,'decoder_hidden_size':64},
 'LSTM':{'encoder_hidden_size':64,'decoder_hidden_size':64},
 'DilatedRNN':{'encoder_hidden_size':64,'decoder_hidden_size':64},
 'BiTCN':{'hidden_size':32},
 'KAN':{'hidden_size':64,'n_hidden_layers':1},
 'DeepNPTS':{'hidden_size':64,'batch_norm':False},
 'Autoformer':{'hidden_size':64,'n_head':4,'conv_hidden_size':64,'MovingAvg_window':5},
 'Informer':{'hidden_size':64,'n_head':4,'conv_hidden_size':64},
 'VanillaTransformer':{'hidden_size':64,'n_head':4,'conv_hidden_size':64},
}

def long_frame(frame,cutoff=None):
    history=frame[[f'eps_lag_{j}' for j in range(31,-1,-1)]].to_numpy(dtype=np.float32)
    scale=frame.scale.to_numpy(dtype=np.float32)
    if cutoff is not None:
        ys=[]
        for j in range(1,5):
            known=frame[f'label_asof_h{j}']<cutoff
            ys.append(frame[f'y_h{j}'].where(known).to_numpy(dtype=np.float32))
        history=np.column_stack([history,*ys])
    y=history/scale[:,None];valid=np.isfinite(y)
    # Identity scaling + real observation loss mask. Imputed zeros are not training truth.
    return pd.DataFrame({'unique_id':np.repeat(frame.sample_id.to_numpy(),y.shape[1]),
                         'ds':np.tile(np.arange(y.shape[1]),len(frame)),
                         'y':np.nan_to_num(y).ravel(),'available_mask':valid.astype(np.float32).ravel()})

class NeuralAdapter(EPSAdapter):
    def __init__(self,name,steps=300):
        super().__init__('nf_'+name,{'family':'NEURALFORECAST','architecture':name,'version':importlib.metadata.version('neuralforecast'),
             'causal':True,'PIT_safe':True,'source':'https://github.com/Nixtla/neuralforecast','masked_missing_targets':True})
        self.name=name;self.steps=steps
    def prepare_data(self,frame): return long_frame(frame)
    def fit(self,frame,target):
        cutoff=target
        from research.eps_model_lab_v1.extra_architecture_specs import EXTRA_UNIVARIATE
        specification=SPECS[self.name] if self.name in SPECS else EXTRA_UNIVARIATE[self.name]
        kwargs=dict(h=4,input_size=16,max_steps=self.steps,learning_rate=.001,batch_size=64,windows_batch_size=256,
           inference_windows_batch_size=256,scaler_type='identity',random_seed=1729,loss=MAE(),val_check_steps=self.steps,
           early_stop_patience_steps=-1,accelerator='gpu',devices=1,precision='32-true',logger=False,
           enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,**specification)
        model=getattr(models,self.name)(**kwargs)
        self.nf=NeuralForecast(models=[model],freq=1)
        self.nf.fit(df=long_frame(frame,cutoff),val_size=0)
        self.fitted=True;return self
    def predict(self,frame,target):
        p=self.path(frame)
        return p.sum(axis=1) if target=='ttm' else p[:,int(target[1:])-1]
    def path(self,frame):
        # TimesNet FFT period selection averages over its batch axis. Keep each
        # inference window independent so future-origin contexts cannot affect it.
        if self.name=='TimesNet':self.nf.models[0].inference_windows_batch_size=1
        pred=self.nf.predict(df=long_frame(frame))
        value=[c for c in pred if c not in ['unique_id','ds']][0]
        grouped=pred.sort_values(['unique_id','ds']).groupby('unique_id')[value].apply(list)
        result=np.array([grouped.loc[i] for i in frame.sample_id])
        if result.shape!=(len(frame),4): raise ValueError('NeuralForecast contract shape mismatch')
        return result*frame.scale.to_numpy()[:,None]
    def save(self,path):
        self.nf.save(path=str(path),overwrite=False,save_dataset=False)
    @classmethod
    def load(cls,path):
        nf=NeuralForecast.load(path=str(path))
        a=cls(type(nf.models[0]).__name__,steps=int(nf.models[0].max_steps))
        a.nf=nf;a.fitted=True;return a

def run(name,smoke=False):
    frame=dataset();outputs=[];audit=[];start=time.perf_counter();torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    for year in ([2019] if smoke else range(2019,2027)):
        cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC')
        train=frame[frame.asof_date<cutoff]
        # At least one available direct future label per training snapshot.
        any_known=np.column_stack([(train[f'label_asof_h{j}']<cutoff)&train[f'y_h{j}'].notna() for j in range(1,5)]).any(axis=1)
        train=train.loc[any_known]
        valid=frame[frame.asof_date.dt.year==year]
        if smoke: train=train.iloc[:128];valid=valid.iloc[:8]
        a=NeuralAdapter(name,steps=5 if smoke else 300)
        began=time.perf_counter();a.fit(train,cutoff);p=a.path(valid)
        assert np.isfinite(p).all()
        duration=time.perf_counter()-began
        if smoke:
            changed=valid.copy()
            for col in changed:
                if col.startswith('y_'): changed[col]=1e12
            np.testing.assert_allclose(p,a.path(changed),rtol=1e-5,atol=1e-5)
        else:
            a.save(RUN/'neural_fitted'/('nf_'+name)/str(year))
            outputs.extend(prediction_frame(valid,f'h{j+1}','nf_'+name,p[:,j],duration,metadata=a.get_metadata()) for j in range(4))
            ttm=prediction_frame(valid,'ttm','nf_'+name,p.sum(axis=1),duration,metadata=a.get_metadata())
            ttm['ttm_prediction_method']='SUM_FORECAST_QUARTERS_WEIGHTED_SHARE_APPROXIMATION';outputs.append(ttm)
        audit.append({'year':year,'train_origins':len(train),'predict_origins':len(valid),'seconds':duration,'cutoff':str(cutoff),'status':'PASS'})
        save_json(RUN/('neural_smoke' if smoke else 'neural_audit')/f'{name}.json',audit)
        print('nf_'+name,year,'SMOKE_PASS' if smoke else 'FOLD_PASS',round(duration,1),flush=True)
        del a;gc.collect();torch.cuda.empty_cache()
    if not smoke:
        save_predictions('nf_'+name,outputs,{'family':'NEURALFORECAST','runtime_seconds':time.perf_counter()-start,
             'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'audit':audit,'smoke_status':'PASS','spec':SPECS[name]})
        evaluate_all()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='+',choices=list(SPECS));parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    save_json(RUN/'ecosystem_discovery/neuralforecast.json',{'version':importlib.metadata.version('neuralforecast'),'models':models.__all__})
    for name in args.names:
        try: run(name,args.smoke)
        except Exception as exc:
            save_json(RUN/('neural_smoke_failures' if args.smoke else 'model_receipts')/f'nf_{name}.json',{'model_id':'nf_'+name,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
            print('nf_'+name,'BROKEN',str(exc),flush=True)
        gc.collect();torch.cuda.empty_cache()
