"""Official multivariate networks on independent, availability-purged company origins.

The channel axis is four quantities of ONE company, never different origin dates
or companies. Official network forward methods are used with a fixed local masked
trainer; this is an architectural adaptation, not the package's default fit recipe.
"""
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
from neuralforecast import models
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all
from research.eps_model_lab_v1.dataset import normalize_eps,causal_fill

SPECS={
 'TSMixerx':{'n_block':2,'ff_dim':64,'dropout':.1,'hist_exog_list':['observed'], 'futr_exog_list':['fiscal_sin','fiscal_cos']},
 'TSMixer':{'n_block':2,'ff_dim':64,'dropout':.1},
 'SOFTS':{'hidden_size':64,'d_core':32,'e_layers':2,'d_ff':128,'dropout':.1},
 'TimeMixer':{'d_model':32,'d_ff':64,'e_layers':2,'moving_avg':5,'down_sampling_layers':2,'dropout':.1},
 'iTransformer':{'hidden_size':64,'n_heads':4,'e_layers':2,'d_ff':128,'dropout':.1},
}
CONTRACT={
 'identity':'EPS_MULTIVARIATE_COMPANY_ORIGIN_V1','channels':['GAAP_diluted_quarter_EPS','native_TTM_EPS','native_YTD_net_income_USD_billions','native_YTD_revenue_USD_billions'],
 'input_size':32,'horizon':4,'n_series':4,'origin_axis':'independent batch element, never series/channel axis',
 'accounting_duration':'Native accession YTD as in frozen fiscal panel, NOT falsely labelled standalone quarter income/revenue',
 'availability':'Each historical fact asof <= origin. Each training label asof < January 1 cutoff. No imputed label.',
 'normalization':'Per-origin/channel median absolute observed history, floor .1; no pooled future fit. EPS and TTM normalized to origin split basis; dollar channels are not EPS conversions.',
 'missing':'Causal seasonal/persistence fill inputs only; masks supplied to TSMixerx. Every architecture uses masked real training labels.',
 'future_exogenous':'Deterministic fiscal quarter sin/cos only, not future realized financial facts',
 'C_prediction':'Fourth horizon of native TTM channel, not sum of quarterly predictions',
 'trainer':{'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'steps':300,'batch_size':128,'loss':'equal-weight observed-cell SmoothL1 beta=1','gradient_clip':1.,'seed':1729,'precision':'FP32'},
 'selection':'All specifications fixed before first candidate score; no research-test tuning',
 'source':'https://github.com/Nixtla/neuralforecast','specifications':SPECS,
}


def build_tensors(frame,panel=None):
    panel=pd.read_parquet(RUN/'data/fiscal_panel.parquet') if panel is None else panel
    n=len(frame);raw=np.full((n,32,4),np.nan);truth=np.full((n,4,4),np.nan)
    available=np.full((n,4,4),np.iinfo(np.int64).max,dtype=np.int64)
    groups={t:{int(r.fiscal_index):r for r in g.itertuples()} for t,g in panel.groupby('ticker')}
    split_cache={}
    for ticker in frame.ticker.unique():
        p=pd.read_parquet(RUN/'data/raw/prices'/f'{ticker}.parquet')
        s=p['Stock Splits'];s.index=s.index.tz_localize(None).normalize();split_cache[ticker]=s[s>0]
    for i,r in enumerate(frame.itertuples()):
        grid=groups[r.ticker];splits=split_cache[r.ticker];origin=r.asof_date.tz_localize(None).normalize()
        def values(v):
            return [normalize_eps(v.eps,v.basis_date,origin,splits),normalize_eps(v.ttm,v.basis_date,origin,splits),v.net_income/1e9,v.revenue/1e9]
        for j,k in enumerate(range(int(r.fiscal_index)-31,int(r.fiscal_index)+1)):
            v=grid.get(k)
            if v is not None and v.asof<=r.asof_date: raw[i,j]=values(v)
        for h in range(4):
            v=grid.get(int(r.fiscal_index)+h+1)
            if v is not None and v.asof>r.asof_date and pd.Timestamp(v.end)>origin:
                truth[i,h]=values(v);available[i,h]=pd.Timestamp(v.asof).value
        # Main targets are verbatim frozen labels, independent of auxiliary support.
        for h in range(4):
            truth[i,h,0]=getattr(r,f'y_h{h+1}')
            a=getattr(r,f'label_asof_h{h+1}')
            available[i,h,0]=pd.Timestamp(a).value if pd.notna(a) else np.iinfo(np.int64).max
        truth[i,3,1]=r.y_ttm
        available[i,3,1]=pd.Timestamp(r.label_asof_ttm).value if pd.notna(r.label_asof_ttm) else np.iinfo(np.int64).max
    expected=frame[[f'eps_lag_{j}' for j in range(31,-1,-1)]].to_numpy(dtype=float)
    np.testing.assert_allclose(raw[:,:,0],expected,equal_nan=True,rtol=1e-12,atol=1e-12)
    mask=np.isfinite(raw);filled=np.empty_like(raw);scale=np.ones((n,4))
    for i in range(n):
        for c in range(4):
            observed=raw[i,:,c][mask[i,:,c]]
            scale[i,c]=max(float(np.median(np.abs(observed))),.1) if len(observed) else .1
            filled[i,:,c]=causal_fill(raw[i,:,c])
    quarter=frame.fiscal_quarter.to_numpy()[:,None]+np.arange(-31,5)[None,:]
    future=np.stack([np.sin(quarter*np.pi/2),np.cos(quarter*np.pi/2)],axis=1)
    future=np.repeat(future[:,:,:,None],4,axis=3)
    return {'x':(filled/scale[:,None,:]).astype('float32'),'observed':mask[:,None].astype('float32'),
            'future':future.astype('float32'),'y':(truth/scale[:,None,:]).astype('float32'),
            'availability':available,'scale':scale.astype('float32')}


def tensors(frame):
    path=RUN/'data/multivariate_company_origin_v1.npz';lock=RUN/'EPS_MULTIVARIATE_GEOMETRY_V1.json'
    if path.exists():
        receipt=json.loads(lock.read_text(encoding='utf-8'))
        if receipt['contract']!=CONTRACT or receipt['tensor_sha256']!=sha(path) or receipt['samples_sha256']!=sha(RUN/'data/samples.parquet'):
            raise RuntimeError('Multivariate frozen geometry drift')
        with np.load(path,allow_pickle=False) as stored: return {k:stored[k] for k in stored.files}
    data=build_tensors(frame)
    # Future observations/labels cannot affect the past feature surface.
    probe=frame[(frame.ticker.isin(['AAPL','JPM','CVX']))&(frame.asof_date.dt.year==2020)].groupby('ticker').head(1)
    panel=pd.read_parquet(RUN/'data/fiscal_panel.parquet')
    for idx,one in probe.iterrows():
        f=frame.loc[[idx]].copy()
        before=build_tensors(f,panel);changed=panel.copy()
        future=(changed.ticker==one.ticker)&(changed['asof']>one.asof_date)
        changed.loc[future,['eps','ttm','net_income','revenue']]=1e12
        after=build_tensors(f,changed)
        for key in ['x','observed','future','scale']: np.testing.assert_array_equal(before[key],after[key])
    np.savez_compressed(path,**data)
    save_json(lock,{'contract':CONTRACT,'tensor_sha256':sha(path),'samples_sha256':sha(RUN/'data/samples.parquet'),
      'source_sha256':sha(__file__),'future_fact_mutation_tickers':probe.ticker.tolist(),'future_fact_mutation_status':'PASS',
      'rows':len(frame),'frozen_before_first_multivariate_score':True})
    return data


class MultivariateAdapter(EPSAdapter):
    def __init__(self,name):
        super().__init__('nf_multivar_'+name,{'family':'NEURALFORECAST_MULTIVARIATE','architecture':name,
          'version':importlib.metadata.version('neuralforecast'),'pit_valid':True,'source':CONTRACT['source'],
          'adaptation':'Official network with custom independent-origin masked trainer','channels':CONTRACT['channels']})
        self.name=name;self.net=None
    def prepare_data(self,frame): return build_tensors(frame)
    def fit(self,frame,target):
        # For this joint-target architecture, target is the annual cutoff year,
        # not a single horizon. Exact row identity binds auxiliary label tensors.
        year=int(target);cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC')
        from research.eps_model_lab_v1.common import require_exact_frozen_frame
        require_exact_frozen_frame(frame)
        data=to_device(tensors(frame))
        train=np.flatnonzero((frame.asof_date<cutoff).to_numpy())
        eligible=((data['availability']<cutoff.value)&torch.isfinite(data['y'])).any(dim=(1,2)).cpu().numpy()
        self.fit_arrays(data,train[eligible[train]],cutoff.value,CONTRACT['trainer']['steps'])
        self.metadata['fit_cutoff_year']=year
        return self
    def initialize(self):
        torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        from research.eps_model_lab_v1.extra_architecture_specs import EXTRA_MULTIVARIATE
        specification=SPECS[self.name] if self.name in SPECS else EXTRA_MULTIVARIATE[self.name]
        self.net=getattr(models,self.name)(h=4,input_size=32,n_series=4,random_seed=1729,**specification).cuda()
        return self
    @staticmethod
    def batch(data,index):
        return {'insample_y':data['x'][index],'hist_exog':data['observed'][index],
                'futr_exog':data['future'][index],'stat_exog':None}
    def fit_arrays(self,data,train_indices,cutoff,steps):
        self.initialize();self.net.train()
        optimizer=torch.optim.AdamW(self.net.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(1729);eligible=data['availability']<cutoff
        y=data['y'];known=eligible&torch.isfinite(y)
        if not known[train_indices].any(): raise RuntimeError('No available training labels')
        for step in range(steps):
            idx=torch.as_tensor(rng.choice(train_indices,size=min(128,len(train_indices)),replace=True),device='cuda')
            prediction=self.net(self.batch(data,idx));mask=known[idx]
            loss=torch.nn.functional.smooth_l1_loss(prediction[mask],y[idx][mask])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.);optimizer.step()
        self.fitted=True;return self
    def path(self,data,indices):
        self.net.eval();out=[]
        size=1 if self.name=='StemGNN' else 128
        with torch.inference_mode():
            for start in range(0,len(indices),size):
                # A length-one reversed view can be C-contiguous yet retain a
                # negative stride. Force ownership before crossing into torch.
                idx=torch.as_tensor(np.array(indices[start:start+size],copy=True,order='C'),device='cuda')
                out.append(self.net(self.batch(data,idx)).cpu().numpy())
        return np.concatenate(out)*data['scale'][indices,None,:]
    def predict(self,frame,target):
        arrays=build_tensors(frame);data=to_device(arrays);p=self.path(data,np.arange(len(frame)))
        return p[:,3,1] if target=='ttm' else p[:,int(target[1:])-1,0]
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'name':self.name,'state_dict':{k:v.cpu() for k,v in self.net.state_dict().items()},'metadata':self.metadata},path)
    @classmethod
    def load(cls,path):
        obj=torch.load(path,map_location='cpu',weights_only=True);a=cls(obj['name']).initialize()
        a.net.load_state_dict(obj['state_dict'],strict=True);a.metadata=obj['metadata'];a.fitted=True;return a


def to_device(arrays):
    return {k:(v if k=='scale' else torch.as_tensor(v,device='cuda')) for k,v in arrays.items()}


def run(name,frame,data,smoke=False):
    began=time.perf_counter();out=[];audit=[];model_id='nf_multivar_'+name
    for year in ([2019] if smoke else range(2019,2027)):
        cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC');train=np.flatnonzero((frame.asof_date<cutoff).to_numpy())
        val=np.flatnonzero((frame.asof_date.dt.year==year).to_numpy())
        eligible=((data['availability']<cutoff.value)&torch.isfinite(data['y'])).any(dim=(1,2)).cpu().numpy()
        train=train[eligible[train]]
        if smoke: train=train[:128];val=val[:8]
        start=time.perf_counter();a=MultivariateAdapter(name).fit_arrays(data,train,cutoff.value,5 if smoke else 300)
        p=a.path(data,val)
        if not np.isfinite(p).all(): raise RuntimeError('Nonfinite prediction')
        path=RUN/('multivariate_smoke_fitted' if smoke else 'multivariate_fitted')/model_id/f'{year}.pt';a.save(path)
        if smoke:
            np.testing.assert_allclose(p,a.path(data,val[::-1])[::-1],rtol=1e-5,atol=1e-5)
            reloaded=MultivariateAdapter.load(path)
            np.testing.assert_allclose(p,reloaded.path(data,val),rtol=0,atol=0)
            changed=dict(data);changed['y']=torch.full_like(data['y'],1e12)
            np.testing.assert_allclose(p,a.path(changed,val),rtol=0,atol=0)
            del reloaded
        else:
            for h in range(4): out.append(prediction_frame(frame.iloc[val],f'h{h+1}',model_id,p[:,h,0],time.perf_counter()-start,metadata=a.get_metadata()))
            c=prediction_frame(frame.iloc[val],'ttm',model_id,p[:,3,1],time.perf_counter()-start,metadata=a.get_metadata())
            c['ttm_prediction_method']='DIRECT_NATIVE_TTM_MULTIVARIATE_CHANNEL';out.append(c)
        audit.append({'year':year,'train_origins':len(train),'predict_origins':len(val),'cutoff':str(cutoff),
          'seconds':time.perf_counter()-start,'status':'PASS','weight_sha256':sha(path)})
        save_json(RUN/('multivariate_smoke' if smoke else 'multivariate_audit')/f'{name}.json',audit)
        print(model_id,year,'SMOKE_PASS' if smoke else 'FOLD_PASS',round(time.perf_counter()-start,2),flush=True)
        del a;gc.collect();torch.cuda.empty_cache()
    if not smoke:
        save_predictions(model_id,out,{'family':'NEURALFORECAST_MULTIVARIATE','smoke_status':'PASS','audit':audit,
          'runtime_seconds':time.perf_counter()-began,'contract':CONTRACT,'spec':SPECS[name]})
        evaluate_all()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('names',nargs='*');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    frame=dataset();data=to_device(tensors(frame))
    for name in args.names or list(SPECS):
        try:
            run(name,frame,data,smoke=True)
            if not args.smoke: run(name,frame,data)
        except Exception as exc:
            save_json(RUN/'multivariate_failures'/f'{name}.json',{'status':'BROKEN','reason':str(exc),'traceback':traceback.format_exc()})
            print(name,'BROKEN',str(exc),flush=True)
        gc.collect();torch.cuda.empty_cache()
