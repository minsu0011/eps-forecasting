"""Official local networks on native EPS/TTM, explicit masks and fiscal timing."""
from pathlib import Path
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import gc
import random
import sys
import time
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from neuralforecast import models
from research.eps_model_lab_v2.common import RUN,V1,read_json,save_json,sha,utcnow,check_stop
from research.eps_model_lab_v2.model_common import dataset,TARGETS,predictions
from research.eps_model_lab_v2.fiscal_builder import normalize_eps


def deterministic(seed=1729):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)


def tensors(frame,panel=None):
    panel=pd.read_parquet(RUN/'data/fiscal_panel_v2.parquet') if panel is None else panel
    groups={t:{int(r.fiscal_index):r for r in g.itertuples()} for t,g in panel.groupby('ticker')}
    splits={}
    for t in frame.ticker.unique():
        prices=pd.read_parquet(V1/'data/raw/prices'/f'{t}.parquet');s=prices['Stock Splits']
        s.index=s.index.tz_localize(None).normalize();splits[t]=s[s>0]
    raw=np.full((len(frame),32,2),np.nan);y=np.full((len(frame),4,2),np.nan)
    availability=np.full(y.shape,np.iinfo(np.int64).max,dtype=np.int64)
    for i,r in enumerate(frame.itertuples()):
        grid=groups[r.ticker];origin=r.asof_date.tz_localize(None).normalize();s=splits[r.ticker]
        for j,k in enumerate(range(int(r.fiscal_index)-31,int(r.fiscal_index)+1)):
            v=grid.get(k)
            if v is not None and v.asof<=r.asof_date:
                raw[i,j]=[normalize_eps(v.eps,v.basis_date,origin,s),normalize_eps(v.ttm,v.basis_date,origin,s)]
        for h in range(4):
            v=grid.get(int(r.fiscal_index)+h+1)
            if v is not None and v.asof>r.asof_date and pd.Timestamp(v.end)>origin:
                y[i,h]=[normalize_eps(v.eps,v.basis_date,origin,s),normalize_eps(v.ttm,v.basis_date,origin,s)]
                availability[i,h]=pd.Timestamp(v.asof).value
            y[i,h,0]=getattr(r,f'y_h{h+1}')
            date=getattr(r,f'label_asof_h{h+1}');availability[i,h,0]=pd.Timestamp(date).value if pd.notna(date) else np.iinfo(np.int64).max
        y[i,3,1]=r.y_ttm;availability[i,3,1]=pd.Timestamp(r.label_asof_ttm).value if pd.notna(r.label_asof_ttm) else np.iinfo(np.int64).max
    expected=frame[[f'eps_lag_{j}' for j in range(31,-1,-1)]].to_numpy(dtype=float)
    np.testing.assert_allclose(raw[:,:,0],expected,equal_nan=True,rtol=1e-12,atol=1e-12)
    mask=np.isfinite(raw);scale=np.ones((len(frame),2));base=np.zeros((len(frame),2));age=np.zeros_like(raw)
    for i in range(len(frame)):
        for c in range(2):
            observed=raw[i,:,c][mask[i,:,c]]
            scale[i,c]=max(float(np.median(np.abs(observed))),.1) if len(observed) else .1
            base[i,c]=float(observed[-1]) if len(observed) else float(frame.iloc[i].last_observed_eps)*(4 if c else 1)
            lag=32
            for j in range(32):
                lag=0 if mask[i,j,c] else min(lag+1,32);age[i,j,c]=lag/32
    quarters=frame.fiscal_quarter.to_numpy()[:,None]+np.arange(-31,1)[None,:]
    calendar=np.stack([np.sin(quarters*np.pi/2),np.cos(quarters*np.pi/2)],axis=-1)
    return {'raw':raw.astype('float32'),'mask':mask.astype('float32'),'age':age.astype('float32'),
        'calendar':calendar.astype('float32'),'y':y.astype('float32'),'availability':availability,
        'scale':scale.astype('float32'),'base':base.astype('float32')}


def frozen_tensors(frame):
    path=RUN/'data/native_tensors_v2.npz';receipt=RUN/'audit/NATIVE_TENSOR_IDENTITY_V2.json'
    if path.exists():
        lock=read_json(receipt)
        assert sha(path)==lock['tensor_sha256'] and sha(RUN/'data/samples_v2.parquet')==lock['samples_sha256']
        with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    data=tensors(frame)
    # Every input is independent of future labels and future financial facts.
    panel=pd.read_parquet(RUN/'data/fiscal_panel_v2.parquet')
    probe=frame[(frame.ticker.isin(['AAPL','AMZN','MCD','COP','HON','UPS']))&(frame.asof_date.dt.year==2017)].groupby('ticker').head(1)
    for idx,row in probe.iterrows():
        one=frame.loc[[idx]].copy();before=tensors(one,panel)
        mutated=panel.copy();future=(mutated.ticker==row.ticker)&(mutated['asof']>row.asof_date)
        mutated.loc[future,['eps','ttm','net_income','revenue','diluted_shares']]=1e12
        for c in one:
            if c.startswith(('y_','label_','target_')):
                # Date metadata retained for auxiliary target materialization;
                # all numeric future truth receives an adversarial mutation.
                if c.startswith('y_'):one[c]=1e12
        after=tensors(one,mutated)
        for key in ['raw','mask','age','calendar','scale','base']:np.testing.assert_array_equal(before[key],after[key])
    np.savez_compressed(path,**data)
    save_json(receipt,{'created_utc':utcnow(),'tensor_sha256':sha(path),'samples_sha256':sha(RUN/'data/samples_v2.parquet'),
        'source_sha256':sha(__file__),'actual_future_mutation_origins':len(probe),'status':'PASS',
        'feature_values':['EPS','TTM'],'context_length':32,'input_missingness':'zero with explicit mask and age; DLinear cannot consume exogenous channels',
        'TTM_auxiliary_targets':'Native ledger next four horizons; h4 is verbatim frozen Lane C truth'},immutable=True)
    return data


class SequenceModel:
    def __init__(self,recipe):
        self.recipe=recipe;self.family=recipe['family'];deterministic()
        params=read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['sequence_parameters'][self.family]
        extra={'n_series':8} if self.family in ['TimeXer','SOFTS','XLinear'] else {}
        if self.family=='LSTM':extra.update(hist_exog_list=['observed','age','fiscal_sin','fiscal_cos'])
        self.net=getattr(models,self.family)(h=4,input_size=32,random_seed=1729,**params,**extra).cuda()
    def prepare(self,data):
        raw=data['raw'].copy();residual=self.recipe['representation']=='residual'
        if residual:raw-=data['base'][:,None,:]
        values=np.nan_to_num(raw/data['scale'][:,None,:],nan=0.)
        x=np.concatenate([values,data['mask'],data['age'],data['calendar']],axis=2)
        y=(data['y']-(data['base'][:,None,:] if residual else 0))/data['scale'][:,None,:]
        return {'x':torch.as_tensor(x,device='cuda'),'y':torch.as_tensor(y,device='cuda'),
            'availability':torch.as_tensor(data['availability'],device='cuda')}
    def forward(self,arrays,index):
        x=arrays['x'][index]
        if self.family in ['TimeXer','SOFTS','XLinear']:
            return self.net({'insample_y':x,'hist_exog':None,'futr_exog':None,'stat_exog':None})[:,:,:2]
        n=len(index);univariate=x[:,:,:2].transpose(1,2).reshape(n*2,32,1)
        exog=torch.stack([x[:,:,2:4].transpose(1,2),x[:,:,4:6].transpose(1,2),
            x[:,:,6].unsqueeze(1).expand(-1,2,-1),x[:,:,7].unsqueeze(1).expand(-1,2,-1)],dim=-1).reshape(n*2,32,4)
        p=self.net({'insample_y':univariate,'hist_exog':exog,'futr_exog':None,'stat_exog':None})
        return p.reshape(n,2,4).transpose(1,2)
    def fit(self,frame,data,year,steps=None):
        cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC');arrays=self.prepare(data)
        valid=(frame.asof_date<cutoff).to_numpy(copy=True)
        if self.recipe['chronological_mode']=='rolling':valid &= (frame.asof_date>=cutoff-pd.DateOffset(years=5)).to_numpy()
        known=torch.isfinite(arrays['y'])&(arrays['availability']<cutoff.value)
        eligible=valid&known.any(dim=(1,2)).cpu().numpy();indices=np.flatnonzero(eligible)
        if len(indices)<50:raise RuntimeError('Insufficient purged sequence origins')
        p=None
        if self.recipe['chronological_mode']=='recency_weighted':
            p=np.exp2(-(cutoff-frame.iloc[indices].asof_date).dt.total_seconds().to_numpy()/(2*365.2425*86400));p/=p.sum()
        rng=np.random.default_rng(1729);self.net.train();optimizer=torch.optim.AdamW(self.net.parameters(),lr=.001,weight_decay=.0001)
        for step in range(steps if steps is not None else self.recipe['steps']):
            if step%50==0:check_stop()
            idx=torch.as_tensor(rng.choice(indices,size=min(128,len(indices)),replace=True,p=p),device='cuda')
            output=self.forward(arrays,idx);mask=known[idx]
            loss=torch.nn.functional.smooth_l1_loss(output[mask],arrays['y'][idx][mask])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite sequence loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.);optimizer.step()
        return {'train_origins':len(indices),'max_train_origin':str(frame.iloc[indices].asof_date.max()),'cutoff':str(cutoff)}
    def predict(self,data,indices):
        arrays=self.prepare(data);self.net.eval();out=[]
        with torch.inference_mode():
            for start in range(0,len(indices),128):
                idx=torch.as_tensor(np.array(indices[start:start+128],copy=True),device='cuda')
                out.append(self.forward(arrays,idx).cpu().numpy())
        p=np.concatenate(out)*data['scale'][indices,None,:]
        if self.recipe['representation']=='residual':p+=data['base'][indices,None,:]
        return p
    def save(self,path):
        if path.exists():raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'recipe':self.recipe,'state_dict':{k:v.cpu() for k,v in self.net.state_dict().items()}},path)
    @classmethod
    def load(cls,path):
        state=torch.load(path,map_location='cpu',weights_only=True);model=cls(state['recipe'])
        model.net.load_state_dict(state['state_dict'],strict=True);return model


def one(recipe,year,phase,frame,data,replicate='main',steps=None):
    check_stop();deterministic();start=time.perf_counter();name=recipe['model_id']
    path=RUN/'models'/phase/name/replicate/f'{year}.pt'
    if path.exists():raise FileExistsError(path)
    torch.cuda.reset_peak_memory_stats();model=SequenceModel(recipe)
    audit=model.fit(frame,data,year,steps)
    indices=np.flatnonzero((frame.asof_date.dt.year==year).to_numpy());point=model.predict(data,indices)
    if not np.isfinite(point).all():raise RuntimeError('Nonfinite full sequence forecast')
    model.save(path)
    probe=indices[:8]
    before=model.predict(data,probe);np.testing.assert_allclose(before,model.predict(data,probe[::-1])[::-1],rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(before[:1],model.predict(data,probe[:1]),rtol=1e-5,atol=1e-5)
    del model;gc.collect();torch.cuda.empty_cache();loaded=SequenceModel.load(path)
    np.testing.assert_array_equal(point,loaded.predict(data,indices));del loaded;gc.collect();torch.cuda.empty_cache()
    out=[];valid=frame.iloc[indices]
    for target in TARGETS:
        values=point[:,3,1] if target=='ttm' else point[:,int(target[1:])-1,0]
        out.append(predictions(valid,target,name,values))
    predpath=RUN/'predictions'/phase/name/replicate/f'{year}.parquet';predpath.parent.mkdir(parents=True,exist_ok=True)
    if predpath.exists():raise FileExistsError(predpath)
    pd.concat(out,ignore_index=True).to_parquet(predpath,index=False)
    audit.update(created_utc=utcnow(),status='PASS',model_id=name,year=year,phase=phase,steps=steps or recipe['steps'],
        seconds=time.perf_counter()-start,model_sha256=sha(path),prediction_sha256=sha(predpath),
        same_process_saved_replay_exact=True,fresh_process_replay_pending=True,
        peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    save_json(path.with_suffix('.receipt.json'),audit,immutable=True)
    print('V2_SEQUENCE_PASS',name,year,round(audit['seconds'],2),flush=True)
    return audit


def run(phase='development',family=None,smoke=False,resume_completed=False):
    deterministic();frame=dataset();data=frozen_tensors(frame)
    recipes=[r for r in read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['recipes'] if r['family'] in ['TimeXer','SOFTS','XLinear','DLinear','LSTM'] and r.get('status') is None and (family is None or r['family']==family)]
    years=[2015,2016,2017,2018]
    if phase!='development':
        selected=read_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')['selected_model_ids']
        recipes=[r for r in recipes if r['model_id'] in selected]
        years=[2019,2020,2021] if phase=='confirmation' else [2022,2023,2024,2025,2026]
    if smoke:recipes=[next(r for r in recipes if r['family']==f) for f in dict.fromkeys(r['family'] for r in recipes)];years=[2015];phase='smoke'
    for recipe in recipes:
        for year in years:
            if resume_completed:
                path=RUN/'models'/phase/recipe['model_id']/'main'/f'{year}.pt'
                receipt=path.with_suffix('.receipt.json')
                if receipt.exists():
                    prior=read_json(receipt)
                    pred=RUN/'predictions'/phase/recipe['model_id']/'main'/f'{year}.parquet'
                    assert prior['status']=='PASS' and sha(path)==prior['model_sha256'] and sha(pred)==prior['prediction_sha256']
                    print('V2_SEQUENCE_VERIFIED_EXISTING_NO_REFIT',recipe['model_id'],year,flush=True)
                    continue
            try:one(recipe,year,phase,frame,data,steps=3 if smoke else None)
            except Exception as exc:
                save_json(RUN/'failures'/f'sequence_{phase}_{recipe["model_id"]}_{year}_{os.getpid()}.json',
                    {'created_utc':utcnow(),'error':str(exc),'traceback':traceback.format_exc()},immutable=True)
                raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='development');p.add_argument('--family');p.add_argument('--smoke',action='store_true');p.add_argument('--resume-completed',action='store_true');a=p.parse_args()
    run(a.phase,a.family,a.smoke,a.resume_completed)
