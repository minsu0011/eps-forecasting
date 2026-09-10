"""Distinct Sundial generative and TabPFN-v2 in-context EPS architectures."""
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['OMP_NUM_THREADS']='4'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
os.environ['HF_HOME']=str(PROJECT.parent/'.cache/eps_models')
os.environ['HF_MODULES_CACHE']=str(PROJECT.parent/'.cache/eps_models/modules')
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,features,split_train,prediction_frame,save_predictions,evaluate_all

SPEC={
 'sundial_base_128m':{'context_quarter':32,'context_ttm':8,'horizon':4,'num_samples':20,
   'point':'Native generated-sample mean','quantiles':'Native20sample empirical P10/P50/P90; finite Monte Carlo precision',
   'missing':'Existing causal seasonal/persistence filled inputs, no missing target imputation; native patch-padding mask is not original missingness mask',
   'seed':'First8hexSHA256(sample_id|lane) modulo2**31; one origin per call, independent of batch/order',
   'precision':'FP32','sampling_steps':50,'weights':'thuml/sundial-base-128m'},
 'tabpfn_v2_panel':{'annual_folds':list(range(2019,2027)),'targets':['h1','h2','h3','h4','ttm'],
   'n_estimators':4,'fit_mode':'fit_preprocessors','random_state':1729,'precision':'FP32',
   'preprocessing':'Same common scaled PIT wide features; target divided by causal origin scale; official preprocessors fitted on purged train only',
   'inference_batch_rows':64,'point':'Native distribution mean','quantiles':'Native bar-distribution P10/P50/P90',
   'weights':'Prior-Labs/TabPFN-v2-reg','checkpoint':'tabpfn-v2-regressor-v2_default.ckpt',
   'adaptation':'Global annual EPS regression using in-context pretrained tabular model, NOT a claimed TabPFN-TS replication'},
}


def weight(repo):
    r=json.loads((RUN/'weight_receipts'/(repo.replace('/','__')+'.json')).read_text(encoding='utf-8'))
    for f in r['files']:
        if sha(f['path'])!=f['sha256']:raise RuntimeError('Pinned weight/code drift '+f['path'])
    return r


def metadata(name,r):
    return {'family':'FOUNDATION_TABULAR_IN_CONTEXT' if name.startswith('tabpfn') else 'FOUNDATION_GENERATIVE_FLOW',
       'pit_valid':False,'causal':True,'zero_shot':name.startswith('sundial'),'finetuned':False,
       'source_commit':r['revision'],'license':r['license'],'version':importlib.metadata.version('tabpfn' if name.startswith('tabpfn') else 'transformers'),
       'evidence_class':'RETROSPECTIVE_RESEARCH_PRETRAINING_OVERLAP_UNRESOLVED','spec':SPEC[name]}


class SundialAdapter(EPSAdapter):
    def __init__(self):
        self.r=weight(SPEC['sundial_base_128m']['weights']);super().__init__('sundial_base_128m',metadata('sundial_base_128m',self.r))
    def prepare_data(self,frame,lane='quarter'):
        cols=[f'filled_lag_{i}' for i in range(31,-1,-1)] if lane=='quarter' else [f'ttm_lag_{i}' for i in range(7,-1,-1)]
        x=frame[cols].to_numpy(dtype=np.float32).copy()
        if lane=='ttm':
            for j in range(x.shape[1]):
                fallback=x[:,j-4] if j>=4 else x[:,j-1] if j else np.zeros(len(x))
                x[:,j]=np.where(np.isfinite(x[:,j]),x[:,j],fallback)
        if not np.isfinite(x).all():raise RuntimeError('Nonfinite causal history')
        return x
    def fit(self,frame=None,target=None):
        audit=json.loads((RUN/'ADDITIONAL_SOURCE_EXECUTION_AUDIT.json').read_text(encoding='utf-8'))
        for p,digest in audit['sundial_python_sha256'].items():
            if sha(Path(self.r['path'])/p)!=digest:raise RuntimeError('Inspected Sundial source drift')
        from transformers import AutoModelForCausalLM
        self.model=AutoModelForCausalLM.from_pretrained(self.r['path'],trust_remote_code=True,local_files_only=True,
           torch_dtype=torch.float32).to('cuda').eval();self.fitted=True;return self
    def forecast(self,frame,lane):
        x=self.prepare_data(frame,lane);points=[];quantiles=[];began=time.perf_counter()
        for i,(identity,row) in enumerate(zip(frame.sample_id,x)):
            seed=int(hashlib.sha256((identity+'|'+lane).encode()).hexdigest()[:8],16)%(2**31)
            with torch.random.fork_rng(devices=[0]),torch.inference_mode():
                torch.manual_seed(seed)
                generated=self.model.generate(torch.tensor(row.copy(),device='cuda')[None],max_new_tokens=4,num_samples=20)
                samples=generated[0].cpu().numpy()
            if samples.shape!=(20,4) or not np.isfinite(samples).all():raise RuntimeError('Native sample contract')
            points.append(samples.mean(axis=0));quantiles.append(np.quantile(samples,[.1,.5,.9],axis=0).T)
            if len(frame)>100 and (i+1)%200==0:print('SUNDIAL_PROGRESS',lane,i+1,len(frame),round(time.perf_counter()-began,1),flush=True)
        return np.asarray(points),np.asarray(quantiles)
    def predict(self,frame,target):return self.forecast(frame,'ttm' if target=='ttm' else 'quarter')[0][:,3 if target=='ttm' else int(target[1:])-1]
    def save(self,path):save_json(Path(path),{'model_id':self.model_id,'weight_receipt_sha256':sha(RUN/'weight_receipts/thuml__sundial-base-128m.json'),'metadata':self.get_metadata()})
    @classmethod
    def load(cls,path):
        r=json.loads(Path(path).read_text(encoding='utf-8'))
        if r['weight_receipt_sha256']!=sha(RUN/'weight_receipts/thuml__sundial-base-128m.json'):raise RuntimeError('Weight reference drift')
        return cls().fit()


class TabPFNAdapter(EPSAdapter):
    def __init__(self):
        self.r=weight(SPEC['tabpfn_v2_panel']['weights']);super().__init__('tabpfn_v2_panel',metadata('tabpfn_v2_panel',self.r))
    def prepare_data(self,frame):return features(frame,scaled=True).to_numpy(dtype=np.float32)
    def fit(self,frame,target):
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        checkpoint=Path(self.r['path'])/SPEC[self.model_id]['checkpoint']
        if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):raise RuntimeError('Unexpected checkpoint pickle globals')
        self.target=target;self.columns=list(features(frame).columns)
        self.model=TabPFNRegressor.create_default_for_version(ModelVersion.V2,model_path=str(checkpoint),
           n_estimators=4,device='cuda',inference_precision=torch.float32,fit_mode='fit_preprocessors',random_state=1729,n_preprocessing_jobs=4)
        self.model.fit(self.prepare_data(frame),(frame['y_'+target]/frame.scale).to_numpy(dtype=np.float32))
        self.metadata.update(fitted_target=target,training_sample_ids=frame.sample_id.tolist());self.fitted=True;return self
    def distribution(self,frame):
        if list(features(frame).columns)!=self.columns:raise RuntimeError('Feature order drift')
        x=self.prepare_data(frame);points=[];quantiles=[]
        for start in range(0,len(x),64):
            pred=self.model.predict(x[start:start+64],output_type='main',quantiles=[.1,.5,.9])
            points.append(pred['mean']);quantiles.append(np.column_stack(pred['quantiles']))
        scale=frame.scale.to_numpy()
        return np.concatenate(points)*scale,np.concatenate(quantiles)*scale[:,None]
    def predict(self,frame,target):
        if target!=self.target:raise RuntimeError('Target mismatch')
        return self.distribution(frame)[0]
    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        self.model.save_fit_state(path/'model.tabpfn_fit')
        save_json(path/'metadata.json',{'target':self.target,'columns':self.columns,'metadata':self.metadata,
           'fit_state_sha256':sha(path/'model.tabpfn_fit')})
    @classmethod
    def load(cls,path):
        from tabpfn import TabPFNRegressor
        path=Path(path);r=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        if sha(path/'model.tabpfn_fit')!=r['fit_state_sha256']:raise RuntimeError('Local fit state drift')
        a=cls();a.model=TabPFNRegressor.load_from_fit_state(path/'model.tabpfn_fit',device='cuda')
        a.target=r['target'];a.columns=r['columns'];a.metadata=r['metadata'];a.fitted=True;return a


def mutation(frame):
    f=frame.copy()
    for c in f:
        if c.startswith(('y_','label_','target_')):f[c]=1e12
    return f


def smoke(name,attempt='initial'):
    frame=dataset();probe=frame[frame.asof_date.dt.year==2019].iloc[:8]
    if name.startswith('sundial'):
        a=SundialAdapter().fit();path=RUN/'additional_smoke/sundial_adapter.json';a.save(path);b=SundialAdapter.load(path)
        for lane in ['quarter','ttm']:
            p,q=a.forecast(probe,lane)
            for f in [mutation(probe),probe.iloc[::-1]]:
                pp,qq=a.forecast(f,lane)
                if f.sample_id.iloc[0]!=probe.sample_id.iloc[0]:pp=pp[::-1];qq=qq[::-1]
                np.testing.assert_allclose(p,pp,rtol=0,atol=0);np.testing.assert_allclose(q,qq,rtol=0,atol=0)
            pp,qq=b.forecast(probe,lane);np.testing.assert_allclose(p,pp,rtol=0,atol=0);np.testing.assert_allclose(q,qq,rtol=0,atol=0)
    else:
        train,_=split_train(frame,2019,'h1');a=TabPFNAdapter().fit(train.iloc[:128],'h1')
        path=RUN/'additional_smoke'/('tabpfn_v2_panel' if attempt=='initial' else 'tabpfn_v2_panel_'+attempt);a.save(path);b=TabPFNAdapter.load(path)
        p,q=a.distribution(probe)
        other=probe.copy();feature_cols=features(other).columns;other.loc[other.index[1:],feature_cols]=other.loc[other.index[1:],feature_cols]*3+7
        for f in [mutation(probe),other]:
            pp,qq=a.distribution(f)
            np.testing.assert_allclose(p[0],pp[0],rtol=2e-5,atol=2e-5);np.testing.assert_allclose(q[0],qq[0],rtol=2e-5,atol=2e-5)
        pp,qq=b.distribution(probe);np.testing.assert_allclose(p,pp,rtol=2e-5,atol=2e-5);np.testing.assert_allclose(q,qq,rtol=2e-5,atol=2e-5)
        if not np.isfinite(p).all() or not np.isfinite(q).all():raise RuntimeError('Nonfinite smoke')
    save_json(RUN/'smoke_receipts'/f'{name}.json',{'status':'PASS','rows':8,'fit_predict_reload':'PASS','future_label_mutation':'PASS',
       'source_sha256':sha(__file__),'spec':SPEC[name]})
    del a,b;gc.collect();torch.cuda.empty_cache();print(name,'SMOKE_PASS',flush=True)


def score(name,smoke_attempt='initial'):
    smoke(name,smoke_attempt);frame=dataset();outputs=[];audits=[];began=time.perf_counter()
    if name.startswith('sundial'):
        a=SundialAdapter().fit();valid=frame[frame.asof_date.dt.year>=2019]
        a.save(RUN/'additional_fitted/sundial_adapter.json')
        for lane in ['quarter','ttm']:
            p,q=a.forecast(valid,lane)
            for target in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
                j=3 if target=='ttm' else int(target[1:])-1
                outputs.append(prediction_frame(valid,target,name,p[:,j],time.perf_counter()-began,q[:,j],a.metadata))
    else:
        for year in range(2019,2027):
            for target in ['h1','h2','h3','h4','ttm']:
                train,valid=split_train(frame,year,target);started=time.perf_counter()
                a=TabPFNAdapter().fit(train,target);p,q=a.distribution(valid)
                if not np.isfinite(p).all() or not np.isfinite(q).all():raise RuntimeError('Nonfinite full prediction')
                path=RUN/'additional_fitted/tabpfn_v2_panel'/str(year)/target;a.save(path)
                outputs.append(prediction_frame(valid,target,name,p,time.perf_counter()-started,q,a.metadata))
                audits.append({'year':year,'target':target,'cutoff':f'{year}-01-01','train_rows':len(train),'prediction_rows':len(valid),
                   'max_training_label_asof':str(train['label_asof_'+target].max()),'seconds':time.perf_counter()-started,
                   'fit_state_sha256':sha(path/'model.tabpfn_fit')})
                save_json(RUN/'additional_audits/tabpfn_v2_panel.json',audits)
                print(name,year,target,'FOLD_PASS',round(time.perf_counter()-started,2),flush=True)
                del a;gc.collect();torch.cuda.empty_cache()
        a=TabPFNAdapter()  # Metadata only, no weights in GPU.
    save_predictions(name,outputs,{'family':a.metadata['family'],'metadata':{k:v for k,v in a.metadata.items() if k!='training_sample_ids'},
       'spec':SPEC[name],'audit':audits,'runtime_seconds':time.perf_counter()-began,'smoke_status':'PASS'})
    del a;gc.collect();torch.cuda.empty_cache();evaluate_all();print(name,'FULL_RESEARCH_SCORED',flush=True)


def replay(name):
    frame=dataset();old=pd.read_parquet(RUN/'predictions'/f'{name}.parquet');records=[]
    if name.startswith('sundial'):
        a=SundialAdapter.load(RUN/'additional_fitted/sundial_adapter.json');valid=frame[frame.asof_date.dt.year>=2019]
        tasks=[(None,target,valid) for target in ['h1','h2','h3','h4','ttm']];cache={}
    else:tasks=[(year,target,frame[frame.asof_date.dt.year==year]) for year in range(2019,2027) for target in ['h1','h2','h3','h4','ttm']]
    for year,target,valid in tasks:
        if name.startswith('sundial'):
            lane='ttm' if target=='ttm' else 'quarter'
            if lane not in cache:cache[lane]=a.forecast(valid,lane)
            pp,qq=cache[lane];j=3 if target=='ttm' else int(target[1:])-1;p=pp[:,j];q=qq[:,j]
        else:
            a=TabPFNAdapter.load(RUN/'additional_fitted/tabpfn_v2_panel'/str(year)/target);p,q=a.distribution(valid)
        reference=old[old.target_key==target].set_index('sample_id').loc[valid.sample_id]
        np.testing.assert_allclose(p,reference.predicted_eps,rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(q,reference[['p10_eps','p50_eps','p90_eps']],rtol=2e-5,atol=2e-5)
        records.append({'year':year,'target':target,'rows':len(valid),'status':'PASS',
          'maximum_point_difference':float(np.max(np.abs(p-reference.predicted_eps))),
          'maximum_quantile_difference':float(np.max(np.abs(q-reference[['p10_eps','p50_eps','p90_eps']].to_numpy())))})
        save_json(RUN/'additional_replay'/f'{name}.json',{'all_pass':True,'records':records,'fresh_process':True,'refit':False,'dataset_sha256':sha(RUN/'data/samples.parquet')})
        if not name.startswith('sundial'):del a;gc.collect();torch.cuda.empty_cache()
        print(name,'FRESH_REPLAY',year,target,'PASS',flush=True)


if __name__=='__main__':
    action=sys.argv[1];torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lock=RUN/'ADDITIONAL_FOUNDATION_PRESCORE_FREEZE.json'
    if action=='score':
        if lock.exists():raise RuntimeError('Explicit candidate resume required; no overwrite')
        save_json(lock,{'specifications':SPEC,'source_sha256':sha(__file__),'samples_sha256':sha(RUN/'data/samples.parquet'),
           'frozen_utc':datetime.now(timezone.utc).isoformat(),'test_scores_consumed':False,'selection':'Distinct official architectures, public ungated weights, no micro-tuning'})
    names=list(SPEC)
    if action=='retry-tabpfn-pathstring':
        import shutil
        name='tabpfn_v2_panel';p=RUN/'model_receipts'/f'{name}.json';prior=json.loads(p.read_text(encoding='utf-8'))
        if prior['status']!='BROKEN' or 'WindowsPath is not JSON serializable' not in prior['failure_reason']:raise RuntimeError('Not the approved bounded path serialization correction')
        if (RUN/'predictions'/f'{name}.parquet').exists():raise RuntimeError('Do not overwrite scored predictions')
        archive=RUN/'audit_corrections/tabpfn_initial_path_serialization_failure';archive.mkdir(parents=True,exist_ok=False)
        shutil.copy2(p,archive/p.name);names=[name]
    if action=='replay-tabpfn':names=['tabpfn_v2_panel']
    for name in names:
        if action=='replay' and not (RUN/'predictions'/f'{name}.parquet').exists():continue
        try:
            if action in ['score','retry-tabpfn-pathstring']:score(name,'pathstring_retry' if action!='score' else 'initial')
            else:replay(name)
        except Exception as exc:
            record={'model_id':name,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc(),'action':action}
            save_json(RUN/('model_receipts' if action in ['score','retry-tabpfn-pathstring'] else 'additional_replay')/f'{name}.json',record)
            print('ADDITIONAL_FOUNDATION_FAILED',name,action,str(exc),flush=True)
        gc.collect();torch.cuda.empty_cache()
