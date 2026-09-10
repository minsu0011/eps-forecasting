"""Audited Time-MoE decoder and MOMENT frozen embeddings with causal fitted heads.

No untrained forecasting head is presented as a pretrained forecast. Both remain
retrospective research because the checkpoint pretraining overlap is unresolved.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
os.environ['OMP_NUM_THREADS']='4'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
os.environ['HF_HOME']=str(PROJECT.parent/'.cache/eps_models')
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,EXTERNAL,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,TARGETS,dataset,split_train,prediction_frame,save_predictions,evaluate_all
from research.eps_model_lab_v1.foundation_models import histories


def weight(repo):
    return json.loads((RUN/'weight_receipts'/(repo.replace('/','__')+'.json')).read_text(encoding='utf-8'))


def metadata(repo):
    r=weight(repo)
    return {'family':'FOUNDATION_EXTERNAL','pit_valid':False,'causal':True,
            'evidence_class':'RETROSPECTIVE_RESEARCH_PRETRAINING_OVERLAP_UNRESOLVED',
            'source_commit':r['revision'],'license':r['license'],'weight_path':r['path']}


class TimeMoeAdapter(EPSAdapter):
    def __init__(self):
        super().__init__('time_moe_50m',metadata('Maple728/TimeMoE-50M'))
        self.metadata.update(zero_shot=True,precision='FP32',decoder='Official generate, eager attention, use_cache=False',
                             input_missingness='Causal seasonal/last input fill, never label fill')
    def fit(self,frame=None,target=None):
        sys.path.insert(0,str(EXTERNAL/'time_moe'))
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction
        self.model=TimeMoeForPrediction.from_pretrained(self.metadata['weight_path'],attn_implementation='eager',torch_dtype=torch.float32).to('cuda').eval()
        self.metadata['local_source_hashes']={str(p.relative_to(EXTERNAL/'time_moe')):sha(p) for p in (EXTERNAL/'time_moe/time_moe/models').glob('*.py')}
        self.fitted=True;return self
    def forecast(self,frame,lane):
        _,filled=histories(frame,lane);parts=[]
        with torch.inference_mode():
            for start in range(0,len(frame),64):
                x=torch.tensor(filled[start:start+64],device='cuda')
                mean=x.mean(-1,keepdim=True);std=x.std(-1,keepdim=True).clamp_min(1e-6)
                z=(x-mean)/std
                out=self.model.generate(z.clone(),max_new_tokens=4,use_cache=False)
                p=out[:,-4:]*std+mean
                parts.append(p.cpu().numpy())
        return np.concatenate(parts)
    def predict(self,frame,target):
        return self.forecast(frame,'ttm' if target=='ttm' else 'quarter')[:,3 if target=='ttm' else int(target[1:])-1]
    def save(self,path): save_json(path,{'model_id':self.model_id,'metadata':self.get_metadata()})
    @classmethod
    def load(cls,path):
        r=json.loads(Path(path).read_text(encoding='utf-8'));a=cls()
        if r['metadata']['source_commit']!=a.metadata['source_commit']: raise RuntimeError('Checkpoint revision drift')
        return a.fit()


class MomentEmbeddingAdapter(EPSAdapter):
    def __init__(self):
        super().__init__('moment_embedding_ridge',metadata('AutonLab/MOMENT-1-small'))
        self.metadata.update(zero_shot=False,finetuned='Ridge readout only; encoder frozen',ridge_alpha=100.,
            embedding_geometry='32 causal-filled fiscal quarters in last 32 of 512 positions; other 480 positions masked; original observation fraction appended',
            training_surface='Annual expanding fit, each horizon labels available strictly before cutoff; scaler fit only on training rows')
    def encoder(self):
        from momentfm import MOMENTPipeline
        self.model=MOMENTPipeline.from_pretrained(self.metadata['weight_path'],model_kwargs={'task_name':'embedding'})
        self.model.init();self.model=self.model.to('cuda').eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        return self
    def prepare_data(self,frame):
        _,filled=histories(frame,'quarter');out=[]
        with torch.inference_mode():
            for start in range(0,len(frame),32):
                v=filled[start:start+32];x=np.pad(v,((0,0),(480,0)),constant_values=0.)
                mask=np.zeros_like(x,dtype=np.int64);mask[:,-32:]=1
                result=self.model.embed(x_enc=torch.tensor(x,device='cuda')[:,None,:],input_mask=torch.tensor(mask,device='cuda'))
                out.append(result.embeddings.float().cpu().numpy())
        embeddings=np.concatenate(out)
        # Preserve EPS level/scale discarded by encoder normalization. No future or price inputs.
        scale=frame.scale.to_numpy()[:,None]
        raw,_=histories(frame,'quarter')
        extra=np.column_stack([filled[:,-1]/scale[:,0],filled.mean(axis=1)/scale[:,0],
                               filled.std(axis=1)/scale[:,0],np.isfinite(raw).mean(axis=1),frame.fiscal_sin,frame.fiscal_cos])
        return np.column_stack([embeddings,extra])
    def fit(self,frame,target):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        self.head=make_pipeline(StandardScaler(),Ridge(alpha=100.))
        self.head.fit(self.prepare_data(frame),frame['y_'+target].to_numpy()/frame.scale.to_numpy())
        self.fitted=True;self.target=target;return self
    def predict(self,frame,target):
        if getattr(self,'target',target)!=target:raise RuntimeError('MOMENT readout target mismatch')
        return self.head.predict(self.prepare_data(frame))*frame.scale.to_numpy()
    def save(self,path):
        import pickle
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        with Path(path).open('wb') as f: pickle.dump({'head':self.head,'metadata':self.get_metadata(),'target':getattr(self,'target',None)},f,protocol=5)
    @classmethod
    def load(cls,path):
        import pickle
        with Path(path).open('rb') as f: r=pickle.load(f)
        a=cls().encoder();a.head=r['head'];a.fitted=True
        a.target=r.get('target') or Path(path).stem.split('_',1)[1];return a


def run_timemoe():
    start=time.perf_counter();frame=dataset();valid=frame[frame.asof_date.dt.year>=2019].reset_index(drop=True)
    a=TimeMoeAdapter().fit();smoke=valid.iloc[:8];p=a.forecast(smoke,'quarter')
    assert p.shape==(8,4) and np.isfinite(p).all()
    changed=smoke.copy()
    for c in changed:
        if c.startswith(('filled_lag_','eps_lag_')): changed.loc[changed.index[1:],c]=777.
    np.testing.assert_allclose(p[0],a.forecast(changed,'quarter')[0],rtol=1e-5,atol=1e-5)
    path=RUN/'foundation_adapters/time_moe_50m.json';a.save(path)
    loaded=TimeMoeAdapter.load(path);np.testing.assert_allclose(p,loaded.forecast(smoke,'quarter'),rtol=0,atol=0);del loaded
    save_json(RUN/'smoke_receipts/time_moe_50m_summary.json',{'status':'PASS','finite':True,'cross_origin_mutation':'PASS','save_load':'EXACT_PASS'})
    outputs=[]
    for lane in ['quarter','ttm']:
        began=time.perf_counter();p=a.forecast(valid,lane);duration=time.perf_counter()-began
        for t in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
            j=3 if t=='ttm' else int(t[1:])-1
            outputs.append(prediction_frame(valid,t,a.model_id,p[:,j],duration,metadata=a.get_metadata()))
    save_predictions(a.model_id,outputs,{'metadata':a.get_metadata(),'family':'FOUNDATION_ZERO_SHOT','runtime_seconds':time.perf_counter()-start,'smoke_status':'PASS'})
    evaluate_all();print(a.model_id,'FULL_SCORE_COMPLETE',time.perf_counter()-start,flush=True)


def run_moment():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    started=time.perf_counter();frame=dataset().reset_index(drop=True);a=MomentEmbeddingAdapter().encoder()
    smoke=frame.iloc[:8];emb=a.prepare_data(smoke)
    assert np.isfinite(emb).all()
    np.testing.assert_allclose(emb,a.prepare_data(smoke.iloc[::-1])[::-1],rtol=2e-5,atol=2e-5)
    x=a.prepare_data(frame);assert np.isfinite(x).all()
    cache=RUN/'data/moment_frozen_embeddings.npy';np.save(cache,x)
    outputs=[];audit=[]
    for year in range(2019,2027):
        for target in TARGETS:
            train,valid=split_train(frame,year,target)
            a.head=make_pipeline(StandardScaler(),Ridge(alpha=100.))
            a.head.fit(x[train.index],train['y_'+target].to_numpy()/train.scale.to_numpy())
            a.target=target;a.fitted=True
            p=a.head.predict(x[valid.index])*valid.scale.to_numpy()
            path=RUN/'moment_fitted'/f'{year}_{target}.pkl';a.save(path)
            outputs.append(prediction_frame(valid,target,a.model_id,p,metadata=a.get_metadata()))
            audit.append({'year':year,'target':target,'train_rows':len(train),'prediction_rows':len(valid),'status':'PASS'})
        print(a.model_id,year,'FOLD_PASS',flush=True)
    # Actual external encoder + locally stored trained head reload, not only metadata roundtrip.
    loaded=MomentEmbeddingAdapter.load(RUN/'moment_fitted/2026_ttm.pkl')
    np.testing.assert_allclose(a.predict(smoke,'ttm'),loaded.predict(smoke,'ttm'),rtol=0,atol=0)
    save_json(RUN/'smoke_receipts/moment_embedding_ridge_summary.json',{'status':'PASS','embedding_shape':list(emb.shape),'batch_reversal':'PASS','save_load':'EXACT_PASS'})
    save_predictions(a.model_id,outputs,{'family':'FOUNDATION_FROZEN_EMBEDDING_CAUSAL_HEAD','metadata':a.get_metadata(),
              'audit':audit,'runtime_seconds':time.perf_counter()-started,'embedding_cache_sha256':sha(cache),'smoke_status':'PASS'})
    evaluate_all();print(a.model_id,'FULL_SCORE_COMPLETE',time.perf_counter()-started,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('models',nargs='+',choices=['time_moe_50m','moment_embedding_ridge']);args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1729);np.random.seed(1729)
    for model in args.models:
        try: run_timemoe() if model=='time_moe_50m' else run_moment()
        except Exception as exc:
            save_json(RUN/'model_receipts'/f'{model}.json',{'model_id':model,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
            print(model,'BROKEN',str(exc),flush=True)
        gc.collect();torch.cuda.empty_cache()
