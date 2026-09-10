"""One fixed official Chronos-2 full-finetuning adaptation, no test tuning.

Joint quarter/native-TTM tasks, raw missing observations and cutoff-masked real
future labels. Official random-window training only sees pre-cutoff information.
Frozen pretraining vintage remains unresolved: this is NOT historical PIT proof.
"""
from __future__ import annotations
import gc
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
from chronos import Chronos2Pipeline
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset,prediction_frame,save_predictions,evaluate_all
from research.eps_model_lab_v1.foundation_models import histories

MODEL='chronos2_eps_joint_finetuned'
SPEC={'model_id':MODEL,'finetune_mode':'full','learning_rate':1e-6,'steps':300,'batch_size':64,
      'context_length':32,'min_past':8,'prediction_length':4,'precision':'FP32','seed':1729,
      'training_tasks':'Quarter32+4 and nativeTTM8+4, raw missing retained, future labels only if released before annual cutoff',
      'weights_initialization':'Same original checkpoint independently for each annual fold, no later-year carryback',
      'model_selection':'None; fixed steps, no validation/test early stopping',
      'pretraining_overlap_unresolved':True,'source':'https://github.com/amazon-science/chronos-forecasting'}


def train_inputs(frame,year):
    cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC');eligible=frame.asof_date<cutoff
    selected=frame.loc[eligible];quarter=histories(selected,'quarter')[0]
    qfuture=np.column_stack([selected[f'y_h{j}'].where(selected[f'label_asof_h{j}']<cutoff).to_numpy() for j in range(1,5)])
    q=np.column_stack([quarter,qfuture]).astype('float32')
    with np.load(RUN/'data/multivariate_company_origin_v1.npz',allow_pickle=False) as tensors:
        ttmfuture=(tensors['y'][:,:,1]*tensors['scale'][:,None,1])[eligible.to_numpy()]
        known=tensors['availability'][:,:,1][eligible.to_numpy()]<cutoff.value
    ttmfuture=np.where(known,ttmfuture,np.nan)
    t=np.column_stack([histories(selected,'ttm')[0],ttmfuture]).astype('float32')
    out=[]
    for block in [q,t]:
        for row in block:
            finite=np.flatnonzero(np.isfinite(row))
            if not len(finite): continue
            row=row[:finite[-1]+1]
            if len(row)>=12 and np.isfinite(row).sum()>=8: out.append(row)
    return out,{'year':year,'cutoff':str(cutoff),'train_origins':len(selected),'actual_train_tasks':len(out),
                'future_labels_after_cutoff':'Masked as missing then trailing missing truncated, never trained as observed'}


def forecast(model,frame,lane):
    # The official fit() returns a pipeline wrapping the last training-mode model;
    # explicitly disable dropout before deterministic inference and reload checks.
    model.model.eval()
    raw=histories(frame,lane)[0];qs=[];ps=[]
    for start in range(0,len(raw),64):
        q,p=model.predict_quantiles(torch.tensor(raw[start:start+64].copy())[:,None,:],prediction_length=4,
              quantile_levels=[.1,.5,.9],batch_size=64,cross_learning=False)
        qs.append(torch.stack(q).squeeze(1).cpu().numpy());ps.append(torch.stack(p).squeeze(1).cpu().numpy())
    return np.concatenate(ps),np.concatenate(qs)


def fit(base,inputs,path,steps):
    return base.fit(inputs,prediction_length=4,finetune_mode='full',context_length=32,min_past=8,
       learning_rate=1e-6,num_steps=steps,batch_size=64,output_dir=path,finetuned_ckpt_name='final',
       validation_inputs=None,bf16=False,tf32=False,fp16=False,seed=1729,data_seed=1729,
       disable_tqdm=True,logging_steps=100,report_to='none',remove_printer_callback=True,
       optim='adamw_torch_fused',dataloader_num_workers=0)


class ChronosFineTunedAdapter(EPSAdapter):
    """Common research API for an actual annual fine-tuned checkpoint."""
    def __init__(self):
        receipt=json.loads((RUN/'weight_receipts/amazon__chronos-2.json').read_text(encoding='utf-8'))
        super().__init__(MODEL,{'family':'FOUNDATION_FINETUNED','pit_valid':False,'causal':True,
          'zero_shot':False,'finetuned':True,'source_commit':receipt['revision'],'license':receipt['license'],
          'evidence_class':'RETROSPECTIVE_RESEARCH_PRETRAINING_OVERLAP_UNRESOLVED'})
        self.pipeline=None
    def prepare_data(self,frame):return histories(frame,'quarter')[0]
    def fit(self,frame,target):
        year=int(target)
        from research.eps_model_lab_v1.common import require_exact_frozen_frame
        require_exact_frozen_frame(frame)
        from research.eps_model_lab_v1.multivariate_models import tensors
        tensors(frame)  # Verify the tensor/geometry/dataset hash binding.
        path=RUN/'chronos_finetuned'/str(year)
        if path.exists():raise RuntimeError('Annual checkpoint exists; load it instead of overwriting')
        r=json.loads((RUN/'weight_receipts/amazon__chronos-2.json').read_text(encoding='utf-8'))
        base=Chronos2Pipeline.from_pretrained(r['path'],device_map='cuda',dtype=torch.float32)
        inputs,_=train_inputs(frame,year);self.pipeline=fit(base,inputs,path,SPEC['steps'])
        self.pipeline.model.eval();self.fitted=True;self.metadata['fit_cutoff_year']=year;return self
    def predict(self,frame,target):
        if not self.fitted:raise RuntimeError('Checkpoint not loaded/fitted')
        p,_=forecast(self.pipeline,frame,'ttm' if target=='ttm' else 'quarter')
        return p[:,3 if target=='ttm' else int(target[1:])-1]
    def save(self,path):
        path=Path(path)
        if path.exists():raise RuntimeError('Save destination already exists; no implicit overwrite')
        if not self.fitted:raise RuntimeError('No fitted model to save')
        self.pipeline.model.save_pretrained(path)
        save_json(path/'EPS_ADAPTER_METADATA.json',self.get_metadata())
    @classmethod
    def load(cls,path):
        path=Path(path);a=cls();a.pipeline=Chronos2Pipeline.from_pretrained(path,device_map='cuda',dtype=torch.float32)
        metadata=path/'EPS_ADAPTER_METADATA.json'
        if metadata.exists():a.metadata=json.loads(metadata.read_text(encoding='utf-8'))
        elif path.parent.name.isdigit():a.metadata['fit_cutoff_year']=int(path.parent.name)
        a.pipeline.model.eval();a.fitted=True;return a


def run():
    lock=RUN/'CHRONOS_EPS_FINETUNE_SPEC_V1.json'
    if lock.exists():
        if json.loads(lock.read_text(encoding='utf-8'))['spec']!=SPEC: raise RuntimeError('Finetune specification drift')
    else: save_json(lock,{'spec':SPEC,'source_sha256':sha(__file__),'frozen_before_candidate_score':True,
                          'multivariate_geometry_sha256':sha(RUN/'EPS_MULTIVARIATE_GEOMETRY_V1.json')})
    receipt=json.loads((RUN/'weight_receipts/amazon__chronos-2.json').read_text(encoding='utf-8'))
    frame=dataset();torch.set_num_threads(4);torch.manual_seed(1729)
    base=Chronos2Pipeline.from_pretrained(receipt['path'],device_map='cuda',dtype=torch.float32)
    inputs,audit=train_inputs(frame,2019);probe=frame[frame.asof_date.dt.year==2019].iloc[:8]
    smoke_path=RUN/'chronos_finetune_smoke_evalmode'
    smoke=fit(base,inputs[:64],smoke_path,3)
    p,q=forecast(smoke,probe,'quarter')
    np.testing.assert_allclose(p,forecast(smoke,probe.iloc[::-1],'quarter')[0][::-1],rtol=2e-5,atol=2e-5)
    reloaded=Chronos2Pipeline.from_pretrained(smoke_path/'final',device_map='cuda',dtype=torch.float32)
    np.testing.assert_allclose(p,forecast(reloaded,probe,'quarter')[0],rtol=0,atol=0)
    changed=probe.copy()
    for c in changed:
        if c.startswith(('y_','label_','target_')): changed[c]=1e12
    np.testing.assert_allclose(p,forecast(smoke,changed,'quarter')[0],rtol=0,atol=0)
    if not np.isfinite(p).all() or not np.isfinite(q).all(): raise RuntimeError('Smoke nonfinite')
    save_json(RUN/'smoke_receipts'/f'{MODEL}.json',{'status':'PASS','actual_fit_steps':3,'reload_exact':True,'cross_origin_reverse':True,'future_label_mutation':True})
    del smoke,reloaded;gc.collect();torch.cuda.empty_cache()
    outputs=[];audits=[];began=time.perf_counter()
    meta={'family':'FOUNDATION_FINETUNED','pit_valid':False,'zero_shot':False,'finetuned':True,'version':'CHRONOS_2_EPS_FT_V1',
          'source_commit':receipt['revision'],'license':receipt['license'],'evidence_class':'RETROSPECTIVE_RESEARCH_PRETRAINING_OVERLAP_UNRESOLVED'}
    for year in range(2019,2027):
        inputs,audit=train_inputs(frame,year);valid=frame[frame.asof_date.dt.year==year];start=time.perf_counter()
        path=RUN/'chronos_finetuned'/str(year);model=fit(base,inputs,path,SPEC['steps'])
        for lane in ['quarter','ttm']:
            p,q=forecast(model,valid,lane)
            if not np.isfinite(p).all() or not np.isfinite(q).all(): raise RuntimeError('Nonfinite full prediction')
            for target in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
                j=3 if target=='ttm' else int(target[1:])-1
                outputs.append(prediction_frame(valid,target,MODEL,p[:,j],time.perf_counter()-start,q[:,j,:],meta))
        weights={str(p.relative_to(path)):sha(p) for p in (path/'final').glob('*') if p.is_file()}
        audit.update(seconds=time.perf_counter()-start,weights=weights,status='PASS')
        audits.append(audit);save_json(RUN/'chronos_finetune_audit.json',audits)
        print(MODEL,year,'FOLD_PASS',round(time.perf_counter()-start,2),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    save_predictions(MODEL,outputs,{'family':'FOUNDATION_FINETUNED','metadata':meta,'audit':audits,'spec':SPEC,
         'runtime_seconds':time.perf_counter()-began,'smoke_status':'PASS'})
    evaluate_all()


if __name__=='__main__':
    try: run()
    except Exception as exc:
        save_json(RUN/'chronos_finetune_failure.json',{'status':'BROKEN','reason':str(exc),'traceback':traceback.format_exc()})
        raise
