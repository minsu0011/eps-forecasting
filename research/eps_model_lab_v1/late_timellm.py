"""Bounded official TimeLLM, frozen GPT-2, independent-origin inference."""
from datetime import datetime,timezone
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast,models
from neuralforecast.losses.pytorch import MAE
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset,prediction_frame,metric_row
from research.eps_model_lab_v1.neural_models import NeuralAdapter,long_frame
from research.eps_model_lab_v1.late_frontier import frozen_hashes

ROOT=RUN/'late_frontier/TimeLLM'
SPEC={'patch_len':4,'stride':2,'d_ff':32,'top_k':5,'d_llm':768,'d_model':32,'n_heads':4,'enc_in':1,'dec_in':1,
    'llm_num_hidden_layers':12,'llm_output_attention':False,'llm_output_hidden_states':False,'dropout':.1,
    'prompt_prefix':'Quarterly signed GAAP diluted earnings per share in USD on the forecast origin share basis. Negative earnings are valid. '}


class FrontierTimeLLM(NeuralAdapter):
    def __init__(self,name='TimeLLM',steps=300):
        super().__init__(name,steps);self.model_id='late_nf_TimeLLM_gpt2_origin_isolated'
        self.metadata.update(post_freeze_diagnostic=True,portfolio_eligible=False,formal_certified=False,pit_valid=False,
            pretraining_overlap_unresolved=True,source_share_unit_feature_used=False)
    def fit(self,frame,target):
        receipt=json.loads((ROOT/'WEIGHTS.json').read_text(encoding='utf-8'))
        kwargs=dict(h=4,input_size=16,max_steps=self.steps,learning_rate=.0001,batch_size=16,windows_batch_size=16,
            inference_windows_batch_size=1,scaler_type='identity',random_seed=1729,loss=MAE(),val_check_steps=self.steps,
            early_stop_patience_steps=-1,accelerator='gpu',devices=1,precision='32-true',logger=False,
            enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,llm=receipt['path'],**SPEC)
        model=models.TimeLLM(**kwargs)
        if len(model.llm.h)!=12 or any(p.requires_grad for p in model.llm.parameters()):raise RuntimeError('Unexpected LLM backbone or unfreezing')
        self.nf=NeuralForecast(models=[model],freq=1);self.nf.fit(df=long_frame(frame,target),val_size=0);self.fitted=True;return self
    def path(self,frame):
        # Official prompt padding has no attention mask. Batch-dependent prompt
        # lengths move patch positions. Singleton windows remove this dependency.
        self.nf.models[0].inference_windows_batch_size=1
        return super().path(frame)


def probe(a,frame):
    p=a.path(frame)
    if not np.isfinite(p).all():raise RuntimeError('Nonfinite prediction')
    np.testing.assert_array_equal(p,a.path(frame.iloc[::-1])[::-1])
    np.testing.assert_array_equal(p[:1],a.path(frame.iloc[:1]))
    changed=frame.copy()
    for c in changed:
        if c.startswith('y_'):changed[c]=1e12
    np.testing.assert_array_equal(p,a.path(changed))
    changed=frame.copy()
    for c in changed:
        if c.startswith('eps_lag_'):changed.loc[changed.index[1:],c]=1e6
    np.testing.assert_array_equal(p[:1],a.path(changed)[:1]);return p


def run(replay=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    frame=dataset();planpath=ROOT/'PLAN.json';output=ROOT/'predictions.parquet'
    receipt=json.loads((ROOT/'WEIGHTS.json').read_text(encoding='utf-8'))
    for r in receipt['files']:assert sha(Path(receipt['path'])/r['name'])==r['sha256']
    if replay:
        original=pd.read_parquet(output);checks=[]
        for year in range(2019,2027):
            valid=frame[frame.asof_date.dt.year==year];a=FrontierTimeLLM.load(ROOT/'fitted'/str(year));p=a.path(valid);probe(a,valid.iloc[:8])
            for target in ['h1','h2','h3','h4','ttm']:
                expected=original[(original.target_key==target)&(original.asof_date.dt.year==year)].set_index('sample_id').loc[valid.sample_id]
                actual=p.sum(axis=1) if target=='ttm' else p[:,int(target[1])-1]
                np.testing.assert_array_equal(actual,expected.predicted_eps.to_numpy())
                checks.append({'year':year,'target':target,'rows':len(valid),'status':'EXACT_SAVED_REPLAY_PASS'})
            del a;gc.collect();torch.cuda.empty_cache()
        plan=json.loads(planpath.read_text(encoding='utf-8'));assert plan['frozen_main_hashes']==frozen_hashes()
        save_json(ROOT/'REPLAY.json',{'status':'PASS','fresh_process':True,'checks':checks,'main_unchanged':True,'four_input_probe_types_times_eight_years':32})
        print('TIMELLM_FRESH_REPLAY_PASS',len(checks),flush=True);return
    if planpath.exists():raise RuntimeError('No post-score overwrite or hyperparameter change')
    save_json(planpath,{'status':'PRE_SCORE_SPECIFICATION_FROZEN','created_utc':datetime.now(timezone.utc).isoformat(),
        'model':'TimeLLM','specification':SPEC,'steps':300,'seed':1729,'precision':'FP32_NO_TF32',
        'batch_size':16,'windows_batch_size':16,'input_size':16,'learning_rate':.0001,
        'frozen_main_hashes':frozen_hashes(),'source_sha256':sha(inspect.getfile(models.TimeLLM)),
        'adapter_source_sha256':sha(__file__),'post_test_inspection_diagnostic':True,
        'independent_confirmation':False,'may_change_main_portfolio':False,'weights_receipt_sha256':sha(ROOT/'WEIGHTS.json'),
        'inference_windows_batch_size':1,'inference_isolation_reason':'Prompt padding without attention mask changes patch positions with other origins',
        'source':'https://nixtlaverse.nixtla.io/neuralforecast/models.timellm.html',
        'target_C':'Sum of four predicted quarterly EPS; weighted-share approximation'})
    results=[];audits=[];began=time.perf_counter()
    for smoke,years in [(True,[2019]),(False,range(2019,2027))]:
        for year in years:
            if not smoke and datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-08T01:38:26+00:00'):
                raise RuntimeError('Finalization boundary reached; no new training fold')
            cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC');train=frame[frame.asof_date<cutoff]
            known=np.column_stack([(train[f'label_asof_h{j}']<cutoff)&train[f'y_h{j}'].notna() for j in range(1,5)]).any(axis=1)
            train=train.loc[known];valid=frame[frame.asof_date.dt.year==year]
            if smoke:train=train.iloc[:128];valid=valid.iloc[:8]
            start=time.perf_counter();a=FrontierTimeLLM(steps=5 if smoke else 300).fit(train,cutoff)
            p=probe(a,valid) if smoke else a.path(valid)
            if not np.isfinite(p).all():raise RuntimeError('Nonfinite full prediction')
            path=ROOT/('smoke_fitted' if smoke else 'fitted')/str(year);a.save(path)
            reloaded=FrontierTimeLLM.load(path);np.testing.assert_array_equal(p,reloaded.path(valid));del reloaded
            if not smoke:
                for target in ['h1','h2','h3','h4','ttm']:
                    pred=p.sum(axis=1) if target=='ttm' else p[:,int(target[1])-1]
                    result=prediction_frame(valid,target,a.model_id,pred,time.perf_counter()-start,metadata=a.get_metadata())
                    result['post_freeze_diagnostic']=True;result['formal_certified']=False
                    if target=='ttm':result['ttm_prediction_method']='SUM_FORECAST_QUARTERS_WEIGHTED_SHARE_APPROXIMATION'
                    results.append(result)
            artifacts=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in path.rglob('*') if p.is_file()]
            audits.append({'year':year,'smoke':smoke,'status':'PASS','train_origins':len(train),'origins':len(valid),
                'seconds':time.perf_counter()-start,'artifacts':artifacts})
            save_json(ROOT/'FOLD_AUDIT.json',audits)
            print('TimeLLM',year,'SMOKE_PASS' if smoke else 'FOLD_PASS',round(time.perf_counter()-start,2),flush=True)
            del a;gc.collect();torch.cuda.empty_cache()
    result=pd.concat(results,ignore_index=True);assert len(result)==10505
    assert not result.duplicated(['sample_id','target_key']).any()
    result.to_parquet(output,index=False);scores=[]
    for (target,split),group in result.groupby(['target_key','split']):
        if split=='VALIDATION_OOF':group=group[group.label_asof<pd.Timestamp('2022-01-01',tz='UTC')]
        scores.append({'model_id':'late_nf_TimeLLM_gpt2_origin_isolated','target':target,'split':split,
            'mask':'AVAILABLE_BEFORE_2022' if split=='VALIDATION_OOF' else 'ALL_MATURE_RESEARCH_TEST',**metric_row(group)})
    pd.DataFrame(scores).to_csv(ROOT/'SCORES.csv',index=False)
    plan=json.loads(planpath.read_text(encoding='utf-8'));assert plan['frozen_main_hashes']==frozen_hashes()
    save_json(ROOT/'COMPLETION.json',{'status':'FULL_POST_FREEZE_RESEARCH_DIAGNOSTIC_SCORED','rows':len(result),
        'runtime_seconds':time.perf_counter()-began,'prediction_sha256':sha(output),'main_unchanged':True,
        'formal_certified':False,'portfolio_member':False,'source_unit_feature_used':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');run(parser.parse_args().replay)
