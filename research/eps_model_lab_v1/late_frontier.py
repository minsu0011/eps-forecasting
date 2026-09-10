"""Post-freeze breadth diagnostics. Never writes into the frozen main portfolio."""
from datetime import datetime, timezone
import argparse
import gc
import inspect
import json
from pathlib import Path
import sys
import time
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
import torch
from neuralforecast import models
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import dataset, prediction_frame, metric_row
from research.eps_model_lab_v1.multivariate_models import MultivariateAdapter, tensors, to_device, CONTRACT

ROOT = RUN/'late_frontier'
SPECS = {'MLPMultivariate': {'hidden_size':128, 'num_layers':2},
         'SOFTSSharp': {'hidden_size':64, 'd_core':32, 'e_layers':2, 'd_ff':128, 'dropout':.1, 'pe_keep_prob':.5}}


def frozen_hashes():
    files = list((RUN/'predictions').glob('*.parquet')) + [RUN/'ENSEMBLE_FREEZE_V1.json', RUN/'data/samples.parquet']
    files += list(RUN.glob('*SHORTLIST*'))
    return {p.relative_to(RUN).as_posix():sha(p) for p in files if p.is_file()}


class FrontierMultivariate(MultivariateAdapter):
    def __init__(self, name):
        super().__init__(name)
        self.model_id = 'late_nf_multivar_'+name
        self.metadata.update(post_freeze_diagnostic=True, portfolio_eligible=False,
            source_share_unit_feature_used=False, formal_certified=False,
            specification=SPECS[name])

    def initialize(self):
        torch.manual_seed(1729); torch.cuda.manual_seed_all(1729)
        self.net = getattr(models,self.name)(h=4,input_size=32,n_series=4,random_seed=1729,**SPECS[self.name]).cuda()
        return self


def probe(adapter, data, indices):
    p = adapter.path(data, indices)
    if not np.isfinite(p).all(): raise RuntimeError('Nonfinite inference')
    np.testing.assert_allclose(p, adapter.path(data,indices[::-1])[::-1],rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(p[:1],adapter.path(data,indices[:1]),rtol=1e-5,atol=1e-5)
    changed = dict(data); changed['y'] = torch.full_like(data['y'],1e12)
    np.testing.assert_array_equal(p,adapter.path(changed,indices))
    altered = dict(data); altered['x'] = data['x'].clone()
    altered['x'][indices[1:]] = 1e6
    np.testing.assert_allclose(p[:1],adapter.path(altered,indices)[:1],rtol=1e-5,atol=1e-5)
    return p


def execute(name, replay=False):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    frame=dataset(); data=to_device(tensors(frame)); root=ROOT/name
    planpath=root/'PLAN.json'; output=root/'predictions.parquet'
    if replay:
        plan=json.loads(planpath.read_text(encoding='utf-8'))
        if plan['frozen_main_hashes']!=frozen_hashes(): raise RuntimeError('Main portfolio changed')
        original=pd.read_parquet(output); rows=[]
        for year in range(2019,2027):
            idx=np.flatnonzero((frame.asof_date.dt.year==year).to_numpy())
            a=FrontierMultivariate.load(root/'fitted'/f'{year}.pt')
            p=a.path(data,idx)
            probe(a,data,idx[:8])
            for target,j,c in [(f'h{j+1}',j,0) for j in range(4)]+[('ttm',3,1)]:
                g=original[(original.target_key==target)&(original.asof_date.dt.year==year)]
                g=g.set_index('sample_id').loc[frame.iloc[idx].sample_id]
                np.testing.assert_array_equal(p[:,j,c].astype(float),g.predicted_eps.to_numpy())
                rows.append({'year':year,'target':target,'rows':len(idx),'status':'EXACT_SAVED_REPLAY_PASS'})
            del a;gc.collect();torch.cuda.empty_cache()
        save_json(root/'REPLAY.json',{'status':'PASS','fresh_process':True,'checks':rows,
            'singleton_reverse_other_origin_future_label_probes':32,'main_unchanged':plan['frozen_main_hashes']==frozen_hashes()})
        print(name,'FRESH_REPLAY_PASS',len(rows),flush=True);return
    if planpath.exists(): raise RuntimeError('No overwrite or post-score specification revision')
    source=Path(inspect.getfile(getattr(models,name)))
    save_json(planpath,{'created_utc':datetime.now(timezone.utc).isoformat(),'model':name,
        'specification':SPECS[name],'trainer':CONTRACT['trainer'],'geometry_contract':CONTRACT,
        'frozen_main_hashes':frozen_hashes(),'source_file':str(source),'source_sha256':sha(source),
        'adapter_source_sha256':sha(__file__),'sources':['https://github.com/Nixtla/neuralforecast',
        'https://nixtlaverse.nixtla.io/neuralforecast/models.mlpmultivariate.html'],
        'status':'PRE_SCORE_SPECIFICATION_FROZEN','post_test_inspection_diagnostic':True,
        'independent_confirmation':False,'may_change_main_portfolio':False,
        'deadline_new_long_experiments':'2026-09-08T01:38:26Z'})
    outputs=[]; audits=[]; began=time.perf_counter()
    for smoke,years in [(True,[2019]),(False,range(2019,2027))]:
        for year in years:
            cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC')
            eligible=((data['availability']<cutoff.value)&torch.isfinite(data['y'])).any(dim=(1,2)).cpu().numpy()
            train=np.flatnonzero((frame.asof_date<cutoff).to_numpy()&eligible)
            idx=np.flatnonzero((frame.asof_date.dt.year==year).to_numpy())
            if smoke: train=train[:128];idx=idx[:8]
            start=time.perf_counter()
            a=FrontierMultivariate(name).fit_arrays(data,train,cutoff.value,5 if smoke else 300)
            p=probe(a,data,idx) if smoke else a.path(data,idx)
            if not np.isfinite(p).all(): raise RuntimeError('Nonfinite full prediction')
            path=root/('smoke_fitted' if smoke else 'fitted')/f'{year}.pt';a.save(path)
            reloaded=FrontierMultivariate.load(path)
            np.testing.assert_array_equal(p,reloaded.path(data,idx));del reloaded
            if not smoke:
                for target,j,c in [(f'h{j+1}',j,0) for j in range(4)]+[('ttm',3,1)]:
                    result=prediction_frame(frame.iloc[idx],target,a.model_id,p[:,j,c],time.perf_counter()-start,metadata=a.get_metadata())
                    result['post_freeze_diagnostic']=True;result['formal_certified']=False
                    if target=='ttm': result['ttm_prediction_method']='DIRECT_NATIVE_TTM_MULTIVARIATE_CHANNEL'
                    outputs.append(result)
            audits.append({'year':year,'smoke':smoke,'status':'PASS','train_origins':len(train),'origins':len(idx),
                'seconds':time.perf_counter()-start,'artifact':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
            save_json(root/'FOLD_AUDIT.json',audits)
            print(name,year,'SMOKE_PASS' if smoke else 'FOLD_PASS',round(time.perf_counter()-start,2),flush=True)
            del a;gc.collect();torch.cuda.empty_cache()
    result=pd.concat(outputs,ignore_index=True)
    assert len(result)==2101*5 and not result.duplicated(['sample_id','target_key']).any()
    result.to_parquet(output,index=False)
    scores=[]
    for (target,split),group in result.groupby(['target_key','split']):
        if split=='VALIDATION_OOF':group=group[group.label_asof<pd.Timestamp('2022-01-01',tz='UTC')]
        scores.append({'model_id':'late_nf_multivar_'+name,'target':target,'split':split,
            'mask':'AVAILABLE_BEFORE_2022' if split=='VALIDATION_OOF' else 'ALL_MATURE_RESEARCH_TEST',**metric_row(group)})
    pd.DataFrame(scores).to_csv(root/'SCORES.csv',index=False)
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    assert plan['frozen_main_hashes']==frozen_hashes()
    save_json(root/'COMPLETION.json',{'status':'FULL_POST_FREEZE_RESEARCH_DIAGNOSTIC_SCORED',
        'prediction_sha256':sha(output),'runtime_seconds':time.perf_counter()-began,
        'rows':len(result),'main_unchanged':True,'formal_certified':False,'portfolio_member':False,
        'source_unit_feature_used':False,'future_share_feature_mutation':'NOT_APPLICABLE_NO_SHARE_CHANNEL',
        'prediction_contract':'pit_valid means inference geometry only; current-vintage source is not certified PIT'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('name',choices=list(SPECS));parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();execute(args.name,args.replay)
