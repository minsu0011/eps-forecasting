"""Remove unreliable accounting auxiliary channels; separate fixed diagnostic only."""
from datetime import datetime,timezone
import argparse
import gc
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from neuralforecast import models
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset,prediction_frame,metric_row
from research.eps_model_lab_v1.multivariate_models import MultivariateAdapter,tensors,to_device,SPECS,build_tensors
from research.eps_model_lab_v1.extra_architecture_specs import EXTRA_MULTIVARIATE
from research.eps_model_lab_v1.late_frontier import frozen_hashes,probe

ROOT=RUN/'eps_only_channel_diagnostics'
NAMES=['TimeXer','SOFTS']


class EPSOnlyAdapter(MultivariateAdapter):
    def __init__(self,name):
        if name not in NAMES:raise ValueError(name)
        super().__init__(name);self.model_id='nf_'+name+'_EPS_TTM_ONLY_DIAGNOSTIC'
        self.metadata.update(channels=['GAAP_diluted_quarter_EPS','native_TTM_EPS'],n_series=2,
            post_freeze_data_quality_diagnostic=True,portfolio_eligible=False,formal_certified=False,
            accounting_NI_revenue_share_features_consumed=False)
    def prepare_data(self,frame):
        return {key:np.array(value[...,:2],copy=True) for key,value in build_tensors(frame).items()}
    def fit(self,frame,target):
        from research.eps_model_lab_v1.common import require_exact_frozen_frame
        require_exact_frozen_frame(frame)
        cutoff=pd.Timestamp(f'{int(target)}-01-01',tz='UTC')
        arrays={key:np.array(value[...,:2],copy=True) for key,value in tensors(frame).items()}
        data=to_device(arrays)
        eligible=((data['availability']<cutoff.value)&torch.isfinite(data['y'])).any(dim=(1,2)).cpu().numpy()
        train=np.flatnonzero((frame.asof_date<cutoff).to_numpy()&eligible)
        self.fit_arrays(data,train,cutoff.value,300);self.metadata['fit_cutoff_year']=int(target);return self
    def predict(self,frame,target):
        p=self.path(to_device(self.prepare_data(frame)),np.arange(len(frame)))
        return p[:,3,1] if target=='ttm' else p[:,int(target[1:])-1,0]
    def initialize(self):
        torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
        spec=SPECS[self.name] if self.name in SPECS else EXTRA_MULTIVARIATE[self.name]
        self.net=getattr(models,self.name)(h=4,input_size=32,n_series=2,random_seed=1729,**spec).cuda();return self


def run(replay=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    frame=dataset();arrays=tensors(frame)
    # Every array's final axis is channel. Keep values/masks/availability and
    # normalization of the native EPS/TTM channels unchanged.
    reduced={key:np.array(value[...,:2],copy=True) for key,value in arrays.items()}
    data=to_device(reduced);planpath=ROOT/'PLAN.json'
    if not replay:
        if planpath.exists():raise RuntimeError('No specification overwrite')
        save_json(planpath,{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PRE_SCORE_FIXED_ABLATION',
            'names':NAMES,'steps':300,'seed':1729,'change':'Drop native NI/revenue channels; keep EPS and TTM values, availability and scaling verbatim. n_series 4 to 2.',
            'known_duration_warning':'SOURCE_ACCOUNTING_DURATION_WARNING_KO.md','post_test_inspection_diagnostic':True,
            'new_portfolio_members':0,'frozen_main_hashes':frozen_hashes(),'source_sha256':sha(__file__),
            'removing_channels_also_changes_auxiliary_supervision_and_parameter_shape':True,
            'pure_causal_effect_of_bad_data_identified':False,'whole_dataset_quality_certified':False})
    plan=json.loads(planpath.read_text(encoding='utf-8'));assert plan['frozen_main_hashes']==frozen_hashes()
    audits=[];scores=[];replays=[]
    for name in NAMES:
        out=[];output=ROOT/f'{name}.parquet';stored=pd.read_parquet(output) if replay else None
        for year in range(2019,2027):
            cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC')
            idx=np.flatnonzero((frame.asof_date.dt.year==year).to_numpy());path=ROOT/'fitted'/name/f'{year}.pt'
            began=time.perf_counter()
            if replay:a=EPSOnlyAdapter.load(path)
            else:
                eligible=((data['availability']<cutoff.value)&torch.isfinite(data['y'])).any(dim=(1,2)).cpu().numpy()
                train=np.flatnonzero((frame.asof_date<cutoff).to_numpy()&eligible)
                a=EPSOnlyAdapter(name).fit_arrays(data,train,cutoff.value,300);a.save(path)
            p=a.path(data,idx);probe(a,data,idx[:8])
            if not np.isfinite(p).all():raise RuntimeError('Nonfinite channel ablation')
            for target,j,c in [(f'h{j+1}',j,0) for j in range(4)]+[('ttm',3,1)]:
                if replay:
                    expected=stored[(stored.target_key==target)&(stored.asof_date.dt.year==year)].set_index('sample_id').loc[frame.iloc[idx].sample_id]
                    np.testing.assert_array_equal(p[:,j,c].astype(float),expected.predicted_eps.to_numpy())
                    replays.append({'name':name,'year':year,'target':target,'rows':len(idx),'status':'EXACT_SAVED_REPLAY_PASS'})
                else:
                    result=prediction_frame(frame.iloc[idx],target,a.model_id,p[:,j,c],time.perf_counter()-began,metadata=a.get_metadata())
                    if target=='ttm':result['ttm_prediction_method']='DIRECT_NATIVE_TTM_MULTIVARIATE_CHANNEL'
                    result['post_freeze_diagnostic']=True;result['formal_certified']=False;out.append(result)
            if not replay:
                audits.append({'name':name,'year':year,'status':'FIT_AND_INPUT_PROBES_PASS','seconds':time.perf_counter()-began,
                    'artifact':path.relative_to(RUN).as_posix(),'sha256':sha(path)})
                save_json(ROOT/'FIT_PROGRESS.json',audits)
            print('EPS_ONLY_CHANNELS',name,year,'REPLAY' if replay else 'FIT',flush=True)
            del a;gc.collect();torch.cuda.empty_cache()
        if not replay:
            result=pd.concat(out,ignore_index=True);assert len(result)==10505
            result.to_parquet(output,index=False)
            for (target,split),g in result.groupby(['target_key','split']):
                if split=='VALIDATION_OOF':g=g[g.label_asof<pd.Timestamp('2022-01-01',tz='UTC')]
                scores.append({'base_name':name,'target':target,'split':split,**metric_row(g)})
    assert plan['frozen_main_hashes']==frozen_hashes()
    if replay:save_json(ROOT/'REPLAY.json',{'status':'PASS','checks':replays,'fresh_process':True,'main_unchanged':True})
    else:
        pd.DataFrame(scores).to_csv(ROOT/'SCORES.csv',index=False)
        save_json(ROOT/'COMPLETION.json',{'status':'TWO_FIXED_CHANNEL_ABLATIONS_TRAINED','trained_annual_models':16,
            'prediction_files':2,'main_unchanged':True,'new_portfolio_members':0,'formal_certified':False,
            'whole_data_quality_certified':False,'removed_known_NI_revenue_feature_channel_consumption':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');run(parser.parse_args().replay)
