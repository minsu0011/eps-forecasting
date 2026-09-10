"""Fresh checkpoint load, full saved prediction/quantile replay, other-origin probe."""
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
if 'tirex2' in sys.argv:os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['TORCHDYNAMO_DISABLE']='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset

GROUPS={
 'primary_gpu':['chronos2_zero_shot','chronos2_past_covariates','chronos_bolt_mini','timesfm_2p5','tirex_zero_shot'],
 'toto':['toto2_4m','toto2_22m'],
 'moirai':['moirai2_small','moirai1p1_small','moirai_moe_small'],
 'lag':['lag_llama_zero_shot'],'tirex2':['tirex2','tirex2_covariates'],
 'granite':['ibm_patchtst_fm','time_moe_50m']}


def adapter(group,path):
    if group in ['primary_gpu','toto']:
        from research.eps_model_lab_v1.foundation_models import FoundationAdapter
        return FoundationAdapter.load(path)
    if group=='moirai':
        from research.eps_model_lab_v1.moirai_models import MoiraiAdapter
        return MoiraiAdapter.load(path)
    if group=='lag':
        from research.eps_model_lab_v1.lag_llama_model import LagAdapter
        return LagAdapter.load(path)
    if group=='tirex2':
        from research.eps_model_lab_v1.tirex2_models import Tirex2Adapter
        return Tirex2Adapter.load(path)
    if path.stem=='time_moe_50m':
        from research.eps_model_lab_v1.additional_foundation import TimeMoeAdapter
        return TimeMoeAdapter.load(path)
    from research.eps_model_lab_v1.ibm_patchtst_model import IBMAdapter
    return IBMAdapter.load(path)


def forecast(a,frame,lane):
    result=a.forecast(frame,lane)
    return result if isinstance(result,tuple) else (result,None)


def run(group):
    torch.set_num_threads(8 if group in ['moirai','lag'] else 4);torch.manual_seed(1729)
    frame=dataset();valid=frame[frame.asof_date.dt.year>=2019].reset_index(drop=True);records=[]
    for name in GROUPS[group]:
        start=time.perf_counter();record={'model_id':name,'status':'RUNNING'}
        try:
            a=adapter(group,RUN/'foundation_adapters'/f'{name}.json');a.get_metadata()
            saved=pd.read_parquet(RUN/'predictions'/f'{name}.parquet');exact=True;maximum=0.;lanes=[]
            for lane in ['quarter','ttm']:
                point,quantile=forecast(a,valid,lane)
                for target in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
                    j=3 if target=='ttm' else int(target[1:])-1
                    expected=saved[saved.target_key==target].set_index('sample_id').loc[valid.sample_id]
                    np.testing.assert_allclose(point[:,j],expected.predicted_eps,rtol=2e-5,atol=2e-5)
                    if quantile is not None:np.testing.assert_allclose(quantile[:,j,:],expected[['p10_eps','p50_eps','p90_eps']],rtol=2e-5,atol=2e-5)
                    exact &= np.array_equal(point[:,j],expected.predicted_eps)
                    maximum=max(maximum,float(np.max(np.abs(point[:,j]-expected.predicted_eps.to_numpy()))))
                # Same batch geometry and RNG seed, OTHER rows' history changes.
                probe=valid.iloc[:8].copy();changed=probe.copy()
                for c in changed:
                    if c.startswith(('eps_lag_','filled_lag_','ttm_lag_')):changed.loc[changed.index[1:],c]=changed.loc[changed.index[1:],c]*3+7
                before=forecast(a,probe,lane)[0][0];after=forecast(a,changed,lane)[0][0]
                mutation_exact=np.array_equal(before,after)
                mutation_close=bool(np.allclose(before,after,rtol=2e-5,atol=2e-5))
                lanes.append({'lane':lane,'other_origin_mutation_close':mutation_close,'mutation_exact':mutation_exact,
                  'maximum_mutation_difference':float(np.max(np.abs(before-after))),
                  'probe_scope':'Stochastic samplers require separate RNG-coupling review if this fails; do not automatically equate Monte Carlo draw changes with conditional-distribution leakage'})
                print('FOUNDATION_REPLAY',name,lane,'FULL_SAVED_ROWS_PASS','MUTATION',mutation_close,flush=True)
            record.update(status='PASS_FULL_POINT_AND_QUANTILE_RELOAD',exact_points=bool(exact),maximum_difference=maximum,
                rows_per_target=len(valid),lanes=lanes,seconds=time.perf_counter()-start,
                prediction_sha256=sha(RUN/'predictions'/f'{name}.parquet'))
            del a
        except Exception as exc:record.update(status='FAIL',reason=str(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start)
        records.append(record);save_json(RUN/'foundation_replay_audits'/f'{group}.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
          'dataset_sha256':sha(RUN/'data/samples.parquet'),'records':records,'expected':len(GROUPS[group]),'completed':len(records),
          'all_reload_pass':all(r['status'].startswith('PASS') for r in records),'rtol':2e-5,'atol':2e-5})
        print('FOUNDATION_REPLAY_FINISHED',name,record['status'],flush=True)
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def replay_chronos_finetuned():
    from research.eps_model_lab_v1.chronos_finetune import ChronosFineTunedAdapter,forecast as ft_forecast
    model='chronos2_eps_joint_finetuned';frame=dataset();saved=pd.read_parquet(RUN/'predictions'/f'{model}.parquet');records=[]
    for year in range(2019,2027):
        try:
            val=frame[frame.asof_date.dt.year==year];path=RUN/'chronos_finetuned'/str(year)/'final'
            a=ChronosFineTunedAdapter.load(path);a.get_metadata();maximum=0.;exact=True
            for lane in ['quarter','ttm']:
                p,q=ft_forecast(a.pipeline,val,lane)
                for target in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
                    j=3 if target=='ttm' else int(target[1:])-1
                    expected=saved[saved.target_key==target].set_index('sample_id').loc[val.sample_id]
                    np.testing.assert_allclose(p[:,j],expected.predicted_eps,rtol=2e-5,atol=2e-5)
                    np.testing.assert_allclose(q[:,j,:],expected[['p10_eps','p50_eps','p90_eps']],rtol=2e-5,atol=2e-5)
                    exact &= np.array_equal(p[:,j],expected.predicted_eps)
                    maximum=max(maximum,float(np.max(np.abs(p[:,j]-expected.predicted_eps.to_numpy()))))
            probe=val.iloc[:8];p=a.predict(probe,'h1');changed=probe.copy()
            for c in changed:
                if c.startswith(('eps_lag_','ttm_lag_')):changed.loc[changed.index[1:],c]=changed.loc[changed.index[1:],c]*3+7
            np.testing.assert_allclose(p[0],a.predict(changed,'h1')[0],rtol=2e-5,atol=2e-5)
            api_save='NOT_REPEATED_EVERY_FOLD'
            if year==2019:
                export=RUN/'adapter_export_smoke/chronos_finetuned_2019'
                a.save(export);restored=ChronosFineTunedAdapter.load(export)
                np.testing.assert_array_equal(p,restored.predict(probe,'h1'));del restored
                api_save='ACTUAL_SAVE_NEW_DIRECTORY_AND_LOAD_EXACT_PASS'
            records.append({'year':year,'status':'PASS','exact_points':bool(exact),'maximum_difference':maximum,
                 'other_origin_mutation':'PASS','common_adapter_API':'PASS','save_export_probe':api_save,'rows_per_target':len(val)})
            del a
        except Exception as exc:records.append({'year':year,'status':'FAIL','reason':str(exc),'traceback':traceback.format_exc()})
        save_json(RUN/'foundation_replay_audits/chronos_finetuned.json',{'model_id':model,'expected':8,'completed':len(records),
           'all_pass':all(r['status']=='PASS' for r in records),'records':records,'rtol':2e-5,'atol':2e-5})
        print('CHRONOS_FT_REPLAY',year,records[-1]['status'],flush=True);gc.collect();torch.cuda.empty_cache()


def replay_moment_heads():
    from research.eps_model_lab_v1.additional_foundation import MomentEmbeddingAdapter
    frame=dataset();saved=pd.read_parquet(RUN/'predictions/moment_embedding_ridge.parquet');records=[]
    for path in sorted((RUN/'moment_fitted').glob('*.pkl')):
        year,target=path.stem.split('_',1)
        try:
            val=frame[frame.asof_date.dt.year==int(year)];a=MomentEmbeddingAdapter.load(path);a.get_metadata()
            point=a.predict(val,target);expected=saved[saved.target_key==target].set_index('sample_id').loc[val.sample_id].predicted_eps
            np.testing.assert_allclose(point,expected,rtol=2e-5,atol=2e-5)
            probe=val.iloc[:8];p=a.predict(probe,target);changed=probe.copy()
            for c in changed:
                if c.startswith(('filled_lag_','eps_lag_')):changed.loc[changed.index[1:],c]=changed.loc[changed.index[1:],c]*3+7
            np.testing.assert_allclose(p[0],a.predict(changed,target)[0],rtol=2e-5,atol=2e-5)
            records.append({'year':int(year),'target':target,'status':'PASS','maximum_difference':float(np.max(np.abs(point-expected))),
              'exact_points':bool(np.array_equal(point,expected)),'other_origin_mutation':'PASS','head_sha256':sha(path)})
            del a
        except Exception as exc:records.append({'year':int(year),'target':target,'status':'FAIL','reason':str(exc),'traceback':traceback.format_exc()})
        save_json(RUN/'foundation_replay_audits/moment_heads.json',{'expected':40,'completed':len(records),
           'all_pass':all(r['status']=='PASS' for r in records),'records':records,'rtol':2e-5,'atol':2e-5})
        print('MOMENT_HEAD_REPLAY',year,target,records[-1]['status'],flush=True);gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    run(sys.argv[1])
    if sys.argv[1]=='primary_gpu':replay_chronos_finetuned()
    if sys.argv[1]=='granite':replay_moment_heads()
