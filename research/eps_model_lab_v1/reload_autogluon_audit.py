"""Fresh-process AutoGluon replay plus virtual fiscal-grid resampling audit."""
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import sys
import time
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset
from research.eps_model_lab_v1.autogluon_models import AutoGluonAdapter,training_segments,contexts


if __name__=='__main__':
    torch.set_num_threads(4);frame=dataset();records=[];grid=[]
    for year in range(2019,2027):
        training,audit=training_segments(year);resampled=training.convert_frequency('QS-JAN')
        if not training.index.equals(resampled.index): raise RuntimeError('Quarterly resampling changed actual grid')
        np.testing.assert_array_equal(training.target,resampled.target)
        val=frame[frame.asof_date.dt.year==year];context=contexts(val);resampled_context=context.convert_frequency('QS-JAN')
        if not context.index.equals(resampled_context.index): raise RuntimeError('Context resampling changed fiscal phase')
        np.testing.assert_array_equal(context.target,resampled_context.target)
        grid.append({'year':year,'training_rows':len(training),'context_rows':len(context),'same_index':True,
                     'same_values':True,'new_missing_rows':0,'status':'PASS'})
        a=AutoGluonAdapter.load(RUN/'autogluon_fitted'/str(year));a.get_metadata()
        for name in a.predictor.model_names():
            start=time.perf_counter();model='ag_'+name
            try:
                prediction=a.predictor.predict(contexts(val),model=name,use_cache=False,random_seed=123).reset_index().groupby('item_id')
                point=np.array([prediction.get_group(i).sort_values('timestamp')['mean'].to_numpy() for i in val.sample_id])
                quantile=np.array([prediction.get_group(i).sort_values('timestamp')[['0.1','0.5','0.9']].to_numpy() for i in val.sample_id])
                saved=pd.read_parquet(RUN/'predictions'/f'{model}.parquet');maximum=0.;exact=True
                for target,h in [('h1',0),('h2',1),('h3',2),('h4',3),('ttm',4)]:
                    expected=saved[saved.target_key==target].set_index('sample_id').loc[val.sample_id]
                    actual=point.sum(axis=1) if target=='ttm' else point[:,h]
                    np.testing.assert_allclose(actual,expected.predicted_eps,rtol=2e-5,atol=2e-5)
                    maximum=max(maximum,float(np.max(np.abs(actual-expected.predicted_eps.to_numpy()))))
                    exact=exact and np.array_equal(actual,expected.predicted_eps)
                    if target!='ttm': np.testing.assert_allclose(quantile[:,h,:],expected[['p10_eps','p50_eps','p90_eps']],rtol=2e-5,atol=2e-5)
                records.append({'model_id':model,'year':year,'status':'PASS','exact_point_predictions':exact,
                     'maximum_point_difference':maximum,'rows_per_target':len(val),'quantile_replay':'PASS','seconds':time.perf_counter()-start})
            except Exception as exc: records.append({'model_id':model,'year':year,'status':'FAIL','reason':str(exc),'traceback':traceback.format_exc()})
            save_json(RUN/'reload_audits/autogluon_no_cache.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'completed':len(records),
              'expected':64,'all_pass':all(r['status']=='PASS' for r in records),'records':records,'rtol':2e-5,'atol':2e-5})
            print('AUTOGLUON_RELOAD',model,year,records[-1]['status'],flush=True)
        save_json(RUN/'AUTOGLUON_FISCAL_GRID_AUDIT.json',{'records':grid,'all_pass':True,
          'explanation':'QS-OCT and QS-JAN frequency labels describe the same Jan/Apr/Jul/Oct start grid here; exact timestamps and values verified, no row added or deleted'})
        del a;gc.collect();torch.cuda.empty_cache()
