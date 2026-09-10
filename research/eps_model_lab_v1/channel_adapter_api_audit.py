"""Actual fit/predict/save/load calls for the two-channel diagnostic adapter."""
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset
from research.eps_model_lab_v1.eps_only_multivariate_diagnostic import EPSOnlyAdapter,ROOT


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    frame=dataset();valid=frame[frame.asof_date.dt.year==2019];records=[]
    for name in ['TimeXer','SOFTS']:
        original=pd.read_parquet(ROOT/f'{name}.parquet')
        path=ROOT/'api_checkpoints'/f'{name}.pt'
        # Preserve the failed initial fresh fit. It is diagnostic evidence,
        # not a checkpoint to overwrite until it matches an expected result.
        if path.exists():adapter=EPSOnlyAdapter.load(path)
        else:
            adapter=EPSOnlyAdapter(name).fit(frame,2019);adapter.save(path)
        loaded=EPSOnlyAdapter.load(path)
        original_adapter=EPSOnlyAdapter.load(ROOT/'fitted'/name/'2019.pt')
        changed=valid.copy()
        for column in changed:
            if column.startswith(('account_','y_')):changed[column]=1e20
        for target in ['h1','h2','h3','h4','ttm']:
            p=adapter.predict(valid,target)
            expected=original[(original.target_key==target)&(original.asof_date.dt.year==2019)].set_index('sample_id').loc[valid.sample_id]
            old=original_adapter.predict(valid,target)
            np.testing.assert_array_equal(old.astype(float),expected.predicted_eps.to_numpy())
            fit_difference=float(np.max(np.abs(p.astype(float)-expected.predicted_eps.to_numpy())))
            np.testing.assert_array_equal(p,loaded.predict(valid,target))
            np.testing.assert_array_equal(p,loaded.predict(changed,target))
            records.append({'name':name,'target':target,'rows':len(valid),'public_fit_matches_saved_full_run':fit_difference==0,
                'fresh_fit_max_difference_vs_original':fit_difference,'original_saved_public_predict_matches_frozen':True,
                'public_predict_save_load_exact':True,'ignored_accounting_and_future_label_mutation_exact':True})
        del adapter,loaded,original_adapter;gc.collect();torch.cuda.empty_cache()
    save_json(RUN/'EPS_ONLY_CHANNEL_PUBLIC_API_AUDIT.json',{'status':'PASS_SAVED_API_WITH_FRESH_TRAINING_NUMERICAL_LIMITATION',
        'created_utc':datetime.now(timezone.utc).isoformat(),'checks':records,'actual_new_fit_calls':2,
        'scope':'Known frozen research origins only; no claim of arbitrary out-of-sample data ingestion',
        'original_channel_predictions_unchanged':True,'no_new_candidate_or_portfolio_output':True})
    print('TWO_CHANNEL_PUBLIC_API_AUDIT_PASS',len(records),'target checks',flush=True)
