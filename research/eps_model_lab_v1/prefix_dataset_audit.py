"""Counterfactual truncated SEC snapshots: future records must not change past inputs."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset,features
from research.eps_model_lab_v1.dataset import one


def task(ticker):
    start=time.perf_counter();original=json.loads((RUN/'data/raw/sec'/ticker/'companyfacts.json').read_text(encoding='utf-8'))
    results=[]
    for year in [2019,2022,2024,2026]:
        cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC');prefix=copy.deepcopy(original);removed=0
        for ns in prefix['facts'].values():
            for tag in ns.values():
                for unit,values in tag.get('units',{}).items():
                    past=[v for v in values if v.get('filed','')<f'{year}-01-01']
                    removed+=len(values)-len(past);tag['units'][unit]=past
        try:
            rebuilt,_,_=one(ticker,prefix)
            frozen=dataset();frozen=frozen[(frozen.ticker==ticker)&(frozen.asof_date<cutoff)].sort_values('sample_id').reset_index(drop=True)
            if rebuilt.empty and frozen.empty:
                results.append({'ticker':ticker,'cutoff':str(cutoff),'status':'NO_PRE_CUTOFF_ORIGINS','removed_future_facts':removed});continue
            rebuilt=rebuilt[rebuilt.asof_date<cutoff].sort_values('sample_id').reset_index(drop=True)
            if not len(frozen) and not len(rebuilt):
                results.append({'ticker':ticker,'cutoff':str(cutoff),'status':'NO_PRE_CUTOFF_ORIGINS','removed_future_facts':removed});continue
            pd.testing.assert_series_equal(frozen.sample_id,rebuilt.sample_id,check_dtype=False)
            pd.testing.assert_frame_equal(features(frozen),features(rebuilt),check_exact=True,check_dtype=False)
            for column in ['asof_date','current_ttm','last_observed_eps','origin_price']:
                pd.testing.assert_series_equal(frozen[column],rebuilt[column],check_exact=True,check_dtype=False)
            results.append({'ticker':ticker,'cutoff':str(cutoff),'status':'PASS_EXACT','past_origins':len(frozen),'removed_future_facts':removed})
        except Exception as exc:
            # A company not yet public can legitimately have zero native records.
            frozen=dataset();count=int(((frozen.ticker==ticker)&(frozen.asof_date<cutoff)).sum())
            if count==0 and str(exc)=='No native GAAP diluted EPS fiscal records':
                results.append({'ticker':ticker,'cutoff':str(cutoff),'status':'NO_PRE_CUTOFF_ORIGINS','removed_future_facts':removed})
            else: results.append({'ticker':ticker,'cutoff':str(cutoff),'status':'FAIL','reason':str(exc),'traceback':traceback.format_exc()})
    return {'ticker':ticker,'seconds':time.perf_counter()-start,'tests':results}


if __name__=='__main__':
    tickers=sorted(dataset().ticker.unique());records=[];before=sha(RUN/'data/samples.parquet')
    with ProcessPoolExecutor(max_workers=12) as pool:
        jobs={pool.submit(task,t):t for t in tickers}
        for future in as_completed(jobs):
            try: records.append(future.result())
            except Exception as exc: records.append({'ticker':jobs[future],'tests':[{'status':'FAIL','reason':str(exc)}]})
            tests=[test for r in records for test in r['tests']]
            save_json(RUN/'SEC_FUTURE_SNAPSHOT_TRUNCATION_AUDIT.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
              'expected_tickers':len(tickers),'completed_tickers':len(records),'all_pass':all(r['status']!='FAIL' for r in tests),
              'records':records,'samples_sha256_before':before,'samples_sha256_after':sha(RUN/'data/samples.parquet'),
              'scope':'Original data object with every fact filed on/after cutoff removed; regenerated only in memory, frozen dataset never overwritten'})
            print('PREFIX_DATASET_AUDIT',len(records),'OF',len(tickers),'FAIL',sum(r['status']=='FAIL' for r in tests),flush=True)
