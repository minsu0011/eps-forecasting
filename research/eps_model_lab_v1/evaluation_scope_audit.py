"""Make target maturity, missing direct Q4s and inference coverage explicit."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset,atomic_csv,TARGETS


def run():
    f=dataset();f=f[f.asof_date.dt.year>=2019];rows=[];inference=[]
    for year,g in f.groupby(f.asof_date.dt.year):
        for target in TARGETS:
            truth=g['y_'+target].notna();label=g['label_asof_'+target]
            rows.append({'origin_year':year,'target':target,'forecast_origins':len(g),'observed_native_truths':int(truth.sum()),
              'truth_observable_fraction':float(truth.mean()),'future_filing_present_but_direct_truth_missing':int((~truth&label.notna()).sum()),
              'no_future_label_in_snapshot':int(label.isna().sum()),'latest_observed_label_available':str(label[truth].max()),
              'negative_truths':int((g['y_'+target]<0).sum()),'zero_truths':int(g['y_'+target].eq(0).sum()),
              'approximate_TTM_truths':int((truth&g.ttm_target_approximate).sum()) if target=='ttm' else None})
    for path in sorted((RUN/'predictions').glob('*.parquet')):
        p=pd.read_parquet(path)
        for (target,year),g in p.groupby(['target_key',p.asof_date.dt.year]):
            inferred=np.isfinite(g.predicted_eps);mature=np.isfinite(g.actual_eps)
            inference.append({'model_id':path.stem,'target':target,'origin_year':year,'intended_prediction_rows':len(g),
              'finite_inference_rows_including_unmatured':int(inferred.sum()),'inference_coverage':float(inferred.mean()),
              'mature_truth_rows':int(mature.sum()),'scorable_rows':int((inferred&mature).sum()),
              'score_coverage_conditional_on_truth':float((inferred&mature).sum()/mature.sum()) if mature.any() else None})
    atomic_csv(pd.DataFrame(rows),RUN/'EPS_TARGET_MATURITY_AUDIT.csv')
    atomic_csv(pd.DataFrame(inference),RUN/'EPS_INFERENCE_VS_SCORE_COVERAGE.csv')
    save_json(RUN/'EPS_EVALUATION_SCOPE_RECEIPT.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
      'dataset_sha256':sha(RUN/'data/samples.parquet'),'snapshot_end':'2026-09-07','models_seen':len({r['model_id'] for r in inference}),
      'distinction':'Score coverage conditions on observed truth; inference coverage includes immature targets. Recent-year TTM truths are right-censored, not zero EPS.',
      'model_comparison_rule':'Pair exact sample IDs per target; compare complete coverage and common masks; do not reward abstention on difficult rows',
      'reporting_rule':'Report actual scorable years/counts for each lane rather than implying every 2026 forecast has a mature Q+4 label',
      'primary_contract':'EPS_DATA_GEOMETRY_LOCK_V1_3.json','supersession_docs':['DATA_CONTRACT_PRESCORE_ADDENDUM.md','PIT_REPAIR_V1_3_ADDENDUM.md']})
    print('EVALUATION_SCOPE',len(rows),'MATURITY_ROWS',len(inference),'MODEL_YEAR_TARGET_ROWS',flush=True)


if __name__=='__main__':run()
