"""Availability-purged four-direct-quarter path diagnostics, distinct from native TTM."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import dataset, atomic_csv


def run():
    source = dataset(); source = source[source.asof_date.dt.year.ge(2019)].set_index('sample_id')
    targets = ['h1', 'h2', 'h3', 'h4']; truth = source[['y_'+t for t in targets]].to_numpy()
    finite = np.isfinite(truth).all(axis=1); labels = source[['label_asof_'+t for t in targets]]
    cutoff = pd.Timestamp('2022-01-01', tz='UTC')
    masks = {'SELECTION_OOF_ALL_FOUR_LABELS_AVAILABLE_BEFORE_2022': source.asof_date.lt(cutoff).to_numpy() & labels.lt(cutoff).all(axis=1).to_numpy(),
             'RESEARCH_TEST_DESCRIPTIVE_ONLY': source.asof_date.ge(cutoff).to_numpy()}
    records = []; counts = {}
    for surface, mask in masks.items():
        complete = mask & finite
        counts[surface] = {'forecast_origins': int(mask.sum()), 'complete_native_quarter_truth_paths': int(complete.sum()),
            'tickers': int(source.loc[complete, 'ticker'].nunique()),
            'origin_year_counts': {str(k): int(v) for k, v in source.loc[complete].asof_date.dt.year.value_counts().sort_index().items()}}
    for path in sorted((RUN/'predictions').glob('*.parquet')):
        pred = pd.read_parquet(path); pred = pred[pred.target_key.isin(targets)]
        if pred.target_key.nunique() != 4: continue
        receipt = json.loads((RUN/'model_receipts'/f'{path.stem}.json').read_text(encoding='utf-8'))
        x = pred.pivot(index='sample_id', columns='target_key', values='predicted_eps').reindex(index=source.index, columns=targets).to_numpy()
        for surface, mask in masks.items():
            complete = mask & finite; valid = complete & np.isfinite(x).all(axis=1)
            error = x[valid]-truth[valid]
            records.append({'model_id': path.stem, 'surface': surface, 'complete_truth_paths': int(complete.sum()),
                'complete_predicted_paths': int(valid.sum()), 'path_coverage': float(valid.sum()/complete.sum()) if complete.any() else 0.,
                'path_mean_MAE': float(np.abs(error).mean()) if valid.any() else None,
                'four_direct_quarter_sum_MAE_NOT_NATIVE_TTM': float(np.abs(error.sum(axis=1)).mean()) if valid.any() else None,
                'model_status': receipt['status'], 'portfolio_eligible': receipt['status'] == 'FULL_RESEARCH_SCORED',
                'used_for_model_selection': False, 'formal_certified': False})
    atomic_csv(pd.DataFrame(records), RUN/'EPS_LANE_B_PURGED_PATH_REVIEW.csv')
    save_json(RUN/'EPS_LANE_B_SCOPE_RECEIPT.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'surface_counts': counts,
        'score_rows': len(records), 'samples_sha256': sha(RUN/'data/samples.parquet'),
        'no_quarter_truth_imputation': True, 'no_native_TTM_equivalence_claim': True,
        'frozen_portfolio_changed': False, 'no_B_winner_selected_from_sparse_research_test': True})
    print('LANE_B_SCOPE_REVIEW', counts, flush=True)


if __name__ == '__main__': run()
