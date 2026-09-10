"""OOF-only Monte Carlo sensitivity; never a new candidate or retuned recipe."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import dataset, atomic_csv

SEEDS = [13, 42, 2027, 2718]


def native(a, one, lane, seed):
    torch.manual_seed(seed)
    if a.model_id.startswith('lag'):
        predictor = a.predictor; data = a.prepare_data(one, lane)
    else:
        y = a.prepare_data(one, lane)
        kwargs = {'prediction_length': 4, 'context_length': y.shape[1], 'target_dim': 1,
                  'feat_dynamic_real_dim': 0, 'past_feat_dynamic_real_dim': 0, 'module': a.module,
                  'patch_size': 8 if a.model_id == 'moirai1p1_small' else 16, 'num_samples': 100}
        predictor = a.klass(**kwargs).eval().create_predictor(batch_size=32, device='cpu')
        data = ListDataset([{'item_id': '0', 'start': pd.Period('2000Q1', freq='Q'), 'target': y[0]}], freq='Q')
    with torch.inference_mode(): forecasts = list(predictor.predict(data))
    if len(forecasts) != 1: raise RuntimeError('One origin per stochastic call required')
    return np.asarray([forecasts[0].quantile(t) for t in [.1, .5, .9]]).T


def run(base):
    torch.set_num_threads(4)
    if base.startswith('lag'):
        from research.eps_model_lab_v1.lag_llama_model import LagAdapter
        a = LagAdapter().fit()
    else:
        from research.eps_model_lab_v1.moirai_models import MoiraiAdapter
        a = MoiraiAdapter(base).fit()
    name = base+'_origin_isolated'; out = RUN/'sampling_stability'/name
    out.mkdir(parents=True, exist_ok=False)
    frame = dataset(); cutoff = pd.Timestamp('2022-01-01', tz='UTC')
    plans = {}; records = []; details = []; started = time.perf_counter()
    for target in ['h1', 'ttm']:
        probe = frame[(frame.asof_date >= pd.Timestamp('2019-01-01', tz='UTC')) & (frame.asof_date < cutoff)
                      & (frame['label_asof_'+target] < cutoff) & frame['y_'+target].notna()]
        plans[target] = probe.sample_id.tolist()
    save_json(out/'PRESCORE_DIAGNOSTIC_PLAN.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'seeds': SEEDS, 'original_seed': 1729, 'samples': plans, 'dataset_sha256': sha(RUN/'data/samples.parquet'),
              'source_sha256': sha(__file__), 'changes_candidate_or_selection': False,
              'native_sample_count_unchanged': 100, 'scope': 'OOF Monte Carlo sampling sensitivity, not epistemic or fresh-PIT certification'})
    for target in ['h1', 'ttm']:
        probe = frame.set_index('sample_id', drop=False).loc[plans[target]]
        lane = 'quarter' if target == 'h1' else 'ttm'; h = 0 if target == 'h1' else 3
        # The diagnostic direct call must exactly reproduce the production
        # corrective wrapper before testing different random draws.
        for i in range(4):
            one = probe.iloc[[i]]
            _, q = a.forecast(one, lane)
            np.testing.assert_array_equal(q[0], native(a, one, lane, 1729))
        for seed in SEEDS:
            ps = []; qs = []; began = time.perf_counter()
            for i in range(len(probe)):
                q = native(a, probe.iloc[[i]], lane, seed)[h]
                ps.append(float(q[1])); qs.append(q)
                if (i+1) % 100 == 0:
                    save_json(out/'PROGRESS.json', {'target': target, 'seed': seed, 'completed': i+1, 'total': len(probe),
                              'updated_utc': datetime.now(timezone.utc).isoformat()})
                    print('SAMPLING', base, target, seed, i+1, len(probe), flush=True)
            p = np.asarray(ps); q = np.asarray(qs); y = probe['y_'+target].to_numpy()
            if not np.isfinite(p).all() or not np.isfinite(q).all(): raise RuntimeError('Nonfinite native draw')
            error = np.abs(p-y)
            records.append({'model_id': name, 'target_key': target, 'seed': seed, 'rows': len(probe),
                            'MAE': float(error.mean()), 'MedianAE': float(np.median(error)),
                            'price_scaled_MAE': float(np.nanmean(error/probe.origin_price.to_numpy())),
                            'interval80_coverage': float(((y >= q[:, 0]) & (y <= q[:, 2])).mean()),
                            'seconds': time.perf_counter()-began, 'surface': 'SELECTION_OOF_AVAILABLE_BEFORE_2022'})
            details.append(pd.DataFrame({'sample_id': probe.sample_id.to_numpy(), 'target_key': target, 'seed': seed,
                                         'predicted_eps': p, 'p10_eps': q[:, 0], 'p50_eps': q[:, 1], 'p90_eps': q[:, 2]}))
            atomic_csv(pd.DataFrame(records), out/'SEED_METRICS.csv')
    detail = pd.concat(details, ignore_index=True); detail.to_parquet(out/'NATIVE_SEED_DRAWS.parquet', index=False)
    save_json(out/'COMPLETION.json', {'status': 'PASS_DIAGNOSTIC_COMPLETE', 'metric_rows': len(records),
              'native_draw_rows': len(detail), 'seconds': time.perf_counter()-started,
              'production_seed_equivalence': 'EXACT_PASS_4_ORIGINS_PER_LANE', 'original_predictions_changed': False,
              'ensemble_members_or_weights_changed': False, 'new_independent_candidates': 0,
              'detail_sha256': sha(out/'NATIVE_SEED_DRAWS.parquet')})
    print('SAMPLING_STABILITY_COMPLETE', name, len(records), flush=True)


if __name__ == '__main__': run(sys.argv[1])
