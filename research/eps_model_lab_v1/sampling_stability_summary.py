"""Summarize fixed OOF seed diagnostics without creating a seed ensemble."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, sha, save_json
from research.eps_model_lab_v1.common import atomic_csv


def run():
    rows = []; bindings = {}
    for base in ['lag_llama_zero_shot', 'moirai1p1_small', 'moirai_moe_small']:
        name = base+'_origin_isolated'; root = RUN/'sampling_stability'/name
        complete = json.loads((root/'COMPLETION.json').read_text(encoding='utf-8'))
        if complete['status'] != 'PASS_DIAGNOSTIC_COMPLETE': raise RuntimeError('Seed diagnostic incomplete')
        raw = pd.read_parquet(root/'NATIVE_SEED_DRAWS.parquet')
        original = pd.read_parquet(RUN/'predictions'/f'{name}.parquet')
        for target in ['h1', 'ttm']:
            draws = raw[raw.target_key == target].pivot(index='sample_id', columns='seed', values='predicted_eps')
            p = original[original.target_key == target].set_index('sample_id').loc[draws.index]
            assert (p.asof_date < pd.Timestamp('2022-01-01', tz='UTC')).all()
            assert (p.label_asof < pd.Timestamp('2022-01-01', tz='UTC')).all()
            draws[1729] = p.predicted_eps; ae = np.abs(draws.to_numpy()-p.actual_eps.to_numpy()[:, None])
            mae = ae.mean(axis=0); std = draws.std(axis=1, ddof=1); reference_mae = float(ae[:, list(draws.columns).index(1729)].mean())
            rows.append({'model_id': name, 'target_key': target, 'rows': len(draws), 'independent_draw_seeds': len(draws.columns),
                         'original_fixed_seed_MAE': reference_mae, 'seed_MAE_min': float(mae.min()), 'seed_MAE_max': float(mae.max()),
                         'seed_MAE_span': float(np.ptp(mae)), 'relative_seed_MAE_span': float(np.ptp(mae)/reference_mae),
                         'median_per_origin_point_std_across_seeds': float(std.median()), 'p95_per_origin_point_std_across_seeds': float(std.quantile(.95)),
                         'point_sign_disagreement_fraction_across_seeds': float((np.sign(draws).nunique(axis=1) > 1).mean()),
                         'surface': 'SELECTION_OOF_AVAILABLE_BEFORE_2022', 'select_best_seed': False, 'formal_certified': False})
        bindings[name] = {'seed_draw_sha256': sha(root/'NATIVE_SEED_DRAWS.parquet'), 'original_prediction_sha256': sha(RUN/'predictions'/f'{name}.parquet')}
    atomic_csv(pd.DataFrame(rows), RUN/'EPS_NATIVE_SAMPLING_STABILITY_SUMMARY.csv')
    save_json(RUN/'SAMPLING_STABILITY_SUMMARY_RECEIPT.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'rows': len(rows), 'bindings': bindings, 'source_sha256': sha(__file__),
              'original_seed_1729_retained': True, 'no_best_seed_selection_or_new_seed_ensemble': True,
              'caveat': 'Monte Carlo repeat sensitivity is not pretrained-vintage safety, epistemic uncertainty, or a new heldout test'})
    print('SAMPLING_STABILITY_SUMMARY', len(rows), flush=True)


if __name__ == '__main__': run()
