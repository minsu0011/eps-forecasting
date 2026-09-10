"""Independent NumPy recomputation of primary saved scores, without common.metric_row."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha


def independent(frame, preserve_native_width_reduction=False):
    truth = np.isfinite(frame.actual_eps.to_numpy(dtype=float))
    valid = truth & frame.prediction_valid.to_numpy(dtype=bool) & np.isfinite(frame.predicted_eps.to_numpy(dtype=float))
    y = frame.actual_eps.to_numpy(dtype=float)[valid]; p = frame.predicted_eps.to_numpy(dtype=float)[valid]
    ae = np.abs(p-y)
    out = {'truth_rows': int(truth.sum()), 'predicted_rows': int(valid.sum()),
        'coverage': float(valid.sum()/truth.sum()) if truth.any() else 0.}
    if not len(y): return out
    price = frame.origin_price.to_numpy(dtype=float)[valid]; price_ok = np.isfinite(price) & (price > 0)
    out.update(MAE=float(ae.mean()), RMSE=float(np.sqrt(np.square(p-y).mean())), MedianAE=float(np.median(ae)),
        bias=float((p-y).mean()), price_scaled_MAE=float((ae[price_ok]/price[price_ok]).mean()) if price_ok.any() else np.nan)
    native_q = frame[['p10_eps', 'p50_eps', 'p90_eps']].to_numpy()[valid]
    q = native_q.astype(float)
    qok = np.isfinite(q).all(axis=1); q = q[qok]; qy = y[qok]
    if len(q):
        errors = qy[:, None]-q; taus = np.array([.1, .5, .9])
        out.update(quantile_rows=len(q), pinball_loss=float(np.maximum(taus*errors, (taus-1)*errors).mean()),
            interval80_coverage=float(((qy >= q[:, 0]) & (qy <= q[:, 2])).mean()),
            interval80_width=float((native_q[qok, 2]-native_q[qok, 0]).mean()) if preserve_native_width_reduction else float((q[:, 2]-q[:, 0]).mean()),
            quantile_crossing_rows=int(((q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2])).sum()))
        for j, name in enumerate(['p10_eps', 'p50_eps', 'p90_eps']): out[name+'_empirical_cdf'] = float((qy <= q[:, j]).mean())
    return out


def run():
    prior = RUN/'INDEPENDENT_NUMPY_METRIC_AUDIT.json'
    if prior.exists() and not json.loads(prior.read_text(encoding='utf-8'))['all_pass']:
        archive = RUN/'audit_corrections/independent_width_dtype_initial_audit.json'
        if not archive.exists():
            archive.parent.mkdir(parents=True, exist_ok=True); archive.write_bytes(prior.read_bytes())
    files = ['EPS_MODEL_ZOO_LEADERBOARD_V1.csv', 'EPS_SELECTION_LEADERBOARD_V1.csv', 'EPS_PROBABILISTIC_LEADERBOARD_V1.csv']
    tables = {name: pd.read_csv(RUN/name) for name in files}; records = []; failures = []
    for path in sorted((RUN/'predictions').glob('*.parquet')):
        pred = pd.read_parquet(path)
        for filename, table in tables.items():
            selected = table[(table.eps_model_id == path.stem) & (table['mask'] == 'COMPLETE_COVERAGE')]
            for row in selected.to_dict('records'):
                q = pred[pred.target_key == row['target_key']]
                if filename == 'EPS_SELECTION_LEADERBOARD_V1.csv':
                    q = q[(q.split == 'VALIDATION_OOF') & (q.label_asof < pd.Timestamp('2022-01-01', tz='UTC'))]
                else: q = q[q.split == row['split']]
                native_width = filename == 'EPS_SELECTION_LEADERBOARD_V1.csv'
                values = independent(q, preserve_native_width_reduction=native_width); bad = []; errors = []
                for key, value in values.items():
                    if key not in row: continue
                    actual = float(row[key])
                    if not np.isclose(value, actual, rtol=1e-11, atol=1e-12, equal_nan=True): bad.append(key)
                    if np.isfinite(value) and np.isfinite(actual): errors.append(abs(value-actual))
                record = {'model': path.stem, 'table': filename, 'target': row['target_key'], 'split': row['split'],
                    'checked_metrics': len([k for k in values if k in row]), 'maximum_absolute_difference': max(errors, default=0.),
                    'pass': not bad, 'failed_metrics': bad, 'width_reduction_profile': 'Saved per-model dtype' if native_width else 'Joint-table float64 promotion'}
                records.append(record)
                if bad: failures.append(record)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'all_pass': not failures and bool(records),
        'prediction_files': len(list((RUN/'predictions').glob('*.parquet'))), 'score_records': len(records),
        'metric_comparisons': sum(r['checked_metrics'] for r in records), 'failures': failures, 'records': records,
        'score_table_hashes': {f: sha(RUN/f) for f in files}, 'source_sha256': sha(__file__),
        'scope': 'Independent NumPy formulas on COMPLETE_COVERAGE masks; not independent truth acquisition or model recertification',
        'initial_discrepancy_review': '94 OOF interval-width fields differed only because initial independent code promoted native float32 to float64. Recompute native subtract/mean exactly; original score bytes and audit failures preserved, tolerance unchanged.',
        'formal_certified': False, 'predictions_or_scores_changed': False}
    save_json(RUN/'INDEPENDENT_NUMPY_METRIC_AUDIT.json', result)
    print('INDEPENDENT_METRIC_AUDIT', result['all_pass'], len(records), 'rows', result['metric_comparisons'], 'metrics', flush=True)
    if not result['all_pass']: raise RuntimeError('Independent score discrepancy')


if __name__ == '__main__': run()
