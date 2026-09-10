"""Post-freeze source-quality diagnostic; no revision of the scored portfolio.

Three fixed representative tabular recipes omit the demonstrably unreliable raw
share-count feature. This is a diagnostic ablation, not test-tuned promotion.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import time
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.core_models import TabularAdapter
from research.eps_model_lab_v1.common import dataset, split_train, prediction_frame, metric_row, atomic_csv

BASES = ['ridge_raw', 'hist_gradient_boosting', 'catboost_mae']
DIRECTORY = RUN/'share_unit_diagnostics'


class NoShareFeatureAdapter(TabularAdapter):
    def __init__(self, base):
        if base not in BASES: raise ValueError(base)
        super().__init__(base)
        self.base_id = base; self.model_id = base+'_no_raw_shares_DIAGNOSTIC'
        self.metadata.update(removed_feature='account_diluted_shares',
            evidence_class='POST_FREEZE_SOURCE_QUALITY_ABLATION_NOT_PORTFOLIO_CANDIDATE', deployable=False)
    def prepare_data(self, frame):
        return super().prepare_data(frame).drop(columns=['account_diluted_shares'])


def train(task):
    base, year, target = task; frame = dataset(); past, valid = split_train(frame, year, target)
    adapter = importlib.import_module('research.eps_model_lab_v1.share_unit_diagnostic').NoShareFeatureAdapter(base)
    start = time.perf_counter(); adapter.fit(past, target); values = adapter.predict(valid, target)
    if not np.isfinite(values).all(): raise RuntimeError('Nonfinite diagnostic')
    changed = valid.copy(); changed['account_diluted_shares'] = -1e30
    for c in changed:
        if c.startswith(('y_', 'label_')): changed[c] = 1e30
    np.testing.assert_array_equal(values, adapter.predict(changed, target))
    path = DIRECTORY/'fitted'/base/f'{year}_{target}.pkl'; adapter.save(path)
    output = prediction_frame(valid, target, adapter.model_id, values, metadata=adapter.get_metadata())
    return output, {'base': base, 'year': year, 'target': target, 'train_rows': len(past),
        'max_train_label': str(past['label_asof_'+target].max()), 'predict_rows': len(valid),
        'seconds': time.perf_counter()-start, 'mutated_share_and_future_fields': 'EXACT_PASS',
        'artifact': str(path.relative_to(RUN)), 'sha256': sha(path)}


def replay(task):
    base, year, target = task; frame = dataset(); _, valid = split_train(frame, year, target)
    path = DIRECTORY/'fitted'/base/f'{year}_{target}.pkl'
    adapter = NoShareFeatureAdapter.load(path); values = adapter.predict(valid, target)
    saved = pd.read_parquet(DIRECTORY/f'{base}.parquet')
    reference = saved[(saved.target_key == target) & (saved.asof_date.dt.year == year)].set_index('sample_id').loc[valid.sample_id].predicted_eps.to_numpy()
    np.testing.assert_array_equal(values, reference)
    return {'base': base, 'year': year, 'target': target, 'rows': len(valid), 'status': 'EXACT_PASS', 'fresh_process_pid': os.getpid()}


def run():
    DIRECTORY.mkdir(exist_ok=False)
    protected = [RUN/'ENSEMBLE_FREEZE_V1.json', RUN/'EPS_RESEARCH_SHORTLIST_V1.json', RUN/'data/samples.parquet',
                 *sorted((RUN/'predictions').glob('*.parquet'))]
    hashes = {str(p.relative_to(RUN)): sha(p) for p in protected}
    tasks = [(b, y, t) for b in BASES for y in range(2019, 2027) for t in ['h1', 'h2', 'h3', 'h4', 'ttm']]
    save_json(DIRECTORY/'PLAN.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'base_ids': BASES,
        'change': 'Remove only account_diluted_shares from common wide inputs; all estimator and train-cutoff settings unchanged',
        'reason': 'Official MCD statements label shares in millions while frozen companyfacts shares field is 732.3/721.9',
        'upstream_urls': ['https://www.sec.gov/Archives/edgar/data/63908/000006390824000072/mcd-20231231.htm',
                          'https://www.sec.gov/Archives/edgar/data/63908/000006390825000012/mcd-20241231.htm'],
        'unit_correction_factors_applied': False, 'post_test_inspection_diagnostic': True,
        'no_promotion_or_ensemble_reselection': True, 'new_independent_architectures': 0,
        'expected_heads': len(tasks), 'workers': 20, 'original_artifact_hashes': hashes, 'source_sha256': sha(__file__)})
    outputs = {b: [] for b in BASES}; trained = []
    with ProcessPoolExecutor(max_workers=20) as pool:
        for future in as_completed([pool.submit(train, task) for task in tasks]):
            prediction, record = future.result(); trained.append(record); outputs[record['base']].append(prediction)
            save_json(DIRECTORY/'TRAIN_PROGRESS.json', {'completed': len(trained), 'expected': len(tasks)})
    for base, frames in outputs.items(): pd.concat(frames, ignore_index=True).to_parquet(DIRECTORY/f'{base}.parquet', index=False)
    save_json(DIRECTORY/'TRAINED_ARTIFACT_INDEX.json', {'records': trained})
    replays = []
    with ProcessPoolExecutor(max_workers=20) as pool:
        for future in as_completed([pool.submit(replay, task) for task in tasks]): replays.append(future.result())
    rows = []
    for base in BASES:
        p = pd.read_parquet(DIRECTORY/f'{base}.parquet')
        old = pd.read_parquet(RUN/'predictions'/f'{base}.parquet')
        for target in ['h1', 'h2', 'h3', 'h4', 'ttm']:
            for surface in ['SELECTION_OOF', 'RESEARCH_TEST']:
                q = p[p.target_key == target]
                if surface == 'SELECTION_OOF': q = q[(q.split == 'VALIDATION_OOF') & (q.label_asof < pd.Timestamp('2022-01-01', tz='UTC'))]
                else: q = q[q.split == 'RESEARCH_TEST']
                aligned = old[old.target_key == target].set_index('sample_id').loc[q.sample_id]
                current, prior = metric_row(q), metric_row(aligned)
                rows.append({'base': base, 'target': target, 'surface': surface, 'rows': current['predicted_rows'],
                    'diagnostic_MAE': current.get('MAE'), 'original_MAE': prior.get('MAE'),
                    'diagnostic_MedianAE': current.get('MedianAE'), 'original_MedianAE': prior.get('MedianAE'),
                    'diagnostic_price_scaled_MAE': current.get('price_scaled_MAE'), 'original_price_scaled_MAE': prior.get('price_scaled_MAE'),
                    'used_for_selection': False, 'formal_certified': False})
    atomic_csv(pd.DataFrame(rows), DIRECTORY/'SCORES_DIAGNOSTIC_ONLY.csv')
    unchanged = all(sha(RUN/p) == digest for p, digest in hashes.items())
    save_json(DIRECTORY/'COMPLETION.json', {'status': 'COMPLETE_DIAGNOSTIC_ONLY', 'trained_heads': len(trained),
        'fresh_replay_heads': len(replays), 'replays': replays, 'all_original_artifacts_unchanged': unchanged,
        'new_portfolio_members': 0, 'completed_utc': datetime.now(timezone.utc).isoformat()})
    if not unchanged: raise RuntimeError('Original scientific artifact changed')
    print('SHARE_FEATURE_DIAGNOSTIC_COMPLETE', len(trained), len(replays), 'original artifacts unchanged', flush=True)


if __name__ == '__main__': run()
