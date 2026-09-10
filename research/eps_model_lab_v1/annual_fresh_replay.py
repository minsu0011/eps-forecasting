"""Actual annual accounting-model replay and up-to-date common annual masks."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
# Deliberately do not expose HVZAdapter as a __main__ global: exercise the
# common trusted legacy-class remapping used by a generic downstream loader.
from research.eps_model_lab_v1 import annual_accounting as annual
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import dataset, EPSAdapter, metric_row, atomic_csv


def run():
    panel = annual.annual_panel(); frame = dataset(); name = 'hvz_gaap_annual_eps_originshares'
    old = pd.read_parquet(RUN/'predictions'/f'{name}.parquet')
    complete = panel[annual.COLS].notna().all(axis=1) & panel.diluted_shares.gt(0); records = []
    for year in range(2019, 2027):
        valid = frame[frame.asof_date.dt.year == year]
        matched = valid[['sample_id', 'origin_accession']].merge(panel.loc[complete], left_on='origin_accession', right_on='accn', how='inner', validate='one_to_one')
        path = RUN/'annual_fitted'/f'{year}.pkl'; adapter = EPSAdapter.load(path)
        values = adapter.predict(matched)
        ref = old[old.asof_date.dt.year == year].set_index('sample_id').loc[matched.sample_id]
        np.testing.assert_array_equal(values, ref.predicted_eps)
        changed = matched.copy()
        changed['future_dollar_earnings'] = 1e12
        np.testing.assert_array_equal(values, adapter.predict(changed))
        changed.loc[changed.index[1:], annual.COLS] = changed.loc[changed.index[1:], annual.COLS]*3+7
        np.testing.assert_allclose(values[0], adapter.predict(changed)[0], rtol=1e-12, atol=1e-12)
        records.append({'model_id': name, 'year': year, 'rows': len(matched), 'status': 'EXACT_PASS',
                        'future_label_mutation': 'EXACT_PASS', 'other_origin_input_mutation': 'PASS', 'weight_sha256': sha(path)})
    ids = set(old.loc[old.prediction_valid, 'sample_id']); scores = []
    for path in sorted((RUN/'predictions').glob('*.parquet')):
        p = pd.read_parquet(path); p = p[(p.target_key == 'ttm') & p.sample_id.isin(ids)]
        receipt = json.loads((RUN/'model_receipts'/f'{path.stem}.json').read_text(encoding='utf-8'))
        surfaces = [(split, g) for split, g in p.groupby('split')]
        surfaces.append(('SELECTION_OOF_AVAILABLE_BEFORE_2022', p[(p.split == 'VALIDATION_OOF') & (p.label_asof < pd.Timestamp('2022-01-01', tz='UTC'))]))
        for split, group in surfaces:
            scores.append({'eps_model_id': path.stem, 'target_key': 'ttm', 'split': split,
                           'mask': 'HVZ_ANNUAL_ACCOUNTING_AVAILABLE_COMMON', 'model_status': receipt['status'],
                           'portfolio_eligible': receipt['status'] == 'FULL_RESEARCH_SCORED', **metric_row(group)})
    atomic_csv(pd.DataFrame(scores), RUN/'EPS_ANNUAL_ACCOUNTING_COMPARISON.csv')
    save_json(RUN/'ANNUAL_ACCOUNTING_FRESH_PROCESS_REPLAY.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'expected': 8, 'completed': len(records), 'all_pass': True, 'records': records,
              'refit': False, 'source_sha256': sha(__file__), 'prediction_sha256': sha(RUN/'predictions'/f'{name}.parquet')})
    print('ANNUAL_ACCOUNTING_FRESH_REPLAY_8_EXACT_PASS', flush=True)


if __name__ == '__main__': run()
