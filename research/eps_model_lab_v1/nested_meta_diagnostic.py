"""Nested chronological meta replay, separate from the final 2022 portfolio.

Re-selects members AND fits weights before each diagnostic forecast year. Never
fills the main ensemble's unavailable meta-fit validation rows. Evaluation of
2020/2021 origins can use their subsequently released labels, but those labels
cannot enter that year's fitting or change the final 2022 freeze.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import dataset, prediction_frame, metric_row, atomic_csv
from research.eps_model_lab_v1.ensemble_models import select, aggregate, TRACKS, TARGETS


def run():
    finalpath = RUN/'ENSEMBLE_FREEZE_V1.json'
    final = json.loads(finalpath.read_text(encoding='utf-8')); final_hash = sha(finalpath)
    path = RUN/'NESTED_META_DIAGNOSTIC_FREEZE.json'
    if path.exists(): raise RuntimeError('No implicit repeated nested diagnostic')
    source = dataset(); pool = {}
    for name, digest in final['candidate_prediction_hashes'].items():
        prediction = RUN/'predictions'/f'{name}.parquet'
        if sha(prediction) != digest: raise RuntimeError('Frozen pool drift')
        pool[name] = pd.read_parquet(prediction)
    plans = []; skipped = []
    for year in [2020, 2021]:
        cutoff = pd.Timestamp(f'{year}-01-01', tz='UTC')
        for track in TRACKS:
            for target in TARGETS:
                try:
                    plan, _ = select(pool, source, target, track, cutoff)
                    plan['forecast_year'] = year; plans.append(plan)
                except RuntimeError as exc:
                    if 'Insufficient eligible OOF selection labels' not in str(exc): raise
                    skipped.append({'year': year, 'track': track, 'target': target, 'reason': str(exc)})
    save_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'plans': plans, 'skipped': skipped,
              'final_2022_freeze_sha256': final_hash, 'candidate_prediction_hashes': final['candidate_prediction_hashes'],
              'source_sha256': sha(__file__), 'nested_scores_consulted_before_freeze': False,
              'changes_final_2022_selection_or_weights': False,
              'scope': 'Retrospective causal meta replay within validation-origin years, not untouched formal validation'})
    scores = []; outputs = []
    for plan in plans:
        year, target, track = plan['forecast_year'], plan['target'], plan['track']
        frame = source[source.asof_date.dt.year == year]; columns = []
        for member in plan['member_ids']:
            p = pool[member]
            columns.append(p[p.target_key == target].set_index('sample_id').reindex(frame.sample_id).predicted_eps.to_numpy())
        x = np.column_stack(columns)
        for method in ['mean', 'median', 'trimmed_mean', 'convex_l1']:
            name = f'nested_{track}_{method}'
            values = aggregate(x, plan['convex_l1_weights'], method)
            pred = prediction_frame(frame, target, name, values, metadata={'pit_valid': track == 'local_causal'})
            pred['nested_fit_cutoff'] = plan['cutoff']; pred['nested_member_ids'] = '|'.join(plan['member_ids'])
            outputs.append(pred)
            scores.append({'model_id': name, 'track': track, 'method': method, 'target_key': target,
                           'forecast_year': year, 'meta_fit_rows': plan['selection_rows'],
                           'maximum_meta_fit_label_availability': plan['maximum_label_availability'],
                           'latest_ex_post_evaluation_label_availability': str(pred.label_asof.max()),
                           'evaluation_labels_available_before_2022': int((pred.label_asof < pd.Timestamp('2022-01-01', tz='UTC')).sum()),
                           'formal_certified': False, **metric_row(pred)})
    directory = RUN/'nested_meta_diagnostic'; directory.mkdir(exist_ok=False)
    pd.concat(outputs, ignore_index=True).to_parquet(directory/'predictions.parquet', index=False)
    atomic_csv(pd.DataFrame(scores), RUN/'EPS_NESTED_META_DIAGNOSTIC_SCORES.csv')
    if sha(finalpath) != final_hash: raise RuntimeError('Final portfolio was changed')
    save_json(RUN/'NESTED_META_DIAGNOSTIC_RECEIPT.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
              'plans': len(plans), 'skipped_plans': len(skipped), 'score_rows': len(scores),
              'diagnostic_predictions_sha256': sha(directory/'predictions.parquet'), 'final_freeze_unchanged': True,
              'main_ensemble_predictions_overwritten': False, 'uses_research_test_origins': False,
              'evaluation_is_ex_post': True, 'formal_certified': False})
    print('NESTED_META_DIAGNOSTIC_COMPLETE', len(plans), 'plans', len(scores), 'scores', flush=True)


if __name__ == '__main__': run()
