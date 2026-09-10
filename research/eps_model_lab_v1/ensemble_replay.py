"""Recompute frozen ensembles in a new process without rewriting predictions."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, sha, save_json
from research.eps_model_lab_v1.common import dataset
from research.eps_model_lab_v1.ensemble_models import aggregate, CUTOFF, TRACKS


def run():
    lockpath = RUN/'ENSEMBLE_FREEZE_V1.json'; lock = json.loads(lockpath.read_text(encoding='utf-8'))
    frame = dataset().loc[lambda f: f.asof_date.dt.year >= 2019]
    pool = {}; records = []
    for name, digest in lock['candidate_prediction_hashes'].items():
        path = RUN/'predictions'/f'{name}.parquet'
        if sha(path) != digest: raise RuntimeError('Frozen candidate changed '+name)
        pool[name] = pd.read_parquet(path)
    for track in TRACKS:
        for method in ['mean', 'median', 'trimmed_mean', 'convex_l1']:
            name = f'ens_{track}_{method}'; path = RUN/'predictions'/f'{name}.parquet'
            receipt = json.loads((RUN/'model_receipts'/f'{name}.json').read_text(encoding='utf-8'))
            if sha(path) != receipt['prediction_sha256'] or sha(lockpath) != receipt['ensemble_freeze_sha256']:
                raise RuntimeError('Ensemble prediction or freeze binding changed')
            old = pd.read_parquet(path)
            for plan in lock['plans']:
                if plan['track'] != track: continue
                target = plan['target']; columns = []
                for member in plan['member_ids']:
                    p = pool[member]
                    columns.append(p[p.target_key == target].set_index('sample_id').reindex(frame.sample_id).predicted_eps.to_numpy())
                values = aggregate(np.column_stack(columns), plan['convex_l1_weights'], method)
                values[(frame.asof_date < CUTOFF).to_numpy()] = np.nan
                expected = old[old.target_key == target].set_index('sample_id').loc[frame.sample_id]
                np.testing.assert_array_equal(values, expected.predicted_eps)
                records.append({'model_id': name, 'target': target, 'rows': len(frame), 'status': 'EXACT_PASS'})
    save_json(RUN/'ENSEMBLE_FRESH_PROCESS_REPLAY.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'expected': 40, 'completed': len(records), 'all_pass': len(records) == 40, 'records': records,
              'freeze_sha256': sha(lockpath), 'predictions_overwritten': False, 'weights_refitted': False})
    print('ENSEMBLE_FRESH_PROCESS_REPLAY', len(records), 'EXACT_PASS', flush=True)


if __name__ == '__main__': run()
