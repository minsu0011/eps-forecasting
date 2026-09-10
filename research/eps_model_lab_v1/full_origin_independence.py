"""All annual neural weights, three deterministic origin probes, strict FP32.

This extends selected-fold diagnostics, not the saved prediction recipe. Earlier
batch failures remain authoritative observations under their original profile.
"""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from research.eps_model_lab_v1.bootstrap import RUN, sha, save_json
from research.eps_model_lab_v1.common import dataset


def checks(before, other, future, singleton):
    return {kind: {'pass': bool(np.allclose(before, value, rtol=2e-5, atol=2e-5)),
                   'maximum_difference': float(np.max(np.abs(before-value)))}
            for kind, value in [('other_origin_values', other), ('future_labels', future), ('singleton', singleton)]}


def run():
    from research.eps_model_lab_v1.neural_models import NeuralAdapter
    from research.eps_model_lab_v1.eps_native_models import NativeAdapter
    from research.eps_model_lab_v1.multivariate_models import MultivariateAdapter, tensors, to_device
    lock = RUN/'ALL_ANNUAL_ORIGIN_INDEPENDENCE_PLAN.json'
    if lock.exists(): raise RuntimeError('No implicit repeat of expanded diagnostic')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    frame = dataset(); data = to_device(tensors(frame)); tasks = []
    for p in sorted((RUN/'neural_fitted').glob('*/*')):
        if p.is_dir() and p.name.isdigit(): tasks.append(('neural', p.parent.name, int(p.name), p))
    for p in sorted((RUN/'native_fitted').glob('*/*.pt')):
        tasks.append(('native', 'epspredict_'+p.parent.name, int(p.stem), p))
    for p in sorted((RUN/'multivariate_fitted').glob('*/*.pt')):
        tasks.append(('multivariate', p.parent.name, int(p.stem), p))
    save_json(lock, {'created_utc': datetime.now(timezone.utc).isoformat(), 'annual_artifacts': len(tasks),
              'probes': ['first annual origin', 'maximum causal origin scale', 'first negative last-observed EPS, otherwise middle origin'],
              'profile': 'FP32; CuDNN and CUDA matmul TF32 disabled; original saved predictions not changed',
              'rtol': 2e-5, 'atol': 2e-5, 'dataset_sha256': sha(RUN/'data/samples.parquet'), 'source_sha256': sha(__file__)})
    records = []
    for family, name, year, path in tasks:
        began = time.perf_counter(); valid = frame[frame.asof_date.dt.year == year]
        negative = valid[valid.last_observed_eps < 0]
        anchors = list(dict.fromkeys([valid.index[0], valid.scale.idxmax(), negative.index[0] if len(negative) else valid.index[len(valid)//2]]))
        a = {'neural': NeuralAdapter, 'native': NativeAdapter, 'multivariate': MultivariateAdapter}[family].load(path)
        for anchor in anchors:
            indices = np.asarray([anchor, *valid.index[valid.index != anchor][:7]], dtype=int)
            probe = frame.loc[indices].copy()
            if family == 'multivariate':
                before = a.path(data, indices)[0]
                changed = dict(data); changed['x'] = data['x'].clone(); changed['x'][indices[1:]] = changed['x'][indices[1:]]*3+7
                other = a.path(changed, indices)[0]
                future_data = dict(data); future_data['y'] = torch.full_like(data['y'], 1e12)
                future = a.path(future_data, indices)[0]; singleton = a.path(data, indices[:1])[0]
                del changed, future_data
            else:
                before = a.path(probe)[0]; changed = probe.copy()
                for column in changed:
                    if column.startswith(('eps_lag_', 'filled_lag_', 'ttm_lag_', 'account_')):
                        changed.loc[changed.index[1:], column] = changed.loc[changed.index[1:], column]*3+7
                other = a.path(changed)[0]
                changed = probe.copy()
                for column in changed:
                    if column.startswith(('y_', 'label_', 'target_')): changed[column] = 1e12
                future = a.path(changed)[0]; singleton = a.path(probe.iloc[:1])[0]
            result = checks(before, other, future, singleton)
            records.append({'model_id': name, 'year': year, 'anchor_sample_id': frame.loc[anchor, 'sample_id'],
                            'anchor_ticker': frame.loc[anchor, 'ticker'], 'checks': result,
                            'all_pass': all(v['pass'] for v in result.values())})
        save_json(RUN/'ALL_ANNUAL_ORIGIN_INDEPENDENCE_AUDIT.json', {'updated_utc': datetime.now(timezone.utc).isoformat(),
                  'expected_annual_artifacts': len(tasks), 'completed_annual_artifacts': len({(r['model_id'], r['year']) for r in records}),
                  'probe_records': len(records), 'records': records,
                  'all_input_value_mutations_pass': all(r['checks']['other_origin_values']['pass'] and r['checks']['future_labels']['pass'] for r in records),
                  'all_singleton_numerics_pass': all(r['checks']['singleton']['pass'] for r in records),
                  'profile': 'STRICT_FP32_SEPARATE_DIAGNOSTIC_NO_PREDICTION_REPLACEMENT', 'formal_certified': False})
        print('ALL_ANNUAL_INPUT_AUDIT', name, year, len(records), round(time.perf_counter()-began, 2), flush=True)
        del a; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__': run()
