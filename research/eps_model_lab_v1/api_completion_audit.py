"""Actual adapter API checks without overwriting scored models or predictions."""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '4'
import numpy as np
import torch
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import dataset, features


def close_record(kind, a, b):
    return {'kind': kind, 'pass': bool(np.allclose(a, b, rtol=2e-5, atol=2e-5)),
            'maximum_absolute_difference': float(np.max(np.abs(np.asarray(a)-np.asarray(b))))}


def tabpfn():
    from research.eps_model_lab_v1.additional_foundation_wave import TabPFNAdapter
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frame = dataset(); records = []
    for year in range(2019, 2027):
        probe = frame[frame.asof_date.dt.year == year].iloc[:16].copy()
        for target in ['h1', 'h2', 'h3', 'h4', 'ttm']:
            a = TabPFNAdapter.load(RUN/'additional_fitted/tabpfn_v2_panel'/str(year)/target)
            p, q = a.distribution(probe)
            changed = probe.copy()
            for col in changed:
                if col.startswith(('y_', 'label_', 'target_')): changed[col] = 1e12
            pp, qq = a.distribution(changed)
            tests = [close_record('all_future_labels_point', p, pp), close_record('all_future_labels_quantiles', q, qq)]
            changed = probe.copy(); cols = features(changed).columns
            changed.loc[changed.index[1:], cols] = changed.loc[changed.index[1:], cols]*3+7
            pp, qq = a.distribution(changed)
            tests += [close_record('other_origin_values_point', p[0], pp[0]), close_record('other_origin_values_quantiles', q[0], qq[0])]
            pp, qq = a.distribution(probe.iloc[::-1])
            tests += [close_record('permutation_point', p, pp[::-1]), close_record('permutation_quantiles', q, qq[::-1])]
            pp, qq = a.distribution(probe.iloc[:1])
            tests += [close_record('singleton_point', p[0], pp[0]), close_record('singleton_quantiles', q[0], qq[0])]
            records.append({'year': year, 'target': target, 'tests': tests, 'all_pass': all(t['pass'] for t in tests)})
            save_json(RUN/'TABPFN_ALL_HEAD_INPUT_INDEPENDENCE_AUDIT.json', {'expected_heads': 40, 'completed_heads': len(records),
                      'records': records, 'all_pass': all(r['all_pass'] for r in records),
                      'prediction_sha256': sha(RUN/'predictions/tabpfn_v2_panel.parquet'), 'refit': False})
            print('TABPFN_API', year, target, records[-1]['all_pass'], flush=True)
            del a; gc.collect(); torch.cuda.empty_cache()


def joint():
    from research.eps_model_lab_v1.neural_models import NeuralAdapter
    from research.eps_model_lab_v1.multivariate_models import MultivariateAdapter, tensors, to_device
    torch.backends.cuda.matmul.allow_tf32 = False
    frame = dataset(); probe = frame[frame.asof_date.dt.year == 2019].iloc[:8]
    records = []
    for name in ['NHITS', 'TFT', 'NLinear']:
        a = NeuralAdapter.load(RUN/'neural_fitted'/('nf_'+name)/'2019')
        path = a.path(probe)
        for target in ['h1', 'h2', 'h3', 'h4', 'ttm']:
            expected = path.sum(axis=1) if target == 'ttm' else path[:, int(target[1:])-1]
            r = close_record('actual_common_predict_'+target, expected, a.predict(probe, target))
            r['model_id'] = a.model_id; records.append(r)
        del a; gc.collect(); torch.cuda.empty_cache()
    # Exercise the new exact-frame joint fit API with the SAME fixed recipe.
    # This verification fit is not registered/scored as another candidate.
    a = MultivariateAdapter('TSMixer').fit(frame, 2019)
    b = MultivariateAdapter.load(RUN/'multivariate_fitted/nf_multivar_TSMixer/2019.pt')
    data = to_device(tensors(frame)); indices = np.flatnonzero((frame.asof_date.dt.year == 2019).to_numpy())[:8]
    expected = b.path(data, indices); actual = a.path(data, indices)
    records.append(close_record('common_fit_same_fixed_recipe_vs_saved_2019', expected, actual))
    for target in ['h1', 'ttm']:
        value = actual[:, 3, 1] if target == 'ttm' else actual[:, 0, 0]
        records.append(close_record('multivariate_common_predict_'+target, value, a.predict(probe, target)))
    out = RUN/'adapter_api_verification/TSMixer_2019.pt'
    if out.exists(): raise RuntimeError('Do not overwrite adapter verification artifact')
    a.save(out); c = MultivariateAdapter.load(out)
    records.append(close_record('actual_common_fit_save_load', actual, c.path(data, indices)))
    save_json(RUN/'JOINT_ADAPTER_API_COMPLETION_AUDIT.json', {'records': records, 'all_pass': all(r['pass'] for r in records),
              'verification_fit_is_new_candidate': False, 'scored_predictions_overwritten': False,
              'sample_sha256': sha(RUN/'data/samples.parquet'), 'source_sha256': sha(__file__),
              'completed_utc': datetime.now(timezone.utc).isoformat()})
    print('JOINT_API_COMPLETE', len(records), all(r['pass'] for r in records), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    {'tabpfn': tabpfn, 'joint': joint}[sys.argv[1]]()
