"""Bounded FP64 diagnosis of the two FP32 singleton failures; no score replacement."""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import dataset, split_train, features
from research.eps_model_lab_v1.additional_foundation_wave import TabPFNAdapter, SPEC
from research.eps_model_lab_v1.api_completion_audit import close_record


def run():
    queue = json.loads((RUN/'API_COMPLETION_QUEUE.json').read_text(encoding='utf-8'))
    if queue['status'] != 'COMPLETE': raise RuntimeError('GPU resource still owned')
    audit = json.loads((RUN/'TABPFN_ALL_HEAD_INPUT_INDEPENDENCE_AUDIT.json').read_text(encoding='utf-8'))
    if audit['completed_heads'] != 40: raise RuntimeError('Input audit incomplete')
    failed = [r for r in audit['records'] if not r['all_pass']]
    for r in failed:
        if any(not t['pass'] and not t['kind'].startswith('singleton_') for t in r['tests']):
            raise RuntimeError('This bounded review does not authorize investigating other-input leakage as rounding')
    lock = RUN/'TABPFN_NUMERICAL_REVIEW_PLAN.json'
    if lock.exists(): raise RuntimeError('No implicit repeated numerical review')
    save_json(lock, {'created_utc': datetime.now(timezone.utc).isoformat(), 'heads': [(r['year'], r['target']) for r in failed],
              'change': 'Official inference_precision float32 to float64 only, unchanged checkpoint/train/features/4 estimators/seed',
              'original_prediction_sha256': audit['prediction_sha256'], 'tolerance_unchanged': {'rtol': 2e-5, 'atol': 2e-5},
              'new_candidate_or_leaderboard_update': False, 'source_sha256': sha(__file__)})
    frame = dataset(); records = []; torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for r in failed:
        year, target = r['year'], r['target']; train, valid = split_train(frame, year, target); probe = valid.iloc[:16]
        a = TabPFNAdapter.load(RUN/'additional_fitted/tabpfn_v2_panel'/str(year)/target)
        p32, q32 = a.distribution(probe); one32, oneq32 = a.distribution(probe.iloc[:1])
        baseline = [close_record('FP32_point_singleton', p32[0], one32[0]), close_record('FP32_quantile_singleton', q32[0], oneq32[0])]
        weight = str(Path(a.r['path'])/SPEC[a.model_id]['checkpoint']); del a; gc.collect(); torch.cuda.empty_cache()
        a = TabPFNAdapter(); a.target = target; a.columns = list(features(train).columns)
        began = time.perf_counter()
        a.model = TabPFNRegressor.create_default_for_version(ModelVersion.V2, model_path=weight,
            n_estimators=4, device='cuda', inference_precision=torch.float64,
            fit_mode='fit_preprocessors', random_state=1729, n_preprocessing_jobs=4)
        a.model.fit(a.prepare_data(train), (train['y_'+target]/train.scale).to_numpy(dtype=np.float32))
        p, q = a.distribution(probe); pp, qq = a.distribution(probe.iloc[:1])
        tests = [close_record('FP64_point_singleton', p[0], pp[0]), close_record('FP64_quantile_singleton', q[0], qq[0])]
        changed = probe.copy(); cols = features(changed).columns
        changed.loc[changed.index[1:], cols] = changed.loc[changed.index[1:], cols]*3+7
        pp, qq = a.distribution(changed)
        tests += [close_record('FP64_other_origin_point', p[0], pp[0]), close_record('FP64_other_origin_quantiles', q[0], qq[0])]
        records.append({'year': year, 'target': target, 'original_FP32': baseline, 'FP64_tests': tests,
                        'all_FP64_pass': all(t['pass'] for t in tests), 'seconds': time.perf_counter()-began,
                        'FP32_to_FP64_point_difference_max': float(np.max(np.abs(p-p32))),
                        'FP32_to_FP64_quantile_difference_max': float(np.max(np.abs(q-q32)))})
        save_json(RUN/'TABPFN_NUMERICAL_REVIEW.json', {'records': records, 'expected_heads': len(failed),
                  'all_FP64_pass': all(x['all_FP64_pass'] for x in records), 'original_scored_predictions_unchanged': True,
                  'original_batch_audit_still_has_failures': True, 'formal_certified': False})
        print('TABPFN_NUMERICAL', year, target, records[-1]['all_FP64_pass'], flush=True)
        del a; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__': run()
