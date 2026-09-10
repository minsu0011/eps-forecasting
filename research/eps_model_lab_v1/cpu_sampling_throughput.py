"""Exact-output CPU throughput profiles for origin-isolated Lag-Llama.

No model/seed/sample-count change. Each worker owns a complete native predictor
and each origin gets an independent seed reset. Benchmarking uses no truth loss.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
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
import psutil
import torch
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import dataset

MODEL = None
FRAME = None


def initialize(threads=1):
    global MODEL, FRAME
    torch.set_num_threads(threads)
    from research.eps_model_lab_v1.lag_llama_model import LagAdapter
    FRAME = dataset(); MODEL = LagAdapter().fit()


def worker(indices):
    results = []
    for index in indices:
        one = FRAME.loc[[index]]; _, quarter = MODEL.forecast(one, 'quarter'); _, ttm = MODEL.forecast(one, 'ttm')
        results.append((int(index), np.concatenate([quarter[0], ttm[0, 3:4]], axis=0)))
    return results


def reference(frame):
    p = pd.read_parquet(RUN/'predictions/lag_llama_zero_shot_origin_isolated.parquet')
    blocks = []
    for target in ['h1', 'h2', 'h3', 'h4', 'ttm']:
        blocks.append(p[p.target_key == target].set_index('sample_id').loc[frame.sample_id, ['p10_eps', 'p50_eps', 'p90_eps']].to_numpy())
    return np.stack(blocks, axis=1)


def compare(values, expected):
    return {'exact': bool(np.array_equal(values, expected)),
            'within_original_2e5_tolerance': bool(np.allclose(values, expected, rtol=2e-5, atol=2e-5)),
            'maximum_difference': float(np.max(np.abs(values-expected)))}


def pool_run(indices, workers, expected, output=None):
    if psutil.virtual_memory().available < 40*1024**3: raise RuntimeError('Insufficient CPU model-worker RAM headroom')
    began = time.perf_counter(); results = {}; chunks = [indices[i:i+8] for i in range(0, len(indices), 8)]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(1,)) as executor:
        futures = [executor.submit(worker, chunk) for chunk in chunks]
        for future in as_completed(futures):
            results.update(dict(future.result()))
            if output:
                save_json(output, {'status': 'RUNNING', 'completed_origins': len(results), 'expected_origins': len(indices),
                          'workers': workers, 'inner_threads': 1, 'seconds': time.perf_counter()-began,
                          'updated_utc': datetime.now(timezone.utc).isoformat()})
    values = np.stack([results[int(index)] for index in indices]); seconds = time.perf_counter()-began
    return {'workers': workers, 'inner_threads': 1, 'origins': len(indices), 'native_calls': len(indices)*2,
            'wall_seconds_including_spawn_and_load': seconds, 'native_calls_per_second_including_startup': len(indices)*2/seconds,
            **compare(values, expected)}


def run():
    global MODEL, FRAME
    lock = RUN/'CPU_SAMPLING_THROUGHPUT_PLAN.json'
    if lock.exists(): raise RuntimeError('No implicit repeated throughput profile')
    frame = dataset(); eligible = frame[frame.asof_date.dt.year >= 2019]
    # Input-only deterministic spread; no label/error-based sample selection.
    probe_indices = eligible.index[np.linspace(0, len(eligible)-1, 16, dtype=int)].to_numpy()
    benchmark_indices = eligible.index[np.linspace(0, len(eligible)-1, 256, dtype=int)].to_numpy()
    save_json(lock, {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_id': 'lag_llama_zero_shot_origin_isolated',
              'threads': [1, 2, 4, 8], 'outer_workers': [4, 12, 20], 'native_sample_count': 100, 'native_seed': 1729,
              'probe_sample_ids': frame.loc[probe_indices, 'sample_id'].tolist(),
              'benchmark_sample_ids': frame.loc[benchmark_indices, 'sample_id'].tolist(),
              'prediction_sha256': sha(RUN/'predictions/lag_llama_zero_shot_origin_isolated.parquet'),
              'selection_uses_forecast_truth_or_accuracy': False, 'source_sha256': sha(__file__),
              'concurrency_note': 'Measured while other EPS CPU diagnostics may be running, not a dedicated-system benchmark'})
    initialize(4); expected = reference(frame.loc[probe_indices]); profiles = []
    for threads in [1, 2, 4, 8]:
        torch.set_num_threads(threads); worker(probe_indices[:1])  # Warm-up excluded.
        began = time.perf_counter(); result = dict(worker(probe_indices))
        seconds = time.perf_counter()-began; values = np.stack([result[int(i)] for i in probe_indices])
        profiles.append({'threads': threads, 'origins': len(probe_indices), 'native_calls': len(probe_indices)*2,
                         'seconds': seconds, 'native_calls_per_second': len(probe_indices)*2/seconds, **compare(values, expected)})
        save_json(RUN/'CPU_SAMPLING_THREAD_PROFILES.json', {'records': profiles, 'unchanged_saved_predictions': True})
        print('SAMPLING_THREAD_PROFILE', profiles[-1], flush=True)
    one = next(r for r in profiles if r['threads'] == 1)
    if not one['exact']:
        save_json(RUN/'CPU_SAMPLING_PARALLEL_REPLAY.json', {'status': 'NOT_ADVANCED_THREAD1_NOT_BIT_EXACT', 'profile': one,
                  'original_4thread_recipe_retained': True}); return
    del MODEL; MODEL = None; FRAME = None; gc.collect()
    expected = reference(frame.loc[benchmark_indices]); timings = []
    for workers in [4, 12, 20]:
        record = pool_run(benchmark_indices, workers, expected); timings.append(record)
        save_json(RUN/'CPU_SAMPLING_PROCESS_PROFILES.json', {'records': timings,
                  'metric': 'Native-call throughput including spawn/load and concurrent-system contention', 'predictions_overwritten': False})
        print('SAMPLING_PROCESS_PROFILE', record, flush=True)
    eligible_profiles = [r for r in timings if r['exact']]
    if not eligible_profiles:
        save_json(RUN/'CPU_SAMPLING_PARALLEL_REPLAY.json', {'status': 'NOT_ADVANCED_NO_EXACT_PROFILE',
                  'original_4thread_recipe_retained': True}); return
    chosen = min(eligible_profiles, key=lambda r: r['wall_seconds_including_spawn_and_load'])['workers']
    indices = eligible.index.to_numpy(); expected = reference(eligible)
    record = pool_run(indices, chosen, expected, RUN/'CPU_SAMPLING_PARALLEL_PROGRESS.json')
    save_json(RUN/'CPU_SAMPLING_PARALLEL_REPLAY.json', {'status': 'PASS_EXACT_ALL_ORIGINS' if record['exact'] else 'DIAGNOSTIC_PROFILE_NOT_EXACT',
              'record': record, 'selected_outer_workers_by_measured_throughput': chosen, 'inner_threads': 1,
              'weights_seed_samples_inputs_unchanged': True, 'original_predictions_overwritten': False,
              'new_model_candidate_count': 0, 'profile_validated_for': 'Full frozen V1.3 sample set only, not arbitrary future machines',
              'source_sha256': sha(__file__)})
    print('CPU_SAMPLING_FULL_PARALLEL_REPLAY', record, flush=True)


if __name__ == '__main__': run()
