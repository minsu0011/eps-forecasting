"""Bounded exact-output CPU scheduling profiles; no model or score selection."""
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
from research.eps_model_lab_v1.cpu_sampling_throughput import compare

MODEL = None
FRAME = None


def initialize(base, threads):
    global MODEL, FRAME
    torch.set_num_threads(threads)
    from research.eps_model_lab_v1.moirai_models import MoiraiAdapter
    FRAME = dataset(); MODEL = MoiraiAdapter(base).fit()


def worker(indices):
    results = []
    for index in indices:
        one = FRAME.loc[[index]]
        _, quarter = MODEL.forecast(one, 'quarter'); _, ttm = MODEL.forecast(one, 'ttm')
        results.append((int(index), np.concatenate([quarter[0], ttm[0, 3:4]], axis=0)))
    return results


def reference(name, frame):
    pred = pd.read_parquet(RUN/'predictions'/f'{name}.parquet')
    return np.stack([pred[pred.target_key == t].set_index('sample_id').loc[frame.sample_id,
        ['p10_eps', 'p50_eps', 'p90_eps']].to_numpy() for t in ['h1', 'h2', 'h3', 'h4', 'ttm']], axis=1)


def pool_run(base, indices, workers, threads, expected, progress):
    if psutil.virtual_memory().available < 45*1024**3: raise RuntimeError('Insufficient worker memory headroom')
    start = time.perf_counter(); results = {}; chunks = [indices[i:i+8] for i in range(0, len(indices), 8)]
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(base, threads)) as executor:
        futures = [executor.submit(worker, chunk) for chunk in chunks]
        for future in as_completed(futures):
            results.update(dict(future.result()))
            save_json(progress, {'status': 'RUNNING', 'completed': len(results), 'expected': len(indices),
                'workers': workers, 'inner_threads': threads, 'seconds': time.perf_counter()-start,
                'updated_utc': datetime.now(timezone.utc).isoformat()})
    values = np.stack([results[int(i)] for i in indices]); elapsed = time.perf_counter()-start
    return {'workers': workers, 'inner_threads': threads, 'origins': len(indices), 'native_calls': 2*len(indices),
        'wall_seconds_including_startup': elapsed, 'calls_per_second': 2*len(indices)/elapsed, **compare(values, expected)}


def run(base):
    global MODEL, FRAME
    if base not in ['moirai1p1_small', 'moirai_moe_small']: raise ValueError(base)
    name = base+'_origin_isolated'; directory = RUN/'cpu_moirai_throughput'/name
    directory.mkdir(parents=True, exist_ok=False)
    prediction = RUN/'predictions'/f'{name}.parquet'; digest = sha(prediction)
    frame = dataset(); eligible = frame[frame.asof_date.dt.year >= 2019]
    probes = eligible.index[np.linspace(0, len(eligible)-1, 16, dtype=int)].to_numpy()
    benchmark = eligible.index[np.linspace(0, len(eligible)-1, 256, dtype=int)].to_numpy()
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_id': name, 'threads': [4, 1, 2, 8],
        'outer_profiles': [[4, 1], [12, 1], [20, 1]], 'prediction_sha256': digest, 'native_samples': 100, 'seed': 1729,
        'probe_ids': frame.loc[probes, 'sample_id'].tolist(), 'benchmark_ids': frame.loc[benchmark, 'sample_id'].tolist(),
        'profile_acceptance': 'Bit-exact original saved quantiles including P50; no relaxed tolerance',
        'score_or_truth_used_for_selection': False, 'no_new_model_identity': True, 'source_sha256': sha(__file__)}
    save_json(directory/'PLAN.json', plan)
    initialize(base, 4); expected = reference(name, frame.loc[probes]); profiles = []
    for threads in plan['threads']:
        torch.set_num_threads(threads); worker(probes[:1])
        start = time.perf_counter(); values = dict(worker(probes)); elapsed = time.perf_counter()-start
        profile = {'threads': threads, 'origins': len(probes), 'seconds': elapsed,
            **compare(np.stack([values[int(i)] for i in probes]), expected)}
        profiles.append(profile); save_json(directory/'THREAD_PROFILES.json', {'records': profiles})
        print('MOIRAI_CPU_THREADS', name, profile, flush=True)
    MODEL = None; FRAME = None; gc.collect()
    if not next(p for p in profiles if p['threads'] == 1)['exact']:
        save_json(directory/'COMPLETION.json', {'status': 'NOT_ADVANCED_THREAD1_NOT_BIT_EXACT',
            'original_recipe_preserved': True, 'original_predictions_unchanged': sha(prediction) == digest,
            'new_candidate_count': 0, 'completed_utc': datetime.now(timezone.utc).isoformat()})
        return
    expected = reference(name, frame.loc[benchmark]); timings = []
    for workers, threads in plan['outer_profiles']:
        record = pool_run(base, benchmark, workers, threads, expected, directory/'PROGRESS.json')
        timings.append(record); save_json(directory/'PROCESS_PROFILES.json', {'records': timings})
        print('MOIRAI_CPU_PROCESS', name, record, flush=True)
    exact = [p for p in timings if p['exact']]
    if not exact: raise RuntimeError('No bit-exact process profile; original outputs retained')
    chosen = min(exact, key=lambda p: p['wall_seconds_including_startup'])
    estimate = chosen['wall_seconds_including_startup']*len(eligible)/len(benchmark)
    remaining = (datetime.fromisoformat('2026-09-08T01:38:26+00:00')-datetime.now(timezone.utc)).total_seconds()
    if estimate*1.5 > remaining: raise RuntimeError('Finalization-window throughput guard')
    record = pool_run(base, eligible.index.to_numpy(), chosen['workers'], 1, reference(name, eligible), directory/'PROGRESS.json')
    save_json(directory/'COMPLETION.json', {'status': 'PASS_EXACT_ALL_ORIGINS' if record['exact'] else 'NOT_EXACT_PROFILE_REJECTED',
        'record': record, 'selected_by_throughput': chosen, 'original_predictions_unchanged': sha(prediction) == digest,
        'weights_seed_samples_unchanged': True, 'new_candidate_count': 0, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'This frozen sample set and current machine; not arbitrary hardware or dedicated-system timing'})
    print('MOIRAI_CPU_COMPLETE', name, record, flush=True)


if __name__ == '__main__': run(sys.argv[1])
