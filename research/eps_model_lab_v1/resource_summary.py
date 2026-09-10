"""Summarize actual system telemetry without inventing model-level utilization."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
from research.eps_model_lab_v1.bootstrap import RUN, ORIGINAL_RUN, save_json, sha


def summarize(records):
    times = np.array([datetime.fromisoformat(r['utc']).timestamp() for r in records])
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('At least two strictly ordered actual telemetry samples required')
    intervals = np.diff(times)
    # Cap interpolation at twice intended cadence; gaps are unobserved, not idle.
    weights = np.minimum(intervals, 60.)
    metrics = {}
    keys = sorted(set().union(*(r.keys() for r in records))-{'utc', 'gpu_sampling_error'})
    for key in keys:
        values = np.array([float(r.get(key, np.nan)) for r in records])
        finite = np.isfinite(values); eligible = finite[:-1]
        if not finite.any(): continue
        metrics[key] = {'samples': int(finite.sum()), 'min': float(np.nanmin(values)),
            'max': float(np.nanmax(values)), 'sample_mean': float(np.nanmean(values)),
            'interval_weighted_mean': float(np.average(values[:-1][eligible], weights=weights[eligible])) if eligible.any() else None}
    return {'samples': len(records), 'first_utc': records[0]['utc'], 'last_utc': records[-1]['utc'],
        'wall_span_hours': float((times[-1]-times[0])/3600), 'observed_interval_hours_capped_60s': float(weights.sum()/3600),
        'long_gaps_over_60s': int((intervals > 60).sum()), 'maximum_gap_seconds': float(intervals.max()),
        'gpu_sample_errors': sum('gpu_sampling_error' in r for r in records), 'metrics': metrics}


def run():
    sources = [p for p in [ORIGINAL_RUN/'RESOURCE_TELEMETRY.jsonl', RUN/'RESOURCE_TELEMETRY.jsonl'] if p.exists()]
    records = []; provenance = []; trailing_bytes = 0
    for source in sources:
        payload = source.read_bytes()
        complete, trailing = payload.rsplit(b'\n', 1)
        rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
        records.extend(rows); trailing_bytes += len(trailing)
        provenance.append({'source':str(source),'complete_samples':len(rows),'ignored_partial_bytes':len(trailing)})
    records.sort(key=lambda row: datetime.fromisoformat(row['utc']))
    copied = RUN/'RESOURCE_TELEMETRY_CAPTURE.jsonl'
    copied.write_text(''.join(json.dumps(row)+'\n' for row in records),encoding='utf-8')
    result = summarize(records)
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), sources=provenance,
        captured_complete_lines_sha256=sha(copied), ignored_partial_trailing_bytes=trailing_bytes,
        measured_scope='Whole system, only monitor-covered period; includes concurrent tasks, display and OS',
        per_model_gpu_active_time_measured=False,
        limitations=['Sampled peaks may miss brief spikes', 'No utilization invented before first sample or after last sample',
            'GPU power is sampled device draw, not wall-plug energy or training-only power',
            'Interval weighted means use previous sample and cap unobserved gaps at 60 seconds'])
    save_json(RUN/'RESOURCE_TELEMETRY_SUMMARY.json', result)
    print('RESOURCE_SUMMARY', result['samples'], result['wall_span_hours'], 'hours', flush=True)


if __name__ == '__main__': run()
