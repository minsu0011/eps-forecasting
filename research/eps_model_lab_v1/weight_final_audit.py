"""Verify every pinned downloaded checkpoint/code file without mutating caches."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, sha, save_json


def run():
    records = []; failures = []; seen = {}
    for path in sorted((RUN/'weight_receipts').glob('*.json')):
        if 'initial_' in path.stem: continue
        receipt = json.loads(path.read_text(encoding='utf-8'))
        for item in receipt.get('files', []):
            p = Path(item['path'])
            if str(p) not in seen: seen[str(p)] = sha(p) if p.is_file() else None
            ok = seen[str(p)] == item['sha256']
            r = {'weight_id': receipt.get('weight_id'), 'revision': receipt.get('revision'),
                 'path': str(p), 'expected_sha256': item['sha256'], 'actual_sha256': seen[str(p)],
                 'status': 'UNCHANGED' if ok else 'DRIFT_OR_MISSING'}
            records.append(r)
            if not ok: failures.append(r)
        print('WEIGHT_FINAL_CHECK', path.name, len(receipt.get('files', [])), flush=True)
    save_json(RUN/'FINAL_PRETRAINED_WEIGHT_HASH_AUDIT.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'all_unchanged': not failures, 'unique_files': len(seen), 'records': records, 'failures': failures,
              'acquisition_status_is_historical': True,
              'actual_execution_status_manifest': 'EPS_WEIGHT_USE_MANIFEST.json',
              'pretraining_corpus_vintage_certified': False})
    if failures: raise RuntimeError('Pinned pretrained checkpoint/code drift')
    print('PRETRAINED_WEIGHT_HASHES_ALL_UNCHANGED', len(seen), flush=True)


if __name__ == '__main__': run()
