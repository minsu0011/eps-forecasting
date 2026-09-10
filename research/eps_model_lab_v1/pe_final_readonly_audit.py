"""Final non-mutating hash check of original PE scope and reused real outputs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha


def run():
    baseline = json.loads((RUN/'PE_READONLY_BASELINE.json').read_text(encoding='utf-8'))
    records = []; errors = []
    def verify(path, expected, scope):
        actual = sha(path) if path.is_file() else None
        record = {'path': str(path), 'expected_sha256': expected, 'actual_sha256': actual,
                  'scope': scope, 'status': 'UNCHANGED' if actual == expected else 'DRIFT_OR_MISSING'}
        records.append(record)
        if actual != expected: errors.append(record)
    for item in baseline['files']:
        verify(Path(baseline['root'])/item['path'], item['sha256'], 'FROZEN_PE_ARTIFACT')
    binding = json.loads((RUN/'pe_integration/READONLY_PE_SOURCE_BINDING.json').read_text(encoding='utf-8'))
    for item in binding['live_C4_verified']:
        verify(PROJECT/item['path'], item['sha256'], 'FROZEN_C4_LIVE_SOURCE')
    upstream = PROJECT.parent/'PE_Regime_Engine_v0.2.0'
    for relative, digest in binding['upstream_input_engine'].items():
        verify(upstream/relative, digest, 'UPSTREAM_INPUT_ENGINE')
    verify(upstream/'config/default.yaml', binding['input_config_sha256'], 'UPSTREAM_INPUT_CONFIG')
    reused = json.loads((RUN/'PE_INPUT_EQUIVALENCE_REUSE_AUDIT.json').read_text(encoding='utf-8'))
    for item in reused['copied_files']:
        verify(RUN/'pe_integration'/item['path'], item['sha256'], 'ACTUAL_REUSED_PE_OUTPUT_OR_EVIDENCE')
    save_json(RUN/'FINAL_PE_READONLY_AUDIT.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'all_unchanged': not errors, 'checked_files': len(records), 'frozen_artifact_count': len(baseline['files']),
              'records': records, 'errors': errors, 'PE_source_or_runtime_modified': False,
              'scope': 'Source/config/frozen artifacts and reused output bytes; not a new PE execution or certification'})
    if errors: raise RuntimeError('PE source/output drift requires investigation')
    print('PE_FINAL_READONLY_ALL_UNCHANGED', len(records), flush=True)


if __name__ == '__main__': run()
