"""Resolve shell stderr presentation separately from actual all-annual checks."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha


def run():
    path = RUN/'ALL_ANNUAL_ORIGIN_INDEPENDENCE_AUDIT.json'
    audit = json.loads(path.read_text(encoding='utf-8'))
    plan = json.loads((RUN/'ALL_ANNUAL_ORIGIN_INDEPENDENCE_PLAN.json').read_text(encoding='utf-8'))
    expected = {(p.parent.name, int(p.name)) for p in (RUN/'neural_fitted').glob('*/*') if p.is_dir() and p.name.isdigit()}
    expected |= {('epspredict_'+p.parent.name, int(p.stem)) for p in (RUN/'native_fitted').glob('*/*.pt')}
    expected |= {(p.parent.name, int(p.stem)) for p in (RUN/'multivariate_fitted').glob('*/*.pt')}
    actual = {(r['model_id'], r['year']) for r in audit['records']}
    assert actual == expected and len(actual) == plan['annual_artifacts'] == audit['completed_annual_artifacts'] == 280
    assert len(audit['records']) == audit['probe_records'] == 840
    assert all(r['all_pass'] and all(c['pass'] for c in r['checks'].values()) for r in audit['records'])
    assert plan['dataset_sha256'] == sha(RUN/'data/samples.parquet')
    log = RUN/'rerun_logs/all_annual_origin_independence.log'
    raw = log.read_bytes(); text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
    assert 'Traceback (most recent call last)' not in text
    assert 'ALL_ANNUAL_INPUT_AUDIT nf_multivar_XLinear 2026 840' in text
    assert 'NativeCommandError' in text and 'Seed set to 1729' in text
    normalized = RUN/'rerun_logs/all_annual_origin_independence_utf8.log'
    normalized.write_text(text, encoding='utf-8')
    save_json(RUN/'ALL_ANNUAL_AUDIT_COMPLETION_REVIEW.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
              'artifact_completion_pass': True, 'annual_artifacts_verified': len(actual), 'explicit_probe_passes': len(audit['records']),
              'reported_shell_exit_code': 1, 'native_python_exit_code_not_independently_captured': True,
              'log_contains_python_traceback': False, 'log_has_all_planned_completion_records': True,
              'reason': 'PowerShell represented Lightning informational stderr Seed set to1729 as NativeCommandError. The saved completed checks, not the shell code, establish diagnostic results.',
              'original_log_sha256': sha(log), 'normalized_utf8_log_sha256': sha(normalized), 'audit_sha256': sha(path)})
    print('ANNUAL_ARTIFACT_COMPLETION_VERIFIED_280_840', flush=True)


if __name__ == '__main__': run()
