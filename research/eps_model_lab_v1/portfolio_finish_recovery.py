"""Explicit post-failure recovery: no re-training, selection or freeze overwrite."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha


def run():
    receipt = RUN/'PORTFOLIO_FINISH_RECOVERY.json'
    if receipt.exists(): raise RuntimeError('Explicit recovery already recorded; inspect before any further action')
    original = RUN/'PORTFOLIO_FINISH_QUEUE.json'
    failed = json.loads(original.read_text(encoding='utf-8'))
    if failed['status'] != 'FAILED_REVIEW_REQUIRED' or failed['steps'][-1]['script'] != 'portable_preflight.py':
        raise RuntimeError('Recovery target is not the diagnosed portable-preflight failure')
    preflight = json.loads((RUN/'PORTABLE_EVALUATOR_PREFLIGHT.json').read_text(encoding='utf-8'))
    if not preflight['all_pass']: raise RuntimeError('Corrected portable execution has not passed')
    immutable = [RUN/'ENSEMBLE_FREEZE_V1.json', RUN/'EPS_RESEARCH_SHORTLIST_V1.json',
                 RUN/'NESTED_META_DIAGNOSTIC_FREEZE.json', RUN/'nested_meta_diagnostic/predictions.parquet',
                 *sorted((RUN/'predictions').glob('*.parquet'))]
    hashes = {str(p.relative_to(RUN)): sha(p) for p in immutable}
    report = {'start_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING',
        'original_failed_queue_sha256': sha(original), 'original_failed_queue_preserved': True,
        'diagnosed_failure': 'Windows MAX_PATH during staged copy; short workspace-local stage and extended copy paths passed',
        'portable_recovery_evidence': 'PORTABLE_EVALUATOR_PREFLIGHT.json',
        'additional_report_failure': 'final_reports.py assumed a dict for all smoke receipts; 11 actual multi-head lists now require every explicit record to PASS',
        'static_combo_executed_separately': 'EPS_PE_COMBINATION_RECEIPT.json',
        'scientific_immutable_hashes_before': hashes, 'steps': []}
    save_json(receipt, report)
    steps = [('resource_summary.py', []), ('operations.py', ['registry']), ('failure_report.py', []),
             ('PYTEST', ['-m', 'pytest', 'research/eps_model_lab_v1', '-q', '--junitxml='+str(RUN/'FINAL_CONTRACT_TESTS.xml')]),
             ('final_verification.py', []), ('final_reports.py', [])]
    for index, (script, args) in enumerate(steps):
        cmd = [sys.executable, '-B', *args] if script == 'PYTEST' else [sys.executable, '-B', str(PROJECT/'research/eps_model_lab_v1'/script), *args]
        log = RUN/'rerun_logs'/f'portfolio_recovery_{index:02d}_{Path(script).stem}.log'
        step = {'script': script, 'command': cmd, 'start_utc': datetime.now(timezone.utc).isoformat()}
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(cmd, cwd=PROJECT, env=dict(os.environ, PYTHONUTF8='1'), stdout=stream,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            step['pid'] = process.pid; code = process.wait()
        step.update(exit_code=code, end_utc=datetime.now(timezone.utc).isoformat(), log=str(log.relative_to(RUN)), log_sha256=sha(log))
        report['steps'].append(step); save_json(receipt, report)
        if code:
            report['status'] = 'FAILED_REVIEW_REQUIRED'; save_json(receipt, report)
            raise RuntimeError('Recovery step failed: '+script)
    unchanged = all(sha(RUN/path) == digest for path, digest in hashes.items()) and sha(original) == report['original_failed_queue_sha256']
    report.update(status='COMPLETE_PREPACKAGE' if unchanged else 'IMMUTABLE_DRIFT_FAILURE',
        completed_utc=datetime.now(timezone.utc).isoformat(), all_scientific_outputs_unchanged=unchanged)
    save_json(receipt, report)
    if not unchanged: raise RuntimeError('Scientific artifact drift during report recovery')
    state = json.loads((RUN/'RUN_STATE.json').read_text(encoding='utf-8'))
    state.update(updated_utc=datetime.now(timezone.utc).isoformat(), status='PREPACKAGE_SCIENTIFIC_REVIEW',
        completion_recovery='PORTFOLIO_FINISH_RECOVERY.json', running_model=None,
        next_actions=['Review final evidence and outcome interpretation without retuning',
                      'Final source/PE/hash verification and compact desktop package', 'Close the user ten-hour run explicitly'])
    save_json(RUN/'RUN_STATE.json', state)
    print('PORTFOLIO_RECOVERY_COMPLETE_ALL_FROZEN_OUTPUTS_UNCHANGED', flush=True)


if __name__ == '__main__': run()
