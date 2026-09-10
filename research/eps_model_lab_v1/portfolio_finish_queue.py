"""Finalize scientific outputs only after all corrected sampler replays complete."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha


def ready():
    for base in ['lag_llama_zero_shot', 'moirai1p1_small', 'moirai_moe_small']:
        path = RUN/'origin_isolation_replay'/f'{base}_origin_isolated.json'
        if not path.exists(): return False
        data = json.loads(path.read_text(encoding='utf-8'))
        if not data.get('all_pass') or len(data.get('records', [])) != 5: return False
        path = RUN/'sampling_stability'/f'{base}_origin_isolated/COMPLETION.json'
        if not path.exists(): return False
        if json.loads(path.read_text(encoding='utf-8'))['status'] != 'PASS_DIAGNOSTIC_COMPLETE': return False
    return True


def run():
    state = RUN/'PORTFOLIO_FINISH_QUEUE.json'
    if state.exists(): raise RuntimeError('No implicit resume or duplicate freeze')
    save_json(state, {'status': 'WAITING_FOR_CORRECTED_REPLAY', 'steps': []})
    while not ready():
        if datetime.now(timezone.utc) >= datetime.fromisoformat('2026-09-08T01:38:26+00:00'):
            raise RuntimeError('Finalization window reached before required sampler evidence')
        time.sleep(10)
    steps = [('result_audit.py', []), ('ensemble_models.py', ['freeze']), ('ensemble_models.py', ['predict']),
             ('result_audit.py', []), ('ensemble_replay.py', []), ('portfolio_analysis.py', []),
             ('shortlist_evidence.py', []), ('static_pe_combo.py', []), ('nested_meta_diagnostic.py', []), ('annual_fresh_replay.py', []),
             ('evaluation_scope_audit.py', []), ('operations.py', ['registry']), ('failure_report.py', []),
             ('PYTEST', ['-m', 'pytest', 'research/eps_model_lab_v1', '-q', '--junitxml='+str(RUN/'FINAL_CONTRACT_TESTS.xml')]),
             ('portable_preflight.py', []), ('final_verification.py', []), ('final_reports.py', [])]
    records = []
    for index, (script, args) in enumerate(steps):
        command = [sys.executable, '-B', *args] if script == 'PYTEST' else [sys.executable, '-B', str(PROJECT/'research/eps_model_lab_v1'/script), *args]
        log = RUN/'rerun_logs'/f'portfolio_finish_{index:02d}_{Path(script).stem}.log'
        record = {'script': script, 'command': command, 'status': 'RUNNING', 'start_utc': datetime.now(timezone.utc).isoformat()}
        records.append(record)
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=dict(os.environ, PYTHONUTF8='1'),
                                       stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            record['pid'] = process.pid; save_json(state, {'status': 'RUNNING', 'steps': records})
            code = process.wait()
        record.update(exit_code=code, status='PROCESS_FINISHED' if code == 0 else 'FAILED_REVIEW_REQUIRED',
                      end_utc=datetime.now(timezone.utc).isoformat(), log=str(log.relative_to(RUN)), log_sha256=sha(log))
        save_json(state, {'status': 'RUNNING' if code == 0 else 'FAILED_REVIEW_REQUIRED', 'steps': records})
        if code != 0: raise RuntimeError('Portfolio completion step failed; inspect before explicit recovery: '+script)
    save_json(state, {'status': 'COMPLETE_PREPACKAGE', 'steps': records,
              'note': 'No desktop package or ten-hour-run closure implied; remaining resource diagnostics and final handoff review still required'})


if __name__ == '__main__': run()
