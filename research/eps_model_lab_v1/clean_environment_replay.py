"""Actually install a new pinned evaluator environment without modifying existing envs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha


def run():
    envdir = (PROJECT.parent/'.venv_eps_handoff_validation_py312').resolve()
    if envdir.parent != PROJECT.parent.resolve() or envdir.exists(): raise RuntimeError('Expected new specifically named EPS environment')
    if shutil.disk_usage(PROJECT).free < 40*1024**3: raise RuntimeError('Disk headroom below guard')
    uv = shutil.which('uv')
    if not uv: raise RuntimeError('Existing uv executable missing')
    lock = RUN/'environments/core_requirements_lock.txt'; digest = sha(lock)
    receipt = RUN/'CLEAN_ENV_REPLAY_QUEUE.json'
    state = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(), 'environment': str(envdir),
        'original_core_lock_sha256': digest, 'existing_environments_modified': False,
        'purpose': 'Verification-only eighth environment, not another training ecosystem', 'steps': []}
    save_json(receipt, state)
    python = envdir/'Scripts/python.exe'
    commands = [[uv, 'venv', str(envdir), '--python', sys.executable],
                [uv, 'pip', 'sync', '--python', str(python), str(lock)],
                [uv, 'pip', 'check', '--python', str(python)],
                [sys.executable, '-B', str(PROJECT/'research/eps_model_lab_v1/portable_preflight.py'),
                 '--python', str(python), '--clean-environment']]
    for index, command in enumerate(commands):
        log = RUN/'rerun_logs'/f'clean_env_replay_{index}.log'
        start = datetime.now(timezone.utc).isoformat()
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=dict(os.environ, PYTHONUTF8='1'),
                stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            state['running_pid'] = process.pid; save_json(receipt, state)
            try: code = process.wait(timeout=1500)
            except subprocess.TimeoutExpired:
                process.terminate(); code = process.wait(timeout=30)
                state['bounded_timeout'] = True
        step = {'command': command, 'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(),
                'exit_code': code, 'log': str(log.relative_to(RUN)), 'log_sha256': sha(log)}
        state['steps'].append(step); save_json(receipt, state)
        if code or state.get('bounded_timeout'):
            state['status'] = 'FAILED_REVIEW_REQUIRED'; save_json(receipt, state); raise RuntimeError('Clean environment step failed')
    installed = subprocess.run([uv, 'pip', 'freeze', '--python', str(python)], capture_output=True, text=True, encoding='utf-8', check=True)
    observed = {s.strip().lower() for s in installed.stdout.splitlines() if s.strip()}
    expected = {s.strip().lower() for s in lock.read_text(encoding='utf-8').splitlines() if s.strip()}
    portable = json.loads((RUN/'CLEAN_ENV_PORTABLE_EVALUATOR_PREFLIGHT.json').read_text(encoding='utf-8'))
    passed = expected == observed and portable['all_pass'] and sha(lock) == digest
    state.update(status='PASS_FRESH_PINNED_ENVIRONMENT' if passed else 'INSTALLED_LOCK_MISMATCH', running_pid=None,
        expected_packages=len(expected), installed_packages=len(observed), missing=sorted(expected-observed), extra=sorted(observed-expected),
        portable_evaluator_pass=portable['all_pass'], completed_utc=datetime.now(timezone.utc).isoformat(),
        same_machine=True, fresh_environment=True, original_lock_unchanged=sha(lock) == digest)
    save_json(receipt, state)
    print('CLEAN_PINNED_ENVIRONMENT_REPLAY', passed, 'packages', len(observed), flush=True)
    if not passed: raise RuntimeError('Fresh environment reproduction not fully passed')


if __name__ == '__main__': run()
