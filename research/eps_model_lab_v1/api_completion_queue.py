"""Single-GPU ownership after TabPFN replay; journal actual adapter checks."""
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


def run():
    state = RUN/'API_COMPLETION_QUEUE.json'
    if state.exists(): raise RuntimeError('Explicit resume required; queue already exists')
    dependency = RUN/'TABPFN_API_RECOVERY_QUEUE.json'
    save_json(state, {'status': 'WAITING_FOR_GPU_OWNER', 'steps': []})
    while not dependency.exists() or json.loads(dependency.read_text(encoding='utf-8'))['status'] != 'COMPLETE':
        if datetime.now(timezone.utc) >= datetime.fromisoformat('2026-09-08T01:38:26+00:00'):
            raise RuntimeError('No new long jobs in finalization window')
        time.sleep(10)
    replay = json.loads((RUN/'additional_replay/tabpfn_v2_panel.json').read_text(encoding='utf-8'))
    if not replay.get('all_pass') or len(replay.get('records', [])) != 40:
        raise RuntimeError('TabPFN actual replay incomplete or failed')
    records = []
    for env, action in [('extra', 'tabpfn'), ('gpu', 'joint')]:
        command = [str(PROJECT.parent/f'.venv_eps_{env}_py312/Scripts/python.exe'), '-B',
                   str(PROJECT/'research/eps_model_lab_v1/api_completion_audit.py'), action]
        log = RUN/'rerun_logs'/f'api_completion_{action}.log'
        record = {'action': action, 'command': command, 'start_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING'}
        records.append(record)
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=dict(os.environ, PYTHONUTF8='1'), stdout=stream,
                                       stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            record['pid'] = process.pid
            save_json(state, {'status': 'RUNNING', 'steps': records})
            code = process.wait()
        record.update(status='PROCESS_FINISHED', exit_code=code, end_utc=datetime.now(timezone.utc).isoformat(),
                      log=str(log.relative_to(RUN)), log_sha256=sha(log))
        save_json(state, {'status': 'RUNNING', 'steps': records})
    save_json(state, {'status': 'COMPLETE', 'steps': records})


if __name__ == '__main__': run()
