"""Sequential bounded CPU profile diagnostics for two already frozen models."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha


def run():
    path = RUN/'MOIRAI_CPU_THROUGHPUT_QUEUE.json'
    if path.exists(): raise RuntimeError('No duplicate diagnostic queue')
    steps = []
    for base in ['moirai1p1_small', 'moirai_moe_small']:
        command = [str(PROJECT.parent/'.venv_eps_moirai_py312/Scripts/python.exe'), '-B',
                   str(PROJECT/'research/eps_model_lab_v1/moirai_cpu_throughput.py'), base]
        log = RUN/'rerun_logs'/f'cpu_throughput_{base}.log'
        step = {'model': base, 'start_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RUNNING', 'command': command}
        steps.append(step)
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=dict(os.environ, PYTHONUTF8='1'),
                stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            step['pid'] = process.pid; save_json(path, {'status': 'RUNNING', 'steps': steps}); code = process.wait()
        step.update(status='COMPLETE' if code == 0 else 'FAILED_REVIEW_REQUIRED', exit_code=code,
            log=str(log.relative_to(RUN)), log_sha256=sha(log), end_utc=datetime.now(timezone.utc).isoformat())
        save_json(path, {'status': 'RUNNING', 'steps': steps})
    save_json(path, {'status': 'COMPLETE' if all(s['exit_code'] == 0 for s in steps) else 'COMPLETE_WITH_DIAGNOSTIC_FAILURE', 'steps': steps})


if __name__ == '__main__': run()
