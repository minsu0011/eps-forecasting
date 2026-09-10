"""Three bounded CPU-only diagnostics, independent from the GPU resource owner."""
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha


def worker(base):
    state = RUN/'sampling_stability_queues'/f'{base}.json'
    if state.exists(): raise RuntimeError('No implicit repeated diagnostic')
    command = [str(PROJECT.parent/'.venv_eps_moirai_py312/Scripts/python.exe'), '-B',
               str(PROJECT/'research/eps_model_lab_v1/sampling_stability_audit.py'), base]
    log = RUN/'rerun_logs'/f'sampling_stability_{base}.log'
    record = {'status': 'RUNNING', 'command': command, 'start_utc': datetime.now(timezone.utc).isoformat()}
    with log.open('w', encoding='utf-8') as stream:
        process = subprocess.Popen(command, cwd=PROJECT, env=dict(os.environ, PYTHONUTF8='1'),
                                   stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        record['pid'] = process.pid; save_json(state, record); code = process.wait()
    record.update(status='PROCESS_FINISHED', exit_code=code, end_utc=datetime.now(timezone.utc).isoformat(),
                  log=str(log.relative_to(RUN)), log_sha256=sha(log))
    save_json(state, record)


if __name__ == '__main__':
    if datetime.now(timezone.utc) >= datetime.fromisoformat('2026-09-08T00:00:00+00:00'):
        raise RuntimeError('Insufficient diagnostic + finalization headroom')
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker, ['lag_llama_zero_shot', 'moirai1p1_small', 'moirai_moe_small']))
