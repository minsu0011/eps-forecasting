"""V2-only paths and durable scoped artifact I/O. No V1 writer imports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

PROJECT = Path(__file__).resolve().parents[2]
LAB = PROJECT/'research/eps_model_lab_v2'
V1 = PROJECT/'outputs/eps_model_lab_v1_20260908T012326_pit_r2'
RUN = Path(os.environ.get('EPS_V2_RUN_DIR', str(PROJECT/'outputs/eps_model_lab_v2_clean_20260908T055713Z'))).resolve()
if RUN.parent != (PROJECT/'outputs').resolve() or not RUN.name.startswith('eps_model_lab_v2_clean_'):
    raise RuntimeError('Explicit V2 output identity required')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RUN):
        raise RuntimeError(f'Writes must stay in V2 output: {path}')
    return path


def save_json(path, payload, immutable=False):
    path = require_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f'Immutable artifact already exists: {path}')
    tmp = path.with_suffix(path.suffix+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    for attempt in range(12):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.025*2**attempt, .5))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check_stop():
    if (RUN/'STOP').exists():
        raise RuntimeError('V2 stopped by explicit scoped signal')
