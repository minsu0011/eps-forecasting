"""Preserve the interim checkpoint and honor the full requested ten-hour window."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json


def run():
    path = RUN/'FINAL_RUN_CLOSURE.json'
    checkpoint = json.loads(path.read_text(encoding='utf-8'))
    archive = RUN/'audit_corrections/interim_early_execution_checkpoint.json'
    if archive.exists(): raise RuntimeError('Interim checkpoint already preserved')
    archive.parent.mkdir(parents=True, exist_ok=True); archive.write_bytes(path.read_bytes())
    now = datetime.now(timezone.utc).isoformat()
    save_json(path, {'status': 'INTERIM_CHECKPOINT_SUPERSEDED_RUN_CONTINUES', 'updated_utc': now,
        'prior_checkpoint': str(archive.relative_to(RUN)), 'requested_hours': 10,
        'deadline_utc': '2026-09-08T02:23:26Z', 'actual_final_closure': False,
        'reason': 'The requested ten-hour execution window is not finished; do not confuse handoff readiness with run completion.'})
    state = json.loads((RUN/'RUN_STATE.json').read_text(encoding='utf-8'))
    state.update(status='TEN_HOUR_WINDOW_CONTINUES_POST_FREEZE_RESEARCH_DIAGNOSTICS', updated_utc=now,
        execution_closed_utc=None, resumed_utc=now, package_status='NOT_YET_FINAL_PACKAGED',
        next_actions=['Bounded additional official-architecture diagnostics on unchanged EPS geometry',
                      'Preserve frozen portfolio and expose all source-quality limitations',
                      'Finalize within requested ten-hour deadline, not an early scientific checkpoint'])
    save_json(RUN/'RUN_STATE.json', state)
    print('TEN_HOUR_WINDOW_CONTINUES', now, flush=True)


if __name__ == '__main__': run()
