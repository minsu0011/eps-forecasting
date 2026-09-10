"""Copy only compact-package inputs and actually run the portable evaluator."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.handoff_package import selection


def run(python=None, clean_environment=False, late_review=False):
    executable = str(Path(python).resolve()) if python else sys.executable
    if not Path(executable).is_file(): raise RuntimeError('Named evaluator Python missing')
    prefix = ('LATE_REVIEW_' if late_review else '') + ('CLEAN_ENV_' if clean_environment else '')
    if late_review and (RUN/(prefix+'PORTABLE_EVALUATOR_PREFLIGHT.json')).exists():
        raise RuntimeError('Late preflight receipt already exists; preserve evidence instead of overwriting logs')
    files, _ = selection()
    # The project name plus a nested output identity can exceed Windows MAX_PATH.
    # Use a short, specifically named sibling inside the same EPS workspace;
    # do not change global Windows long-path settings.
    stage = Path(tempfile.mkdtemp(prefix='eps_pf_', dir=PROJECT.parent)).resolve()
    if stage.parent != PROJECT.parent.resolve() or not stage.name.startswith('eps_pf_'):
        raise RuntimeError('Stage outside named EPS workspace')
    for relative, source in files.items():
        target = stage/relative
        if not target.resolve().is_relative_to(stage): raise RuntimeError('Package path traversal')
        target.parent.mkdir(parents=True, exist_ok=True)
        src = '\\\\?\\'+str(source.resolve()) if os.name == 'nt' else str(source)
        dst = '\\\\?\\'+str(target.resolve()) if os.name == 'nt' else str(target)
        shutil.copy2(src, dst)
    stage_run = stage/'outputs'/RUN.name
    if (stage_run/'data/raw/sec').exists(): raise RuntimeError('Unexpected full SEC data in compact stage')
    before = {p.name: sha(p) for p in (stage_run/'predictions').glob('*.parquet')}
    commands = [
        [executable, '-B', 'research/eps_model_lab_v1/result_audit.py'],
        [executable, '-B', 'research/eps_model_lab_v1/ensemble_replay.py'],
        [executable, '-B', '-m', 'pytest', 'research/eps_model_lab_v1/test_meta_contracts.py',
         'research/eps_model_lab_v1/test_native_crps.py', 'research/eps_model_lab_v1/test_frozen_frame_guard.py', '-q']]
    if late_review:
        commands += [[executable,'-B','research/eps_model_lab_v1/late_frontier_review.py'],
            [executable,'-B','research/eps_model_lab_v1/training_seed_review.py'],
            [executable,'-B','research/eps_model_lab_v1/eps_only_channel_review.py'],
            [executable,'-B','-m','pytest','research/eps_model_lab_v1/test_next_wave_context_selector.py',
             'research/eps_model_lab_v1/test_accounting_context_audit.py','-q']]
    records = []; env = dict(os.environ, PYTHONUTF8='1')
    env.pop('EPS_LAB_RUN_DIR', None)  # Never write the original run from the staged source.
    for i, command in enumerate(commands):
        log = RUN/'rerun_logs'/f'{prefix.lower()}portable_preflight_{i}.log'
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=stage, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW, timeout=600)
        records.append({'command': command, 'exit_code': result.returncode, 'log': str(log.relative_to(RUN)), 'log_sha256': sha(log)})
    after = {p.name: sha(p) for p in (stage_run/'predictions').glob('*.parquet')}
    audit = json.loads((stage_run/'EPS_RESULT_INTEGRITY_AUDIT.json').read_text(encoding='utf-8'))
    replay = json.loads((stage_run/'ENSEMBLE_FRESH_PROCESS_REPLAY.json').read_text(encoding='utf-8'))
    passed = all(r['exit_code'] == 0 for r in records) and before == after and audit['all_pass'] and replay['all_pass']
    save_json(RUN/(prefix+'PORTABLE_EVALUATOR_PREFLIGHT.json'), {'created_utc': datetime.now(timezone.utc).isoformat(),
              'all_pass': passed, 'stage_path': str(stage), 'files_copied': len(files), 'commands': records,
              'predictions_unchanged': before == after, 'prediction_hashes': before,
              'samples_sha256': sha(stage_run/'data/samples.parquet'), 'model_integrity_count': audit['model_count'],
              'raw_SEC_and_full_weights_present': False,
              'scope': ('Actual isolated-directory evaluation/replay with fresh pinned core environment on this same Windows machine; not cross-machine validation'
                        if clean_environment else 'Actual isolated-directory evaluation/replay with current pinned core environment; not a fresh dependency installation on another machine'),
              'evaluator_python': executable,
              'stage_retained_for_inspection': True})
    print('PORTABLE_EVALUATOR_PREFLIGHT', passed, str(stage), flush=True)
    if not passed: raise RuntimeError('Portable preflight failure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--python'); parser.add_argument('--clean-environment', action='store_true')
    parser.add_argument('--late-review',action='store_true')
    args = parser.parse_args(); run(args.python, args.clean_environment,args.late_review)
