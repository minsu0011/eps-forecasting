"""Document exact isolated optional-compiler patch and verify unchanged mLSTM code."""
from datetime import datetime,timezone
import difflib
import json
from pathlib import Path
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha


def run():
    root=RUN/'late_frontier/environment';envdir=PROJECT.parent/'.venv_eps_frontier_py312'
    original=envdir/'Lib/site-packages/xlstm';vendor=PROJECT.parent/'.eps_xlstm_frontier_vendor/xlstm'
    records=[];changes=[]
    for source in sorted(original.rglob('*')):
        if not source.is_file() or '__pycache__' in source.parts:continue
        rel=source.relative_to(original);copy=vendor/rel
        if not copy.is_file():raise RuntimeError('Incomplete isolated package copy '+str(rel))
        same=sha(source)==sha(copy)
        records.append({'path':rel.as_posix(),'original_sha256':sha(source),'copied_sha256':sha(copy),'unchanged':same})
        if not same:changes.append(rel.as_posix())
    if changes!=['blocks/slstm/src/cuda_init.py']:raise RuntimeError('Unexpected scientific source change '+str(changes))
    rel=changes[0];before=(original/rel).read_text(encoding='utf-8');after=(vendor/rel).read_text(encoding='utf-8')
    patch=''.join(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='original/xlstm/'+rel,tofile='isolated_vendor/xlstm/'+rel))
    (root/'XLSTM_OPTIONAL_COMPILER_PATCH.txt').write_text(patch,encoding='utf-8')
    python=envdir/'Scripts/python.exe'
    command=[str(python),'-B','-c',f"import sys;sys.path.insert(0,{str(vendor.parent)!r});import torch;import neuralforecast.models.xlstm as m;assert m.IS_XLSTM_INSTALLED;assert torch.cuda.is_available();print(torch.__version__,m.IS_XLSTM_INSTALLED)"]
    checked=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=90)
    (root/'isolated_import_recovery.log').write_text(checked.stdout+checked.stderr,encoding='utf-8')
    if checked.returncode:raise RuntimeError('Isolated import remains broken')
    frozen=subprocess.run(['uv','pip','freeze','--python',str(python)],capture_output=True,text=True,check=True).stdout
    (root/'requirements_lock.txt').write_text(frozen,encoding='utf-8')
    base=RUN/'environments/gpu_requirements_lock.txt'
    expected={s.lower().strip() for s in base.read_text(encoding='utf-8').splitlines() if s.strip()}
    observed={s.lower().strip() for s in frozen.splitlines() if s.strip()}
    if expected-observed:raise RuntimeError('Original base package version changed')
    save_json(root/'RECOVERY.json',{'status':'PASS_ISOLATED_IMPORT_ONLY_PATCH','created_utc':datetime.now(timezone.utc).isoformat(),
        'original_installation_failure_preserved':'INSTALLATION.json','environment':str(envdir),
        'base_packages_unchanged':True,'extra_packages':sorted(observed-expected),'files':records,
        'exact_changed_paths':changes,'mLSTM_forward_training_math_changed':False,'sLSTM_executed':False,
        'reason':'Unused sLSTM import eagerly asks for an absent CUDA development toolkit; defer discovery until compiled sLSTM load is actually requested',
        'patch':'XLSTM_OPTIONAL_COMPILER_PATCH.txt','import_probe_exit_code':checked.returncode,
        'reproduction':'Install exact requirements_lock.txt. Copy installed xlstm to an isolated vendor directory, apply the recorded minimal patch, prepend only that directory for this experiment. Do not alter the installed original or claim stock-package import compatibility.',
        'existing_seven_training_environments_modified':False,'full_upstream_source_redistributed':False})
    print('XLSTM_ISOLATED_ENVIRONMENT_RECOVERY_PASS',len(records),'source files, one optional initialization patch',flush=True)


if __name__=='__main__':run()
