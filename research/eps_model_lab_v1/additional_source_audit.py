"""Record inspected remote-code surface before Sundial execution."""
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha

if __name__=='__main__':
    receipt=json.loads((RUN/'weight_receipts/thuml__sundial-base-128m.json').read_text(encoding='utf-8'))
    folder=Path(receipt['path']);sources={};imports={}
    allowed={'typing','torch','transformers','math','warnings','configuration_sundial','ts_generation_mixin','flow_loss'}
    for path in sorted(folder.glob('*.py')):
        tree=ast.parse(path.read_text(encoding='utf-8'));roots=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):roots.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom):roots.add(node.module.split('.')[0])
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ['eval','exec','open','__import__']:
                raise RuntimeError('Unexpected dynamic/system call')
        if not roots<=allowed:raise RuntimeError('Unexpected source import '+str(roots-allowed))
        sources[path.name]=sha(path);imports[path.name]=sorted(roots)
    if set(sources)!={'configuration_sundial.py','modeling_sundial.py','flow_loss.py','ts_generation_mixin.py'}:raise RuntimeError('Unexpected source set')
    tab=json.loads((RUN/'weight_receipts/Prior-Labs__TabPFN-v2-reg.json').read_text(encoding='utf-8'))
    checkpoint=Path(tab['path'])/'tabpfn-v2-regressor-v2_default.ckpt'
    unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    if unsafe:raise RuntimeError('Unexpected pickle globals')
    save_json(RUN/'ADDITIONAL_SOURCE_EXECUTION_AUDIT.json',{'inspected_utc':datetime.now(timezone.utc).isoformat(),
       'sundial_python_sha256':sources,'imports':imports,'sundial_revision':receipt['revision'],
       'manual_review':'All four pinned Python files read. Transformer causal temporal attention, per-row RevIN, TimeFlow sampling. No system/network/file execution hooks identified. This is not a formal security proof.',
       'sundial_missingness':'Native patch embedding masks padding only; input missing values are filled causally by wrapper and disclosed, not hidden as observed training labels.',
       'sundial_execution':'Only inspected local_files_only snapshot, unchanged upstream source',
       'tabpfn_checkpoint_sha256':sha(checkpoint),'tabpfn_unsafe_pickle_globals':unsafe,
       'tabpfn_installed_source_review':'v2 test queries attend only to train positions; preprocessing reuses train statistics. Native prediction mutation tested in smoke.',
       'licenses':{'sundial':'Apache-2.0','TabPFN_v2':'PriorLabs-1.1; not latest gated v3'},
       'cloud_inference':False,'telemetry_disabled':True,'strict_historical_pretraining_certified':False})
    print('ADDITIONAL_SOURCE_REVIEW_RECORDED',len(sources),'PYTHON_FILES',flush=True)
