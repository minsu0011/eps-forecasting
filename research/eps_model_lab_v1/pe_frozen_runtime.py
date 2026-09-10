"""EPS-owned read-only invocation of exact frozen v04 and C4-R2 code.

Only real-market research inputs produced by pe_real_inputs are opened. No
heldout, reserve generation, formal evaluator, promotion, or PE registry write.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
DEST=RUN/'pe_integration';ARCHIVE=DEST/'v04_exact_archive'
sys.path.insert(0,str(ARCHIVE/'src'))
import numpy as np
import pandas as pd


def binding():
    receipt=json.loads((DEST/'READONLY_PE_SOURCE_BINDING.json').read_text(encoding='utf-8'))
    for r in receipt['extracted_v04']:
        if sha(ARCHIVE/r['path'])!=r['sha256']: raise RuntimeError('Exact archived v04 source drift')
    for r in receipt['live_C4_verified']:
        if sha(PROJECT/r['path'])!=r['sha256']: raise RuntimeError('Frozen live C4 source drift')
    import pe_regime_v04.pipeline as p
    if ARCHIVE.resolve() not in Path(p.__file__).resolve().parents: raise RuntimeError('Wrong v04 import binding')
    return receipt


def v04(ticker):
    from pe_regime_v04.config import load_config
    from pe_regime_v04.pipeline import apply_v04_layers
    outpath=DEST/'v04'/f'{ticker}.parquet'
    if outpath.exists():
        r=json.loads(outpath.with_suffix('.json').read_text(encoding='utf-8'))
        if sha(outpath)!=r['sha256']: raise RuntimeError('v04 research output hash drift')
        return pd.read_parquet(outpath)
    inputpath=DEST/'canonical'/f'{ticker}.parquet';frame=pd.read_parquet(inputpath)
    cfg=load_config(ARCHIVE/'config/v04_bottleneck.yaml');execution=copy.deepcopy(cfg)
    # Execution only, supported public option; all model/fold/seed parameters fixed.
    execution['expected_pe']['outer_n_jobs']=20
    began=time.perf_counter();result,diagnostics=apply_v04_layers(frame,execution)
    outpath.parent.mkdir(exist_ok=True);result.to_parquet(outpath,index=False)
    save_json(outpath.with_suffix('.json'),{'ticker':ticker,'model_id':'v04_frozen_production_archive',
      'status':'RESEARCH_REAL_INPUT_EXECUTED','rows':len(result),'positive_finite_rows':int((np.isfinite(result.v04_expected_pe)&(result.v04_expected_pe>0)).sum()),
      'sha256':sha(outpath),'input_sha256':sha(inputpath),'config_sha256':sha(ARCHIVE/'config/v04_bottleneck.yaml'),
      'execution_overrides':{'expected_pe.outer_n_jobs':20},'runtime':sys.executable,
      'seconds':time.perf_counter()-began,'diagnostics':diagnostics,'formal_certification':False,'source_mutated':False})
    print('FROZEN_V04_REAL',ticker,'COMPLETE',round(time.perf_counter()-began,2),flush=True)
    return result


def c4(ticker,full):
    from research.model_zoo.pe_c4_r2_numerical_robustness_v1.runner import run_source_task
    from research.model_zoo.hierarchical_observable_fair_value_state_v7.dgp_r4 import R4_CANONICAL_COLUMNS
    outpath=DEST/'c4'/f'{ticker}.parquet'
    if outpath.exists(): return
    # Calendar-contiguous latest 1800 rows, never filter by positive earnings.
    # The production C4 gate must reject incompatible/nonfinite domains itself.
    block=full.iloc[-1800:].reset_index(drop=True)
    canonical=block[list(R4_CANONICAL_COLUMNS)].copy()
    overlay=block[['date','symbol','v04_expected_pe']].copy()
    for frame in [canonical,overlay]: frame['date']=pd.to_datetime(frame.date).dt.strftime('%Y-%m-%d')
    cp=DEST/'c4_inputs'/f'{ticker}_canonical.csv';op=cp.with_name(f'{ticker}_v04.csv');cp.parent.mkdir(exist_ok=True)
    canonical.to_csv(cp,index=False,float_format='%.17g');overlay.to_csv(op,index=False,float_format='%.17g')
    result=run_source_task(cp.read_bytes(),op.read_bytes(),variant_id='c4_r2_e_irls80_block_v04',
      task_label='EPS_REAL_MARKET_RESEARCH_'+ticker,seed_alias='REAL_MARKET_NO_SYNTHETIC_SEED',
      dgp_id='REAL_MARKET_NOT_DGP',spent_task=None,fold_indices=None)
    rows=result.pop('rows');outpath.parent.mkdir(exist_ok=True);pd.DataFrame(rows).to_parquet(outpath,index=False)
    save_json(outpath.with_suffix('.json'),{**result,'model_id':'c4_r2_hofs_irls80_block_v04_w0500',
       'ticker':ticker,'sha256':sha(outpath),'canonical_sha256':sha(cp),'v04_overlay_sha256':sha(op),
       'runtime':sys.executable,'formal_certification':False,'source_mutated':False,
       'input_scope':'Real market research only; generic frozen fold-plan constructor, no qualification reserve/API used'})
    print('FROZEN_C4_REAL',ticker,'COMPLETE',len(rows),'FALLBACK_FOLDS',result['fallback_fold_count'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('tickers',nargs='*');args=parser.parse_args()
    binding()
    for ticker in args.tickers or ['AAPL','MSFT','JPM','KO','CVX']:
        try:
            full=v04(ticker)
        except Exception as exc:
            save_json(DEST/'runtime_failures'/f'{ticker}_v04.json',{'ticker':ticker,'status':'BROKEN_OR_DATA_BLOCKED','reason':str(exc),'traceback':traceback.format_exc()})
            print(ticker,'V04_FAILED_CLOSED',str(exc),flush=True);continue
        try:
            import subprocess
            subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('c4_isolated_runtime.py')),ticker],
                check=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        except Exception as exc:
            save_json(DEST/'runtime_failures'/f'{ticker}_c4.json',{'ticker':ticker,'status':'BROKEN_OR_DATA_BLOCKED','reason':str(exc),'traceback':traceback.format_exc()})
            print(ticker,'C4_FAILED_CLOSED',str(exc),flush=True)
    binding()
