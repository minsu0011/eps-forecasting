"""Reuse actual frozen PE outputs only when every consumed input is unchanged."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,ORIGINAL_RUN,save_json,sha


def run():
    target=RUN/'pe_integration'
    if target.exists(): raise RuntimeError('PE reuse destination already exists; inspect instead of overwrite')
    old=pd.read_parquet(ORIGINAL_RUN/'data/fiscal_panel.parquet')
    new=pd.read_parquet(RUN/'data/fiscal_panel.parquet')
    columns=['accepted','effective','end','ttm','ttm_method','timestamp_exact','ttm_approximate','accn','fiscal_quarter']
    records=[]
    for ticker in ['AAPL','MSFT','JPM','KO','CVX']:
        a=old.loc[(old.ticker==ticker)&old.ttm.notna(),columns].sort_values('accn').reset_index(drop=True)
        b=new.loc[(new.ticker==ticker)&new.ttm.notna(),columns].sort_values('accn').reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b,check_exact=True,check_dtype=False)
        records.append({'ticker':ticker,'consumed_native_events':len(a),'event_fields':columns,'status':'EXACT_EQUAL'})
    for ticker in ['AAPL','MSFT','JPM','KO','CVX','QQQ']:
        rel=Path('data/raw/prices')/f'{ticker}.parquet'
        if sha(ORIGINAL_RUN/rel)!=sha(RUN/rel): raise RuntimeError('Raw price changed '+ticker)
        records.append({'ticker':ticker,'raw_price_sha256':sha(RUN/rel),'status':'BYTE_IDENTICAL'})
    binding=json.loads((ORIGINAL_RUN/'pe_integration/READONLY_PE_SOURCE_BINDING.json').read_text(encoding='utf-8'))
    for item in binding['live_C4_verified']:
        if sha(PROJECT/item['path'])!=item['sha256']: raise RuntimeError('Frozen C4 source drift')
    upstream=PROJECT.parent/'PE_Regime_Engine_v0.2.0'
    for relative,digest in binding['upstream_input_engine'].items():
        if sha(upstream/relative)!=digest: raise RuntimeError('Upstream input engine drift')
    if sha(upstream/'config/default.yaml')!=binding['input_config_sha256']: raise RuntimeError('Input configuration drift')
    baseline=json.loads((RUN/'PE_READONLY_BASELINE.json').read_text(encoding='utf-8'))
    for item in baseline['files']:
        if sha(Path(baseline['root'])/item['path'])!=item['sha256']: raise RuntimeError('Frozen PE artifact drift '+item['path'])
    shutil.copytree(ORIGINAL_RUN/'pe_integration',target)
    files=[{'path':str(p.relative_to(target)),'sha256':sha(p)} for p in target.rglob('*') if p.is_file()]
    save_json(RUN/'PE_INPUT_EQUIVALENCE_REUSE_AUDIT.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
       'status':'PASS_EXACT_INPUT_EQUIVALENCE','original_run':str(ORIGINAL_RUN),'new_run':str(RUN),
       'events_and_prices':records,'original_samples_retired_but_consumed_PE_events_unchanged':True,
       'frozen_PE_baseline_files_verified':len(baseline['files']),'copied_files':files,
       'scope':'Reuse of actual v04 outputs and actual C4 rejection evidence under identical consumed data and source, not a new PE fit or certification',
       'eps_prediction_combinations':'Must be regenerated from new-run EPS predictions and new OOF member freeze'})
    print('PE_INPUT_EQUIVALENCE_PASS',len(records),'FROZEN_FILES',len(baseline['files']),'COPIED',len(files),flush=True)


if __name__=='__main__':run()
