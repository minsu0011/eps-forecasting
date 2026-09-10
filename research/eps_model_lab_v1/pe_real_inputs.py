"""Real-market input preparation using unchanged upstream PE functions.

This is a new EPS-owned input adapter, not a modification of frozen PE models.
The native first-filing ledger and retrospective corporate-action limitations
remain explicit. No certified market-feed or downstream price-performance claim.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
import zipfile
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
OLD=PROJECT.parent/'PE_Regime_Engine_v0.2.0';sys.path.insert(0,str(OLD/'src'))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from pe_regime_engine.config import load_config
from pe_regime_engine.eps import attach_pit_eps_to_prices,_consensus_details
from pe_regime_engine.features import build_market_features,select_feature_frame,attach_stock_features,merge_benchmark_features
from pe_regime_engine.regime import rule_regime_probabilities,walk_forward_latent_model,apply_sideways_gate_to_two_state_sjm,ensemble_regime_probabilities,walk_forward_regime_forecaster
from pe_regime_engine.valuation import add_soft_regime_pe_statistics,walk_forward_expected_pe_model,finalize_valuation

DEST=RUN/'pe_integration';TICKERS=['AAPL','MSFT','JPM','KO','CVX']


def inventory():
    frozen=RUN.parent/'c4_r2_final_freeze_20260907_run01'
    manifest=json.loads((frozen/'C4_R2_FINAL_SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    zinfo=manifest['champion_archive'];archive=frozen/zinfo['path']
    if sha(archive)!=zinfo['sha256']: raise RuntimeError('Frozen v04 archive hash drift')
    root=DEST/'v04_exact_archive';root.mkdir(parents=True,exist_ok=True)
    records=[]
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            parts=Path(entry.filename).parts
            if len(parts)<3 or parts[1] not in ['src','config']: continue
            relative=Path(*parts[1:]);target=(root/relative).resolve()
            if root.resolve() not in target.parents: raise RuntimeError('Unsafe archive member')
            if entry.is_dir(): continue
            raw=z.read(entry)
            if target.exists() and target.read_bytes()!=raw: raise RuntimeError('Extracted frozen source drift')
            if not target.exists(): target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            records.append({'path':str(relative),'sha256':sha(target)})
    live=[]
    for record in manifest['files']:
        p=PROJECT/record['path']
        if sha(p)!=record['sha256']: raise RuntimeError('C4 live source differs from frozen closure: '+record['path'])
        live.append({'path':record['path'],'sha256':record['sha256']})
    sources={str(p.relative_to(OLD)):sha(p) for p in (OLD/'src/pe_regime_engine').glob('*.py')}
    save_json(DEST/'READONLY_PE_SOURCE_BINDING.json',{'archive':zinfo,'extracted_v04':records,'live_C4_verified':live,
        'upstream_input_engine':sources,'input_config_sha256':sha(OLD/'config/default.yaml'),
        'note':'Exact frozen PE source; new first-native EPS data surface is RESEARCH_ONLY, not a certified vendor vintage.'})


def price(ticker):
    raw=pd.read_parquet(RUN/'data/raw/prices'/f'{ticker}.parquet')
    # pandas 3/Yahoo may retain millisecond indices; the frozen upstream
    # merge_asof contract uses nanosecond filing dates. Normalize only here.
    dates=raw.index.tz_localize(None).normalize().as_unit('ns')
    ratios=raw['Stock Splits'].where(raw['Stock Splits']>0,1.).to_numpy()
    subsequent=np.cumprod(ratios[::-1])[::-1]/ratios
    frame=pd.DataFrame({'date':dates,'symbol':ticker,'stock_split':raw['Stock Splits'].to_numpy(),
                        'volume':raw.Volume.to_numpy()})
    for name in ['open','high','low','close']: frame[name]=raw[name.title()].to_numpy()*subsequent
    return frame


def benchmark():
    path=DEST/'benchmark_regime.parquet'
    if path.exists():
        cached=pd.read_parquet(path)
        cached['date']=cached['date'].astype('datetime64[ns]')
        return cached
    c=load_config(OLD/'config/default.yaml');r=c['regime'];seed=int(c['project']['random_seed'])
    # Execution-only thread controls, statistical hyperparameters unchanged.
    b=build_market_features(price('QQQ'),price_basis='contemporaneous');indexed=b.set_index('date')
    x=select_feature_frame(b,r['feature_columns']);returns=indexed.return_1.reindex(x.index)
    models={'rule':rule_regime_probabilities(indexed,temperature=r['rule']['temperature'])};audit={}
    for name,kind,n,delta in [('sjm3','sjm',3,0),('sjm2','sjm',2,100000),('hmm3','hmm',3,200000)]:
        began=time.perf_counter()
        m=walk_forward_latent_model(X=x,returns=returns,model_kind=kind,model_config=r[kind],
          train_window=r['train_window'],min_train=r['min_train'],refit_every=r['refit_every'],
          context_window=r['context_window'],seed=seed+delta,n_components=n,fold_n_jobs=16,fold_backend='loky')
        audit[name]={'available':m.available,'error':m.error,'seconds':time.perf_counter()-began,'diagnostics':m.diagnostics}
        save_json(DEST/'REGIME_INPUT_AUDIT.json',audit)
        if m.available:
            models['sjm2_gate' if name=='sjm2' else name]=apply_sideways_gate_to_two_state_sjm(m.probabilities,indexed,r['sjm2_gate']) if name=='sjm2' else m.probabilities
        print('REAL_PE_REGIME',name,'COMPLETE',m.available,flush=True)
    ensemble=ensemble_regime_probabilities(models,benchmark_close=indexed.close.reindex(x.index),ensemble_config=r['ensemble'])
    forecast_cfg={**r['forecaster'],'n_jobs':1}
    forecast,diag=walk_forward_regime_forecaster(x,ensemble,forecast_cfg,seed=seed+300000)
    out=b.merge(ensemble.join(forecast).reset_index(),on='date',validate='one_to_one')
    out.to_parquet(path,index=False)
    save_json(DEST/'BENCHMARK_INPUT_RECEIPT.json',{'benchmark':'QQQ','rows':len(out),'sha256':sha(path),'config':c,
       'execution_overrides':{'latent_fold_n_jobs':16,'latent_backend':'loky','forecaster_n_jobs':1},'forecaster_diagnostics':diag})
    return out


def events(ticker):
    panel=pd.read_parquet(RUN/'data/fiscal_panel.parquet');p=panel[(panel.ticker==ticker)&panel.ttm.notna()].copy()
    rows=[]
    for r in p.itertuples():
        methods={r.ttm_method:float(r.ttm)}
        _,confidence,disagreement=_consensus_details(methods,bool(r.timestamp_exact),True)
        rows.append({'available_at':r.accepted,'effective_date':pd.Timestamp(r.effective),'filed_at':r.accepted,
          'period_end':pd.Timestamp(r.end),'eps_ttm_raw':r.ttm,'eps_method':'SEC_NATIVE_FIRST_FILING_LEDGER',
          'eps_primary_method':r.ttm_method,'eps_confidence':confidence,'eps_disagreement':disagreement,
          'timestamp_exact':bool(r.timestamp_exact),'eps_method_count':1,'eps_method_values':json.dumps(methods),
          'eps_approximation_flag':bool(r.ttm_approximate),'eps_reconstruction_approximate':bool(r.ttm_approximate),
          'eps_quarter_sources':'[]','accession':r.accn,'fiscal_period':'FY' if r.fiscal_quarter==4 else 'Q'+str(r.fiscal_quarter),
          'source_tag':'EarningsPerShareDiluted','eps_source_tag':'EarningsPerShareDiluted','shares_source_tag':None,
          'event_source':'EPS_LAB_NATIVE_LEDGER','eps_definition':'GAAP_DILUTED_TTM'})
    return pd.DataFrame(rows)


def canonical(ticker,b):
    path=DEST/'canonical'/f'{ticker}.parquet'
    if path.exists(): print(ticker,'EXISTING_CANONICAL_SKIP',flush=True);return
    c=load_config(OLD/'config/default.yaml');v=c['valuation'];start=time.perf_counter()
    ledger=events(ticker)
    if ledger.empty:
        save_json(DEST/'canonical'/f'{ticker}_receipt.json',{'ticker':ticker,'status':'DATA_BLOCKED',
          'reason':'No finite native first-filing diluted TTM events in frozen dataset; no substitute EPS definition allowed'})
        print('REAL_PE_CANONICAL',ticker,'DATA_BLOCKED_NO_TTM',flush=True);return
    stock=attach_pit_eps_to_prices(price(ticker),ledger,price_basis='contemporaneous')
    stock=attach_stock_features(stock)
    prob=[col for col in b if col.startswith(('p_','forecast_','weight_','loss_')) or col in ['market_regime','regime_entropy','regime_confidence']]
    base=b[[col for col in b if col not in prob]]
    stock=merge_benchmark_features(stock,base)
    stock=stock.merge(b[['date',*prob]],on='date',validate='one_to_one')
    stock=add_soft_regime_pe_statistics(stock,lookback=v['stats_lookback'],min_history=v['min_history'])
    cfg={**v['expected_pe_model'],'n_jobs':1,'fold_n_jobs':16}
    pe,diag,features=walk_forward_expected_pe_model(stock,cfg,seed=int(c['project']['random_seed'])+400000)
    stock['ml_expected_pe']=pe;stock=finalize_valuation(stock,v)
    stock=stock[stock.date>=pd.Timestamp('2012-01-01')].reset_index(drop=True)
    path.parent.mkdir(exist_ok=True);stock.to_parquet(path,index=False)
    save_json(DEST/'canonical'/f'{ticker}_receipt.json',{'ticker':ticker,'rows':len(stock),'sha256':sha(path),
      'seconds':time.perf_counter()-start,'upstream_model_diagnostics':diag,'features':features,
      'basis':'CONTEMPORANEOUS_PRICE_AND_GAAP_DILUTED_TTM','basis_certified':False,
      'limitation':'Current Yahoo action history and current SEC accession snapshot; not certified release-vintage feed',
      'eps_approximation_preserved':True,'statistical_config':cfg})
    print('REAL_PE_CANONICAL',ticker,len(stock),'COMPLETE',time.perf_counter()-start,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('tickers',nargs='*');args=parser.parse_args()
    inventory();b=benchmark()
    for ticker in (args.tickers or TICKERS): canonical(ticker,b)
