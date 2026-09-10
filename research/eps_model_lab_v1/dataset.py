"""Accession-bound fiscal panel. No model scores are consulted by this module."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
OLD_EPS=PROJECT.parent/'PE_Regime_Engine_v0.2.0/src/pe_regime_engine/eps.py'
spec=importlib.util.spec_from_file_location('readonly_pe_eps',OLD_EPS)
pit=importlib.util.module_from_spec(spec); spec.loader.exec_module(pit)

GEOMETRY={
 'identity':'EPS_PIT_FISCAL_PANEL_V1_3_NATIVE_CONTEXT_CALENDAR','frozen_before_any_model_score':True,
 'history_start':'2009-01-01','minimum_observed_history':8,'context_quarters':32,
 'train_origins_start':'2012-01-01','validation_origin_years':[2019,2020,2021],
 'research_test_origin_years':[2022,2023,2024,2025,2026],
 'fit_policy':'Expanding annual refit at January 1 00:00 UTC; train origin AND each horizon label effective close strictly before cutoff. Test-year past labels may enter later annual fits; hyperparameters remain frozen.',
 'data_snapshot_end':'2026-09-07','selection':'Validation years only; research test descriptive, not formal reserve',
 'quarter_truth':'First native-period 10-Q/K accession direct 65-110 day EarningsPerShareDiluted USD/shares; no subtraction-derived Q4. First within statutory filings, not necessarily earlier 8-K earnings release.',
 'fiscal_geometry':'Quarter from same-accession native EPS/NetIncomeLoss/CFO YTD context duration (65-110/150-215/235-305 days) and annual 10-K duration (330-390). Financial dollar amounts only establish context dates, never EPS values. Fiscal cycle ordinal from native YTD start relative to first available native fiscal-start anchor, rounded elapsed years; <=21-day alignment tolerance. fiscal_year is normalized fiscal START-year ordinal, not SEC FiscalYearFocus. Reported SEC fy/fp retained as untrusted metadata, not index. No compression of missing quarters. First-available collision policy never erases a past record; incompatible calendar transitions fail closed.',
 'feature_vintages':'First native statutory-filing values (never revised historical backfill); financial features only same native accession and end date.',
 'availability':'Read-only existing EPS engine acceptance map and XNYS first usable session close; date-only uses conservative next session.',
 'A':'Direct next-quarter EPS only',
 'B':'Direct quarterly labels by horizon 1..4; missing Q4 retained as missing and never treated as observed; complete path separately scored',
 'C':'NATIVE_LEDGER_PE_METHOD_TTM_AT_Q_PLUS_4: existing read-only PE engine method priority/consensus applied to first-native statutory-filing ledger: annual direct, then four direct quarters, then annual-plus-YTD-minus-prior-YTD EPS bridge. Retain method/approximation; annual/direct and approximate surfaces scored separately. It is not falsely labeled a sum of four direct EPS observations.',
 'C_STRICT_PATH_DIAGNOSTIC':'Four future direct EPS sum only when all four exist; no substitution from engine approximate TTM.',
 'share_basis':'First statutory fact assumed as-filed share basis as in existing engine. Historical split ratios only through origin adjust inputs. Future splits used only to express future labels in origin basis.',
 'basis_limitation':'Yahoo splits are retrospective action evidence, not a certified vintage feed. Announced-before-effective accounting rebasing and spin-off adjustments require filing-level audit before strict deployment/combination certification.',
 'price_policy':'Only evaluator scale diagnostic; no price, return, sector or macro feature in V1. Undo Yahoo retrospective split adjustment for contemporaneous origin close.',
 'missing_input_policy':'Quarter grid retained. Impute only model input: same-season past observation, else last past observed EPS, else zero. Masks retained. No imputed value becomes label.',
 'features':'32 quarter EPS and observation masks, 8 engine TTM lags, fiscal sin/cos, historical scale/growth/volatility/sign, native-accession accounting values/ratios/missing indicators.',
 'known_limits':['Current-company cohort has survivorship selection','Only GAAP diluted USD tag, no basic/adjusted mixing','Not all SEC financial facts have a public original XBRL processing vintage; accession values used from current API snapshot','Foundation pretraining contamination assessed separately'],
 'data_repair':'V1.1 retired for future fiscal-index collisions. Unscored V1.2 keep-first staging exposed broader SEC fiscal metadata errors. V1.3 uses native period contexts and prefix-only calendar mapping, solely data-contract driven, no model metric tuning.',
 'fiscal_year_overrides':{'HON/0000773840-22-000018':{'reported_xbrl_fy':2020,'cover_fiscal_year':2021,'period_end':'2021-12-31',
   'cover_source':'https://www.sec.gov/Archives/edgar/data/773840/000077384022000018/hon-20211231.htm',
   'erroneous_dei_source':'https://www.sec.gov/Archives/edgar/data/773840/000077384022000018/R1.htm'}}}

def split_product(splits,start,end):
    if end<=start: return 1.0
    return float(splits.loc[(splits.index>start)&(splits.index<=end)].prod())

def normalize_eps(value,basis,origin,splits):
    if not np.isfinite(value): return np.nan
    return value/split_product(splits,basis,origin) if origin>=basis else value*split_product(splits,origin,basis)

def causal_fill(values):
    out=np.asarray(values,dtype=float).copy()
    for j in range(len(out)):
        if not np.isfinite(out[j]):
            # Only observations/imputations from earlier fiscal periods.
            out[j]=out[j-4] if j>=4 else (out[j-1] if j else 0.)
    return out

ACCOUNTING={
 'assets':(['Assets'],False),'liabilities':(['Liabilities'],False),
 'equity':(['StockholdersEquity','StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],False),
 'cash':(['CashAndCashEquivalentsAtCarryingValue'],False),
 'revenue':(['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet','RevenueFromContractWithCustomerIncludingAssessedTax'],True),
 'operating_income':(['OperatingIncomeLoss'],True),
 'net_income':(['NetIncomeLoss'],True),
 'common_net_income':(['NetIncomeLossAvailableToCommonStockholdersDiluted','NetIncomeLossAvailableToCommonStockholdersBasic'],True),
 'cfo':(['NetCashProvidedByUsedInOperatingActivities'],True),
 'capex':(['PaymentsToAcquirePropertyPlantAndEquipment'],True),
 'dividends':(['PaymentsOfDividendsCommonStock','PaymentsOfDividends'],True),
 'diluted_shares':(['WeightedAverageNumberOfDilutedSharesOutstanding'],True),
}

def native_accounting(us,accn,end):
    result={}
    for name,(tags,duration) in ACCOUNTING.items():
        found=[]
        for tag in tags:
            units=us.get(tag,{}).get('units',{})
            vals=units.get('shares' if name=='diluted_shares' else 'USD',[])
            found=[v for v in vals if v.get('accn')==accn and v.get('end')==end and bool(v.get('start'))==duration]
            if found: break
        # Same accession native YTD financial statements. Quarterly EPS remains separate.
        if found:
            found.sort(key=lambda v:v.get('start',''))
            starts=found[0].get('start')
            values={float(v['val']) for v in found if v.get('start')==starts}
            result[name]=next(iter(values)) if len(values)==1 else np.nan
        else: result[name]=np.nan
    return result

def native_ttm(grid,current,splits):
    """Prefix-only reconstruction. Unrelated future tag conflicts cannot erase past features."""
    k=int(current['fiscal_index']);basis=current['basis_date'];asof=current['asof']
    def known(index,column):
        r=grid.get(index)
        if r is None or r['asof']>asof: return np.nan
        return normalize_eps(float(r[column]),r['basis_date'],basis,splits)
    methods={}
    if current['fiscal_quarter']==4 and np.isfinite(current['eps_ytd']):
        methods['annual_direct']=float(current['eps_ytd'])
    discrete=[known(j,'eps') for j in range(k-3,k+1)]
    if np.isfinite(discrete).all(): methods['standalone_4q_discrete']=float(np.sum(discrete))
    if current['fiscal_quarter']!=4:
        previous_annual=int(current['fiscal_year'])*4-1
        terms=[known(previous_annual,'eps_ytd'),known(k,'eps_ytd'),known(k-4,'eps_ytd')]
        if np.isfinite(terms).all(): methods['eps_bridge_approx_cumulative_eps']=terms[0]+terms[1]-terms[2]
    value,confidence,_=pit._consensus_details(methods,bool(current['timestamp_exact']),True)
    method=pit._primary_method_name(methods) or 'MISSING'
    return value,method,method in pit._APPROXIMATE_METHODS,confidence

def native_period_context(values,form):
    """Infer fiscal phase from native duration, never SEC fy/fp metadata."""
    buckets={1:(65,110),2:(150,215),3:(235,305),4:(330,390)}
    candidates=[]
    for value in values:
        days=(pd.Timestamp(value['end'])-pd.Timestamp(value['start'])).days
        for quarter,(lower,upper) in buckets.items():
            if lower<=days<=upper and ((quarter==4)==(form in ('10-K','10-KT'))):
                candidates.append((quarter,value['start'],value))
    if not candidates: return None
    quarter=max(v[0] for v in candidates)
    chosen=[v for v in candidates if v[0]==quarter]
    starts={v[1] for v in chosen}
    if len(starts)!=1: return None
    amounts={float(v[2]['val']) for v in chosen}
    return quarter,starts.pop(),next(iter(amounts)) if len(amounts)==1 else np.nan


def calendar_index(start,anchor_start,anchor_year,quarter):
    elapsed=(pd.Timestamp(start)-pd.Timestamp(anchor_start)).days
    cycle=round(elapsed/365.2425)
    if abs(elapsed-cycle*365.2425)>21: return None
    return (anchor_year+cycle)*4+quarter-1


def one(ticker,companyfacts_override=None):
    dest=RUN/'data/raw/sec'/ticker
    cf=companyfacts_override if companyfacts_override is not None else json.loads((dest/'companyfacts.json').read_bytes())
    submissions=[json.loads((dest/'submissions.json').read_bytes())]
    submissions.extend(json.loads(p.read_bytes()) for p in sorted(dest.glob('CIK*.json')))
    amap=pit.build_acceptance_map(submissions)
    filings={}
    for payload in submissions:
        table=payload.get('filings',{}).get('recent',payload)
        for i,accn in enumerate(table.get('accessionNumber',[])):
            form=table.get('form',[])[i]
            if form not in ('10-Q','10-K','10-QT','10-KT'): continue
            end=table.get('reportDate',[])[i]
            filed=table['filingDate'][i]
            if not end or end<'2009-01-01': continue
            accepted=amap.get(accn,pd.Timestamp(filed,tz='UTC'))
            effective=pit._effective_date(accepted,accn in amap)
            close=pit._xnys_calendar().session_close(effective)
            filings[accn]={'accn':accn,'end':end,'filed':filed,'accepted':accepted,'effective':effective,'asof':close,'timestamp_exact':accn in amap,'form':form}
    pricepath=RUN/'data/raw/prices'/f'{ticker}.parquet'
    prices=pd.read_parquet(pricepath) if pricepath.exists() else pd.DataFrame()
    if not prices.empty:
        prices.index=prices.index.tz_localize(None).normalize()
        splits=prices.loc[prices['Stock Splits']!=0,'Stock Splits']
    else: splits=pd.Series(dtype=float,index=pd.DatetimeIndex([]))
    splitframe=pd.DataFrame({'date':splits.index,'stock_split':splits.values})
    us=cf['facts']['us-gaap']
    facts=us.get('EarningsPerShareDiluted',{}).get('units',{}).get('USD/shares',[])
    # Versioned table preserved; select first native filing for each period.
    native={}
    for v in facts:
        filing=filings.get(v.get('accn'))
        if filing is None or v.get('end')!=filing['end'] or not v.get('start'): continue
        native.setdefault(v['accn'],[]).append(v)
    records=[];context_rejections=[]
    for accn,values in native.items():
        f=filings[accn]
        context_values=list(values)
        for tag in ['NetIncomeLoss','NetCashProvidedByUsedInOperatingActivities']:
            context_values.extend(v for v in us.get(tag,{}).get('units',{}).get('USD',[])
               if v.get('accn')==accn and v.get('end')==f['end'] and v.get('start'))
        context=native_period_context(context_values,f['form'])
        if context is None:
            context_rejections.append({'accn':accn,'end':f['end'],'reason':'NO_UNAMBIGUOUS_NATIVE_YTD_DURATION'});continue
        q,fiscal_start,_=context
        eps_ytd_values={float(v['val']) for v in values if v['start']==fiscal_start}
        eps_ytd=next(iter(eps_ytd_values)) if len(eps_ytd_values)==1 else np.nan
        direct=[v for v in values if 65<=(pd.Timestamp(v['end'])-pd.Timestamp(v['start'])).days<=110]
        vals={float(v['val']) for v in direct}
        eps=next(iter(vals)) if len(vals)==1 else np.nan
        row={**f,'ticker':ticker,'fiscal_quarter':q,'native_fiscal_start':fiscal_start,
             'reported_fy':','.join(sorted({str(v.get('fy','')) for v in values})),
             'reported_fp':','.join(sorted({str(v.get('fp','')) for v in values})),
             'eps':eps,
             'eps_ytd':eps_ytd,
             'quarter_source':'DIRECT_NATIVE_GAAP_DILUTED' if np.isfinite(eps) else 'MISSING_DIRECT_QUARTER',
             'basis_date':pd.Timestamp(f['filed']),**native_accounting(us,accn,f['end'])}
        records.append(row)
    panel=pd.DataFrame(records)
    if panel.empty: raise ValueError('No native GAAP diluted EPS fiscal records')
    panel=panel.sort_values(['asof','accn'],kind='stable').drop_duplicates('end',keep='first')
    anchor=panel.iloc[0]
    # Normalized year names fiscal START year, not SEC FiscalYearFocus.
    anchor_year=pd.Timestamp(anchor.native_fiscal_start).year
    indices=[calendar_index(v.native_fiscal_start,anchor.native_fiscal_start,anchor_year,v.fiscal_quarter) for v in panel.itertuples()]
    panel['fiscal_index']=indices
    calendar_rejections=panel.loc[panel.fiscal_index.isna(),['accn','end','native_fiscal_start','reported_fy','reported_fp']].to_dict('records')
    panel=panel.loc[panel.fiscal_index.notna()].copy()
    panel['fiscal_index']=panel.fiscal_index.astype(int)
    panel['fiscal_year']=panel.fiscal_index//4
    # A later conflicting filing may be quarantined, but must never erase an
    # already-known earlier record. Global duplicated(keep=False) violated PIT.
    fiscal_collisions=panel.loc[panel.fiscal_index.duplicated(keep='first'),['accn','end','fiscal_index']].to_dict('records')
    panel=panel.drop_duplicates('fiscal_index',keep='first').sort_values('fiscal_index')
    engine_status={}
    try:
        # Independent reference only. No full-snapshot engine status/value enters native features.
        reference=RUN/'data/engine_events'/f'{ticker}.parquet'
        events=pd.read_parquet(reference)
        eventmap={str(v['accession']):v for _,v in events.iterrows()}
        engine_status={'status':'PASS','events':len(events)}
    except Exception as exc:
        eventmap={}; engine_status={'status':'BLOCKED','reason':f'{type(exc).__name__}: {exc}'}
    native_grid={int(v['fiscal_index']):v for _,v in panel.iterrows()}
    reconstructed=[native_ttm(native_grid,v,splits) for _,v in panel.iterrows()]
    panel['ttm']=[r[0] for r in reconstructed]
    panel['ttm_method']=[r[1] for r in reconstructed]
    panel['ttm_approximate']=[r[2] for r in reconstructed]
    reference_values=np.array([float(eventmap[a]['eps_ttm_raw']) if a in eventmap else np.nan for a in panel.accn])
    overlap=np.isfinite(reference_values)&np.isfinite(panel.ttm.to_numpy())
    engine_status.update(reference_only=True,common_rows=int(overlap.sum()),
       equal_rows=int(np.isclose(reference_values[overlap],panel.ttm.to_numpy()[overlap],rtol=1e-9,atol=1e-10).sum()),
       differences_policy='Expected possible differences: native first-report ledger excludes subsequent restatements/comparisons; NI/basic-and-diluted alternatives and inferred Q4 omitted.')
    grid={int(v['fiscal_index']):v for _,v in panel.iterrows()}
    rows=[]
    for k,current in sorted(grid.items()):
        if current['end']<'2012-01-01': continue
        origin=current['asof']; origin_day=current['effective']
        obs=[]; ttm=[]
        for j in range(k-31,k+1):
            r=grid.get(j)
            obs.append(normalize_eps(r['eps'],r['basis_date'],origin_day,splits) if r is not None and r['asof']<=origin else np.nan)
            ttm.append(normalize_eps(r['ttm'],r['basis_date'],origin_day,splits) if r is not None and r['asof']<=origin else np.nan)
        if np.isfinite(obs).sum()<GEOMETRY['minimum_observed_history']: continue
        filled=causal_fill(obs)
        finite=np.asarray(obs)[np.isfinite(obs)]
        scale=max(float(np.median(np.abs(finite))),.1)
        row={'sample_id':f'{ticker}_{current["end"]}_{current["accn"]}','ticker':ticker,'asof_date':origin,
             'origin_period_end':current['end'],'origin_accession':current['accn'],'fiscal_year':current['fiscal_year'],
             'fiscal_quarter':current['fiscal_quarter'],'fiscal_index':k,'timestamp_exact':current['timestamp_exact'],
             'history_observed':int(np.isfinite(obs).sum()),'scale':scale,'last_observed_eps':float(finite[-1]),
             'current_ttm':ttm[-1],'current_ttm_method':current['ttm_method'],'current_ttm_approximate':current['ttm_approximate'],
             'split_basis_status':'AS_FILED_ASSUMPTION_WITH_RETROSPECTIVE_ACTION_EVIDENCE','pit_valid':True,
             'currency':'USD','eps_definition':'GAAP_DILUTED','share_basis':'FORECAST_ORIGIN','origin_price':np.nan}
        for lag in range(32):
            row[f'eps_lag_{lag}']=obs[-1-lag]
            row[f'filled_lag_{lag}']=float(filled[-1-lag])
            row[f'observed_lag_{lag}']=int(np.isfinite(obs[-1-lag]))
        for lag in range(8): row[f'ttm_lag_{lag}']=ttm[-1-lag]
        row.update(eps_growth=float(filled[-1]-filled[-5]),eps_acceleration=float(filled[-1]-2*filled[-5]+filled[-9]),
                   eps_volatility=float(np.std(finite[-8:])),eps_negative=int(finite[-1]<0),
                   fiscal_sin=math.sin(current['fiscal_quarter']*math.pi/2),fiscal_cos=math.cos(current['fiscal_quarter']*math.pi/2))
        for name in ACCOUNTING: row['account_'+name]=float(current[name])
        assets=current['assets']; revenue=current['revenue']; equity=current['equity']
        for name,num,den in [('accruals_to_assets',current['net_income']-current['cfo'],assets),('leverage',current['liabilities'],assets),
                             ('operating_margin',current['operating_income'],revenue),('roe',current['net_income'],equity),('cash_to_assets',current['cash'],assets)]:
            row[name]=float(num/den) if np.isfinite(num) and np.isfinite(den) and abs(den)>1 else np.nan
        if not prices.empty:
            pp=prices.loc[prices.index<=origin_day]
            if len(pp) and (origin_day-pp.index[-1]).days<=7:
                row['origin_price']=float(pp.iloc[-1]['Close']*splits.loc[splits.index>pp.index[-1]].prod())
        targets=[]
        for h in range(1,5):
            future=grid.get(k+h)
            valid=future is not None and future['asof']>origin and pd.Timestamp(future['end'])>origin.tz_localize(None).normalize()
            y=normalize_eps(future['eps'],future['basis_date'],origin_day,splits) if valid else np.nan
            row[f'y_h{h}']=y; targets.append(y)
            row[f'label_asof_h{h}']=future['asof'] if valid else pd.NaT
            row[f'target_period_end_h{h}']=future['end'] if valid else None
            row[f'target_accession_h{h}']=future['accn'] if valid else None
        future=grid.get(k+4)
        row['y_ttm']=normalize_eps(future['ttm'],future['basis_date'],origin_day,splits) if future is not None and future['asof']>origin else np.nan
        row['label_asof_ttm']=future['asof'] if future is not None and future['asof']>origin else pd.NaT
        row['ttm_target_method']=future['ttm_method'] if future is not None else 'MISSING'
        row['ttm_target_approximate']=bool(future['ttm_approximate']) if future is not None else True
        row['y_direct_4q_sum']=float(np.sum(targets)) if np.isfinite(targets).all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows),panel,{'ticker':ticker,'native_periods':len(panel),'direct_quarters':int(panel.eps.notna().sum()),'origins':len(rows),'engine':engine_status,
      'later_fiscal_collisions_quarantined':fiscal_collisions,'native_context_rejections':context_rejections,
      'incompatible_calendar_rejections':calendar_rejections,
      'prefix_calendar_anchor':{'accession':anchor.accn,'native_fiscal_start':anchor.native_fiscal_start,'normalized_start_year':anchor_year}}

def run():
    if (RUN/'predictions').exists(): raise RuntimeError('Dataset is frozen after first scoring; new identity required')
    lock=RUN/'EPS_DATA_GEOMETRY_LOCK_V1_3.json'
    if lock.exists():
        prior=json.loads(lock.read_text(encoding='utf-8'))
        if prior!=GEOMETRY: raise RuntimeError('Frozen geometry drift')
    else: save_json(lock,GEOMETRY)
    (RUN/'data/engine_events').mkdir(parents=True,exist_ok=True)
    manifest=json.loads((RUN/'SEC_ACQUISITION_MANIFEST.json').read_text(encoding='utf-8'))
    tickers=[r['ticker'] for r in manifest['companies'] if r['status'].endswith('CACHED')]
    frames=[];panels=[];audit=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(one,t):t for t in tickers}
        for future in as_completed(jobs):
            ticker=jobs[future]
            try:
                f,p,r=future.result();frames.append(f);panels.append(p);audit.append(r)
                print(ticker,r['origins'],r['engine']['status'],flush=True)
            except Exception as exc:
                audit.append({'ticker':ticker,'status':'DATASET_BLOCKED','failure':str(exc),'traceback':traceback.format_exc()})
                print(ticker,'BLOCKED',str(exc),flush=True)
            save_json(RUN/'DATASET_BUILD_AUDIT.json',audit)
    if not frames: raise RuntimeError('No valid companies')
    frame=pd.concat(frames,ignore_index=True).sort_values(['asof_date','ticker']).reset_index(drop=True)
    panel=pd.concat(panels,ignore_index=True).sort_values(['ticker','fiscal_index'])
    frame.to_parquet(RUN/'data/samples.parquet',index=False)
    panel.to_parquet(RUN/'data/fiscal_panel.parquet',index=False)
    save_json(RUN/'EPS_DATASET_MANIFEST_V1.json',{'frozen_before_scoring':True,'created_utc':datetime.now(timezone.utc).isoformat(),
       'samples':len(frame),'companies':frame.ticker.nunique(),'labels':{c:int(frame[c].notna().sum()) for c in ['y_h1','y_h2','y_h3','y_h4','y_ttm','y_direct_4q_sum']},
       'samples_sha256':sha(RUN/'data/samples.parquet'),'panel_sha256':sha(RUN/'data/fiscal_panel.parquet'),
       'geometry_sha256':sha(lock),'readonly_eps_source':str(OLD_EPS),'readonly_eps_source_sha256':sha(OLD_EPS),
       'origin_year_counts':{str(k):int(v) for k,v in frame.groupby(frame.asof_date.dt.year).size().items()}})
    state=json.loads((RUN/'RUN_STATE.json').read_text(encoding='utf-8'))
    state.update(status='DATASET_READY_FOR_BASELINES',updated_utc=datetime.now(timezone.utc).isoformat(),download_status='SEC and Yahoo cached',
                 next_actions=['Validate causality/label cutoffs and missing-quarter masks','CPU throughput benchmark','Core baseline/ML/statistical scores','GPU external adapters'])
    save_json(RUN/'RUN_STATE.json',state)
    print('DATASET_COMPLETE',len(frame),flush=True)

if __name__=='__main__': run()
