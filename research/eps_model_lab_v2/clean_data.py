"""Rebuild source-bound V2 data; unknown accounting units stay quarantined.

Exact source context validity and full historical data-vintage certification are
different properties. Track N retains explicitly partial native-API provenance.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from pathlib import Path
import sys
import traceback
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v2.common import RUN,V1,sha,save_json,read_json,utcnow,check_stop
from research.eps_model_lab_v2 import fiscal_builder as builder


def exact_candidate(us,tags,unit,accn,end,start,filed,duration):
    for tag in tags:
        found=[v for v in us.get(tag,{}).get('units',{}).get(unit,[])
            if v.get('accn')==accn and v.get('end')==end
            and (v.get('start')==start if duration else not v.get('start'))
            and v.get('filed') and v['filed']<=filed]
        finite=[]
        for v in found:
            try:
                if np.isfinite(float(v['val'])):finite.append(v)
            except (TypeError,ValueError,KeyError):pass
        if not finite:continue
        values={float(v['val']) for v in finite}
        if len(values)!=1:return {'status':'CONFLICT_QUARANTINED','tag':tag,'value':None,'records':len(finite)}
        return {'status':'EXACT_CONTEXT_UNIT_UNVERIFIED','tag':tag,'value':values.pop(),'records':len(finite)}
    return {'status':'NO_EXACT_CONTEXT','tag':None,'value':None,'records':0}


def filing_metadata(ticker):
    root=V1/'data/raw/sec'/ticker
    paths=[root/'submissions.json',*sorted(root.glob('CIK*.json'))]
    out={}
    for path in paths:
        raw=read_json(path);table=raw.get('filings',{}).get('recent',raw)
        for i,a in enumerate(table.get('accessionNumber',[])):
            out[a]={k:table[k][i] for k in ['form','filingDate','reportDate','primaryDocument'] if k in table}
    return out


def build_one(ticker,cutoff=None,include_ledger=True):
    check_stop()
    source=V1/'data/raw/sec'/ticker/'companyfacts.json'
    cf=read_json(source);source_sha256=sha(source);metadata=filing_metadata(ticker)
    if cutoff is not None:
        # Remove future accession facts before ANY native fiscal construction.
        allowed={a for a,v in metadata.items() if v['filingDate']<cutoff[:10]}
        cf=copy.deepcopy(cf)
        for namespace in cf.get('facts',{}).values():
            for concept in namespace.values():
                for unit,values in concept.get('units',{}).items():
                    concept['units'][unit]=[v for v in values if v.get('accn') in allowed and v.get('filed','9999')<cutoff[:10]]
    # Metadata/fact publication disagreement must not enter the native calendar
    # earlier than its own filed date. Quarantine these records consistently in
    # both calendar and accounting selectors (IBM 2019 Q3 is a concrete case).
    source_date_disagreements=[]
    for concept,entry in cf['facts']['us-gaap'].items():
        for unit,values in entry.get('units',{}).items():
            retained=[]
            for v in values:
                m=metadata.get(v.get('accn'))
                if m is not None and (not v.get('filed') or v['filed']>m['filingDate']):
                    source_date_disagreements.append({'concept':concept,'unit':unit,'accession':v.get('accn'),
                        'fact_filed':v.get('filed'),'submission_filed':m['filingDate']})
                else:retained.append(v)
            entry['units'][unit]=retained
    us=cf['facts']['us-gaap'];cik=int(cf['cik']);ledger=[]
    overrides=read_json(RUN/'data/SOURCE_AUTHORITY_OVERRIDES.json') if (RUN/'data/SOURCE_AUTHORITY_OVERRIDES.json').exists() else []
    authority={(r['ticker'],r['accession'],r['concept'],r['start'],r['end'],r['unit']):r for r in overrides}
    ix_authority=[]
    for p in (RUN/'data/source_filings').glob('*.facts.json'):
        ix_authority.extend(r for r in read_json(p) if r.get('ticker')==ticker and r.get('status')=='VERIFIED')
    lookup={}
    for tag in {tag for tags,_ in builder.ACCOUNTING.values() for tag in tags}|{'EarningsPerShareDiluted'}:
        for unit,values in us.get(tag,{}).get('units',{}).items():
            for v in values:
                lookup.setdefault((v.get('accn'),v.get('end')),{}).setdefault(tag,{'units':{}})['units'].setdefault(unit,[]).append(v)
    def accounting(_us,accn,end):
        current=lookup.get((accn,end),{})
        f=metadata[accn]
        periods=[]
        for tag,unit in [('EarningsPerShareDiluted','USD/shares'),('NetIncomeLoss','USD'),('NetCashProvidedByUsedInOperatingActivities','USD')]:
            periods.extend(v for v in current.get(tag,{}).get('units',{}).get(unit,[])
                if v.get('start') and v.get('filed','9999')<=f['filingDate'])
        context=builder.native_period_context(periods,f['form'])
        if context is None:raise RuntimeError('Native context vanished between selectors')
        quarter,start,_=context
        result={}
        for feature,(tags,duration) in builder.ACCOUNTING.items():
            unit='shares' if feature=='diluted_shares' else 'USD'
            candidate=exact_candidate(current,tags,unit,accn,end,start,f['filingDate'],duration)
            source_start=start if duration else None
            record={'ticker':ticker,'cik':cik,'accession':accn,'feature':feature,
                'concept':candidate['tag'],'start':source_start,'end':end,'fiscal_period':f'Q{quarter}',
                'form':f['form'],'filed':f['filingDate'],'accepted':None,'source_available_at':None,
                'unitRef':None,'unit':unit,'scale':None,'decimals':None,
                'source_value':candidate['value'],'canonical_value':None,
                'candidate_status':candidate['status'],'quality_tier':'QUARANTINED',
                'quality_reason':'Unit/scale not independently established from filing; unavailable means missing, not guessed',
                'source_json_sha256':source_sha256,'duration_contract':'NATIVE_FISCAL_YTD' if duration else 'PERIOD_END_INSTANT',
                'authority_tier':None,'historical_publication_vintage_certified':False}
            exact_ix=[r for r in ix_authority if r['accession']==accn and r['end']==end and
                r['start']==source_start and r['concept'].split(':')[-1] in tags and r['unit']==unit.lower()]
            exact_values={r['canonical_value'] for r in exact_ix}
            verified=exact_ix[0] if len(exact_values)==1 else None
            if verified:
                record.update(canonical_value=verified['canonical_value'],unitRef=verified['unitRef'],
                    scale=verified['scale'],decimals=verified['decimals'],source_value=verified['source_value'],
                    concept=verified['concept'].split(':')[-1],quality_tier='VERIFIED',authority_tier=1,
                    quality_reason='Exact original iXBRL context and scale',source_url=verified['source_url'])
            elif candidate['tag'] is not None:
                override=authority.get((ticker,accn,candidate['tag'],source_start,end,unit))
                if override:
                    record.update({k:override[k] for k in ['canonical_value','unitRef','scale','decimals','source_value','quality_tier','authority_tier','source_url']})
                    record['quality_reason']=override['basis']
            result[feature]=record['canonical_value'] if record['quality_tier']=='VERIFIED' else np.nan
            if include_ledger:ledger.append(record)
        return result
    builder.native_accounting=accounting
    frame,panel,audit=builder.one(ticker,cf)
    audit['source_date_disagreements_quarantined']=source_date_disagreements
    if include_ledger:
        for r in panel.itertuples():
            for feature,start,value in [('native_eps',r.direct_quarter_start,r.eps),('native_ytd_eps',r.native_fiscal_start,r.eps_ytd)]:
                ledger.append({'ticker':ticker,'cik':cik,'accession':r.accn,'feature':feature,
                    'concept':'EarningsPerShareDiluted','start':start,'end':r.end,
                    'fiscal_period':f'Q{r.fiscal_quarter}','form':r.form,'filed':r.filed,
                    'accepted':r.accepted.isoformat(),'source_available_at':r.asof.isoformat(),
                    'unitRef':None,'unit':'USD/shares','scale':None,'decimals':None,
                    'source_value':value if np.isfinite(value) else None,
                    'canonical_value':value if np.isfinite(value) else None,
                    'candidate_status':'NATIVE_SAME_ACCESSION_EXACT_PERIOD',
                    'quality_tier':'PARTIALLY_VERIFIED' if np.isfinite(value) else 'QUARANTINED',
                    'quality_reason':'Native API EPS, exact accession and context; original XBRL scale/historical vintage not universally verified',
                    'source_json_sha256':source_sha256,
                    'duration_contract':'DIRECT_NATIVE_QUARTER' if feature=='native_eps' else 'NATIVE_FISCAL_YTD',
                    'authority_tier':None,'historical_publication_vintage_certified':False})
    amap={r.accn:r for r in panel.itertuples()}
    for record in ledger:
        r=amap.get(record['accession'])
        if r is not None:
            record['accepted']=r.accepted.isoformat();record['source_available_at']=r.asof.isoformat()
            record['selected_in_panel']=True
        else:record['selected_in_panel']=False
    quality={(r['accession'],r['feature']):r['quality_tier'] for r in ledger}
    if include_ledger:
        for feature in builder.ACCOUNTING:
            panel[f'quality_{feature}']=[quality.get((a,feature),'QUARANTINED') for a in panel.accn]
            frame[f'quality_account_{feature}']=[quality.get((a,feature),'QUARANTINED') for a in frame.origin_accession]
    panel['native_quality_tier']='PARTIALLY_VERIFIED'
    frame['data_quality_tier']='PARTIALLY_VERIFIED_NATIVE_WITH_QUARANTINED_ACCOUNTING'
    frame['data_identity']='EPS_DATA_IDENTITY_V2_CLEAN'
    frame['historical_vintage_certified']=False
    frame['formal_certified']=False
    # Numeric zero is never a replacement for a missing observation without its
    # explicit mask. Staleness is measured only from already available lags.
    obs=frame[[f'observed_lag_{j}' for j in range(32)]].to_numpy()
    frame['eps_staleness_quarters']=np.argmax(obs>0,axis=1)
    return frame,panel,ledger,audit


def build():
    if (RUN/'EPS_DATASET_MANIFEST_V2.json').exists():raise RuntimeError('V2 dataset already frozen')
    check_stop()
    # Match the prespecified V1 supported cohort, not all downloaded identifiers
    # (V has no native diluted facts; XOM had no supported sample origins).
    tickers=sorted(pd.read_parquet(V1/'data/samples.parquet',columns=['ticker']).ticker.unique())
    frames=[];panels=[];ledger=[];audits=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        jobs={pool.submit(build_one,t):t for t in tickers}
        for future in as_completed(jobs):
            t=jobs[future]
            try:
                f,p,l,a=future.result();frames.append(f);panels.append(p);ledger.extend(l);audits.append(a)
                print('V2_REBUILT',t,len(f),len(l),flush=True)
            except Exception as exc:
                save_json(RUN/'failures'/f'data_{t}_{os.getpid()}.json',{'error':str(exc),'traceback':traceback.format_exc(),'created_utc':utcnow()},immutable=True)
                raise
    frame=pd.concat(frames,ignore_index=True).sort_values(['asof_date','ticker']).reset_index(drop=True)
    panel=pd.concat(panels,ignore_index=True).sort_values(['ticker','fiscal_index']).reset_index(drop=True)
    context=pd.DataFrame(ledger).sort_values(['ticker','source_available_at','accession','feature']).reset_index(drop=True)
    frame.to_parquet(RUN/'data/samples_v2.parquet',index=False)
    panel.to_parquet(RUN/'data/fiscal_panel_v2.parquet',index=False)
    context.to_parquet(RUN/'EPS_SOURCE_CONTEXT_LEDGER_V2.parquet',index=False)
    context.to_parquet(RUN/'data/source_context_ledger_v2.parquet',index=False)
    save_json(RUN/'audit/DATA_BUILD_V2.json',{'created_utc':utcnow(),'companies':audits,'samples':len(frame),
        'panel_rows':len(panel),'ledger_rows':len(context),'unit_quality_counts':context.quality_tier.value_counts().to_dict(),
        'status':'BUILT_NOT_YET_FROZEN_PENDING_AUDITS'},immutable=True)
    print('V2_DATA_STAGING_COMPLETE',len(frame),len(panel),len(context),flush=True)


if __name__=='__main__':build()
