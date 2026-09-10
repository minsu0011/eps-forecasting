"""Full-cohort historical mutation and honest source-quality audit before scoring."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import sys
import json
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v2.common import RUN,V1,sha,save_json,read_json,utcnow
from research.eps_model_lab_v2.clean_data import build_one,exact_candidate
from research.eps_model_lab_v2.fiscal_builder import ACCOUNTING

CUTOFFS=['2015-01-01T00:00:00Z','2019-01-01T00:00:00Z','2022-01-01T00:00:00Z','2025-01-01T00:00:00Z']


def prefix_audit(ticker):
    baseline=pd.read_parquet(RUN/'data/samples_v2.parquet')
    baseline=baseline[baseline.ticker==ticker]
    features=[c for c in baseline.columns if not c.startswith(('y_','label_','target_','ttm_target_','quality_'))]
    results=[]
    for cutoff in CUTOFFS:
        expected=baseline[baseline.asof_date<pd.Timestamp(cutoff)].set_index('sample_id').sort_index()
        if expected.empty:
            results.append({'ticker':ticker,'cutoff':cutoff,'past_samples':0,'pass':True,
                'scope':'VACUOUS_NO_PAST_SAMPLE_ORIGINS','actual_rebuild_performed':False})
            continue
        truncated,panel,_,_=build_one(ticker,cutoff,include_ledger=False)
        actual=truncated[truncated.asof_date<pd.Timestamp(cutoff)].set_index('sample_id').sort_index()
        columns=[c for c in features if c!='sample_id']
        try:
            pd.testing.assert_frame_equal(expected[columns],actual[columns],check_exact=True,check_dtype=False)
            results.append({'ticker':ticker,'cutoff':cutoff,'past_samples':len(expected),'features':len(columns),'pass':True})
        except AssertionError as exc:
            results.append({'ticker':ticker,'cutoff':cutoff,'past_samples':len(expected),'pass':False,'error':str(exc)[:2500]})
    return results


def run():
    frame=pd.read_parquet(RUN/'data/samples_v2.parquet')
    panel=pd.read_parquet(RUN/'data/fiscal_panel_v2.parquet')
    ledger=pd.read_parquet(RUN/'EPS_SOURCE_CONTEXT_LEDGER_V2.parquet')
    selected=ledger[ledger.selected_in_panel]
    account=selected[selected.feature.isin(ACCOUNTING)]
    issues=[]
    if frame.sample_id.duplicated().any():issues.append('Duplicate sample IDs')
    if panel.duplicated(['ticker','fiscal_index']).any():issues.append('Duplicate company fiscal slots')
    pmap=panel.set_index(['ticker','accn'])
    for row in account.itertuples():
        native=pmap.loc[(row.ticker,row.accession)]
        if ACCOUNTING[row.feature][1] and row.start!=native.native_fiscal_start:
            issues.append(f'Duration mismatch {row.ticker}/{row.accession}/{row.feature}')
        if pd.Timestamp(row.source_available_at)>native['asof']:issues.append('Unavailable accounting feature')
        value=native[row.feature]
        if pd.notna(value) and row.quality_tier!='VERIFIED':issues.append('Unverified accounting consumed')
    for target in ['h1','h2','h3','h4','ttm']:
        valid=frame['y_'+target].notna()
        if not (frame.loc[valid,'label_asof_'+target]>frame.loc[valid,'asof_date']).all():issues.append('Nonfuture target '+target)
    # Exactly 100 deterministic random contexts plus all known-problem issuers.
    # This verifies API context linkage, NOT a nonexistent full original-XBRL audit.
    random=selected.sample(n=min(100,len(selected)),random_state=1729)
    random.to_csv(RUN/'audit/RANDOM_100_SOURCE_CONTEXTS.csv',index=False)
    known=selected[selected.ticker.isin(['MCD','AMZN','COP','HON','UPS'])]
    known.to_parquet(RUN/'audit/KNOWN_PROBLEM_SOURCE_CONTEXTS.parquet',index=False)
    source_checks=[];raw_cache={}
    for row in random.itertuples():
        if row.ticker not in raw_cache:raw_cache[row.ticker]=read_json(V1/'data/raw/sec'/row.ticker/'companyfacts.json')['facts']['us-gaap']
        unit=row.unit;tags=[row.concept] if pd.notna(row.concept) else []
        match=exact_candidate(raw_cache[row.ticker],tags,unit,row.accession,row.end,row.start,row.filed,True if row.feature.startswith('native_') else ACCOUNTING[row.feature][1])
        expected=row.source_value
        # Explicit statement repairs have independent display-scale evidence.
        equal=(pd.isna(expected) and match['value'] is None) or (pd.notna(expected) and match['value']==expected)
        source_checks.append({'ticker':row.ticker,'accession':row.accession,'feature':row.feature,
            'api_context_value_match':bool(equal),'original_filing_unit_verified':row.quality_tier=='VERIFIED',
            'quality_tier':row.quality_tier})
    results=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        jobs={pool.submit(prefix_audit,t):t for t in sorted(frame.ticker.unique())}
        for future in as_completed(jobs):
            part=future.result();results.extend(part)
            print('V2_PREFIX_AUDIT',jobs[future],sum(r['pass'] for r in part),'/',len(part),flush=True)
    violations=[r for r in results if not r['pass']]
    save_json(RUN/'audit/PIT_MUTATION_AUDIT_V2.json',{'created_utc':utcnow(),'status':'PASS' if not violations else 'FAIL',
        'checks':results,'violations':violations,'companies':int(frame.ticker.nunique()),'cutoffs':CUTOFFS,
        'scope':'Future companyfacts accession removal and complete past predictive surface rebuild, not historical-vintage certification'},immutable=True)
    save_json(RUN/'EPS_SHARE_UNIT_AUDIT_V2.json',{'created_utc':utcnow(),
        'status':'PASS_QUARANTINE_POLICY_PARTIAL_SOURCE_REPAIR',
        'source_original_unit_audit_complete':False,'automatic_size_rescaling':False,'NI_over_EPS_inference':False,
        'verified_accounting_cells':int((account.quality_tier=='VERIFIED').sum()),
        'quarantined_accounting_cells':int((account.quality_tier=='QUARANTINED').sum()),
        'share_quality_counts':account[account.feature=='diluted_shares'].quality_tier.value_counts().to_dict(),
        'exact_statement_repairs':read_json(RUN/'data/SOURCE_AUTHORITY_OVERRIDES.json'),
        'limitation':'Direct original SEC HTTP 403 and corporate IR download timeouts retained; unknown accounting omitted from Track A'},immutable=True)
    save_json(RUN/'EPS_DURATION_CONTEXT_AUDIT_V2.json',{'created_utc':utcnow(),'status':'PASS' if not issues else 'FAIL',
        'structural_issues':issues,'checked_accounting_cells':len(account),
        'random_contexts':source_checks,'random_original_context_audit_complete':False,
        'original_source_audit_not_equivalent_to_API_equality':True,
        'long_quarter_rows':int(((pd.to_datetime(panel.end)-pd.to_datetime(panel.direct_quarter_start)).dt.days>110).fillna(False).sum()),
        'source_available_quarantine_policy':True},immutable=True)
    if issues or violations:raise RuntimeError('Data audit failed; no model training permitted')
    old=pd.read_parquet(V1/'data/samples.parquet')
    common=old.merge(frame,on='sample_id',suffixes=('_old','_v2'))
    differences={}
    for target in ['h1','h2','h3','h4','ttm']:
        a=common['y_'+target+'_old'].to_numpy();b=common['y_'+target+'_v2'].to_numpy()
        differences[target]=int((~np.isclose(a,b,equal_nan=True,rtol=0,atol=0)).sum())
    save_json(RUN/'audit/V1_V2_DATA_DIFF.json',{'v1_origins':len(old),'v2_origins':len(frame),'common_origins':len(common),
        'label_differences_on_common_origins':differences,'v2_label_counts':{t:int(frame['y_'+t].notna().sum()) for t in differences},
        'basis':'Data contract changes only; no V2 model scores yet'},immutable=True)
    print('V2_DATA_AUDITS_COMPLETE',len(results),'mutation checks',len(issues),'structural issues',flush=True)


if __name__=='__main__':
    try:run()
    except Exception as exc:
        import traceback
        save_json(RUN/'failures'/f'data_audit_{os.getpid()}.json',{'created_utc':utcnow(),
            'error':str(exc),'traceback':traceback.format_exc()},immutable=True)
        raise
