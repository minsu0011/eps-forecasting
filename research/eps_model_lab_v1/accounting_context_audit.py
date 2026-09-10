"""Read-only same-accession accounting context and dimensional suspicion audit."""
from collections import defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.dataset import ACCOUNTING


def unit_ratio(income, shares, eps):
    # A suspicion ratio, NOT an identity: attribution and dilution adjustments
    # can legitimately differ. Never use this to manufacture or rescale EPS.
    if not all(np.isfinite(v) for v in [income,shares,eps]) or shares<=0 or abs(eps)<.1:return np.nan
    return income/(shares*eps)


def run():
    paths=[RUN/'data/samples.parquet',RUN/'data/fiscal_panel.parquet',RUN/'ENSEMBLE_FREEZE_V1.json',
           *list((RUN/'predictions').glob('*.parquet'))]
    before={p.relative_to(RUN).as_posix():sha(p) for p in paths}
    panel=pd.read_parquet(RUN/'data/fiscal_panel.parquet');records=[];ratios=[];raw_hashes=[]
    for ticker,group in panel.groupby('ticker'):
        source=RUN/'data/raw/sec'/ticker/'companyfacts.json'
        raw=json.loads(source.read_bytes())['facts']['us-gaap'];indexes={}
        raw_hashes.append({'ticker':ticker,'sha256':sha(source)})
        for feature,(tags,duration) in ACCOUNTING.items():
            for tag in tags:
                unit='shares' if feature=='diluted_shares' else 'USD'
                indexed=defaultdict(list)
                for fact in raw.get(tag,{}).get('units',{}).get(unit,[]):
                    if bool(fact.get('start'))==duration:indexed[(fact.get('accn'),fact.get('end'))].append(fact)
                indexes[tag]=indexed
        for row in group.itertuples():
            for feature,(tags,duration) in ACCOUNTING.items():
                value=getattr(row,feature)
                if not np.isfinite(value):continue
                selected=[];selected_tag=None
                for tag in tags:
                    selected=indexes[tag].get((row.accn,str(row.end)),[])
                    if selected:selected_tag=tag;break
                if not selected:raise RuntimeError('Nonmissing panel fact has no raw accession context')
                first=min(f.get('start','') for f in selected)
                native=[f for f in selected if f.get('start','')==first]
                values={float(f['val']) for f in native}
                if values!={float(value)}:raise RuntimeError('Panel-source value mismatch')
                gap=(pd.Timestamp(first)-pd.Timestamp(row.native_fiscal_start)).days if duration else None
                records.append({'ticker':ticker,'accn':row.accn,'period_end':row.end,'asof':str(row.asof),
                    'feature':feature,'source_tag':selected_tag,'unit':'shares' if feature=='diluted_shares' else 'USD',
                    'selected_start':first if duration else None,'native_fiscal_start':row.native_fiscal_start,
                    'duration_start_gap_days':gap,'exact_source_value_reproduced':True,
                    'context_start_review_required':duration and gap!=0,'value':value})
            income=row.common_net_income if np.isfinite(row.common_net_income) else row.net_income
            ratio=unit_ratio(income,row.diluted_shares,row.eps_ytd)
            if np.isfinite(ratio):
                ratios.append({'ticker':ticker,'accn':row.accn,'period_end':row.end,'asof':str(row.asof),
                    'income_tag_scope':'common_net_income' if np.isfinite(row.common_net_income) else 'net_income',
                    'income':income,'raw_shares':row.diluted_shares,'native_eps_ytd':row.eps_ytd,
                    'income_over_shares_times_eps_ratio':ratio,'factor_magnitude_review_required':abs(ratio)>10 or abs(ratio)<.1,
                    'sign_review_required':ratio<0,'automatic_scale_correction_allowed':False})
    cells=pd.DataFrame(records);dim=pd.DataFrame(ratios)
    cells.to_csv(RUN/'ACCOUNTING_ACCESSION_CONTEXT_CELLS.csv',index=False)
    dim.to_csv(RUN/'ACCOUNTING_DIMENSIONAL_SUSPICION_SCREEN.csv',index=False)
    issues=cells[cells.context_start_review_required]
    issues.to_csv(RUN/'ACCOUNTING_DURATION_START_REVIEW.csv',index=False)
    assert all(sha(RUN/path)==digest for path,digest in before.items())
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'READ_ONLY_CONTEXT_AUDIT_COMPLETE',
        'companies':len(raw_hashes),'panel_rows':len(panel),'nonmissing_accounting_cells':len(cells),
        'exact_source_value_match_cells':int(cells.exact_source_value_reproduced.sum()),
        'duration_start_mismatch_cells':len(issues),'duration_start_mismatch_tickers':sorted(issues.ticker.unique()),
        'duration_start_gaps':issues.duration_start_gap_days.value_counts().to_dict(),
        'dimensional_screen_rows':len(dim),'factor_magnitude_suspicion_rows':int(dim.factor_magnitude_review_required.sum()),
        'factor_magnitude_suspicion_tickers':sorted(dim.loc[dim.factor_magnitude_review_required,'ticker'].unique()),
        'raw_sources':raw_hashes,'all_frozen_artifacts_unchanged':True,'automatic_corrections':0,
        'formal_certified':False,'source': 'Frozen official SEC companyfacts, same accession and period end',
        'scope_limitations':['Matching a cached source does not establish correct dimensional scale or historical vintage',
            'A different duration start requires original context review; it is not automatically a data error',
            'Income attribution, preferred dividends and dilutive adjustments mean NI / shares need not equal diluted EPS',
            'Ratio flags are triage only. No source shares or EPS labels are inferred from this arithmetic.',
            'COP original HTML browser exceeded 4 MiB; direct conventional HTTP request returned 403. No bypass or automatic scaling attempted.']}
    save_json(RUN/'ACCOUNTING_CONTEXT_AND_UNIT_REVIEW.json',result)
    print('ACCOUNTING_CONTEXT_REVIEW',len(cells),'source cells;',len(issues),'duration mismatches;',result['factor_magnitude_suspicion_rows'],'ratio flags',flush=True)


if __name__=='__main__':run()
