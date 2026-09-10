"""Unintegrated next-wave exact-context selector. Never rewrites V1.3 data."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.dataset import ACCOUNTING


def exact_context_fact(us, tags, unit, accession, period_end, fiscal_start, accepted, origin, duration):
    origin=pd.Timestamp(origin);accepted=pd.Timestamp(accepted)
    if origin.tzinfo is None or accepted.tzinfo is None:raise ValueError('Explicit UTC-aware publication/origin timestamps required')
    if accepted>origin:return {'status':'ACCESSION_NOT_YET_AVAILABLE','value':None}
    if duration and (not fiscal_start or pd.Timestamp(fiscal_start)>pd.Timestamp(period_end)):
        raise ValueError('Explicit valid fiscal start required for duration feature')
    for tag in tags:
        matches=[]
        for fact in us.get(tag,{}).get('units',{}).get(unit,[]):
            if fact.get('accn')!=accession or fact.get('end')!=period_end:continue
            if duration and fact.get('start')!=fiscal_start:continue
            if not duration and fact.get('start'):continue
            # The original filing's accepted timestamp is necessary, but reject
            # an internally inconsistent later-filed cached record as well.
            filed=fact.get('filed')
            if not filed or pd.Timestamp(filed).date()>origin.date():continue
            try:value=float(fact['val'])
            except (KeyError,TypeError,ValueError):continue
            if math.isfinite(value):matches.append((value,fact))
        if not matches:continue
        values={value for value,_ in matches}
        if len(values)!=1:return {'status':'CONFLICTING_EXACT_CONTEXT_FAIL_CLOSED','value':None,'tag':tag,'values':sorted(values)}
        return {'status':'EXACT_CONTEXT_UNIT_VINTAGE_UNCERTIFIED','value':values.pop(),'tag':tag,'unit':unit,
            'accession':accession,'start':fiscal_start if duration else None,'end':period_end,
            'matching_records':len(matches),'unit_scale_verified':False,'source_vintage_verified':False}
    return {'status':'NO_EXACT_CONTEXT_KEEP_MISSING','value':None}


def shadow_review():
    source=RUN/'data/fiscal_panel.parquet';before=sha(source);panel=pd.read_parquet(source)
    output=RUN/'next_wave_context_selector';output.mkdir(exist_ok=True);rows=[];examples=[]
    for ticker,group in panel.groupby('ticker'):
        raw=json.loads((RUN/'data/raw/sec'/ticker/'companyfacts.json').read_bytes())['facts']['us-gaap']
        # Per-accession/end projection keeps the exact selector semantics while
        # avoiding repeatedly scanning thousands of unrelated source facts.
        lookup={}
        for tag in {tag for tags,_ in ACCOUNTING.values() for tag in tags}:
            for unit,facts in raw.get(tag,{}).get('units',{}).items():
                for fact in facts:
                    key=(fact.get('accn'),fact.get('end'))
                    lookup.setdefault(key,{}).setdefault(tag,{'units':{}})['units'].setdefault(unit,[]).append(fact)
        for r in group.itertuples():
            for feature,(tags,duration) in ACCOUNTING.items():
                previous=float(getattr(r,feature))
                result=exact_context_fact(lookup.get((r.accn,str(r.end)),{}),tags,'shares' if feature=='diluted_shares' else 'USD',
                    r.accn,str(r.end),str(r.native_fiscal_start),r.accepted,r.asof,duration)
                value=result['value']
                equal=(not math.isfinite(previous) and value is None) or (value is not None and previous==value)
                record={'ticker':ticker,'accession':r.accn,'end':str(r.end),'feature':feature,'old_value':previous,
                    'shadow_value':value,'status':result['status'],'same_value':equal,'unit_scale_certified':False}
                rows.append(record)
                if not equal:examples.append({**record,'source_selection':result})
    table=pd.DataFrame(rows);table.to_csv(output/'SHADOW_CONTEXT_COMPARISON.csv',index=False)
    save_json(output/'CHANGED_CELL_PROVENANCE.json',{'cells':examples,'not_applied_to_data':True})
    assert sha(source)==before
    save_json(RUN/'NEXT_WAVE_CONTEXT_SELECTOR_REVIEW.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_SHADOW_ONLY_NOT_INTEGRATED','compared_cells':len(table),'changed_cells':len(examples),
        'status_counts':table.status.value_counts().to_dict(),'source_unchanged':True,'source_sha256':before,
        'new_dataset_created':False,'models_retrained_on_shadow_values':False,'source_share_units_repaired':False,
        'formal_certified':False,'selector_source_sha256':sha(__file__),
        'boundary':'Exact duration context selection implemented and tested for next wave; unit scale and historical vintage are still unverified. No production or frozen-data replacement.'})
    print('NEXT_WAVE_SHADOW_SELECTOR',len(table),'cells;',len(examples),'differences; no dataset change',flush=True)


if __name__=='__main__':shadow_review()
