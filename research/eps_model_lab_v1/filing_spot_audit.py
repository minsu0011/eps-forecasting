"""Read-only original filing spot checks around announced/effective stock splits."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import urllib.request
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha

CASES=[('AAPL','2014-03-29'),('AAPL','2020-06-27'),('NVDA','2021-05-02'),('NVDA','2024-04-28'),
       ('TSLA','2020-06-30'),('TSLA','2022-06-30'),('GE','2021-06-30'),('GOOGL','2022-03-31')]


def primary(ticker,accn):
    root=RUN/'data/raw/sec'/ticker;sub=json.loads((root/'submissions.json').read_text(encoding='utf-8'))
    cik=int(sub['cik']);tables=[sub['filings']['recent']]
    tables.extend(json.loads(p.read_text(encoding='utf-8')) for p in root.glob('CIK*.json'))
    for table in tables:
        if accn in table.get('accessionNumber',[]):
            i=table['accessionNumber'].index(accn);doc=table.get('primaryDocument',[])[i]
            return f'https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace("-","")}/{doc}'
    raise RuntimeError('Accession document mapping absent')


def parse_inline(raw,end):
    soup=BeautifulSoup(raw,'html.parser');contexts={}
    for context in soup.find_all(lambda t:t.name.lower().split(':')[-1]=='context'):
        start=context.find(lambda t:t.name.lower().split(':')[-1]=='startdate')
        stop=context.find(lambda t:t.name.lower().split(':')[-1]=='enddate')
        if start is None or stop is None: continue
        dimensions=context.find_all(lambda t:t.name.lower().split(':')[-1] in ['explicitmember','typedmember'])
        contexts[context.get('id')]={'start':start.get_text(strip=True),'end':stop.get_text(strip=True),'has_dimensions':bool(dimensions)}
    matched=[]
    for fact in soup.find_all(lambda t:t.name.lower().split(':')[-1]=='nonfraction'):
        if fact.get('name','').lower()!='us-gaap:earningspersharediluted': continue
        context=contexts.get(fact.get('contextref'))
        if context is None or context['end']!=end or context['has_dimensions']: continue
        days=(pd.Timestamp(context['end'])-pd.Timestamp(context['start'])).days
        if not 65<=days<=110: continue
        text=fact.get_text('',strip=True).replace(',','').replace('$','').replace('\u2212','-')
        try:
            value=float(text)*10**int(fact.get('scale','0'))
            if fact.get('sign')=='-': value=-abs(value)
            matched.append({'value':value,'context':context,'unit_ref':fact.get('unitref'),'id':fact.get('id')})
        except ValueError: pass
    return matched


if __name__=='__main__':
    panel=pd.read_parquet(RUN/'data/fiscal_panel.parquet');records=[]
    for ticker,end in CASES:
        record={'ticker':ticker,'period_end':end}
        try:
            selected=panel[(panel.ticker==ticker)&(panel.end==end)]
            if len(selected)!=1: raise RuntimeError('Expected native filing absent/ambiguous in frozen ledger')
            row=selected.iloc[0];url=primary(ticker,row.accn);record.update(accession=row.accn,source=url,native_eps=float(row.eps))
            path=RUN/'data/raw/filing_spot_checks'/f'{ticker}_{end}.html';path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                request=urllib.request.Request(url,headers={'User-Agent':'EPSModelLab public academic-style research client','Accept':'text/html'})
                with urllib.request.urlopen(request,timeout=45) as response: raw=response.read()
                path.write_bytes(raw);time.sleep(.4)
            matched=parse_inline(path.read_bytes(),end);values=sorted(set(r['value'] for r in matched))
            record.update(original_filing_sha256=sha(path),inline_facts=matched)
            if len(values)==1 and np.isclose(values[0],row.eps,rtol=0,atol=1e-12): record['status']='PASS_ORIGINAL_INLINE_VALUE'
            elif not matched: record['status']='NO_UNAMBIGUOUS_INLINE_FACT_REQUIRES_MANUAL_OR_INSTANCE_AUDIT'
            else: record['status']='VALUE_OR_CONTEXT_MISMATCH_REQUIRES_INVESTIGATION'
        except Exception as exc: record.update(status='ORIGINAL_DOCUMENT_ACCESS_OR_MAPPING_BLOCKED',reason=str(exc))
        records.append(record);save_json(RUN/'ORIGINAL_FILING_SPOT_AUDIT.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'records':records,
           'scope':'Predetermined split-event spot checks only, not full cohort vintage certification; original documents are never used to rewrite frozen V1 values'})
        print('ORIGINAL_FILING_AUDIT',ticker,end,record['status'],flush=True)
