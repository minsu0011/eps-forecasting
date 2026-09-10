"""Cache accession-versioned official SEC data; no current consensus backfill."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha

# Fixed score-blind, deliberately broad current-company research cohort.
# Current constituent mapping is IDENTIFIER DISCOVERY ONLY, never a PIT feature.
TICKERS='AAPL MSFT NVDA AMZN GOOGL META ORCL CRM ADBE INTC AMD IBM CSCO QCOM TXN MU JPM BAC WFC C GS MS AXP V MA UNH JNJ PFE MRK ABBV BMY LLY AMGN GILD CVS WMT COST TGT HD LOW PG KO PEP MDLZ NKE MCD SBUX DIS NFLX CMCSA T VZ XOM CVX COP SLB CAT DE BA GE HON UPS FDX DAL UAL LUV GM F TSLA UBER PYPL SQ'.split()
RAW=RUN/'data/raw'


def fetch(url,path):
    if path.exists(): return
    request=urllib.request.Request(url,headers={'User-Agent':'EPSModelLab research client','Accept':'application/json,text/csv,*/*'})
    with urllib.request.urlopen(request,timeout=60) as stream: content=stream.read()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream: stream.write(content)
    time.sleep(0.3)


def run():
    RAW.mkdir(parents=True,exist_ok=True)
    mapping_url='https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'
    mapping=RAW/'identifier_mapping.csv'
    fetch(mapping_url,mapping)
    identifiers={r['Symbol']:r for r in csv.DictReader(io.StringIO(mapping.read_text(encoding='utf-8')))}
    manifest=[]
    for ticker in TICKERS:
        record={'ticker':ticker,'source':'SEC_EDGAR_OFFICIAL','status':'PENDING'}
        try:
            if ticker not in identifiers: raise RuntimeError('Not in frozen identifier mapping; no inferred CIK')
            cik=int(identifiers[ticker]['CIK']); record['cik']=cik
            destination=RAW/'sec'/ticker
            sub=destination/'submissions.json'
            facts=destination/'companyfacts.json'
            fetch(f'https://data.sec.gov/submissions/CIK{cik:010d}.json',sub)
            metadata=json.loads(sub.read_bytes())
            if ticker not in metadata.get('tickers',[]): raise RuntimeError('SEC ticker/CIK identity mismatch')
            fetch(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json',facts)
            history=[]
            for item in metadata['filings'].get('files',[]):
                if item.get('filingTo','')<'2009-01-01': continue
                path=destination/item['name']
                fetch('https://data.sec.gov/submissions/'+item['name'],path)
                history.append({'path':str(path.relative_to(RUN)),'sha256':sha(path)})
            record.update(status='OFFICIAL_FACTS_AND_SUBMISSIONS_CACHED',company=metadata.get('name'),
                          facts_path=str(facts.relative_to(RUN)),facts_sha256=sha(facts),
                          submissions_sha256=sha(sub),historical_submission_files=history)
        except Exception as exc:
            record.update(status='DATA_ACQUISITION_BLOCKED',failure_reason=f'{type(exc).__name__}: {exc}')
        manifest.append(record)
        save_json(RUN/'SEC_ACQUISITION_MANIFEST.json',{'universe_selection':'fixed current-company cohort; survivorship/selection limitation; no historical index membership claim',
                  'identifier_source':mapping_url,'identifier_mapping_sha256':sha(mapping),
                  'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),'companies':manifest})
        print(ticker,record['status'],flush=True)


if __name__=='__main__': run()
