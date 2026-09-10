"""Cache split-adjusted Yahoo history with actions; never use adjusted close as PIT features."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import yfinance as yf

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha

def one(ticker):
    record={'ticker':ticker,'source':'Yahoo Finance via yfinance','status':'PENDING'}
    try:
        path=RUN/'data/raw/prices'/f'{ticker}.parquet'
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            frame=yf.Ticker(ticker).history(start='2009-01-01',end='2026-09-07',auto_adjust=False,actions=True,repair=False)
            if frame.empty: raise ValueError('Empty history')
            frame.to_parquet(path)
        import pandas as pd
        frame=pd.read_parquet(path)
        record.update(status='CACHED',path=str(path.relative_to(RUN)),sha256=sha(path),rows=len(frame),
                      first=str(frame.index.min()),last=str(frame.index.max()),splits=int((frame['Stock Splits']!=0).sum()))
    except Exception as exc:
        record.update(status='DATA_BLOCKED',failure_reason=f'{type(exc).__name__}: {exc}')
    return record

if __name__=='__main__':
    companies=json.loads((RUN/'SEC_ACQUISITION_MANIFEST.json').read_text(encoding='utf-8'))['companies']
    result=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one,c['ticker']) for c in companies if c['status'].endswith('CACHED')]):
            r=future.result(); result.append(r); print(r['ticker'],r['status'],flush=True)
            save_json(RUN/'PRICE_ACQUISITION_MANIFEST.json',{'records':result,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                'price_policy':'Close is retrospectively split-adjusted, NOT unadjusted despite auto_adjust=False. Reconstruct contemporaneous price by multiplying subsequent split ratios; price is used only for diagnostics/evaluation in V1.',
                'limitations':['Yahoo action history is not an exchange-certified corporate-action vintage feed','Spin-offs/special adjustments may not be fully represented; fail strict PE combination on uncertain basis','No Adj Close features; future split ratios only undo retrospective price adjustment or normalize evaluation labels']})
