"""Explicit, accession-specific statement-unit evidence. Never infer from EPS."""
from pathlib import Path
import sys
import urllib.request
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v2.common import RUN, utcnow, save_json, sha

SOURCES = [
    {'key':'MCD_2023_IR','url':'https://corporate.mcdonalds.com/content/dam/sites/corp/nfl/pdf/2023%20Annual%20Report_vf.pdf',
     'sec_url':'https://www.sec.gov/Archives/edgar/data/63908/000006390824000072/mcd-20231231.htm',
     'accession':'0000063908-24-000072','start':'2023-01-01','end':'2023-12-31','display_value':732.3,'statement_page':38},
    {'key':'MCD_2024_IR','url':'https://corporate.mcdonalds.com/content/dam/sites/corp/nfl/pdf/McD%20-%202024%20Annual%20Report%20to%20Shareholders.pdf',
     'sec_url':'https://www.sec.gov/Archives/edgar/data/63908/000006390825000012/mcd-20241231.htm',
     'accession':'0000063908-25-000012','start':'2024-01-01','end':'2024-12-31','display_value':721.9,'statement_page':40}]


def run():
    receipts=[];authority=[]
    for item in SOURCES:
        path=RUN/'data/source_filings'/f"{item['key']}.pdf"
        result={**item,'created_utc':utcnow(),'authority_tier':2,
            'evidence':'Same-accession SEC statement viewed through web tool; corporate IR PDF corroboration',
            'unit_display':'In millions except per-share data','scale':6,'original_ixbrl_unitRef_not_acquired':True}
        try:
            req=urllib.request.Request(item['url'],headers={'User-Agent':'EPSModelLabV2 local research'})
            with urllib.request.urlopen(req,timeout=30) as response:payload=response.read(16*1024*1024)
            if not payload.startswith(b'%PDF'):raise ValueError('Not a PDF')
            if path.exists():raise FileExistsError(path)
            path.write_bytes(payload)
            result.update(pdf_status='DOWNLOADED',path=str(path),sha256=sha(path))
        except Exception as exc:
            result.update(pdf_status='UNAVAILABLE',error=f'{type(exc).__name__}: {exc}')
        receipts.append(result)
        # This is an exact statement observation, not a blanket ticker multiplier.
        authority.append({'ticker':'MCD','accession':item['accession'],
            'concept':'WeightedAverageNumberOfDilutedSharesOutstanding','start':item['start'],'end':item['end'],
            'unit':'shares','unitRef':None,'scale':6,'decimals':None,'source_value':item['display_value'],
            'canonical_value':item['display_value']*1e6,'quality_tier':'VERIFIED','authority_tier':2,
            'source_url':item['sec_url'],'corroborating_url':item['url'],
            'source_document_sha256':result.get('sha256'),
            'basis':'Original same-accession financial statement specifies millions; not NI/EPS reverse inference',
            'historical_publication_vintage_certified':False})
    save_json(RUN/'audit/STATEMENT_SOURCE_RECEIPTS.json',receipts,immutable=True)
    save_json(RUN/'data/SOURCE_AUTHORITY_OVERRIDES.json',authority,immutable=True)
    print('EXACT_STATEMENT_AUTHORITIES',len(authority),'PDFs',sum(r['pdf_status']=='DOWNLOADED' for r in receipts),flush=True)


if __name__=='__main__':run()
