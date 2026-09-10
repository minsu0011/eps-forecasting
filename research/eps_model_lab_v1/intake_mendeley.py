"""Author replication source intake from public Mendeley API, no code execution."""
import json
from pathlib import Path
import sys
import requests
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,EXTERNAL,save_json,sha

if __name__=='__main__':
    url='https://data.mendeley.com/public-api/datasets/2pg764w6gs/files?folder_id=cf765d11-3ca3-496b-924c-e627a412fee5&version=1'
    response=requests.get(url,headers={'Accept':'application/vnd.mendeley-public-dataset.1+json'},timeout=45);response.raise_for_status();files=response.json()
    target=EXTERNAL/'CHLW_Mendeley_v1';target.mkdir(exist_ok=True)
    records=[]
    for f in files:
        name=f['filename'];dest=target/name
        if dest.resolve().parent!=target.resolve(): raise RuntimeError('Unsafe source filename')
        if f['size']>10*2**20 or dest.suffix.lower() not in ['.py','.do','.r','.sas','.md','.txt','.pdf']:
            records.append({'filename':name,'status':'NOT_DOWNLOADED_NOT_CODE_OR_TOO_LARGE'});continue
        if not dest.exists():
            stream=requests.get(f['content_details']['download_url'],timeout=45);stream.raise_for_status();dest.write_bytes(stream.content)
        expected=f['content_details']['sha256_hash'];actual=sha(dest)
        if actual!=expected: raise RuntimeError('Mendeley source hash mismatch '+name)
        records.append({'filename':name,'sha256':actual,'bytes':dest.stat().st_size,'source_url':f['content_details']['download_url'],'status':'DOWNLOADED_NOT_EXECUTED'})
    save_json(RUN/'source_receipts/CHLW_Mendeley.json',{'source':'CHLW_Mendeley','repository':'https://data.mendeley.com/datasets/2pg764w6gs/1',
        'DOI':'10.17632/2pg764w6gs.1','license':'CC-BY-4.0','status':'SOURCE_DOWNLOADED_DATA_AUDIT_PENDING','files':records,'path':str(target)})
    print([(r['filename'],r['status']) for r in records],flush=True)
