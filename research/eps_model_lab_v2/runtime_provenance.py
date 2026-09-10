"""Hash installed model implementation files, without fitting or device use."""
from pathlib import Path
import argparse
import importlib
import importlib.metadata as metadata
import inspect
import platform
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v2.common import RUN,V1,read_json,save_json,sha,utcnow


def run(lane):
    names={'core':[('catboost','CatBoostRegressor'),('sklearn.ensemble','HistGradientBoostingRegressor'),('sklearn.linear_model','Ridge')],
        'prob':[('ngboost','NGBRegressor'),('ngboost.distns','Laplace')],
        'gpu':[('chronos','Chronos2Pipeline')]+[('neuralforecast.models',name) for name in ['TimeXer','SOFTS','XLinear','DLinear','LSTM']]}[lane]
    records=[]
    for module,name in names:
        cls=getattr(importlib.import_module(module),name);path=Path(inspect.getfile(cls))
        records.append({'class':module+'.'+name,'source_path':str(path),'source_sha256':sha(path),'source_bytes':path.stat().st_size})
    report={'created_utc':utcnow(),'lane':lane,'python':sys.executable,'python_version':platform.python_version(),
        'model_implementation_files':records,'scope':'Resolved implementation entry files; not a hash of every transitive binary dependency',
        'trained_or_allocated_GPU':False}
    if lane=='gpu':
        weight=read_json(V1/'weight_receipts/amazon__chronos-2.json')
        for file in weight['files']:
            if sha(file['path'])!=file['sha256']:raise AssertionError('Original Chronos weight changed')
        report['read_only_original_foundation_receipt']=weight
    save_json(RUN/'audit'/f'RUNTIME_SOURCE_PROVENANCE_{lane}.json',report,immutable=True)
    print('V2_RUNTIME_SOURCE_PROVENANCE',lane,len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lane',required=True);a=p.parse_args();run(a.lane)
