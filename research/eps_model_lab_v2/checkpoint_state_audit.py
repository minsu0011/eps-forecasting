"""Resolve CatBoost serialization metadata differences without changing models."""
from pathlib import Path
import hashlib
import json
import pickle
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v2.common import RUN,read_json,save_json,utcnow,sha


def normalized(path):
    with path.open('rb') as stream:model=pickle.load(stream)['model']
    destination=RUN/'audit/catboost_native_state'/path.relative_to(RUN/'models').with_suffix('.json')
    if destination.exists():return read_json(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    model.save_model(str(destination),format='json')
    return read_json(destination)


def run():
    results=[]
    for rep in ['repeat1','repeat2']:
        report=read_json(RUN/'audit'/f'FRESH_RETRAIN_CatBoost_{rep}.json')
        for r in report['results']:
            root=RUN/'models'/r['phase']/r['model_id'];stem=f"{r['year']}_{r['target']}.pkl"
            a=normalized(root/'main'/stem);b=normalized(root/rep/stem)
            info_a=a.pop('model_info',{});info_b=b.pop('model_info',{})
            differing=[k for k in set(info_a)|set(info_b) if info_a.get(k)!=info_b.get(k)]
            if a!=b:raise AssertionError('CatBoost learned tree/feature state differs')
            if set(differing)-{'model_guid','train_finish_time'}:raise AssertionError('Unexplained CatBoost metadata difference '+str(differing))
            digest=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            results.append({'model_id':r['model_id'],'year':r['year'],'target':r['target'],'phase':r['phase'],
                'replicate':rep,'status':'PASS','all_non_model_info_JSON_fields_exact':True,
                'differing_metadata_keys':sorted(differing),'canonical_learned_state_sha256':digest})
    save_json(RUN/'audit/CATBOOST_CHECKPOINT_STATE_V2.json',{'created_utc':utcnow(),'status':'PASS',
        'actual_native_model_JSON_comparisons':len(results),'results':results,
        'original_checkpoint_bytes_not_modified':True,'metadata_keys_excluded':['model_guid','train_finish_time']},immutable=True)
    print('V2_CATBOOST_LEARNED_STATE_EXACT',len(results),flush=True)


if __name__=='__main__':run()
