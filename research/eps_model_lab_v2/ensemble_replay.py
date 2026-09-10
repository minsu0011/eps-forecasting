"""Separate-process deterministic composition replay; no ensemble fitting."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v2.common import RUN,read_json,save_json,sha,utcnow
from research.eps_model_lab_v2.evaluation import load_predictions


def run():
    plan=read_json(RUN/'EPS_V2_ENSEMBLE_PLAN.json');results=[]
    for phase in ['confirmation','monitor']:
        table,_=load_predictions(phase);stored=pd.read_parquet(RUN/'predictions'/f'ensemble_{phase}.parquet')
        for r in plan['plans']:
            if r['status']!='FROZEN_FROM_DEVELOPMENT_ONLY':continue
            members=r['member_ids'];source=table[(table.target==r['target'])&table.eps_model_id.isin(members)]
            matrix=source.pivot(index='sample_id',columns='eps_model_id',values='forecast_eps').sort_index()[members]
            values=matrix.to_numpy();n=len(members);trim=int(.2*n)
            variants={'mean':values.mean(axis=1),'median':np.median(values,axis=1),
                'trimmed_mean':np.sort(values,axis=1)[:,trim:n-trim].mean(axis=1),
                'convex_l1':values@np.asarray(r['convex_l1_weights'])}
            for method,p in variants.items():
                name=f'ENS_{r["track"]}_{r["target"]}_{method}'
                expected=stored[stored.eps_model_id==name].set_index('sample_id').loc[matrix.index,'forecast_eps'].to_numpy()
                np.testing.assert_array_equal(p,expected)
                results.append({'surface':phase,'model_id':name,'rows':len(p),'status':'PASS','point_arrays_exact':True})
    save_json(RUN/'audit/FRESH_PROCESS_ENSEMBLE_REPLAY.json',{'created_utc':utcnow(),'status':'PASS',
        'frozen_plan_sha256':sha(RUN/'EPS_V2_ENSEMBLE_PLAN.json'),'results':results,'no_refitting':True},immutable=True)
    print('V2_ENSEMBLE_REPLAY_PASS',len(results),flush=True)


if __name__=='__main__':run()
