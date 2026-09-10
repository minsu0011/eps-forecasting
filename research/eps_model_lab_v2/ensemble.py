"""Apply already-frozen development members, weights and interval corrections."""
from pathlib import Path
import argparse
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v2.common import RUN,read_json,save_json,sha,utcnow
from research.eps_model_lab_v2.evaluation import load_predictions,detailed_scores
from research.eps_model_lab_v2.model_common import point_metrics


def run(phase):
    if phase not in ['confirmation','monitor']:raise RuntimeError('Main ensemble results must be out of development fitting surface')
    plan=read_json(RUN/'EPS_V2_ENSEMBLE_PLAN.json');selection=read_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')
    assert sha(RUN/'EPS_V2_ENSEMBLE_PLAN.json')==selection['ensemble_plan_sha256']
    table,_=load_predictions(phase);output=[];calibration=[]
    for recipe in plan['plans']:
        if recipe['status']!='FROZEN_FROM_DEVELOPMENT_ONLY':continue
        target=recipe['target'];members=recipe['member_ids']
        subset=table[(table.target==target)&table.eps_model_id.isin(members)]
        pivot=subset.pivot(index='sample_id',columns='eps_model_id',values='forecast_eps').sort_index()
        if not set(members)<=set(pivot) or pivot[members].isna().any().any():raise RuntimeError('Missing frozen ensemble component')
        base=subset[subset.eps_model_id==members[0]].set_index('sample_id').loc[pivot.index].reset_index()
        matrix=pivot[members].to_numpy();n=len(members);trim=int(np.floor(.2*n))
        variants={'mean':matrix.mean(axis=1),'median':np.median(matrix,axis=1),
            'trimmed_mean':np.sort(matrix,axis=1)[:,trim:n-trim].mean(axis=1),
            'convex_l1':matrix@np.asarray(recipe['convex_l1_weights'])}
        for method,values in variants.items():
            result=base.copy();result['forecast_eps']=values
            result['eps_model_id']=f'ENS_{recipe["track"]}_{target}_{method}'
            result['model_track']=recipe['track'];result['ensemble_member_ids']='|'.join(members)
            for col in ['q10','q50','q90']:
                if col in result:result[col]=np.nan
            output.append(result)
    for entry in plan['calibrators']:
        raw=table[(table.eps_model_id==entry['model_id'])&(table.target==entry['target'])].copy()
        if raw.empty:continue
        raw['q10']-=entry['interval80_additive_expansion'];raw['q90']+=entry['interval80_additive_expansion']
        calibration.append({'model_id':entry['model_id'],'target':entry['target'],'surface':phase,
            'calibration_source_year':entry['calibration_year'],'interval80_additive_expansion':entry['interval80_additive_expansion'],
            **point_metrics(raw)})
    if output:
        combined=pd.concat(output,ignore_index=True);path=RUN/'predictions'/f'ensemble_{phase}.parquet'
        if path.exists():raise FileExistsError(path)
        combined.to_parquet(path,index=False)
        scores,years,subs,tickers=detailed_scores(combined,phase)
        scores.to_csv(RUN/f'EPS_V2_ENSEMBLE_{phase.upper()}_RESULTS.csv',index=False)
        years.to_csv(RUN/f'EPS_V2_ENSEMBLE_{phase.upper()}_YEAR_RESULTS.csv',index=False)
        subs.to_csv(RUN/f'EPS_V2_ENSEMBLE_{phase.upper()}_SUBGROUP_RESULTS.csv',index=False)
    pd.DataFrame(calibration).to_csv(RUN/f'EPS_V2_CALIBRATION_{phase.upper()}_RESULTS.csv',index=False)
    save_json(RUN/'audit'/f'ENSEMBLE_{phase}_APPLICATION.json',{'created_utc':utcnow(),'frozen_plan_sha256':sha(RUN/'EPS_V2_ENSEMBLE_PLAN.json'),
        'method_outputs':len(output),'calibrated_model_targets':len(calibration),'members_or_weights_refitted':False},immutable=True)
    print('V2_FROZEN_ENSEMBLES_APPLIED',phase,len(output),'calibrators',len(calibration),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);a=p.parse_args();run(a.phase)
