"""Development-only selection and fixed diagnostic definitions; no monitor tuning."""
from pathlib import Path
import argparse
import itertools
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from research.eps_model_lab_v2.common import RUN,read_json,save_json,sha,utcnow
from research.eps_model_lab_v2.model_common import TARGETS,surface,point_metrics


def load_predictions(phase):
    paths=sorted((RUN/'predictions'/phase).glob('*/main/*.parquet'))
    if not paths:raise RuntimeError('No predictions for '+phase)
    frames=[pd.read_parquet(p) for p in paths]
    frames.extend(pd.read_parquet(p) for p in sorted((RUN/'predictions/baselines').glob('*.parquet')))
    table=pd.concat(frames,ignore_index=True)
    table=surface(table,phase)
    if table.duplicated(['sample_id','target','eps_model_id']).any():raise RuntimeError('Duplicate scored identity')
    return table,paths


def subgroups(table):
    truth=table.actual_eps.to_numpy();current=table.baseline_persistence.to_numpy();scale=np.maximum(np.abs(current),.1)
    return {'negative':truth<0,'near_zero':np.abs(truth)<=.1,'positive':truth>0,
        'high_growth':(truth-current)/scale>.5,'low_growth':(truth-current)/scale<=.1,
        'loss_to_profit':(current<0)&(truth>0),'profit_to_loss':(current>0)&(truth<0)}


def detailed_scores(table,phase):
    recipes={r['model_id']:r for r in read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['recipes']}
    summary=[];years=[];subs=[];companies=[]
    for (model,target),group in table.groupby(['eps_model_id','target']):
        meta={'model_id':model,'target':target,'surface':phase,'family':recipes.get(model,{}).get('family','BASELINE'),
            'mode':recipes.get(model,{}).get('chronological_mode','NO_FIT'),'model_track':group.model_track.iloc[0]}
        yearly=[];company=[];sub=[]
        for y,g in group.groupby(group.asof_date.dt.year):
            row={**meta,'year':int(y),**point_metrics(g)};years.append(row)
            if row.get('MAE') is not None:yearly.append(row)
        for ticker,g in group.groupby('ticker'):
            row={**meta,'ticker':ticker,**point_metrics(g)};companies.append(row)
            if row.get('MAE') is not None:company.append(row)
        for name,mask in subgroups(group).items():
            row={**meta,'subgroup':name,'uses_realized_future_truth_not_origin_router':True,**point_metrics(group.loc[mask])}
            subs.append(row);sub.append(row)
        metrics=point_metrics(group)
        if metrics.get('MAE') is not None:
            finite=group[group.actual_eps.notna()].copy()
            finite['absolute_error']=(finite.forecast_eps-finite.actual_eps).abs()
            joint=finite.groupby(['ticker',finite.asof_date.dt.year]).absolute_error.agg(['mean','count'])
            worst_pair=joint['mean'].idxmax()
            harm=[r['MAE']/r['baseline_MAE'] for r in yearly if r['baseline_MAE']>0]
            danger=[r['MAE']/r['baseline_MAE'] for r in sub if r['subgroup'] in ['negative','loss_to_profit','profit_to_loss'] and r['truth_rows']>=20 and r.get('baseline_MAE',0)>0]
            worst=max(company,key=lambda r:r['MAE']) if company else None
            metrics.update(worst_year_harm_ratio=max(harm) if harm else None,
                negative_transition_max_harm_ratio=max(danger) if danger else None,
                catastrophic_subgroup_supported_count=len(danger),
                worst_year=max(yearly,key=lambda r:r['MAE'])['year'] if yearly else None,
                worst_ticker=worst['ticker'] if worst else None,
                worst_ticker_year=f'{worst_pair[0]}/{worst_pair[1]}',
                worst_ticker_year_MAE=float(joint.loc[worst_pair,'mean']),
                worst_ticker_year_rows=int(joint.loc[worst_pair,'count']),
                top_error_ticker_removed_MAE=point_metrics(group[group.ticker!=worst['ticker']]).get('MAE') if worst else None)
        summary.append({**meta,**metrics})
    return pd.DataFrame(summary),pd.DataFrame(years),pd.DataFrame(subs),pd.DataFrame(companies)


def gate(row):
    reasons=[]
    if row.get('coverage',0)<1:reasons.append('COVERAGE')
    if pd.isna(row.get('MAE_gain_fraction')) or row['MAE_gain_fraction']<.03:reasons.append('MAE_GAIN_LT_3_PERCENT')
    if row.get('MedianAE',np.inf)>row.get('baseline_MedianAE',0):reasons.append('MEDIAN_HARM')
    if row.get('p99_AE',np.inf)>1.25*row.get('baseline_p99_AE',0):reasons.append('P99_TAIL_HARM')
    if pd.isna(row.get('worst_year_harm_ratio')) or row['worst_year_harm_ratio']>1.30:reasons.append('WORST_YEAR_HARM')
    if pd.notna(row.get('negative_transition_max_harm_ratio')) and row['negative_transition_max_harm_ratio']>1.50:reasons.append('NEGATIVE_TRANSITION_HARM')
    return reasons


def stack_weights(matrix,y):
    n,m=matrix.shape
    objective=np.r_[np.zeros(m),np.ones(n)/n]
    constraint=np.r_[np.c_[matrix,-np.eye(n)],np.c_[-matrix,-np.eye(n)]]
    bounds=[(0,1)]*m+[(0,None)]*n
    fit=linprog(objective,A_ub=constraint,b_ub=np.r_[y,-y],A_eq=np.r_[np.ones(m),np.zeros(n)][None,:],b_eq=[1],bounds=bounds,method='highs')
    if not fit.success:raise RuntimeError('Development-only convex stack did not solve')
    return fit.x[:m].tolist()


def evaluate(phase='development',select=False):
    table,paths=load_predictions(phase);scores,years,subs,tickers=detailed_scores(table,phase)
    name={'development':'EPS_V2_DEVELOPMENT_CV.csv','confirmation':'EPS_V2_LOCKED_OOF_RESULTS.csv','monitor':'EPS_V2_RESEARCH_MONITOR_RESULTS.csv'}[phase]
    scores.to_csv(RUN/name,index=False)
    for suffix,frame in [('TIME_ROBUSTNESS',years),('SUBGROUP_RESULTS',subs),('COMPANY_ROBUSTNESS',tickers)]:
        frame.to_csv(RUN/f'EPS_V2_{phase.upper()}_{suffix}.csv',index=False)
    if not select:return
    if phase!='development':raise RuntimeError('Selection forbidden outside development')
    expected=[r for r in read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['recipes'] if not r.get('status')]
    for r in expected:
        group=table[table.eps_model_id==r['model_id']]
        if set(group.target)!=set(TARGETS):raise RuntimeError('Incomplete candidate '+r['model_id'])
        for target in TARGETS:
            reference=table[(table.eps_model_id=='persistence_observed')&(table.target==target)&table.actual_eps.notna()]
            actual=group[(group.target==target)&group.actual_eps.notna()]
            if set(reference.sample_id)!=set(actual.sample_id):
                raise RuntimeError('Candidate omitted fixed evaluation truth rows '+r['model_id']+'/'+target)
        # TTM 2018 labels are unavailable before 2019 by design, not failed inference.
        if set(group[group.target=='h1'].asof_date.dt.year)!={2015,2016,2017,2018}:raise RuntimeError('Missing development fold '+r['model_id'])
    recipes={r['model_id']:r for r in expected};candidates=scores[(scores.family!='BASELINE')&scores.target.isin(['h1','ttm'])].copy()
    candidates['gate_reasons']=[gate(row) for row in candidates.to_dict('records')]
    representatives=[];diagnostic_representatives=[]
    for (target,family),group in candidates.groupby(['target','family']):
        passing=group[group.gate_reasons.map(len)==0].sort_values(['MAE','MedianAE','model_id'])
        representative=(passing if len(passing) else group.sort_values(['MAE','MedianAE','model_id'])).iloc[0]
        record={'target':target,'family':family,'model_id':representative.model_id,
            'development_gate_pass':len(representative.gate_reasons)==0,'failure_reasons':representative.gate_reasons,
            'MAE':float(representative.MAE),'gain':float(representative.MAE_gain_fraction)}
        (representatives if record['development_gate_pass'] else diagnostic_representatives).append(record)
    # Confirm one preregistered development representative per family/lane even
    # after a development failure, but never call it a survivor or use monitor to rescue it.
    selected=sorted({r['model_id'] for r in representatives+diagnostic_representatives})
    correlations=[];plans=[];calibrators=[]
    for target in ['h1','ttm']:
        origin=table[(table.target==target)&table.actual_eps.notna()]
        pivot=origin.pivot(index='sample_id',columns='eps_model_id',values='forecast_eps').sort_index()
        truth=origin.drop_duplicates('sample_id').set_index('sample_id').loc[pivot.index,'actual_eps'].to_numpy()
        for a,b in itertools.combinations(selected,2):
            if a not in pivot or b not in pivot:continue
            ea=pivot[a].to_numpy()-truth;eb=pivot[b].to_numpy()-truth
            paired=np.abs((pivot[a].to_numpy()+pivot[b].to_numpy())/2-truth).mean()
            correlations.append({'target':target,'model_a':a,'model_b':b,'residual_corr':float(np.corrcoef(ea,eb)[0,1]),
                'absolute_error_corr':float(np.corrcoef(np.abs(ea),np.abs(eb))[0,1]),'equal_weight_MAE':float(paired),
                'gain_over_better_member':float(min(np.abs(ea).mean(),np.abs(eb).mean())-paired),
                'gain_over_global_single_leader':float(candidates[candidates.target==target].MAE.min()-paired)})
        for track in ['LOCAL_CAUSAL_RESEARCH','MIXED_RETROSPECTIVE_FOUNDATION']:
            supported=[r for r in representatives if r['target']==target and (track.startswith('MIXED') or recipes[r['model_id']]['family'] not in ['Chronos2','Chronos2ZeroShot'])]
            supported=sorted(supported,key=lambda r:(r['MAE'],r['model_id']))
            members=[];families=set()
            for r in supported:
                family='Chronos2' if r['family']=='Chronos2ZeroShot' else r['family']
                if family in families:continue
                families.add(family);members.append(r['model_id'])
                if len(members)==8:break
            # No forced weak member to make up four. A small/empty ensemble is
            # explicitly limited; baseline remains a separate portfolio control.
            if len(members)<2:
                plans.append({'target':target,'track':track,'status':'INSUFFICIENT_DEVELOPMENT_SURVIVORS','member_ids':members})
                continue
            matrix=pivot[members].to_numpy()
            plans.append({'target':target,'track':track,'status':'FROZEN_FROM_DEVELOPMENT_ONLY','member_ids':members,
                'convex_l1_weights':stack_weights(matrix,truth),'development_rows':len(truth),
                'member_count_below_four':len(members)<4,'family_cap':1})
        # All selected probabilistic models receive a fixed dev-only additive
        # conformal correction; no independence or exchangeability certification.
        for model in selected:
            group=origin[origin.eps_model_id==model]
            if not len(group) or 'q10' not in group or group.q10.isna().all():continue
            calibration_year=2018 if target=='h1' else 2017
            cal=group[group.asof_date.dt.year==calibration_year].dropna(subset=['q10','q90'])
            if not len(cal):continue
            nonconformity=np.maximum(cal.q10-cal.actual_eps,cal.actual_eps-cal.q90).to_numpy()
            quantile=min(np.ceil((len(cal)+1)*.8)/len(cal),1.0)
            correction=max(0.,float(np.quantile(nonconformity,quantile,method='higher')))
            calibrators.append({'model_id':model,'target':target,'calibration_year':calibration_year,'rows':len(cal),
                'interval80_additive_expansion':correction,'fit_surface':'PRE_2019_DEVELOPMENT_PREDICTION_RESIDUALS',
                'guaranteed_exchangeable_coverage':False})
    # Write locks LAST, only after every declared development candidate is present.
    save_json(RUN/'EPS_V2_ENSEMBLE_PLAN.json',{'created_utc':utcnow(),'plans':plans,'calibrators':calibrators,
        'selection_uses_confirmation_or_monitor':False,'development_scores_sha256':sha(RUN/name)},immutable=True)
    save_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json',{'created_utc':utcnow(),'selected_model_ids':selected,
        'development_gate_passing_representatives':representatives,'diagnostic_only_family_representatives':diagnostic_representatives,
        'all_development_prediction_hashes':{str(p.relative_to(RUN)):sha(p) for p in paths},
        'ensemble_plan_sha256':sha(RUN/'EPS_V2_ENSEMBLE_PLAN.json'),
        'confirmation_or_monitor_results_seen':False,'no_future_weight_updates':True},immutable=True)
    pd.DataFrame(correlations).to_csv(RUN/'EPS_V2_ERROR_CORRELATION.csv',index=False)
    print('V2_DEVELOPMENT_SELECTION_FROZEN',len(selected),'confirmation representatives;',len(representatives),'lane-family gates passing',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='development');p.add_argument('--select',action='store_true');a=p.parse_args();evaluate(a.phase,a.select)
