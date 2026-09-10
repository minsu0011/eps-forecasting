"""Paired evidence and diversity diagnostics; never retunes frozen ensemble weights."""
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import atomic_csv,metric_row
from research.eps_model_lab_v1.ensemble_models import family


def cluster_ci(frame,differences,column,draws=2000):
    d=pd.DataFrame({'cluster':frame[column].to_numpy(),'delta':differences})
    groups=d.groupby('cluster').delta.agg(['sum','count'])
    if len(groups)<2: return [None,None]
    rng=np.random.default_rng(1729);pick=rng.integers(0,len(groups),size=(draws,len(groups)))
    estimate=groups['sum'].to_numpy()[pick].sum(axis=1)/groups['count'].to_numpy()[pick].sum(axis=1)
    return [float(v) for v in np.quantile(estimate,[.025,.975])]


def run():
    pool={p.stem:pd.read_parquet(p) for p in sorted((RUN/'predictions').glob('*.parquet'))}
    eligibility={name:json.loads((RUN/'model_receipts'/f'{name}.json').read_text(encoding='utf-8')).get('status')=='FULL_RESEARCH_SCORED' for name in pool}
    selection=pd.read_csv(RUN/'EPS_SELECTION_LEADERBOARD_V1.csv')
    baseline_ids=['persistence_observed','drift_observed','seasonal_observed','seasonal_drift_observed','mean_observed']
    baseline_choices={};pairs=[];diversity=[];slices=[];roles=[]
    for target in ['h1','h2','h3','h4','ttm']:
        b=selection[(selection.target_key==target)&selection.eps_model_id.isin(baseline_ids)].copy()
        # The strongest available baseline is chosen on OOF, never on test.
        # Require the maximum available coverage among observed-only controls.
        b=b[b.coverage==b.coverage.max()]
        b['rank']=b[['MAE','MedianAE','price_scaled_MAE']].rank(pct=True).mean(axis=1)
        reference=b.sort_values(['rank','eps_model_id']).iloc[0].eps_model_id;baseline_choices[target]=reference
        ref=pool[reference];ref=ref[ref.target_key==target].set_index('sample_id')
        for model,p in pool.items():
            q=p[p.target_key==target].set_index('sample_id')
            if q.empty: continue
            for surface in ['SELECTION_OOF','RESEARCH_TEST']:
                if surface=='SELECTION_OOF':
                    g=q[(q.split=='VALIDATION_OOF')&(q.label_asof<pd.Timestamp('2022-01-01',tz='UTC'))]
                else: g=q[q.split=='RESEARCH_TEST']
                compare=ref.reindex(g.index)
                mask=np.isfinite(g.actual_eps)&np.isfinite(g.predicted_eps)&np.isfinite(compare.predicted_eps)
                g=g[mask];compare=compare[mask]
                if not len(g): continue
                model_ae=np.abs(g.predicted_eps-g.actual_eps).to_numpy();base_ae=np.abs(compare.predicted_eps-g.actual_eps).to_numpy()
                delta=model_ae-base_ae;g=g.assign(origin_year=g.asof_date.dt.year)
                ci=cluster_ci(g,delta,'ticker');time_ci=cluster_ci(g,delta,'origin_year')
                pairs.append({'eps_model_id':model,'target_key':target,'surface':surface,'baseline':reference,'paired_rows':len(g),
                    'model_MAE':float(model_ae.mean()),'baseline_MAE':float(base_ae.mean()),'relative_MAE_gain':float(1-model_ae.mean()/base_ae.mean()),
                    'paired_MAE_delta':float(delta.mean()),'ticker_cluster_ci95_low':ci[0],'ticker_cluster_ci95_high':ci[1],
                    'year_cluster_ci95_low':time_ci[0],'year_cluster_ci95_high':time_ci[1],
                    'row_win_rate':float((model_ae<base_ae).mean()),'independent_ticker_clusters':g.ticker.nunique(),
                    'independent_year_clusters':g.origin_year.nunique(),'inference_pit_flag':bool(g.pit_valid.all()),
                    'uncertainty_scope':'Exploratory paired resampling, not multiplicity-adjusted certification; labels overlap across origins'})
                for kind,col in [('ORIGIN_YEAR','origin_year'),('TICKER','ticker')]:
                    for key,sub in g.groupby(col):
                        slices.append({'eps_model_id':model,'target_key':target,'surface':surface,'slice_type':kind,'slice':str(key),**metric_row(sub)})
        # Correlations/oracle use only the purged OOF surface for portfolio decisions.
        vectors={};truth=None
        for model,p in pool.items():
            if model.startswith('ens_') or not eligibility[model]: continue
            q=p[(p.target_key==target)&(p.split=='VALIDATION_OOF')&(p.label_asof<pd.Timestamp('2022-01-01',tz='UTC'))].set_index('sample_id')
            if q.empty: continue
            vectors[model]=q.predicted_eps
            if truth is None: truth=q.actual_eps
        matrix=pd.DataFrame(vectors);truth=truth.reindex(matrix.index)
        # Some first sorted specialists could have shorter target rows. Rebuild
        # truth from all aligned immutable labels without choosing a model's mask.
        truth=pd.concat([p[(p.target_key==target)&(p.split=='VALIDATION_OOF')&(p.label_asof<pd.Timestamp('2022-01-01',tz='UTC'))]
                        [['sample_id','actual_eps']] for p in pool.values()]).drop_duplicates('sample_id').set_index('sample_id').actual_eps.reindex(matrix.index)
        for first,second in itertools.combinations(matrix.columns,2):
            mask=np.isfinite(matrix[first])&np.isfinite(matrix[second])&np.isfinite(truth)
            if mask.sum()<50: continue
            a=matrix.loc[mask,first].to_numpy();b=matrix.loc[mask,second].to_numpy();y=truth[mask].to_numpy()
            ea=a-y;eb=b-y
            def corr(x,z):
                c=np.corrcoef(x,z)[0,1]
                return float(c) if np.isfinite(c) else None
            oracle=float(np.minimum(np.abs(ea),np.abs(eb)).mean())
            diversity.append({'target_key':target,'model_a':first,'model_b':second,'rows':int(mask.sum()),
              'prediction_corr':corr(a,b),'residual_corr':corr(ea,eb),'absolute_error_corr':corr(np.abs(ea),np.abs(eb)),
              'mean_absolute_prediction_disagreement':float(np.abs(a-b).mean()),'oracle_pair_MAE_NOT_EXECUTABLE_MODEL':oracle,
              'prediction_sign_disagreement_frequency':float((np.sign(a)!=np.sign(b)).mean()),
              'numerically_distinct_prediction_frequency':float((~np.isclose(a,b,rtol=1e-6,atol=1e-8)).mean()),
              'oracle_gain_over_best_single_NOT_DEPLOYABLE':float(min(np.abs(ea).mean(),np.abs(eb).mean())-oracle),
              'equal_pair_actual_MAE':float(np.abs((a+b)/2-y).mean()),'identical_predictions':bool(np.array_equal(a,b)),
              'surface':'SELECTION_OOF_AVAILABLE_BEFORE_2022'})
    pair_frame=pd.DataFrame(pairs);atomic_csv(pair_frame,RUN/'EPS_PAIRED_BASELINE_COMPARISON.csv')
    atomic_csv(pd.DataFrame(slices),RUN/'EPS_YEAR_TICKER_SLICES.csv')
    atomic_csv(pd.DataFrame(diversity),RUN/'EPS_COMPLEMENTARITY_OOF.csv')
    lock=json.loads((RUN/'ENSEMBLE_FREEZE_V1.json').read_text(encoding='utf-8'))
    members={n for p in lock['plans'] for n in p['member_ids']}
    for name,p in pool.items():
        oof=pair_frame[(pair_frame.eps_model_id==name)&(pair_frame.surface=='SELECTION_OOF')]
        gains=oof.relative_MAE_gain.dropna();coverage=selection[selection.eps_model_id==name].coverage
        if not eligibility[name]: role='DIAGNOSTIC_ONLY_INELIGIBLE_FOR_PORTFOLIO'
        elif name.startswith('ens_'): role='FROZEN_ENSEMBLE_RESEARCH_ONLY'
        elif name in baseline_ids: role='OBSERVED_BASELINE_COMPARATOR'
        elif name in members: role='FROZEN_ENSEMBLE_MEMBER_RESEARCH_SURVIVOR'
        elif len(gains) and (gains>0).any(): role='LANE_SPECIFIC_RESEARCH_SURVIVOR'
        elif len(coverage) and (coverage<.95).any(): role='NARROW_SCOPE_OR_AVAILABILITY_LIMITED'
        else: role='NO_STANDALONE_OOF_GAIN_IN_V1'
        roles.append({'model_id':name,'family':family(name),'portfolio_role':role,'selection_basis':'Available-before-2022 OOF only',
            'positive_gain_lanes':oof.loc[oof.relative_MAE_gain>0,'target_key'].tolist(),
            'pretraining_overlap_unresolved':not bool(p.pit_valid.all()),'deployable':False,'formal_promoted':False})
    save_json(RUN/'EPS_PORTFOLIO_ROLES_V1.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'roles':roles,
       'baseline_choices_by_target':baseline_choices,'role_selection_uses_test':False,
       'caveat':'A positive OOF gain alone is an exploratory survivor rule, not a statistical or deployment gate'})
    print('PORTFOLIO_ANALYSIS_COMPLETE','paired',len(pairs),'diversity',len(diversity),'roles',len(roles),flush=True)


if __name__=='__main__': run()
