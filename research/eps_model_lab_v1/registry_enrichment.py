"""Bind model intake to executed artifacts; no scoring or model-selection changes."""
import json
from pathlib import Path
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,LAB,sha

CHECKPOINTS={
 'lag_llama_zero_shot_origin_isolated':'time-series-foundation-models/Lag-Llama',
 'moirai1p1_small_origin_isolated':'Salesforce/moirai-1.1-R-small',
 'moirai_moe_small_origin_isolated':'Salesforce/moirai-moe-1.0-R-small',
 'sundial_base_128m':'thuml/sundial-base-128m','tabpfn_v2_panel':'Prior-Labs/TabPFN-v2-reg',
 'chronos2_zero_shot':'amazon/chronos-2','chronos2_past_covariates':'amazon/chronos-2',
 'chronos2_eps_joint_finetuned':'amazon/chronos-2','chronos_bolt_mini':'amazon/chronos-bolt-mini',
 'timesfm_2p5':'google/timesfm-2.5-200m-pytorch','tirex_zero_shot':'NX-AI/TiRex',
 'tirex2':'NX-AI/TiRex-2','tirex2_covariates':'NX-AI/TiRex-2',
 'toto2_4m':'Datadog/Toto-2.0-4m','toto2_22m':'Datadog/Toto-2.0-22m',
 'moirai2_small':'Salesforce/moirai-2.0-R-small','moirai1p1_small':'Salesforce/moirai-1.1-R-small',
 'moirai_moe_small':'Salesforce/moirai-moe-1.0-R-small',
 'lag_llama_zero_shot':'time-series-foundation-models/Lag-Llama',
 'time_moe_50m':'Maple728/TimeMoE-50M','moment_embedding_ridge':'AutonLab/MOMENT-1-small',
 'ibm_patchtst_fm':'ibm-granite/granite-timeseries-patchtst-fm-r1'}


def implementation(name):
    # environment, primary installed distribution, reference source, local adapter
    if name.startswith('ens_'):return 'core','scipy',None,'ensemble_models.py'
    if name.startswith('ngboost'):return 'prob','ngboost','ngboost','probabilistic_models.py'
    if name in ['bayesian_ridge_distribution','gaussian_process_matern_distribution']:return 'prob','scikit-learn',None,'probabilistic_models.py'
    if name.endswith('_origin_isolated'):return 'moirai',('gluonts' if name.startswith('lag') else 'uni2ts'),('lag_llama' if name.startswith('lag') else 'moirai'),'origin_isolated_foundations.py'
    if name.startswith('sundial'):return 'extra','transformers','sundial','additional_foundation_wave.py'
    if name.startswith('tabpfn'):return 'extra','tabpfn','tabpfn','additional_foundation_wave.py'
    if name.startswith('nf_multivar_'):return 'gpu','neuralforecast','neuralforecast','multivariate_models.py'
    if name.startswith('nf_'):return 'gpu','neuralforecast','neuralforecast','neural_models.py'
    if name.startswith('ag_'):return 'ts','autogluon.timeseries','autogluon','autogluon_models.py'
    if name.startswith('darts_'):return 'ts','darts','darts','darts_models.py'
    if name.startswith('epspredict_'):return 'gpu','torch','epspredict','eps_native_models.py'
    if name.startswith('mlforecast_'):return 'core','mlforecast','mlforecast','mlforecast_models.py'
    if name.startswith('sf_'):return 'core','statsforecast','statsforecast','statistical_models.py'
    if name in ['foster','brown_rozeff','griffin_watts']:return 'core','statsmodels',None,'statistical_models.py'
    if name.startswith('hvz_'):return 'core','scikit-learn',None,'annual_accounting.py'
    if name.startswith('moirai'):return 'moirai','uni2ts','moirai','moirai_models.py'
    if name.startswith('lag_llama'):return 'moirai','gluonts','lag_llama','lag_llama_model.py'
    if name.startswith('tirex2'):return 'granite','tirex-2','tirex2','tirex2_models.py'
    if name.startswith('tirex_'):return 'gpu','tirex-ts','tirex','foundation_models.py'
    if name.startswith('time_moe'):return 'granite','transformers','time_moe','additional_foundation.py'
    if name.startswith('moment_'):return 'granite','momentfm','moment','additional_foundation.py'
    if name.startswith('ibm_'):return 'granite','granite-tsfm','granite_tsfm','ibm_patchtst_model.py'
    if name.startswith('chronos'):return 'gpu','chronos-forecasting','chronos',('chronos_finetune.py' if name=='chronos2_eps_joint_finetuned' else 'foundation_models.py')
    if name.startswith('timesfm'):return 'gpu','timesfm','timesfm','foundation_models.py'
    if name.startswith('toto'):return 'ts','toto-2','toto','foundation_models.py'
    if name.startswith('afden'):return 'core','xgboost','afden','core_models.py'
    if name.startswith('xgboost'):return 'core','xgboost',None,'core_models.py'
    if name=='lightgbm':return 'core','lightgbm',None,'core_models.py'
    if name.startswith('catboost'):return 'core','catboost',None,'core_models.py'
    return 'core','scikit-learn',None,'core_models.py'


def enrich(records,scores):
    sources={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (RUN/'source_receipts').glob('*.json')}
    packages={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (RUN/'package_source_manifests').glob('*.json')}
    environments={p.stem:json.loads(p.read_text(encoding='utf-8')) for p in (RUN/'environments').glob('*.json')}
    rolespath=RUN/'EPS_PORTFOLIO_ROLES_V1.json'
    roles={r['model_id']:r for r in json.loads(rolespath.read_text(encoding='utf-8'))['roles']} if rolespath.exists() else {}
    aliases={'HVZ_accounting_earnings_to_EPS':'hvz_gaap_annual_eps_originshares',
      'moment_embedding':'moment_embedding_ridge',**{'nf_'+n:'nf_multivar_'+n for n in ['TSMixerx','TSMixer','SOFTS','TimeMixer','iTransformer','TimeXer','RMoK','XLinear','StemGNN']}}
    diversitypath=RUN/'EPS_COMPLEMENTARITY_OOF.csv'
    diversity=pd.read_csv(diversitypath) if diversitypath.exists() else pd.DataFrame()
    best={}
    if not scores.empty and 'MAE' in scores:
        eligible_ids={name for name,r in records.items() if r['status']=='FULL_RESEARCH_SCORED'}
        eligible=scores[scores.coverage.ge(.999)&scores.MAE.notna()&scores.eps_model_id.isin(eligible_ids)]
        for target,g in eligible.groupby('target_key'):best[target]=g.sort_values(['MAE','eps_model_id']).iloc[0].eps_model_id
    for name,r in records.items():
        r.update(formal_certified=False,deployable=False,formal_promoted=False,run_identity=RUN.name,
          dataset_sha256=sha(RUN/'data/samples.parquet'),prior_research_test_seen_before_data_repair=True,
          source_vintage_certified=False,share_basis_vintage_certified=False)
        if name in aliases:
            r.update(status='ALIAS_SEE_EXECUTED_ADAPTATION',alias_of=aliases[name],eligible_forecaster=False,
               failure_reason='Not an additional independent scored model. See actual named adaptation.');continue
        if name=='ibm_ttm':
            r.update(status='DATA_GEOMETRY_BLOCKED',portfolio_role='DATA_BLOCKED',
               failure_reason='Official TTM minimum context 52 exceeds frozen 32 fiscal-quarter geometry; no invented observations or silent contract expansion.');continue
        if name=='moment_forecasting':
            r.update(status='UNTRAINED_HEAD_NOT_A_VALID_FORECAST',eligible_forecaster=False,
               failure_reason='Official embedding checkpoint is not an EPS-trained forecasting head; actual causal Ridge readout is moment_embedding_ridge.');continue
        if name=='nf_HINT':
            r.update(status='DATA_GEOMETRY_BLOCKED',portfolio_role='DATA_BLOCKED',
               failure_reason='No authorized additive hierarchy for diluted per-share EPS across companies. Company EPS cannot be summed into an economically coherent earnings hierarchy.');continue
        if name in ['darts_Croston','ag_Croston','ag_ADIDA','ag_IMAPA']:
            r.update(status='NOT_APPLICABLE_INTERMITTENT_NONNEGATIVE_DEMAND',eligible_forecaster=False,
               failure_reason='Intermittent nonnegative-demand specification does not preserve signed EPS contract; no positive clipping or silent reinterpretation.');continue
        if name=='nf_TimeLLM':
            r.update(status='DEFERRED_LLM_BACKBONE_RESOURCE_SCOPE',
               failure_reason='LLM-backbone integration is outside small checkpoint priority for this run; not falsely labeled import or model failure.');continue
        diagnostic=r['status'].startswith('FULL_DIAGNOSTIC_')
        if r['status']!='FULL_RESEARCH_SCORED' and not diagnostic:continue
        receiptpath=RUN/r['receipt'];receipt=json.loads(receiptpath.read_text(encoding='utf-8'));meta=receipt.get('metadata',receipt.get('adapter',{}))
        env,distribution,source,adapter=implementation(name)
        provenance=sources.get(source,{})
        r.update(environment=env,primary_distribution=distribution,adapter_path='research/eps_model_lab_v1/'+adapter,
           source=meta.get('source') or receipt.get('source') or r.get('source'),
           adapter_sha256=sha(LAB/adapter),receipt_sha256=sha(receiptpath),family=receipt.get('family',r['family']),
           features=meta.get('input_geometry') or meta.get('embedding_geometry') or meta.get('target_scope') or receipt.get('contract',{}).get('channels') or
             ('Frozen EPS/accounting feature allowlist' if adapter in ['core_models.py','mlforecast_models.py'] else 'Native causal EPS histories; see adapter and frozen geometry'),
           covariates=meta.get('covariates') or receipt.get('contract',{}).get('future_exogenous') or 'No future realized covariates',
           reference_repository=provenance.get('repository'),reference_git_commit=provenance.get('commit'),
           reference_source_receipt=('source_receipts/'+source+'.json') if source else None,
           executed_code_git_commit=provenance.get('commit') if source in ['epspredict','lag_llama','time_moe'] else None,
           commit=None,commit_semantics='No conflation: checkpoint revision, reference Git HEAD and executed wheel hashes are separate fields',
           zero_shot=meta.get('zero_shot',name in CHECKPOINTS and name not in ['chronos2_eps_joint_finetuned','moment_embedding_ridge']),
           finetuned=meta.get('finetuned',False),runtime=receipt.get('runtime_seconds'),
           GPU_time=receipt.get('GPU_time_seconds_including_load'),peak_gpu_allocated_gib=receipt.get('peak_gpu_allocated_gib'),
           GPU_time_scope='When present, adapter walltime including load/CPU preparation, not CUDA-event active-device time',
           RAM_measurement='Process/system telemetry shared with concurrent jobs; no unsupported per-model allocation attribution')
        if source:r['source']=provenance.get('repository',r.get('source'))
        if env in packages:
            match=next((p for p in packages[env]['packages'] if p['distribution']==distribution),None)
            r['actual_package_source_manifest']='package_source_manifests/'+env+'.json'
            r['actual_package_manifest_sha256']=sha(RUN/r['actual_package_source_manifest'])
            if match:r['version']=match['version'];r['executed_installed_source_status']=match['status']
        dist=next((p for p in environments.get(env,{}).get('packages',[]) if p['name'].lower().replace('_','-')==distribution.lower().replace('_','-')),None)
        r['package_license']=dist.get('license') if dist else None
        r['license']=meta.get('license') or provenance.get('license') or r.get('package_license') or 'UNRESOLVED'
        if name in CHECKPOINTS:
            wr=RUN/'weight_receipts'/(CHECKPOINTS[name].replace('/','__')+'.json');w=json.loads(wr.read_text(encoding='utf-8'))
            r.update(weight_id=w['weight_id'],weight_revision=w['revision'],weight_license=w.get('license'),
               weight_receipt=str(wr.relative_to(RUN)),weight_receipt_sha256=sha(wr),
               weights_sha256={Path(f['path']).name:f['sha256'] for f in w['files']},
               checkpoint_pretraining_PIT_status=w.get('pretraining_pit_status','UNRESOLVED'),
               reference_checkpoint_creation=w.get('created_at'))
        predpath=RUN/'predictions'/f'{name}.parquet';p=pd.read_parquet(predpath)
        r.update(prediction_sha256=sha(predpath),inference_PIT_flag=bool(p.pit_valid.all()),
           PIT_safe=False,PIT_safe_semantics='Strict certification is false; inference_PIT_flag reports implemented causal-input contract only',
           causal=True,point_forecast=True,native_quantile_rows=int(p[['p10_eps','p50_eps','p90_eps']].notna().all(axis=1).sum()),
           target_lane=receipt.get('target_lane',['A','B','C']),coverage_scope=receipt.get('target_lane') or 'All native evaluation origins; missing truth/prediction masks retained')
        if name in roles:r.update({k:v for k,v in roles[name].items() if k not in ['model_id','family']})
        r['portfolio_eligible']=not diagnostic
        r['candidate_unit']='ORIGIN_ISOLATION_CORRECTION_NOT_NEW_ARCHITECTURE' if name.endswith('_origin_isolated') else ('FROZEN_ENSEMBLE' if name.startswith('ens_') else 'CONFIGURED_RESEARCH_RECIPE_NOT_ASSUMED_INDEPENDENT_ARCHITECTURE')
        r['ensemble_pool_eligible']=(not diagnostic and not name.startswith('ens_') and name not in {
            'random_walk','random_walk_drift','seasonal_random_walk','seasonal_drift','historical_mean'})
        r['numerical_certified']=False
        if receipt.get('numerical_limitation'):
            r['numerical_limitation']=receipt['numerical_limitation']
            r['numerical_scope']=receipt.get('numerical_scope')
        if diagnostic:
            r.update(portfolio_role='DIAGNOSTIC_ONLY_INELIGIBLE_FOR_PORTFOLIO',
                smoke_status=receipt.get('smoke_status'),failure_reason=receipt.get('failure_reason'))
        if not scores.empty:
            selected=scores[(scores.eps_model_id==name)&(scores.split=='SELECTION_OOF_AVAILABLE_BEFORE_2022')]
            r['OOF_metrics_by_target']={z['target_key']:{k:z.get(k) for k in ['MAE','RMSE','MedianAE','price_scaled_MAE','bias','coverage']} for z in selected.to_dict('records')}
        if not diversity.empty:
            pairs=diversity[(diversity.model_a==name)|(diversity.model_b==name)]
            if len(pairs):r['minimum_absolute_OOF_residual_correlation']=float(pairs.residual_corr.abs().min())
            correlations={}
            for target,leader in best.items():
                q=pairs[(pairs.target_key==target)&((pairs.model_a==leader)|(pairs.model_b==leader))]
                if name==leader:correlations[target]={'reference_model':leader,'signed_residual_corr':1.0}
                elif len(q):correlations[target]={'reference_model':leader,'signed_residual_corr':float(q.iloc[0].residual_corr)}
            r['error_corr_best_by_target']=correlations
            r['error_corr_best']=correlations.get('h1',{}).get('signed_residual_corr')
    quality = RUN/'SOURCE_SHARE_UNIT_QUALITY_AUDIT.json'
    if quality.exists():
        audit = json.loads(quality.read_text(encoding='utf-8'))
        for entry in audit['consumers']:
            name = entry['model_id']
            if name in records:
                records[name].update(source_share_unit_warning=entry['source_unit_warning'],
                    source_share_unit_exposure=entry['exposure'], source_share_unit_audit=quality.name,
                    clean_share_feature_claim_allowed=entry['clean_share_feature_claim_allowed'],
                    current_status_means_scored_not_data_quality_certified=True)
    return records
