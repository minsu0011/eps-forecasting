"""Evidence aggregation and fail-closed research freeze; never trains or tunes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import sys
import shutil
import xml.etree.ElementTree as ET
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v2.common import RUN,V1,LAB,read_json,save_json,sha,utcnow
from research.eps_model_lab_v2.model_common import dataset,TARGETS,baseline
from research.eps_model_lab_v2.evaluation import load_predictions,gate

PHASES=['development','confirmation','monitor']


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def diagnostics():
    lock=read_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')
    plan=read_json(RUN/'EPS_V2_ENSEMBLE_PLAN.json')
    assert sha(RUN/'EPS_V2_DEVELOPMENT_CV.csv')==plan['development_scores_sha256']
    assert sha(RUN/'EPS_V2_ENSEMBLE_PLAN.json')==lock['ensemble_plan_sha256']
    coverage=[];scores=[];ttm_methods=[];frozen_frame=dataset()
    recipes=read_json(RUN/'EPS_V2_MODEL_RECIPES.json')['recipes']
    for phase in PHASES:
        table,paths=load_predictions(phase)
        expected=[r['model_id'] for r in recipes if not r.get('status')] if phase=='development' else lock['selected_model_ids']
        years={'development':range(2015,2019),'confirmation':range(2019,2022),'monitor':range(2022,2027)}[phase]
        reference=frozen_frame[frozen_frame.asof_date.dt.year.isin(years)]
        for model in expected:
            files=list((RUN/'predictions'/phase/model/'main').glob('*.parquet'))
            actual=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True)
            for t in TARGETS:
                group=actual[actual.target==t]
                if set(group.sample_id)!=set(reference.sample_id) or group.sample_id.duplicated().any():raise AssertionError('Missing/duplicate raw inference rows')
                if not np.isfinite(group.forecast_eps).all():raise AssertionError('Nonfinite raw inference')
                scored=table[(table.eps_model_id==model)&(table.target==t)&table.actual_eps.notna()]
                base=table[(table.eps_model_id=='persistence_observed')&(table.target==t)&table.actual_eps.notna()]
                if set(scored.sample_id)!=set(base.sample_id):raise AssertionError('Changed truth scoring denominator')
                coverage.append({'surface':phase,'model_id':model,'target':t,'raw_origin_rows':len(group),
                    'finite_predictions':len(group),'truth_rows':len(scored),'fixed_truth_mask_exact':True,'coverage_all_origins':1.0})
        name={'development':'EPS_V2_DEVELOPMENT_CV.csv','confirmation':'EPS_V2_LOCKED_OOF_RESULTS.csv','monitor':'EPS_V2_RESEARCH_MONITOR_RESULTS.csv'}[phase]
        score=pd.read_csv(RUN/name)
        # Earlier union-of-columns metric code populated interval coverage=0
        # for point-only models. Correct only the normalized reporting surface;
        # immutable development scores and point-selection hashes stay intact.
        probabilistic={r['model_id'] for r in recipes if r['family'] in ['NGBoostLaplace','Chronos2','Chronos2ZeroShot']}
        columns=[c for c in ['interval80_coverage','interval80_mean_width','mean_pinball_loss'] if c in score]
        score.loc[~score.model_id.isin(probabilistic),columns]=np.nan
        scores.append(score)
        from research.eps_model_lab_v2.model_common import point_metrics
        for (model,method),group in table[(table.target=='ttm')&table.actual_eps.notna()].groupby(['eps_model_id','ttm_target_method']):
            ttm_methods.append({'surface':phase,'model_id':model,'TTM_method':method,**point_metrics(group)})
    pd.DataFrame(coverage).to_csv(RUN/'EPS_V2_FULL_COVERAGE_AUDIT.csv',index=False)
    pd.concat(scores,ignore_index=True).to_csv(RUN/'EPS_V2_ALL_SURFACE_RESULTS.csv',index=False)
    pd.DataFrame(ttm_methods).to_csv(RUN/'EPS_V2_TTM_METHOD_DIAGNOSTICS.csv',index=False)
    for kind in ['SUBGROUP_RESULTS','TIME_ROBUSTNESS','COMPANY_ROBUSTNESS']:
        pd.concat([pd.read_csv(RUN/f'EPS_V2_{p.upper()}_{kind}.csv') for p in PHASES],ignore_index=True).to_csv(RUN/f'EPS_V2_{kind}.csv',index=False)
    pd.concat([pd.read_csv(RUN/f'EPS_V2_ENSEMBLE_{p.upper()}_RESULTS.csv') for p in PHASES[1:]],ignore_index=True).to_csv(RUN/'EPS_V2_ENSEMBLE_RESULTS.csv',index=False)
    frame=dataset();observed=np.isfinite(frame.current_ttm.to_numpy());b=baseline(frame,'ttm')
    np.testing.assert_array_equal(b[observed],frame.current_ttm.to_numpy()[observed])
    dev=frame[(frame.asof_date<pd.Timestamp('2019-01-01',tz='UTC'))&frame.y_ttm.notna()&(frame.label_asof_ttm<pd.Timestamp('2019-01-01',tz='UTC'))]
    save_json(RUN/'audit/LANE_C_FORMULATION_V2.json',{'created_utc':utcnow(),'status':'PASS_ALGEBRA_WITH_EXPLICIT_LIMITATION',
        'C_DIRECT':'Actual separate direct recipes fitted on native TTM truth',
        'C_RESIDUAL':'Actual separate residual recipes fitted against last observed native TTM persistence',
        'C_DELTA':'Exactly the same estimator as persistence residual where current native TTM is observed; not an independent diversity member',
        'all_origins':len(frame),'current_TTM_observed_origins':int(observed.sum()),'missing_current_TTM_origins':int((~observed).sum()),
        'development_train_truth_rows':len(dev),'development_current_TTM_observed_rows':int(dev.current_ttm.notna().sum()),
        'missing_current_TTM_policy':'last observed native TTM, or explicit 4*last EPS prediction fallback; never truth imputation',
        'path_derived_TTM':'NOT_EXECUTED: historical common share-basis vintage not certified; no false native TTM equivalence',
        'new_candidates_after_confirmation':False},immutable=True)
    save_json(RUN/'audit/FINAL_PREDICTION_COVERAGE_V2.json',{'created_utc':utcnow(),'status':'PASS','checks':len(coverage),
        'all_declared_year_target_origins_present':True,'all_forecasts_finite':True,
        'fixed_truth_masks_identical_to_baseline':True,'development_locks_unchanged':True,
        'interval_reporting_correction':'Point-only interval columns blanked in ALL_SURFACE_RESULTS; immutable development source unchanged; no effect on model or point gates'},immutable=True)
    print('V2_FINAL_DIAGNOSTICS_PASS',len(coverage),flush=True)


def preservation(receipt_name='FINAL_READONLY_PRESERVATION_V2.json'):
    spec=read_json(RUN/'audit/READONLY_BASELINE.json');differences=[]
    expected={r['path'] for r in spec['v1']};actual={str(p) for p in V1.rglob('*') if p.is_file()}
    if expected!=actual:differences.append({'v1_added':sorted(actual-expected),'v1_removed':sorted(expected-actual)})
    def verify(r):
        p=Path(r['path'])
        if not p.exists():return {'path':str(p),'error':'MISSING'}
        st=p.stat()
        if 'bytes' in r and (st.st_size!=r['bytes'] or st.st_mtime_ns!=r['mtime_ns']):return {'path':str(p),'error':'METADATA_CHANGED'}
        if r.get('sha256') and sha(p)!=r['sha256']:return {'path':str(p),'error':'CONTENT_CHANGED'}
        return None
    with ThreadPoolExecutor(max_workers=8) as pool:differences.extend(r for r in pool.map(verify,spec['v1']+spec['PE']) if r)
    report={'created_utc':utcnow(),'status':'PASS' if not differences else 'FAIL','v1_files':len(spec['v1']),
        'v1_content_hash_count':spec['v1_content_hash_count'],'PE_content_hash_files':len(spec['PE']),
        'v1_large_weight_scope':spec['large_v1_weights_protection'],'differences':differences}
    save_json(RUN/'audit'/receipt_name,report,immutable=True)
    if differences:raise RuntimeError('Read-only preservation failed')
    print('V2_READONLY_PRESERVATION_PASS',flush=True)


def freeze():
    spec=read_json(RUN/'prescore/RETRAINING_VERIFICATION_PLAN.json')
    replay=[];repeat=[]
    for phase in PHASES:
        for suffix in [f'TABULAR_REPLAY_{phase}_prob',f'TABULAR_REPLAY_{phase}_CatBoost',f'NEURAL_REPLAY_{phase}',f'CHRONOS_REPLAY_{phase}']:
            path=RUN/'audit'/f'FRESH_PROCESS_{suffix}.json';r=read_json(path)
            if r['status']!='PASS':raise RuntimeError('Replay gate failed '+str(path))
            expected_counts={'development':[300,100,80,24],'confirmation':[90,30,15,9],'monitor':[150,50,25,15]}
            index=0 if suffix.startswith('TABULAR') and suffix.endswith('prob') else 1 if suffix.startswith('TABULAR') else 2 if suffix.startswith('NEURAL') else 3
            if len(r['results'])!=expected_counts[phase][index]:raise RuntimeError('Missing saved replay identities')
            replay.append({'path':str(path.relative_to(RUN)),'sha256':sha(path),**r})
    for kind in ['prob','CatBoost','neural','chronos']:
        for rep in ['repeat1','repeat2']:
            path=RUN/'audit'/f'FRESH_RETRAIN_{kind}_{rep}.json';r=read_json(path)
            if r['status']!='PASS':raise RuntimeError('Retraining gate failed '+str(path))
            repeat.append({'path':str(path.relative_to(RUN)),'sha256':sha(path),**r})
    for audit in ['FINAL_PREDICTION_COVERAGE_V2.json','FINAL_READONLY_PRESERVATION_V2.json','PIT_MUTATION_AUDIT_V2.json']:
        if read_json(RUN/'audit'/audit)['status']!='PASS':raise RuntimeError(audit)
    ensemble_replay=read_json(RUN/'audit/FRESH_PROCESS_ENSEMBLE_REPLAY.json')
    if ensemble_replay['status']!='PASS' or len(ensemble_replay['results'])!=32:raise RuntimeError('Ensemble replay incomplete')
    tests=ET.parse(RUN/'audit/FINAL_CONTRACT_TESTS.xml').getroot()[0].attrib
    if int(tests['tests'])!=35 or any(int(tests[key]) for key in ['failures','errors','skipped']):raise RuntimeError('Contract regression test gate failed')
    unit=read_json(RUN/'EPS_SHARE_UNIT_AUDIT_V2.json')
    if unit['automatic_size_rescaling'] or unit['NI_over_EPS_inference']:raise RuntimeError('Forbidden unit repair')
    save_json(RUN/'EPS_V2_REPLAY_AUDIT.json',{'created_utc':utcnow(),'status':'PASS','actual_new_process_reports':replay,
        'ensemble_replay':ensemble_replay,
        'scope':'All declared saved annual models/heads across development, confirmation and monitor'},immutable=True)
    repeated=[r for report in repeat for r in report['results']]
    cat_state=read_json(RUN/'audit/CATBOOST_CHECKPOINT_STATE_V2.json')
    expected_cat=sum(r['family']=='CatBoost' for r in repeated)
    if cat_state['status']!='PASS' or cat_state['actual_native_model_JSON_comparisons']!=expected_cat:raise RuntimeError('CatBoost learned-state validation incomplete')
    for model in spec['model_ids']:
        rows=[r for r in repeated if r['model_id']==model]
        for phase,years in spec['years_by_phase'].items():
            for year in years:
                for rep in ['repeat1','repeat2']:
                    group=[r for r in rows if r['phase']==phase and r['year']==year and r['replicate']==rep]
                    if len(group) not in [1,5] or not all(r['status']=='PASS' for r in group):raise RuntimeError('Incomplete repeat coverage')
    save_json(RUN/'EPS_V2_RETRAINING_REPRODUCIBILITY.json',{'created_utc':utcnow(),'status':'PASS','reports':repeat,
        'actual_additional_fit_units':sum(r['actual_new_fit'] for r in repeated),
        'zero_shot_inference_units':sum(r['zero_shot_inference_only'] for r in repeated),
        'all_predictions_exact':True,'same_hardware_environment_only':True,'universal_reproducibility_claim':False,
        'checkpoint_byte_mismatches':sum(r['checkpoint_file_bytes_equal'] is False for r in repeated),
        'CatBoost_native_learned_state_audit':cat_state,
        'repeated_years':[2019,2020,2021,2022,2023,2024,2025,2026],
        'new_fit_repeats':2,'baseline_main_fit_additional':True},immutable=True)
    survivors=[]
    for record in spec['individual_gates']:
        if record['confirmation_gate_pass']:
            survivors.append({**record,'role':'RESEARCH_SURVIVOR','formal_certified':False,'production':False,
                'source_quality':'PARTIALLY_VERIFIED_NATIVE','saved_replay':'PASS','fresh_retraining':'PASS_TWO_ADDITIONAL_FITS' if not record['family'].endswith('ZeroShot') else 'ZERO_SHOT_INFERENCE_ONLY_NO_FIT',
                'model_track':'RETROSPECTIVE_FOUNDATION' if record['family'].startswith('Chronos') else 'LOCAL_CAUSAL_RESEARCH'})
    ensembles=[{**r,'status':'FROZEN_RESEARCH_ENSEMBLE','development_status':'MEMBERS_AND_WEIGHTS_FITTED_ONLY_ON_DEVELOPMENT; NO_UNBIASED_DEVELOPMENT_STACK_SCORE',
        'formal_certified':False} for r in spec['ensemble_gates'] if r['confirmation_gate_pass']]
    # A frozen ensemble is not a separately dev-gate-certified single model.
    c=[r for r in survivors if r['target']=='ttm']
    result={'created_utc':utcnow(),'status':'RESEARCH_ONLY_WITH_SOURCE_LIMITATIONS','individual_survivors':survivors,
        'frozen_ensemble_candidates':ensembles,'LANE_C_STATUS':'RESEARCH_SURVIVORS' if c else 'LANE_C_NOT_READY',
        'Track_A':'DATA_BLOCKED_NO_VERIFIED_PRE2019_ACCOUNTING','Track_M':'DISABLED',
        'formal_certified':False,'production_promotions':0,'confirmation_already_viewed':True,
        'monitor_descriptive_only':True,'source_original_audit_complete':False,
        'protocol_sha256':sha(RUN/'prescore/RESEARCH_PROTOCOL_V2.json'),
        'recipes_sha256':sha(RUN/'EPS_V2_MODEL_RECIPES.json'),'selection_sha256':sha(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')}
    save_json(RUN/'EPS_V2_FROZEN_RESEARCH_SURVIVORS.json',result,immutable=True)
    registry=read_json(RUN/'EPS_V2_MODEL_REGISTRY.json');save_json(RUN/'audit/PRESCORE_MODEL_REGISTRY_COPY.json',registry,immutable=True)
    supported={r['model_id'] for r in survivors};selected=set(read_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')['selected_model_ids'])
    development_scores=pd.read_csv(RUN/'EPS_V2_DEVELOPMENT_CV.csv')
    for r in registry['models']:
        if r.get('feature_track')=='A':r['status']='DATA_BLOCKED_NO_VERIFIED_DEVELOPMENT_ACCOUNTING'
        else:r['status']='FROZEN_RESEARCH_SURVIVOR' if r['model_id'] in supported else 'RESEARCH_DIAGNOSTIC_GATE_FAILED' if r['model_id'] in selected else 'DEVELOPMENT_ONLY_NOT_SELECTED'
        r['survivor_targets']=[s['target'] for s in survivors if s['model_id']==r['model_id']]
        r['actual_development_prediction_files']=len(list((RUN/'predictions/development'/r['model_id']/'main').glob('*.parquet')))
        r['development_gate_details']=[{'target':row['target'],'MAE':row['MAE'],
            'MAE_gain_fraction':row['MAE_gain_fraction'],'failure_reasons':gate(row)}
            for row in development_scores[(development_scores.model_id==r['model_id'])&development_scores.target.isin(['h1','ttm'])].to_dict('records')]
        r['formal_certified']=False
    registry['updated_utc']=utcnow();registry['production_promotions']=0
    save_json(RUN/'EPS_V2_MODEL_REGISTRY.json',registry)
    roles={'created_utc':utcnow(),'survivor_roles':survivors,'frozen_ensembles':ensembles,
        'simple_baseline':['persistence_observed','seasonal_observed'],
        'confirmation_gates_of_development_passes':spec['individual_gates'],
        'development_failed_diagnostic_representatives':read_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')['diagnostic_only_family_representatives'],
        'primary_role_mapping_from_development':{'EPS_A_CORE':'Chronos2_N_joint_rolling',
            'EPS_A_LOCAL_CORE':'HistGB_N_direct','EPS_ROBUST':'NGBoostLaplace_N_direct',
            'EPS_DIVERSITY_TEMPORAL':'LSTM_N_residual_recency','EPS_TABULAR_COMPLEMENT':'CatBoost_N_direct',
            'EPS_FOUNDATION_ZERO_SHOT_CONTROL':'Chronos2ZeroShot_N_native','EPS_C_CORE':None},
        'unsupported_roles_not_filled':['C_CORE'] if not c else [],'same_family_cap':1,'monitor_not_a_selection_surface':True}
    save_json(RUN/'EPS_V2_PORTFOLIO_ROLES.json',roles,immutable=True)
    print('V2_RESEARCH_FREEZE',len(survivors),'individual lane/model roles;',len(ensembles),'ensemble candidates;',result['LANE_C_STATUS'],flush=True)


def seal():
    if not (RUN/'STOP_MONITOR').exists():raise RuntimeError('Stop telemetry writer before checksums')
    if read_json(RUN/'audit/FINAL_PROCESS_QUIESCENCE.json')['status']!='PASS':raise RuntimeError('V2 process still active')
    preservation('SEAL_READONLY_RECHECK_V2.json')
    sources=sorted(p for p in LAB.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    source_records=[{'path':str(p.relative_to(PROJECT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sources]
    save_json(RUN/'EPS_V2_SOURCE_MANIFEST.json',{'created_utc':utcnow(),'files':source_records,
        'read_only_imports':'V2 fiscal builder uses existing PE pure helpers; no PE writer invoked',
        'external_sources':'Original model/package identities are in ENVIRONMENT_LOCK and model receipts',
        'runtime_provenance':{lane:read_json(RUN/'audit'/f'RUNTIME_SOURCE_PROVENANCE_{lane}.json') for lane in ['core','prob','gpu']}},immutable=True)
    required=['EPS_DATA_V2_CONTRACT.json','EPS_SOURCE_CONTEXT_LEDGER_V2.parquet','EPS_SHARE_UNIT_AUDIT_V2.json',
        'EPS_DURATION_CONTEXT_AUDIT_V2.json','EPS_DATA_GEOMETRY_LOCK_V2.json','EPS_DATASET_MANIFEST_V2.json',
        'EPS_V2_MODEL_RECIPES.json','EPS_V2_MODEL_REGISTRY.json','EPS_V2_DEVELOPMENT_CV.csv','EPS_V2_LOCKED_OOF_RESULTS.csv',
        'EPS_V2_RESEARCH_MONITOR_RESULTS.csv','EPS_V2_SUBGROUP_RESULTS.csv','EPS_V2_TIME_ROBUSTNESS.csv',
        'EPS_V2_ERROR_CORRELATION.csv','EPS_V2_ENSEMBLE_PLAN.json','EPS_V2_ENSEMBLE_RESULTS.csv','EPS_V2_PORTFOLIO_ROLES.json',
        'EPS_V2_REPLAY_AUDIT.json','EPS_V2_RETRAINING_REPRODUCIBILITY.json','EPS_V2_ENVIRONMENT_LOCK.json','EPS_V2_SOURCE_MANIFEST.json',
        'EPS_PE_COMPATIBILITY_CONTRACT_V2.md','EPS_PE_READY_OUTPUT_SCHEMA.json','EPS_V2_FROZEN_RESEARCH_SURVIVORS.json','EPS_MODEL_LAB_V2_FINAL_REPORT.md']
    if any(not (RUN/name).is_file() for name in required):raise RuntimeError('Missing required deliverable')
    save_json(RUN/'RUN_STATE.json',{'created_utc':utcnow(),'status':'CLOSED_RESEARCH_WAVE_WITH_DECLARED_SOURCE_BLOCKERS',
        'required_artifacts_present':len(required),'pending_training_jobs':0,'formal_certified':False,
        'Track_A':'DATA_BLOCKED','next_wave_requires_new_authority':True})
    paths=sorted(p for p in RUN.rglob('*') if p.is_file() and p.name not in ['CHECKSUMS.sha256','CHECKSUM_VERIFICATION.json'])
    with ThreadPoolExecutor(max_workers=4) as pool:digests=list(pool.map(sha,paths))
    lines=[f'{digest}  {p.relative_to(RUN).as_posix()}' for p,digest in zip(paths,digests)]
    (RUN/'CHECKSUMS.sha256').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with ThreadPoolExecutor(max_workers=4) as pool:repeated=list(pool.map(sha,paths))
    if repeated!=digests:raise RuntimeError('Artifact changed during checksum verification')
    for record in source_records:
        if sha(PROJECT/record['path'])!=record['sha256']:raise RuntimeError('Source changed during seal')
    save_json(RUN/'audit/CHECKSUM_VERIFICATION.json',{'created_utc':utcnow(),'status':'PASS','verified_files':len(paths),
        'verified_bytes':sum(p.stat().st_size for p in paths),'checksum_sha256':sha(RUN/'CHECKSUMS.sha256'),
        'excluded_self_referential_files':['CHECKSUMS.sha256','audit/CHECKSUM_VERIFICATION.json'],
        'source_files_reverified':len(source_records)},immutable=True)
    print('V2_FINAL_SEAL_PASS',len(paths),'files',flush=True)


def quiescence():
    import os
    import psutil
    active=[];self_and_launchers={os.getpid()}|{p.pid for p in psutil.Process().parents()}
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            if process.pid in self_and_launchers or not process.info['name'].lower().startswith('python'):continue
            args=process.info['cmdline'] or []
            if any('research/eps_model_lab_v2/' in s.replace('\\','/') for s in args):
                active.append({'pid':process.pid,'command':args})
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    report={'created_utc':utcnow(),'status':'PASS' if not active else 'FAIL','active_V2_python_processes':active,
        'scoped_to_V2_only':True,'unrelated_user_processes_not_stopped':True}
    save_json(RUN/'audit/FINAL_PROCESS_QUIESCENCE.json',report,immutable=True)
    if active:raise RuntimeError('V2 jobs still running')
    print('V2_PROCESS_QUIESCENCE_PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['diagnostics','preservation','freeze','seal','quiescence']);a=p.parse_args();globals()[a.action]()
