"""Durable environment/discovery/registry/checkpoint receipts for autonomous handoff."""
from datetime import datetime,timezone
import argparse
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,EXTERNAL,save_json,sha

def environment(name):
    import psutil
    distributions=[]
    for d in importlib.metadata.distributions():
        distributions.append({'name':d.metadata['Name'],'version':d.version,'license':d.metadata.get('License-Expression') or d.metadata.get('License','UNKNOWN')})
    data={'name':name,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
          'packages':sorted(distributions,key=lambda d:d['name'].lower()),'snapshot_utc':datetime.now(timezone.utc).isoformat(),
          'cpu_count':psutil.cpu_count(),'ram_total_gib':psutil.virtual_memory().total/2**30}
    try:
        import torch
        data.update(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    except ImportError: pass
    save_json(RUN/'environments'/f'{name}.json',data)
    freeze=subprocess.run(['uv','pip','freeze','--python',sys.executable],capture_output=True,text=True,encoding='utf-8',check=True)
    (RUN/'environments'/f'{name}_requirements_lock.txt').write_text(freeze.stdout,encoding='utf-8')
    if name=='ts':
        import darts.models as dm
        import autogluon.timeseries.models as am
        save_json(RUN/'ecosystem_discovery/darts.json',{'version':importlib.metadata.version('darts'),
            'models':[n for n in dir(dm) if inspect.isclass(getattr(dm,n))]})
        save_json(RUN/'ecosystem_discovery/autogluon.json',{'version':importlib.metadata.version('autogluon.timeseries'),
            'models':[n for n in dir(am) if n.endswith('Model') and inspect.isclass(getattr(am,n))]})
    print(name,'ENVIRONMENT_LOCKED',len(distributions),flush=True)

def clean(value):
    import math
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def registry():
    import pandas as pd
    from research.eps_model_lab_v1.core_models import CORE_SPECS,BASELINES
    # Registry generation must not import GPU runtimes into the CPU control env.
    import ast
    parsed=ast.parse((PROJECT/'research/eps_model_lab_v1/foundation_models.py').read_text(encoding='utf-8'))
    FS=next(ast.literal_eval(node.value) for node in parsed.body if isinstance(node,ast.Assign)
       and any(isinstance(t,ast.Name) and t.id=='SPECS' for t in node.targets))
    records={}
    def add(model,family,**more):
        records[model]={'model_id':model,'family':family,'architecture':model,'source':None,'commit':None,'license':None,
         'target_lane':['A','B','C'],'features':None,'covariates':None,'zero_shot':False,'finetuned':False,
         'PIT_safe':None,'causal':None,'deployable':False,'MAE':None,'RMSE':None,'MedianAE':None,'price_scaled_MAE':None,
         'bias':None,'coverage':None,'error_corr_best':None,'runtime':None,'GPU_time':None,'RAM':None,
         'status':'DISCOVERED_ADAPTER_PENDING','portfolio_role':'RESEARCH_ONLY','failure_reason':None,**more}
    for m in BASELINES: add(m,'ACCOUNTING_BASELINE')
    for m in CORE_SPECS: add(m,'TABULAR_ACCOUNTING',features='Frozen native-PIT wide feature allowlist')
    for m,(kind,repo,package) in FS.items(): add(m,'FOUNDATION_ZERO_SHOT',source='https://huggingface.co/'+repo,zero_shot=True,PIT_safe=False,causal=True)
    for name in ['LSTM','CNNLSTM','MultiFreqLSTM','MultiFreqCNNLSTM']: add('epspredict_'+name,'EPS_NATIVE_SEQUENCE',source='https://github.com/seferlab/epspredict',license='UNKNOWN_RESEARCH_ONLY')
    for name in ['foster','brown_rozeff','griffin_watts']: add(name,'ACCOUNTING_SARIMA')
    for name in ['ridge','lightgbm','xgboost','catboost']: add('mlforecast_'+name,'MLFORECAST')
    add('CHLW_analyst_bias_correction','EPS_NATIVE_ANALYST',source='https://data.mendeley.com/datasets/2pg764w6gs/1',
        status='DATA_BLOCKED',portfolio_role='DATA_BLOCKED',failure_reason='No point-in-time historical analyst consensus / I/B/E/S equivalent found. Current estimates cannot backfill history.')
    add('HVZ_accounting_earnings_to_EPS','CROSS_SECTIONAL_ACCOUNTING',source='https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/03/Zhang2015Jan.pdf',
        status='SPECIFICATION_INTAKEN_ADAPTER_PENDING',failure_reason='Published HVZ predicts annual dollar earnings; native annual accounting/share adapter required, not mislabeled quarterly EPS regression.')
    for ecosystem,prefix in [('statsforecast','sf_'),('neuralforecast','nf_'),('darts','darts_'),('autogluon','ag_')]:
        path=RUN/'ecosystem_discovery'/f'{ecosystem}.json'
        if not path.exists(): continue
        discovery=json.loads(path.read_text(encoding='utf-8'))
        for name in discovery['models']:
            # Discovery of a Python class is not discovery of an eligible EPS forecaster.
            excluded=('Classifier' in name or name.endswith('Filter') or name in {
                'EnsembleModel','MultivariateModel','RegressionModel','SKLearnModel','StatsForecastModel'})
            # Public AutoGluon classes have a Model suffix; prediction artifacts use display names.
            identity=name[:-5] if ecosystem=='autogluon' and name.endswith('Model') else name
            add(prefix+identity,ecosystem.upper(),version=discovery['version'],
                status='NOT_APPLICABLE_HELPER_OR_CLASSIFIER' if excluded else 'IMPORTED_ADAPTER_PENDING',
                eligible_forecaster=not excluded)
    for m in ['moirai2_small','moirai_moe_small','moirai1p1_small','lag_llama_zero_shot','ibm_ttm','ibm_patchtst_fm','moment_forecasting','moment_embedding','time_moe_50m','tirex2','sundial_base_128m','tabpfn_v2_panel',
       'moirai1p1_small_origin_isolated','moirai_moe_small_origin_isolated','lag_llama_zero_shot_origin_isolated']:
        add(m,'FOUNDATION_EXTERNAL',zero_shot=True,PIT_safe=False)
    for m in ['ngboost_normal','ngboost_laplace','bayesian_ridge_distribution','gaussian_process_matern_distribution']:
        add(m,'NATIVE_PROBABILISTIC_TABULAR',zero_shot=False,PIT_safe=False)
    leaderboard=RUN/'EPS_SELECTION_LEADERBOARD_V1.csv'
    scores=pd.read_csv(leaderboard) if leaderboard.exists() else pd.DataFrame(columns=['eps_model_id','target_key','split','mask'])
    for p in sorted((RUN/'model_receipts').glob('*.json')):
        receipt=json.loads(p.read_text(encoding='utf-8'));model=receipt.get('model_id')
        if model is None or 'initial_' in p.stem: continue
        if model not in records: add(model,receipt.get('family','EXTERNAL'))
        r=records[model];r.update(status=receipt.get('status','UNKNOWN'),receipt=str(p.relative_to(RUN)),
             runtime=receipt.get('runtime_seconds'),GPU_time=receipt.get('GPU_time_seconds_including_load'),failure_reason=receipt.get('failure_reason'),
             metadata=receipt.get('metadata',{}))
        if r['status']=='FULL_RESEARCH_SCORED':
            val=scores[(scores.eps_model_id==model)&(scores.target_key=='h1')&(scores.split=='SELECTION_OOF_AVAILABLE_BEFORE_2022')&(scores['mask']=='COMPLETE_COVERAGE')]
            if len(val):
                s=val.iloc[0].to_dict()
                for key in ['MAE','RMSE','MedianAE','price_scaled_MAE','bias','coverage']: r[key]=s.get(key)
                r['registry_metric_surface']='SELECTION_OOF_AVAILABLE_BEFORE_2022_LANE_A; origin and label both pre-cutoff, never research-test selection'
            meta=receipt.get('metadata',{})
            r['PIT_safe']=bool(meta.get('pit_valid',not r.get('zero_shot',False)));r['causal']=True
            for key in ['license','source_commit','zero_shot','finetuned']:
                if key in meta: r['commit' if key=='source_commit' else key]=meta[key]
    from research.eps_model_lab_v1.registry_enrichment import enrich
    records=enrich(records,scores)
    ordered=sorted(records.values(),key=lambda r:r['model_id'])
    save_json(RUN/'EPS_MODEL_REGISTRY_V1.json',clean({'updated_utc':datetime.now(timezone.utc).isoformat(),'evidence_class':'RESEARCH_ONLY_NOT_FORMAL_CERTIFICATION','models':ordered}))
    pd.json_normalize(ordered).to_csv(RUN/'EPS_EXTERNAL_MODEL_INTAKE.csv',index=False)
    sources=[json.loads(p.read_text(encoding='utf-8')) for p in (RUN/'source_receipts').glob('*.json') if 'initial_' not in p.stem]
    for source in sources:
        users=[r for r in ordered if r.get('reference_repository')==source.get('repository') and r['status']=='FULL_RESEARCH_SCORED']
        source.update(executed_model_ids=[r['model_id'] for r in users],environment=sorted({r['environment'] for r in users}),
          adapter=sorted({r['adapter_path'] for r in users}),weight_id=sorted({r['weight_id'] for r in users if r.get('weight_id')}),
          reference_clone_status_preserved=source.get('status'),
          actual_execution_role='SOURCE_CODE_USED_THROUGH_NEW_ADAPTER' if any(r.get('executed_code_git_commit') for r in users) else
            ('OFFICIAL_REFERENCE_FOR_HASHED_INSTALLED_PACKAGE' if users else 'INTAKE_REFERENCE_ONLY_NO_SCORED_MODEL'))
    pd.DataFrame(sources).to_csv(RUN/'EPS_EXTERNAL_SOURCE_MANIFEST.csv',index=False)
    weight_use=[]
    for p in (RUN/'weight_receipts').glob('*.json'):
        w=json.loads(p.read_text(encoding='utf-8'))
        if not w.get('weight_id') or 'initial_' in p.stem:continue
        users=[r['model_id'] for r in ordered if r.get('weight_id')==w['weight_id'] and r['status']=='FULL_RESEARCH_SCORED']
        weight_use.append({'weight_id':w['weight_id'],'revision':w.get('revision'),'license':w.get('license'),
          'receipt':str(p.relative_to(RUN)),'receipt_sha256':sha(p),'executed_scored_models':users,
          'status':'ACTUAL_SCORED_INFERENCE' if users else 'CACHED_OR_BLOCKED_NOT_SCORED'})
    save_json(RUN/'EPS_WEIGHT_USE_MANIFEST.json',{'weights':weight_use})
    envs=[json.loads(p.read_text(encoding='utf-8')) for p in (RUN/'environments').glob('*.json')]
    save_json(RUN/'ENVIRONMENT_MANIFEST.json',{'environments':envs})
    state=json.loads((RUN/'RUN_STATE.json').read_text(encoding='utf-8'))
    state.update(updated_utc=datetime.now(timezone.utc).isoformat(),status='BROAD_MODEL_SCORING',
      completed_models=[r['model_id'] for r in ordered if r['status']=='FULL_RESEARCH_SCORED'],
      diagnostic_only_models=[r['model_id'] for r in ordered if r['status'].startswith('FULL_DIAGNOSTIC_')],
      blocked_models=[r['model_id'] for r in ordered if r['status'] in ['BROKEN','DATA_BLOCKED','PLATFORM_BLOCKED','DATA_GEOMETRY_BLOCKED']],
      model_count=len(ordered),eligible_forecaster_count=sum(r.get('eligible_forecaster',True) for r in ordered),current_environment=[e['executable'] for e in envs],
      current_leaderboard='EPS_MODEL_ZOO_LEADERBOARD_V1.csv',download_status='See per-checkpoint weight_receipts for actual cached/blocked revision and hashes',
      next_actions=['Complete fixed-recipe data-repair queues; original scores retired','Coverage/fresh reload and pretraining-overlap audits',
                    'Availability-purged OOF ensemble/diversity experiment','Regenerate EPS combinations with unchanged read-only PE outputs','Finalize in last 45 minutes'])
    queues={}
    for path in [*RUN.glob('RERUN_QUEUE_*.json'),RUN/'GPU_POST_QUEUE.json',*RUN.glob('FOUNDATION_REPLAY_QUEUE_*.json'),RUN/'GPU_MAINTENANCE_QUEUE.json',RUN/'ADDITIONAL_FOUNDATION_QUEUE.json',RUN/'TABPFN_API_RECOVERY_QUEUE.json',RUN/'PROBABILISTIC_QUEUE.json',RUN/'API_COMPLETION_QUEUE.json',*(RUN/'origin_isolation_queues').glob('*.json'),*(RUN/'sampling_stability_queues').glob('*.json')]:
        if not path.exists():continue
        q=json.loads(path.read_text(encoding='utf-8'));steps=q.get('steps',[])
        queues[str(path.relative_to(RUN))]={'status':q['status'],'latest_step':steps[-1] if steps else None}
    state.update(resource_queues=queues,active_dataset_geometry='EPS_DATA_GEOMETRY_LOCK_V1_3.json',
       original_score_wave_retired=True,data_prefix_audit='SEC_FUTURE_SNAPSHOT_TRUNCATION_AUDIT.json',
       saved_tabular_replay='TABULAR_FRESH_PROCESS_RELOAD_AUDIT.json',
       user_budget_hours=10,deadline_utc='2026-09-08T02:23:26Z')
    if all(json.loads(p.read_text(encoding='utf-8')).get('status')=='COMPLETE' for p in RUN.glob('RERUN_QUEUE_*.json')):
        state['status']='PORTFOLIO_AND_REPORT_VERIFICATION' if (RUN/'ENSEMBLE_FREEZE_V1.json').exists() else 'SAVED_MODEL_AND_INPUT_INDEPENDENCE_QA'
    save_json(RUN/'RUN_STATE.json',state)
    print('REGISTRY',len(ordered),'FULL_SCORED',len(state['completed_models']),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['environment','registry']);parser.add_argument('--name');a=parser.parse_args()
    environment(a.name) if a.action=='environment' else registry()
