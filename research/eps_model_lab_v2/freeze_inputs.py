"""Freeze audited data, bounded recipes and environment identity before scores."""
from pathlib import Path
import sys
import subprocess
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import pandas as pd
from research.eps_model_lab_v2.common import RUN,V1,sha,save_json,read_json,utcnow,check_stop


def recipes():
    out=[]
    def add(family,variant,track='N',mode='expanding',representation='direct',**options):
        out.append({'model_id':f'{family}_{track}_{variant}','family':family,'feature_track':track,
            'chronological_mode':mode,'representation':representation,'seed':1729,**options})
    for family in ['Ridge','HistGB','CatBoost','NGBoostLaplace']:
        add(family,'direct',normalization='origin_historical_robust')
        add(family,'asinh',representation='asinh',normalization='train_only_asinh_scale')
        add(family,'residual',representation='residual',normalization='origin_historical_robust')
        add(family,'residual_rolling',mode='rolling',representation='residual',normalization='origin_historical_robust')
        add(family,'residual_recency',mode='recency_weighted',representation='residual',normalization='origin_historical_robust')
    for family in ['TimeXer','SOFTS','XLinear','DLinear','LSTM']:
        add(family,'direct',steps=300,normalization='origin_historical_robust')
        add(family,'residual',representation='residual',steps=300,normalization='origin_historical_robust')
        add(family,'residual_rolling',mode='rolling',representation='residual',steps=300,normalization='origin_historical_robust')
        add(family,'residual_recency',mode='recency_weighted',representation='residual',steps=300,normalization='origin_historical_robust')
        if family in ['TimeXer','SOFTS']:
            add(family,'verified_accounting',track='A',status='DATA_BLOCKED_NO_VERIFIED_DEVELOPMENT_ACCOUNTING',steps=300)
    add('Chronos2','joint',steps=300,representation='joint',evidence_track='RETROSPECTIVE_FOUNDATION')
    add('Chronos2','native',steps=300,representation='native_history_only',evidence_track='RETROSPECTIVE_FOUNDATION')
    add('Chronos2','residual',steps=300,representation='residual',evidence_track='RETROSPECTIVE_FOUNDATION')
    add('Chronos2','joint_rolling',mode='rolling',steps=300,representation='joint',evidence_track='RETROSPECTIVE_FOUNDATION')
    add('Chronos2','joint_recency',mode='recency_weighted',steps=300,representation='joint',evidence_track='RETROSPECTIVE_FOUNDATION')
    add('Chronos2ZeroShot','native',steps=0,evidence_track='RETROSPECTIVE_FOUNDATION',
        family_equivalence_group='Chronos2',chronological_mode_note='No trained history weights; zero-shot is not three fake refit modes')
    for family in ['Ridge','HistGB','CatBoost','NGBoostLaplace','Chronos2']:
        out.append({'model_id':family+'_A_VERIFIED_PENDING','family':family,'feature_track':'A',
            'status':'DATA_BLOCKED_NO_VERIFIED_DEVELOPMENT_ACCOUNTING','not_an_executed_variant':True})
    return out


def run():
    check_stop()
    pit=read_json(RUN/'audit/PIT_MUTATION_AUDIT_V2.json')
    duration=read_json(RUN/'EPS_DURATION_CONTEXT_AUDIT_V2.json')
    assert pit['status']=='PASS' and duration['status']=='PASS'
    assert all(r['api_context_value_match'] for r in duration['random_contexts']), 'Random API source review mismatch'
    frame=pd.read_parquet(RUN/'data/samples_v2.parquet')
    ledger=pd.read_parquet(RUN/'EPS_SOURCE_CONTEXT_LEDGER_V2.parquet')
    dev_account=ledger[(ledger.quality_tier=='VERIFIED')&(ledger.source_available_at<'2019-01-01')]
    assert dev_account.empty, 'Accounting-source scope changed; prescore recipe routing must be reviewed before freeze'
    contract={'created_utc':utcnow(),'identity':'EPS_DATA_IDENTITY_V2_CLEAN',
        'clean_means':'Unverified accounting values excluded, not universal original-source verification',
        'feature_tracks':{'N':{'allowed':['native EPS','native ledger TTM with approximation flags','lags','observed masks','age','fiscal calendar'],
            'quality':'PARTIALLY_VERIFIED_NATIVE','source_vintage_certified':False},
            'A':{'allowed_quality':['VERIFIED'],'unverified_as':'MISSING_QUARANTINED',
                 'development_verified_cells':len(dev_account),'model_status':'DATA_BLOCKED'},
            'M':{'enabled':False}},
        'quarter_truth':'Direct exact native context contiguous with previous disclosed quarter end; 13/14/17 week contexts allowed by native calendar',
        'TTM_truth':'Native first-filing EPS ledger; annual/direct/bridge approximation method retained; not NI/shares-generated truth',
        'share_basis':'Forecast-origin basis using retrospective corporate-action evidence; historical basis vintage not certified',
        'label_missingness':'No synthetic Q4, no zero/negative clipping, no truth imputation',
        'unit_authority_ladder':['Exact filing iXBRL context/unitRef/scale','Same accession original statement display unit','Same filing cross-check','MISSING_QUARANTINE'],
        'temporal_semantics':'Available-by-origin session-close snapshots; future accessions never erase past facts',
        'official_source_access_limitations':'Original SEC HTTP403; corporate IR PDF timeout; two exact MCD statement scale observations only',
        'formal_certification':False}
    save_json(RUN/'EPS_DATA_V2_CONTRACT.json',contract,immutable=True)
    files=[RUN/'data/samples_v2.parquet',RUN/'data/fiscal_panel_v2.parquet',RUN/'EPS_SOURCE_CONTEXT_LEDGER_V2.parquet',RUN/'data/source_context_ledger_v2.parquet']
    geometry={'created_utc':utcnow(),'frozen_before_scoring':True,'samples':len(frame),'companies':int(frame.ticker.nunique()),
        'files':{str(p.relative_to(RUN)):sha(p) for p in files},'protocol_sha256':sha(RUN/'prescore/RESEARCH_PROTOCOL_V2.json'),
        'data_contract_sha256':sha(RUN/'EPS_DATA_V2_CONTRACT.json'),'mutation_audit_sha256':sha(RUN/'audit/PIT_MUTATION_AUDIT_V2.json'),
        'development_years':[2015,2016,2017,2018],'confirmation_years':[2019,2020,2021],
        'monitor_years':[2022,2023,2024,2025,2026],'feature_accounting_original_full_audit_complete':False}
    save_json(RUN/'EPS_DATA_GEOMETRY_LOCK_V2.json',geometry,immutable=True)
    save_json(RUN/'EPS_DATASET_MANIFEST_V2.json',{**geometry,'identity':'EPS_DATA_IDENTITY_V2_CLEAN',
        'label_counts':{t:int(frame['y_'+t].notna().sum()) for t in ['h1','h2','h3','h4','ttm']},
        'origin_year_counts':{str(y):int(n) for y,n in frame.groupby(frame.asof_date.dt.year).size().items()}},immutable=True)
    model_recipes={'created_utc':utcnow(),'source_sha256':sha(__file__),'selection_uses_development_only':True,
        'recipes':recipes(),'baseline_names':['persistence_observed','seasonal_observed','random_walk','seasonal_random_walk','simple_drift'],
        'baseline_alias_semantics':'Observed random walk equals observed persistence; observed seasonal random walk equals seasonal observed. Report identical forecasts as aliases, not diversity.',
        'tabular_parameters':{'Ridge':{'alpha':100.0},'HistGB':{'max_iter':250,'loss':'absolute_error','max_leaf_nodes':15,'min_samples_leaf':20,'l2_regularization':5,'early_stopping':False},
            'CatBoost':{'iterations':400,'depth':5,'learning_rate':.04,'loss_function':'MAE','l2_leaf_reg':5,'thread_count':1},
            'NGBoostLaplace':{'n_estimators':500,'learning_rate':.01,'tree_depth':3,'min_samples_leaf':5,'natural_gradient':True}},
        'sequence_parameters':{'TimeXer':{'patch_len':4,'hidden_size':64,'n_heads':4,'e_layers':2,'d_ff':128},
            'SOFTS':{'hidden_size':64,'d_core':32,'e_layers':2,'d_ff':128,'dropout':.1},
            'XLinear':{'hidden_size':64,'temporal_ff':128,'channel_ff':8},
            'DLinear':{'moving_avg_window':5},'LSTM':{'encoder_hidden_size':64,'encoder_n_layers':1,'decoder_hidden_size':64}},
        'sequence_trainer':{'steps':300,'batch_size':128,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'loss':'masked SmoothL1 beta1','clip_gradient':1.0},
        'Chronos_parameters':{'steps':300,'learning_rate':1e-6,'batch_size':64,'context_length':32,'prediction_length':4,'min_past':8,'precision':'FP32','validation_early_stopping':False},
        'calibration':'Past development residual quantiles only; 2018 h1/2017 TTM calibration not used to retune point forecasts; finite-sample plus-one quantile, report dependent-year limitations'}
    save_json(RUN/'EPS_V2_MODEL_RECIPES.json',model_recipes,immutable=True)
    envs=[]
    for name in ['core','gpu','prob']:
        executable=PROJECT.parent/f'.venv_eps_{name}_py312/Scripts/python.exe'
        lock=subprocess.run([str(executable),'-B','-m','pip','freeze','--all'],capture_output=True,text=True,timeout=40)
        if lock.returncode:
            # uv-created environments may intentionally lack pip; metadata is
            # still authoritative for installed package versions.
            lock=subprocess.run([str(executable),'-B','-c',"import importlib.metadata as m; print('\\n'.join(sorted(d.metadata['Name']+'=='+d.version for d in m.distributions())))"],capture_output=True,text=True,check=True,timeout=40)
        envs.append({'name':name,'python':str(executable),'packages':lock.stdout.splitlines(),'modified':False})
    save_json(RUN/'EPS_V2_ENVIRONMENT_LOCK.json',{'created_utc':utcnow(),'environments':envs,
        'existing_PE_environment_modified':False,'deterministic_profile':{'CUBLAS_WORKSPACE_CONFIG':':4096:8','torch_deterministic_algorithms':True,
            'cudnn_deterministic':True,'cudnn_benchmark':False,'TF32':False,'seed':1729,'same_machine_only_claim':True}},immutable=True)
    save_json(RUN/'EPS_V2_MODEL_REGISTRY.json',{'created_utc':utcnow(),'models':[
        {**r,'status':r.get('status','PRESCORE_RECIPE_FROZEN_NOT_TRAINED')} for r in model_recipes['recipes']],
        'formal_certified':0})
    save_json(RUN/'RUN_STATE.json',{'updated_utc':utcnow(),'status':'DATA_AND_RECIPES_FROZEN_READY_FOR_DEVELOPMENT',
        'development_only_training_first':True,'Track_A_blocked_for_insufficient_verified_source_contexts':True,
        'V1_PE_readonly':True,'formal_certified':False})
    print('V2_DATA_AND_RECIPES_FROZEN',len(frame),len(model_recipes['recipes']),flush=True)


if __name__=='__main__':run()
