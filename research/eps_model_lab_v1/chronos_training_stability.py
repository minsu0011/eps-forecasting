"""Fixed five-seed OOF training sensitivity, not best-seed selection or test tuning."""
from datetime import datetime,timezone
import argparse
import gc
import json
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset,prediction_frame,metric_row
from research.eps_model_lab_v1.chronos_finetune import train_inputs,forecast,SPEC
from research.eps_model_lab_v1.late_frontier import frozen_hashes

ROOT=RUN/'training_seed_stability/chronos2_eps_joint_finetuned'
SEEDS=[1729,2027,3407,31415,27182]


def fit(base,inputs,path,seed):
    return base.fit(inputs,prediction_length=4,finetune_mode='full',context_length=32,min_past=8,
        learning_rate=1e-6,num_steps=300,batch_size=64,output_dir=path,finetuned_ckpt_name='final',
        validation_inputs=None,bf16=False,tf32=False,fp16=False,seed=seed,data_seed=seed,
        disable_tqdm=True,logging_steps=100,report_to='none',remove_printer_callback=True,
        optim='adamw_torch_fused',dataloader_num_workers=0)


def run(replay=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    frame=dataset();receipt=json.loads((RUN/'weight_receipts/amazon__chronos-2.json').read_text(encoding='utf-8'))
    planpath=ROOT/'PLAN.json'
    if not replay:
        if planpath.exists():raise RuntimeError('No overwrite of fixed seed experiment')
        save_json(planpath,{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PRE_SCORE_FROZEN',
            'seeds':SEEDS,'years':[2019,2020,2021],'original_recipe':SPEC,'steps':300,
            'only_changed_training_setting':'seed and data_seed','test_scores_consumed':False,
            'post_freeze_diagnostic':True,'best_seed_selection_allowed':False,'frozen_main_hashes':frozen_hashes(),
            'source_sha256':sha(__file__),'original_checkpoint_revision':receipt['revision'],
            'purpose':'Quantify training randomness around the already chosen OOF leading architecture; no new model IDs or portfolio weights'})
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    assert plan['frozen_main_hashes']==frozen_hashes()
    original=pd.read_parquet(RUN/'predictions/chronos2_eps_joint_finetuned.parquet')
    records=[];score_rows=[];started=time.perf_counter()
    for seed in SEEDS:
        output=ROOT/f'seed_{seed}.parquet';outputs=[]
        stored=pd.read_parquet(output) if replay else None
        for year in [2019,2020,2021]:
            if not replay and datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-08T01:38:26+00:00'):
                raise RuntimeError('Finalization boundary: no new training fold')
            valid=frame[frame.asof_date.dt.year==year];path=ROOT/'fitted'/str(seed)/str(year)
            began=time.perf_counter()
            if replay:model=Chronos2Pipeline.from_pretrained(path/'final',device_map='cuda',dtype=torch.float32)
            else:
                if path.exists():raise RuntimeError('No checkpoint overwrite')
                base=Chronos2Pipeline.from_pretrained(receipt['path'],device_map='cuda',dtype=torch.float32)
                inputs,info=train_inputs(frame,year);model=fit(base,inputs,path,seed);del base
            model.model.eval()
            for lane in ['quarter','ttm']:
                p,q=forecast(model,valid,lane)
                if not np.isfinite(p).all() or not np.isfinite(q).all():raise RuntimeError('Nonfinite seed forecast')
                for target in (['h1','h2','h3','h4'] if lane=='quarter' else ['ttm']):
                    j=3 if target=='ttm' else int(target[1])-1
                    if replay:
                        g=stored[(stored.target_key==target)&(stored.asof_date.dt.year==year)].set_index('sample_id').loc[valid.sample_id]
                        np.testing.assert_array_equal(p[:,j].astype(float),g.predicted_eps.to_numpy())
                        np.testing.assert_array_equal(q[:,j,:],g[['p10_eps','p50_eps','p90_eps']].to_numpy())
                        records.append({'seed':seed,'year':year,'target':target,'rows':len(valid),'status':'EXACT_SAVED_REPLAY_PASS'})
                    else:
                        result=prediction_frame(valid,target,'chronos2_eps_joint_finetuned',p[:,j],time.perf_counter()-began,q[:,j,:],
                            {'pit_valid':False,'evidence_class':'POST_FREEZE_TRAINING_SEED_DIAGNOSTIC_PRETRAINING_OVERLAP_UNRESOLVED'})
                        result['diagnostic_seed']=seed;result['formal_certified']=False;outputs.append(result)
            if not replay:
                artifacts=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in (path/'final').rglob('*') if p.is_file()]
                records.append({'seed':seed,'year':year,'status':'FIT_AND_FORECAST_PASS','seconds':time.perf_counter()-began,'artifacts':artifacts,**info})
                save_json(ROOT/'FIT_PROGRESS.json',records)
            print('CHRONOS_TRAINING_SEED',seed,year,'REPLAY' if replay else 'FIT',round(time.perf_counter()-began,2),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        if not replay:
            combined=pd.concat(outputs,ignore_index=True);assert len(combined)==797*5
            combined.to_parquet(output,index=False)
            for target,g in combined.groupby('target_key'):
                eligible=g[g.label_asof<pd.Timestamp('2022-01-01',tz='UTC')]
                score_rows.append({'seed':seed,'target_key':target,**metric_row(eligible)})
            pd.DataFrame(score_rows).to_csv(ROOT/'OOF_SCORES.csv',index=False)
    assert plan['frozen_main_hashes']==frozen_hashes()
    if replay:
        save_json(ROOT/'REPLAY.json',{'status':'PASS','fresh_process':True,'checks':records,'main_unchanged':True})
    else:
        seed0=pd.read_parquet(ROOT/'seed_1729.parquet')
        g=original.set_index(['sample_id','target_key']).loc[pd.MultiIndex.from_frame(seed0[['sample_id','target_key']])]
        pointdiff=float(np.max(np.abs(seed0.predicted_eps.to_numpy()-g.predicted_eps.to_numpy())))
        qdiff=float(np.max(np.abs(seed0[['p10_eps','p50_eps','p90_eps']].to_numpy()-g[['p10_eps','p50_eps','p90_eps']].to_numpy())))
        save_json(ROOT/'COMPLETION.json',{'status':'FIVE_FIXED_SEED_OOF_TRAINING_COMPLETE','trained_annual_models':15,
            'prediction_files':5,'main_unchanged':True,'new_portfolio_members':0,'best_seed_selected':False,
            'research_test_forecast_or_score_computed':False,'runtime_seconds':time.perf_counter()-started,
            'seed1729_vs_original_max_point_difference':pointdiff,'seed1729_vs_original_max_quantile_difference':qdiff,
            'seed1729_original_bit_exact':pointdiff==0 and qdiff==0,'formal_certified':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');run(parser.parse_args().replay)
