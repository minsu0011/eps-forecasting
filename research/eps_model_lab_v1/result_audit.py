"""Reconcile saved outputs with frozen origins/truth and complete-path diagnostics."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset,evaluate_all,atomic_csv


def run():
    source=dataset();valid=source[source.asof_date.dt.year>=2019].set_index('sample_id');ids=set(valid.index)
    audits=[];paths=[];selection=[]
    for path in sorted((RUN/'predictions').glob('*.parquet')):
        p=pd.read_parquet(path);receipt=json.loads((RUN/'model_receipts'/f'{path.stem}.json').read_text(encoding='utf-8'))
        errors=[]
        if receipt.get('dataset_sha256')!=sha(RUN/'data/samples.parquet'):errors.append('DATASET_HASH_BINDING_MISMATCH')
        if receipt.get('dataset_run_identity')!=RUN.name:errors.append('DATASET_RUN_RECEIPT_MISMATCH')
        if 'dataset_run_identity' not in p or not p.dataset_run_identity.eq(RUN.name).all():errors.append('PREDICTION_RUN_IDENTITY_MISMATCH')
        if sha(path)!=receipt.get('prediction_sha256'): errors.append('PREDICTION_HASH_MISMATCH')
        if p.duplicated(['sample_id','target_key']).any(): errors.append('DUPLICATE_KEY')
        if not set(p.eps_model_id)=={path.stem}: errors.append('MODEL_IDENTITY_MISMATCH')
        if not (p.prediction_valid==np.isfinite(p.predicted_eps)).all(): errors.append('VALIDITY_FLAG_MISMATCH')
        for target,g in p.groupby('target_key'):
            if set(g.sample_id)!=ids: errors.append('ORIGIN_COVERAGE_MISMATCH_'+target)
            aligned=valid.reindex(g.sample_id)
            try:
                np.testing.assert_allclose(g.actual_eps,aligned['y_'+target],rtol=0,atol=0,equal_nan=True)
                if not (pd.to_datetime(g.asof_date).to_numpy()==aligned.asof_date.to_numpy()).all(): errors.append('ORIGIN_TIMESTAMP_MISMATCH_'+target)
                if not (g.ticker.to_numpy()==aligned.ticker.to_numpy()).all(): errors.append('TICKER_MISMATCH_'+target)
            except AssertionError: errors.append('TRUTH_MISMATCH_'+target)
            # An origin in 2021 can have a Q+4 label released in 2023. Such a
            # label cannot select a model/stack deployed at the start of 2022.
            from research.eps_model_lab_v1.common import metric_row
            cutoff=pd.Timestamp('2022-01-01',tz='UTC')
            known=g[(g.split=='VALIDATION_OOF')&(g.label_asof<cutoff)&(g.asof_date<cutoff)]
            selection.append({'eps_model_id':path.stem,'target_key':target,'split':'SELECTION_OOF_AVAILABLE_BEFORE_2022',
                  'model_status':receipt.get('status'),'portfolio_eligible':receipt.get('status')=='FULL_RESEARCH_SCORED',
                  'mask':'COMPLETE_COVERAGE','checkpoint_pretraining_overlap_unresolved':not bool(g.pit_valid.all()),**metric_row(known)})
        quarter=p[p.target_key.isin(['h1','h2','h3','h4'])]
        if quarter.target_key.nunique()==4:
            truth=quarter.pivot(index='sample_id',columns='target_key',values='actual_eps')
            forecast=quarter.pivot(index='sample_id',columns='target_key',values='predicted_eps')
            complete=np.isfinite(truth).all(axis=1)&np.isfinite(forecast).all(axis=1)
            for split,origin_ids in p.groupby('split').sample_id.unique().items():
                selected=truth.index.intersection(origin_ids);ok=selected[complete.reindex(selected).to_numpy()]
                error=forecast.loc[ok]-truth.loc[ok]
                paths.append({'eps_model_id':path.stem,'split':split,'complete_truth_paths':int(np.isfinite(truth.loc[selected]).all(axis=1).sum()),
                    'complete_prediction_paths':len(ok),'path_mean_MAE':float(error.abs().to_numpy().mean()) if len(ok) else None,
                    'four_direct_quarter_sum_MAE':float(error.sum(axis=1).abs().mean()) if len(ok) else None,
                    'surface':'STRICT_FOUR_DIRECT_QUARTER_TRUTHS_NOT_NATIVE_TTM'})
        audits.append({'model_id':path.stem,'status':'PASS' if not errors else 'FAIL','errors':errors,'rows':len(p),
          'targets':sorted(p.target_key.unique()),'years':sorted(int(y) for y in p.asof_date.dt.year.unique()),
          'valid_prediction_rows':int(p.prediction_valid.sum()),'prediction_sha256':sha(path),
          'inference_pit_flag':bool(p.pit_valid.all()),'formal_certified':False})
    save_json(RUN/'EPS_RESULT_INTEGRITY_AUDIT.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
           'models':audits,'all_pass':all(r['status']=='PASS' for r in audits),'model_count':len(audits),
           'scope':'Predictions/identity/truth/fold geometry; not source-vintage or pretrained-corpus certification'})
    atomic_csv(pd.DataFrame(paths),RUN/'EPS_COMPLETE_PATH_LEADERBOARD.csv')
    atomic_csv(pd.DataFrame(selection),RUN/'EPS_SELECTION_LEADERBOARD_V1.csv')
    evaluate_all()
    print('RESULT_AUDIT',len(audits),'PASS',sum(r['status']=='PASS' for r in audits),'FAIL',[r['model_id'] for r in audits if r['status']!='PASS'],flush=True)


if __name__=='__main__': run()
