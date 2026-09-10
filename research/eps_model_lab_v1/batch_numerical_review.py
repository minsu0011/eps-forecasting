"""Separate value dependence from numerical/RNG effects of batch-size changes.

This does not rewrite forecasts or relax the existing failed singleton gate.
"""
from datetime import datetime,timezone
import gc
from pathlib import Path
import sys
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset
from research.eps_model_lab_v1.batch_causality_audit import perturb
from research.eps_model_lab_v1.neural_models import NeuralAdapter
from research.eps_model_lab_v1.multivariate_models import MultivariateAdapter,tensors,to_device


def run():
    torch.set_num_threads(4);frame=dataset();data=to_device(tensors(frame));records=[]
    for precision in ['runtime_default','strict_fp32_diagnostic']:
        if precision=='strict_fp32_diagnostic':
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        settings={'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'matmul_precision':torch.get_float32_matmul_precision()}
        for name in ['nf_BiTCN','nf_Informer','nf_multivar_TimeMixer']:
            for year in [2019,2022,2026]:
                val=frame[frame.asof_date.dt.year==year].iloc[:8];row={'model_id':name,'year':year,'precision':precision,'settings':settings}
                try:
                    if name.startswith('nf_multivar_'):
                        a=MultivariateAdapter.load(RUN/'multivariate_fitted'/name/f'{year}.pt');indices=val.index.to_numpy()
                        changed=dict(data);changed['x']=data['x'].clone();changed['x'][indices[1:]]=changed['x'][indices[1:]]*3+7
                        def predict(kind):
                            return a.path(changed if kind=='other_value_mutation' else data,
                              indices[:1] if kind=='singleton' else indices[::-1].copy() if kind=='permuted' else indices)
                    else:
                        a=NeuralAdapter.load(RUN/'neural_fitted'/name/str(year))
                        def predict(kind):return a.path(perturb(val) if kind=='other_value_mutation' else val.iloc[:1] if kind=='singleton' else val.iloc[::-1] if kind=='permuted' else val)
                    torch.manual_seed(123);before=predict('base')[0];checks={}
                    for kind in ['same_batch_repeat','other_value_mutation','singleton','permuted']:
                        torch.manual_seed(123);p=predict(kind);after=p[-1] if kind=='permuted' else p[0]
                        checks[kind]={'passes_original_2e5_gate':bool(np.allclose(before,after,rtol=2e-5,atol=2e-5)),
                           'exact':bool(np.array_equal(before,after)),'max_absolute_difference':float(np.max(np.abs(before-after)))}
                    row.update(status='DIAGNOSTICS_COMPLETE',checks=checks)
                    del a
                except Exception as exc:row.update(status='DIAGNOSTIC_FAILED',reason=str(exc),traceback=traceback.format_exc())
                records.append(row)
                save_json(RUN/'BATCH_NUMERICAL_REVIEW.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
                   'dataset_sha256':sha(RUN/'data/samples.parquet'),'records':records,'predictions_rewritten':False,
                   'scope':'Distinct counterfactuals; strict-FP32 is diagnostic only. Original singleton failures remain recorded, no tolerance relaxed.'})
                print('BATCH_REVIEW',name,year,precision,row['status'],row.get('checks'),flush=True)
                gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':run()
