"""Seven new official architectures after fixed-recipe repair; same target data."""
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import sys
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.extra_architecture_specs import EXTRA_UNIVARIATE,EXTRA_MULTIVARIATE
from research.eps_model_lab_v1.common import dataset


if __name__=='__main__':
    from research.eps_model_lab_v1 import neural_models as n,multivariate_models as m
    lock=RUN/'EXTRA_ARCHITECTURE_PRESCORE_FREEZE.json'
    specification={'univariate':EXTRA_UNIVARIATE,'multivariate':EXTRA_MULTIVARIATE,'steps':300,
      'samples_sha256':sha(RUN/'data/samples.parquet'),'selection':'Remaining explicitly requested installed architectures, API/geometry support only, no test-score filtering',
      'original_93_model_recipes_unchanged':True,'created_utc':datetime.now(timezone.utc).isoformat(),
      'inference_isolation':'TimesNet inference_windows_batch_size=1; StemGNN company-origin batch size=1; training only pre-cutoff data',
      'source':'Official installed neuralforecast3.2.1; package_source_manifests/gpu.json',
      'candidate_specs_sha256':sha(PROJECT/'research/eps_model_lab_v1/extra_architecture_specs.py')}
    if lock.exists():raise RuntimeError('Extra wave already frozen; explicit candidate resume required')
    save_json(lock,specification);torch.set_num_threads(4)
    # Replace the model dictionary, do not mutate the existing tensor contract's
    # original specification object or any old architecture configuration.
    n.SPECS={**n.SPECS,**EXTRA_UNIVARIATE};m.SPECS={**m.SPECS,**EXTRA_MULTIVARIATE}
    frame=dataset();data=None
    for name in [*EXTRA_UNIVARIATE,*EXTRA_MULTIVARIATE]:
        if datetime.now(timezone.utc)>=datetime.fromisoformat('2026-09-08T01:38:26+00:00'):break
        model=('nf_' if name in EXTRA_UNIVARIATE else 'nf_multivar_')+name
        try:
            if name in EXTRA_UNIVARIATE:n.run(name,True);n.run(name,False)
            else:
                if data is None:data=m.to_device(m.tensors(frame))
                m.run(name,frame,data,True);m.run(name,frame,data,False)
        except Exception as exc:
            save_json(RUN/'model_receipts'/f'{model}.json',{'model_id':model,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc(),
              'debug_policy':'Bounded breadth wave: preserve failure and proceed; no score-driven tuning'})
            print('EXTRA_ARCHITECTURE_BLOCKED',model,str(exc),flush=True)
        gc.collect();torch.cuda.empty_cache()
