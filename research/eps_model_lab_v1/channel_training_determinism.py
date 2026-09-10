"""Prespecified deterministic-kernel retraining diagnostic, not a score replacement."""
from datetime import datetime,timezone
import argparse
import gc
import json
import os
from pathlib import Path
import sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset
from research.eps_model_lab_v1.eps_only_multivariate_diagnostic import EPSOnlyAdapter,ROOT


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('replicate',choices=['a','b']);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    output=ROOT/'deterministic_training'/args.replicate
    if output.exists():raise RuntimeError('No deterministic replicate overwrite')
    output.mkdir(parents=True)
    save_json(output/'PLAN.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'replicate':args.replicate,
        'models':['TimeXer','SOFTS'],'year':2019,'seed':1729,'steps':300,'device':'CUDA FP32',
        'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],'deterministic_algorithms':True,
        'cudnn_deterministic':True,'cudnn_benchmark':False,'precision':'highest, TF32 disabled',
        'new_kernel_profile_may_differ_from_original_training':True,'main_or_ablation_forecasts_overwritten':False,
        'reason':'Public API rebuild reproduced input tensors exactly but fresh TimeXer training weights differed. Saved checkpoint replay remains exact.',
        'source':'https://docs.pytorch.org/docs/stable/notes/randomness.html'})
    frame=dataset();valid=frame[frame.asof_date.dt.year==2019];records=[]
    for name in ['TimeXer','SOFTS']:
        a=EPSOnlyAdapter(name).fit(frame,2019);a.save(output/f'{name}.pt')
        values=np.column_stack([a.predict(valid,target) for target in ['h1','h2','h3','h4','ttm']])
        np.save(output/f'{name}.npy',values,allow_pickle=False)
        old=EPSOnlyAdapter.load(ROOT/'fitted'/name/'2019.pt')
        prior=np.column_stack([old.predict(valid,target) for target in ['h1','h2','h3','h4','ttm']])
        records.append({'name':name,'status':'DETERMINISTIC_FIT_AND_PREDICT_COMPLETE','rows':len(valid),
            'checkpoint_sha256':sha(output/f'{name}.pt'),'prediction_sha256':sha(output/f'{name}.npy'),
            'different_kernel_profile_max_difference_vs_original':float(np.max(np.abs(values-prior)))})
        del a,old;gc.collect();torch.cuda.empty_cache()
    save_json(output/'COMPLETION.json',{'status':'COMPLETE','replicate':args.replicate,'records':records,'pid':os.getpid()})
    print('DETERMINISTIC_TRAINING_REPLICATE_COMPLETE',args.replicate,flush=True)
