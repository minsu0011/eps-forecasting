"""Localize the public API discrepancy without changing any diagnostic forecasts."""
from datetime import datetime,timezone
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import dataset
from research.eps_model_lab_v1.multivariate_models import tensors,to_device
from research.eps_model_lab_v1.eps_only_multivariate_diagnostic import ROOT,EPSOnlyAdapter


if __name__=='__main__':
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    frame=dataset();valid=frame[frame.asof_date.dt.year==2019];idx=np.flatnonzero((frame.asof_date.dt.year==2019).to_numpy())
    arrays={k:np.array(v[...,:2],copy=True) for k,v in tensors(frame).items()}
    original=EPSOnlyAdapter.load(ROOT/'fitted/TimeXer/2019.pt');new=EPSOnlyAdapter.load(ROOT/'api_checkpoints/TimeXer.pt')
    rebuilt=original.prepare_data(valid);inputs=[]
    for key in arrays:
        left=arrays[key][idx];right=rebuilt[key]
        exact=np.array_equal(left,right,equal_nan=True)
        maximum=float(np.nanmax(np.abs(left.astype(float)-right.astype(float))))
        inputs.append({'key':key,'equal':exact,'max_difference':maximum,'shape':list(left.shape)})
    weights=[]
    for key,value in original.net.state_dict().items():
        difference=float((value-new.net.state_dict()[key]).abs().max())
        if difference:weights.append({'key':key,'max_difference':difference})
    original_cached=original.path(to_device(arrays),idx)
    new_cached=new.path(to_device(arrays),idx)
    original_rebuilt=original.path(to_device(rebuilt),np.arange(len(valid)))
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'DISCREPANCY_LOCALIZATION',
        'input_arrays':inputs,'changed_state_keys':weights,
        'original_vs_new_weights_cached_prediction_max_difference':float(np.max(np.abs(original_cached-new_cached))),
        'cached_vs_rebuilt_input_prediction_max_difference':float(np.max(np.abs(original_cached-original_rebuilt))),
        'old_fitted_sha256':sha(ROOT/'fitted/TimeXer/2019.pt'),'new_api_fitted_sha256':sha(ROOT/'api_checkpoints/TimeXer.pt'),
        'original_predictions_overwritten':False}
    save_json(ROOT/'PUBLIC_API_INITIAL_FAILURE_DIAGNOSIS.json',result)
    print(result,flush=True)
