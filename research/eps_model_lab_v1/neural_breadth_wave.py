"""Additional real architectures, fixed common budget; no score-driven micro-tuning."""
import gc
import json
from pathlib import Path
import sys
import traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import torch
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.neural_models import run,SPECS

NAMES=['NBEATS','TiDE','MLP','RNN','LSTM','DilatedRNN','BiTCN','KAN','DeepNPTS','Autoformer','Informer','VanillaTransformer']

if __name__=='__main__':
    lock=RUN/'NEURAL_BREADTH_WAVE_SPEC.json'
    if not lock.exists(): save_json(lock,{'models':{n:SPECS[n] for n in NAMES},'steps':300,'selection':'Smoke contract pass only, no metric filtering',
                                          'source_sha256':sha(PROJECT/'research/eps_model_lab_v1/neural_models.py')})
    for name in NAMES:
        receipt=RUN/'model_receipts'/f'nf_{name}.json'
        if receipt.exists(): print(name,'EXISTING_RECEIPT_SKIP',flush=True);continue
        try:
            run(name,smoke=True)
            run(name,smoke=False)
        except Exception as exc:
            save_json(receipt,{'model_id':'nf_'+name,'status':'BROKEN','failure_reason':str(exc),'traceback':traceback.format_exc()})
            print(name,'BROKEN',str(exc),flush=True)
        gc.collect();torch.cuda.empty_cache()
