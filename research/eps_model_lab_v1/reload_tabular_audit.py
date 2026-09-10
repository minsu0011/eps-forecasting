"""Fresh-process reload of every saved core/MLForecast head, all forecast rows."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import os
from pathlib import Path
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import EPSAdapter,dataset


def verify(path):
    path=Path(path);began=time.perf_counter();digest=sha(path)
    try:
        year,target=path.stem.split('_');a=EPSAdapter.load(path);metadata=a.get_metadata()
        model=metadata['model_id'];f=dataset();f=f[f.asof_date.dt.year==int(year)]
        expected=pd.read_parquet(RUN/'predictions'/f'{model}.parquet')
        expected=expected[expected.target_key==target].set_index('sample_id').loc[f.sample_id].predicted_eps.to_numpy()
        actual=a.predict(f,target)
        np.testing.assert_array_equal(actual,expected)
        return {'path':str(path.relative_to(RUN)),'sha256':digest,'model_id':model,'year':int(year),'target':target,
          'rows':len(f),'status':'PASS_EXACT_ALL_ROWS','seconds':time.perf_counter()-began,'metadata_api':'PASS'}
    except Exception as exc:
        return {'path':str(path.relative_to(RUN)),'sha256':digest,'status':'FAIL','reason':str(exc),'traceback':traceback.format_exc()}


if __name__=='__main__':
    paths=sorted((RUN/'fitted_models').glob('*/*.pkl'))+sorted((RUN/'mlforecast_fitted').glob('*/*.pkl'))
    records=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(verify,str(p)) for p in paths]
        for job in as_completed(jobs):
            records.append(job.result())
            if len(records)%20==0 or records[-1]['status']=='FAIL':
                save_json(RUN/'TABULAR_FRESH_PROCESS_RELOAD_AUDIT.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
                    'expected_artifacts':len(paths),'completed':len(records),'records':records,'all_pass':all(r['status'].startswith('PASS') for r in records)})
                print('TABULAR_RELOAD',len(records),'OF',len(paths),'FAIL',sum(r['status']=='FAIL' for r in records),flush=True)
    save_json(RUN/'TABULAR_FRESH_PROCESS_RELOAD_AUDIT.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
       'expected_artifacts':len(paths),'completed':len(records),'records':records,'all_pass':all(r['status'].startswith('PASS') for r in records),
       'scope':'Every saved core/MLForecast head, every intended prediction origin; exact equality, local artifact hashes observed now, not backdated'})
