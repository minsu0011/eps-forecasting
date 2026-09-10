"""Measured process throughput on identical synthetic compute, not target performance."""
from concurrent.futures import ProcessPoolExecutor
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[name]='1'
from pathlib import Path
import sys
import time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json

def job(seed):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    r=np.random.default_rng(seed);x=r.normal(size=(2500,100));y=x[:,0]+r.normal(size=2500)
    with threadpool_limits(1):
        model=HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=15,random_state=seed).fit(x,y)
        return float(model.predict(x[:10]).sum())

if __name__=='__main__':
    import psutil
    results=[]
    for workers in [20,24,28,32]:
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool: values=list(pool.map(job,range(64)))
        duration=time.perf_counter()-start
        results.append({'workers':workers,'inner_threads':1,'tasks':64,'seconds':duration,'tasks_per_second':64/duration,
                        'checksum':sum(values),'system_used_ram_gib':psutil.virtual_memory().used/2**30})
        save_json(RUN/'CPU_THROUGHPUT_BENCHMARK.json',{'results':results,'selected_workers':max(results,key=lambda r:r['tasks_per_second'])['workers']})
        print(workers,round(duration,2),flush=True)
