"""Create a separate data-repair run; never replace the flawed first-run evidence."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import ORIGINAL_RUN,RUN,save_json,sha

if __name__=='__main__':
    if RUN==ORIGINAL_RUN or RUN.exists(): raise RuntimeError('Repair requires a new, absent EPS run directory')
    RUN.mkdir()
    copied=[]
    for relative in ['data/raw','data/engine_events','source_receipts','weight_receipts','environments','package_source_manifests','ecosystem_discovery']:
        source=ORIGINAL_RUN/relative
        if source.exists():
            shutil.copytree(source,RUN/relative)
            copied.extend({'path':str(p.relative_to(RUN)),'sha256':sha(p)} for p in (RUN/relative).rglob('*') if p.is_file())
    for name in ['SEC_ACQUISITION_MANIFEST.json','CPU_THROUGHPUT_BENCHMARK.json','PE_READONLY_BASELINE.json',
                 'EPS_SYSTEM_CONTRACT_V1.json','EPS_TARGET_CONTRACT_V1.md','ENVIRONMENT_MANIFEST.json']:
        p=ORIGINAL_RUN/name
        if p.exists(): shutil.copy2(p,RUN/name);copied.append({'path':name,'sha256':sha(RUN/name)})
    state=json.loads((ORIGINAL_RUN/'RUN_STATE.json').read_text(encoding='utf-8'))
    state.update(status='PIT_REPAIR_DATASET_BUILD',updated_utc=datetime.now(timezone.utc).isoformat(),completed_models=[],blocked_models=[],
       predecessor_run=str(ORIGINAL_RUN),predecessor_retired_reason='HON future fiscal-index collision erased an earlier native fiscal record',
       next_actions=['Build V1.2 prefix-safe dataset','Run 69-company/three-cutoff truncation audit before scoring',
                     'Re-execute identical fixed model recipes; no test-driven hyperparameter change','Refreeze OOF ensembles under new data identity'])
    save_json(RUN/'RUN_STATE.json',state)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'original_run':str(ORIGINAL_RUN),'new_run':str(RUN),
      'original_samples_sha256':sha(ORIGINAL_RUN/'data/samples.parquet'),
      'repair_cause':'Original Honeywell FY2021 filing incorrectly tags dei fiscal-year focus as 2020. Global duplicate removal then deleted the earlier FY2020 record using future knowledge.',
      'source_cover':'https://www.sec.gov/Archives/edgar/data/773840/000077384022000018/hon-20211231.htm',
      'source_dei_typo':'https://www.sec.gov/Archives/edgar/data/773840/000077384022000018/R1.htm',
      'fix':'Keep earliest-available fiscal identity on collision; accession-scoped HON fiscal-year correction 2020 to 2021 based on original annual cover.',
      'model_hyperparameters_changed':False,'original_predictions_preserved':True,'original_source_snapshot':'source_pre_pit_repair',
      'prior_test_seen':True,'new_results_formal_certifiable':False,
      'test_reuse_caveat':'Same research periods reused solely for documented data repair, not a fresh heldout. Original results not a valid strict-PIT benchmark.',
      'copied_input_files':copied}
    save_json(RUN/'PIT_REPAIR_LINEAGE.json',receipt)
    save_json(ORIGINAL_RUN/'PIT_AUDIT_RETIREMENT.json',{k:v for k,v in receipt.items() if k!='copied_input_files'})
    print('PIT_REPAIR_RUN_CREATED',RUN,flush=True)
