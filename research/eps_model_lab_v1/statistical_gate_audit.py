"""Correct false blanket smoke-PASS claims; preserve full diagnostic predictions."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.statistical_models import SF_NAMES,ACCOUNTING


if __name__=='__main__':
    records=[]
    for model in ['sf_'+n for n in SF_NAMES]+list(ACCOUNTING):
        smoke=RUN/'statistical_smoke'/f'{model}.json';checks=json.loads(smoke.read_text(encoding='utf-8'))
        errors=[{'ticker':r.get('ticker'),'failure':e} for r in checks for e in r.get('failures',[])]
        errors.extend({'ticker':r.get('ticker'),'failure':r['failure_reason']} for r in checks if r.get('failure_reason'))
        path=RUN/'model_receipts'/f'{model}.json';receipt=json.loads(path.read_text(encoding='utf-8'))
        if errors:
            archive=RUN/'audit_corrections/original_model_receipts'/path.name
            if not archive.exists():archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,archive)
            receipt.update(status='FULL_DIAGNOSTIC_SCORED_AFTER_FAILED_SMOKE',smoke_status='FAILED_SOME_NATIVE_SMOKE_ROWS',
                portfolio_eligible=False,full_score_was_generated_before_gate_defect_found=True,
                failure_reason='Original runner failed to enforce per-row smoke convergence gate; full predictions retained only as diagnostic evidence.',
                smoke_failure_details=errors,predictions_unchanged=True)
            save_json(path,receipt)
        records.append({'model_id':model,'smoke_status':'PASS' if not errors else 'FAIL',
          'failed_rows':len(errors),'errors':errors,'portfolio_eligible':not errors,'prediction_sha256':receipt['prediction_sha256']})
    save_json(RUN/'STATISTICAL_SMOKE_GATE_AUDIT.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'records':records,
       'corrected_models':[r['model_id'] for r in records if not r['portfolio_eligible']],
       'policy':'Failed-smoke full outputs were accidentally generated. Preserved diagnostic evidence, excluded from successful full-score count and ensemble selection. No future row fallback or optimizer cap increase.'})
    print('STATISTICAL_GATE_AUDIT','DIAGNOSTIC_ONLY',[r['model_id'] for r in records if not r['portfolio_eligible']],flush=True)
