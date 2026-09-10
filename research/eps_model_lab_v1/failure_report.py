"""Evidence-backed blockers and recovered failures; do not erase initial attempts."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v1.bootstrap import RUN,ORIGINAL_RUN,save_json

EVENTS=[
 ('SOURCE_SHARE_UNIT','CONFIRMED_UNRESOLVED_IN_FROZEN_DATA','MCD original statements show shares in millions while cached shares fields retain display-scale values. HVZ dollar-to-EPS division explodes; 16 wide-input models and all8frozen ensembles also consume this field directly/indirectly.',
  'SOURCE_SHARE_UNIT_QUALITY_AUDIT.json','Preserve original scores with explicit quality warnings. Three fixed no-share-feature diagnostic recipes trained120heads and replayed120heads separately, without portfolio changes. Full source-context unit repair needs a new data identity; no heuristic rescaling or test-based row deletion.'),
 ('PORTABLE_WINDOWS_PATH','RECOVERED','The long project name plus nested handoff paths exceeded Windows MAX_PATH during staged copy.',
  'PORTFOLIO_FINISH_RECOVERY.json','Short workspace-local staging and extended copy paths; staged evaluator passed, then fresh pinned core environment replay passed. Original failed queue and partial stage remain preserved.'),
 ('REPORT_SCHEMA','RECOVERED','Final report initially assumed every smoke receipt was a dictionary, but eleven are lists of per-head results.',
  'PORTFOLIO_FINISH_RECOVERY.json','Explicit nonempty all-PASS list handling with regression tests. No model, metric or smoke status was fabricated.'),
 ('INDEPENDENT_METRIC_DTYPE','EXPLAINED_NO_SCORE_CHANGE','94 OOF interval-width comparisons differed after the independent audit promoted saved float32 quantiles to float64.',
  'INDEPENDENT_NUMPY_METRIC_AUDIT.json','Independent native-dtype subtraction/mean reproduces every original field; all18104metric comparisons pass, same tolerance. Original initial audit retained under audit_corrections; no original score change.'),
 ('NESTED_COUNT_METADATA','CLARIFIED_NO_SCORE_CHANGE','Nested evaluation_labels_available_before_2022 counted filing timestamps even when direct native EPS was missing.',
  'NESTED_META_LABEL_COUNT_CLARIFICATION.csv','Separate forecast-origin, available-filing, finite-truth and scored-truth counts. Existing predictions, score fields and member freeze are unchanged.'),
 ('TABPFN_NUMERICS','RESEARCH_ONLY_UNRESOLVED','Two of forty annual heads exceeded the fixed singleton-size numerical tolerance; other-input and future-label mutation tests passed.',
  'TABPFN_NUMERICAL_LIMITATION_REVIEW.json','FP64 did not repair the first quantile failure and the second head exceeded GPU memory. No more V1 retries, no relaxed tolerance, no replacement forecasts. Fixed-batch FP32 research scores are not deployable numerical certification.'),
 ('ADAPTER_API','RECOVERED','TabPFN official save-fit-state initially rejected a WindowsPath value in JSON metadata.',
  'audit_corrections/tabpfn_initial_path_serialization_failure','The same checkpoint path was passed as a string. Actual40-head full scoring and fresh saved-model replay completed; no training recipe change.'),
 ('NEURAL_BATCH_NUMERICS','DIAGNOSED_NOT_REWRITTEN','Some BiTCN/Informer/TimeMixer singleton-size tests failed under original CuDNN TF32 settings.',
  'BATCH_NUMERICAL_REVIEW.json','All nine strict-FP32 diagnostic cases passed unchanged tolerance; original failures and original scored forecasts remain. Other-origin value mutations were invariant.'),
 ('STOCHASTIC_BATCH','CORRECTIVE_INFERENCE_WAVE','Lag-Llama and two Moirai checkpoints changed first-origin Monte Carlo point forecasts when other input rows changed.',
  'foundation_replay_audits/moirai.json','Original forecasts diagnostic only; origin-isolated inference has a new identity. This does not by itself prove future-informed conditional distributions. See actual isolation smoke/full/replay receipts.'),
 ('SMOKE_ADVANCEMENT','DIAGNOSTIC_ONLY_EXCLUDED','Foster and Griffin-Watts full scoring advanced despite native smoke convergence failures.',
  'STATISTICAL_SMOKE_GATE_AUDIT.json','Predictions retained as diagnostics; receipts corrected and excluded from successful candidate count, selection and portfolio. Future advancement now fails closed.'),
 ('SCIENTIFIC_PIT','RETIRED_AND_REPAIRED','Original SEC fiscal metadata collision removed a past HON record using future knowledge.',
  'PIT_REPAIR_LINEAGE.json','Native-context calendar and full69x4 truncation audit; all original scores remain retired. Research test reuse is not fresh heldout.'),
 ('DATA_SCOPE','DATA_BLOCKED','No historical PIT analyst consensus for CHLW replication.',
  'source_receipts/CHLW_Mendeley.json','Worth revisiting only with licensed historical analyst vintage data; current consensus backfill prohibited.'),
 ('DATA_GEOMETRY','DATA_GEOMETRY_BLOCKED','IBM TTM minimum52 exceeds fixed32 fiscal-quarter history.',
  'EPS_MODEL_REGISTRY_V1.json','Potential separate long-history geometry wave; do not invent52 observations or silently change this dataset.'),
 ('DATA_CONTEXT','QUARANTINED','UPS2013Q1 native EPS/NI/CFO contexts disagree on Jan1/Jan2 start.',
  'DATASET_BUILD_AUDIT.json','Original filing/context audit may restore the single period in a separately versioned data wave. No arbitrary conflict resolution.'),
 ('SOURCE_ACCESS','PARTIALLY_VERIFIED','Direct SEC Archives HTTP requests returned403 in filing spot audit.',
  '../eps_model_lab_v1_20260908T012326/ORIGINAL_FILING_SPOT_AUDIT.json','Web-accessible official AAPL2020 and NVDA2024 originals checked; this is not a full-cohort vintage certification.'),
 ('WINDOWS_ATOMIC_IO','RECOVERED','Transient file lock interrupted core collector and CPU queue JSON replacement.',
  'CORE_SAVED_HEAD_RECOVERY.json','440 already-trained new-run heads recovered without refit; bounded atomic-replace retry and explicit resume added.'),
 ('MODEL_INFERENCE_MODE','RECOVERED','Chronos2 full fine-tuning returned training-mode model with dropout.',
  '../eps_model_lab_v1_20260908T012326/chronos_finetune_failure.json','Explicit eval mode; original failed smoke retained; no model hyperparameter change.'),
 ('SERIALIZATION','RECOVERED','Spawned Windows workers stored adapter classes under __mp_main__.',
  'TABULAR_FRESH_PROCESS_RELOAD_AUDIT.json','Narrow local trusted-class remapping, then600fresh-process exact replays. Loader is not an untrusted pickle sandbox.'),
 ('NUMERICAL_REPLAY','RECOVERED','Darts Kalman smoke differed by one float64ULP (4.44e-16).',
  '../eps_model_lab_v1_20260908T012326/darts_smoke/KalmanForecaster_initial_exact_bit_failure.json','Declared1e-12 replay tolerance; no accuracy tuning.'),
 ('PLATFORM','RECOVERED_ON_CPU','Legacy Uni2TS/LagLlama torch dependencies do not support the RTX5080 build.',
  'environments/moirai.json','Isolated CPU environment; never upgraded frozen PE or forced incompatible CUDA binaries.'),
 ('PLATFORM','RECOVERED_ON_CPU','TiRex2 CUDA import required unavailable toolkit/compiler path.',
  'environments/granite.json','Official CPU backend withCUDA_VISIBLE_DEVICES=-1; no CUDA driver/compiler installation.'),
 ('PRETRAINING','NOT_STRICT_PIT_CERTIFIED','Foundation checkpoints may include information after historical evaluation origins.',
  'EPS_WEIGHT_USE_MANIFEST.json','Keep retrospective track separate. Causal inference and EPS fine-tuning do not erase pretrained information.'),
 ('SOURCE_LICENSE','RESEARCH_ONLY','seferlab source has no resolved redistribution license; some checkpoints have noncommercial/community terms.',
  'EPS_EXTERNAL_SOURCE_MANIFEST.csv','Do not mark publicly deployable or redistribute restricted upstream code/weights in handoff ZIP.'),
 ('PE_RUNTIME','ISOLATION_RECOVERED','Loading v04 before C4 introduced extra numerical backends rejected by exact frozen runtime.',
  'pe_integration/c4_domain_audit','Clean C4 process passed its original pinned runtime without changing frozen environments/tolerances.'),
 ('PE_INPUT_CONTRACT','C4_CONNECTION_BLOCKED','Frozen C4 requires invalid prefix positions(0,1,2); real four-ticker inputs have only(0). CVX also has invalid v04 fallback values.',
  'pe_integration/runtime_failures','No fabricated missing rows or relaxed C4 gate. Next separately authorized PE validation/adapter wave is needed.'),
 ('PE_BASIS','STRICT_COMBINATION_BLOCKED','Native SEC/Yahoo data do not certify original share-basis processing vintage.',
  'PE_INPUT_EQUIVALENCE_REUSE_AUDIT.json','Actual v04 static scenarios remain research-only; no certified implied-price accuracy claim.'),
]


def run():
    records=[]
    for category,status,reason,evidence,next_step in EVENTS:
        p=RUN/evidence
        records.append({'category':category,'status':status,'reason':reason,'evidence':evidence,
          'evidence_exists':p.exists(),'next_step_or_resolution':next_step})
    registry=RUN/'EPS_MODEL_REGISTRY_V1.json'
    if registry.exists():
        for model in json.loads(registry.read_text(encoding='utf-8'))['models']:
            if model['status'] in ['BROKEN','DATA_BLOCKED','PLATFORM_BLOCKED','DATA_GEOMETRY_BLOCKED'] or model['status'].startswith('FULL_DIAGNOSTIC_'):
                records.append({'category':'MODEL_STATUS','status':model['status'],'model_id':model['model_id'],
                   'reason':model.get('failure_reason'),'evidence':model.get('receipt','EPS_MODEL_REGISTRY_V1.json'),
                   'next_step_or_resolution':'See exact adapter receipt. No unattempted model is counted as a successful experiment.'})
    save_json(RUN/'EPS_MODEL_FAILURES_AND_BLOCKERS.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),'records':records})
    lines=['# EPS Model Lab failures, blockers, and recoveries','',
      'Initial failures remain preserved. A recovered platform problem is distinct from a failed scientific contract.','']
    for r in records:
        lines += [f"## {r['category']} — {r['status']}",'',str(r.get('model_id',''))+' '+str(r['reason']),'',
           'Evidence: `'+str(r['evidence'])+'`.','',str(r['next_step_or_resolution']),'']
    (RUN/'EPS_MODEL_FAILURES_AND_BLOCKERS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('FAILURE_REPORT',len(records),flush=True)


if __name__=='__main__':run()
