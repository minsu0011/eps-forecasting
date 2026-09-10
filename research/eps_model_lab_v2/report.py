"""Human handoff from sealed evidence, with explicit scientific limitations."""
from pathlib import Path
import json
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v2.common import RUN,LAB,read_json,save_json,sha,utcnow
from research.eps_model_lab_v2.pe_contract import REQUIRED,compatibility


def md(frame,columns):
    sub=frame[columns].copy()
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in sub.itertuples(index=False,name=None):
        values=[f'{v:.6f}' if isinstance(v,(float,np.floating)) and pd.notna(v) else ('—' if pd.isna(v) else str(v)) for v in row]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def run():
    freeze=read_json(RUN/'EPS_V2_FROZEN_RESEARCH_SURVIVORS.json')
    retrain=read_json(RUN/'EPS_V2_RETRAINING_REPRODUCIBILITY.json')
    replay=read_json(RUN/'EPS_V2_REPLAY_AUDIT.json');source=read_json(RUN/'EPS_SHARE_UNIT_AUDIT_V2.json')
    scores=pd.read_csv(RUN/'EPS_V2_ALL_SURFACE_RESULTS.csv');ens=pd.read_csv(RUN/'EPS_V2_ENSEMBLE_RESULTS.csv')
    plan=read_json(RUN/'prescore/DEVELOPMENT_SELECTION_FREEZE.json')
    survivors=freeze['individual_survivors'];ids=[r['model_id'] for r in survivors]
    paths=[];exports=[]
    for phase in ['confirmation','monitor']:
        for r in survivors:
            for p in (RUN/'predictions'/phase/r['model_id']/'main').glob('*.parquet'):
                f=pd.read_parquet(p);f=f[f.target==r['target']].copy()
                if f.empty:continue
                f['research_survivor_target']=r['target'];f['evidence_surface']=phase
                if not set(REQUIRED)<=set(f):raise RuntimeError('Missing export fields')
                exports.append(f);paths.append(str(p.relative_to(RUN)))
    export=pd.concat(exports,ignore_index=True)
    path=RUN/'EPS_V2_FROZEN_SURVIVOR_PREDICTIONS.parquet'
    if path.exists():raise FileExistsError(path)
    export.to_parquet(path,index=False)
    if export.duplicated(['sample_id','target','eps_model_id']).any():raise RuntimeError('Duplicate handoff identity')
    # These are Lane A forecasts; annual PE must reject them. We actually
    # exercise the fail-closed contract rather than mark schema-only as deployable.
    rejected=0
    for row in export.to_dict('records'):
        pe={**row,'expected_pe':10.0}
        result=compatibility(row,pe)
        if result['PE_READY']:raise RuntimeError('Quarterly EPS accidentally admitted to annual PE')
        rejected+=1
    save_json(RUN/'audit/PE_EXPORT_CONTRACT_AUDIT_V2.json',{'created_utc':utcnow(),'status':'PASS_FAIL_CLOSED',
        'export_rows':len(export),'required_fields':REQUIRED,'all_fields_present':True,'all_signed_forecasts_preserved':True,
        'quarterly_to_annual_PE_rejections':rejected,'strict_certified_combinations':0,
        'actual_PE_runtime_invoked':False,'export_sha256':sha(path)},immutable=True)
    telemetry_path=RUN/'audit/RESOURCE_TELEMETRY_V2.jsonl'
    telemetry=pd.DataFrame(json.loads(line) for line in telemetry_path.read_text(encoding='utf-8').splitlines() if line.strip())
    resource={'created_utc':utcnow(),'samples':len(telemetry),'first_sample_utc':telemetry.utc.iloc[0],
        'last_sample_utc':telemetry.utc.iloc[-1],'scope':'Started mid-run; samples include idle and audit intervals; not whole-run utilization',
        'metrics':{c:{'median':float(telemetry[c].median()),'p95':float(telemetry[c].quantile(.95)),
            'max':float(telemetry[c].max())} for c in ['CPU_percent','RAM_used_GiB','GPU_percent','VRAM_used_MiB','GPU_power_watts']}}
    save_json(RUN/'audit/RESOURCE_SUMMARY_V2.json',resource,immutable=True)
    # Completion is a transparent deliverable checklist, not confidence or a
    # claim of forecasting readiness. Future-data/vintage blockers count false.
    checks={
        'Data quality':{'separate_frozen_identity':True,'context_ledger':True,'source_unit_quarantine':True,
            'exact_duration_selection':True,'full_prefix_mutation_audit':True,'100_API_context_checks':True,
            'target_method_and_basis_flags':True,'100_original_filing_audits':False,'historical_vintage_certified':False,'verified_pre2019_accounting_support':False},
        'Lane A':{'bounded_native_training':True,'development_gain':True,'confirmation_gain':True,'median_tail_gates':True,
            'full_inference_coverage':True,'saved_replay':True,'two_new_training_repeats':True,'local_and_foundation_survivors':bool(survivors),
            'fully_verified_source_vintage':False,'untouched_confirmation':False},
        'Lane C':{'direct_and_residual_fit':True,'delta_equivalence_audit':True,'chronological_comparison':True,
            'full_coverage_saved_replay':True,'tail_subgroup_diagnostics':True,'dev_and_confirmation_survivor':False,
            'survivor_retraining':False,'verified_historical_basis':False,'ready_PE_model_pair':False,'untouched_confirmation':False},
        'Model portfolio':{'bounded_11_families':True,'registered_46_native_variants':True,'track_separation':True,
            'frozen_ensemble_weights':True,'diversity_diagnostics':True,'time_tail_coverage':True,
            'research_freeze_and_retraining':True,'PE_contract_and_preservation':True,'supported_clean_accounting_models':False,'Lane_C_core':False},
        'PE compatibility':{'required_output_fields':True,'currency_check':True,'basis_check':True,'horizon_origin_check':True,
            'GAAP_diluted_TTM_checks':True,'negative_forecast_retention':True,'mismatch_fail_closed_tests':True,
            'PE_readonly_audit':True,'actual_compatible_finalist_pair':False,'basis_and_model_formal_certification':False},
        'Formal certification readiness':{'untouched_evidence_reserved_and_scored':False,'original_source_vintage_verified':False,
            'historical_share_basis_verified':False,'formal_model_and_PE_joint_gate':False}}
    completion={name:{'passed':sum(items.values()),'total':len(items),'percent':100*sum(items.values())/len(items),'checks':items} for name,items in checks.items()}
    save_json(RUN/'EPS_V2_COMPLETION_CHECKLIST.json',{'created_utc':utcnow(),'meaning':'Engineering/research deliverable checklist only, not statistical confidence or operational readiness','areas':completion},immutable=True)
    a=scores[(scores.target=='h1')&scores.model_id.isin(ids+['persistence_observed'])]
    c=scores[(scores.target=='ttm')&scores.model_id.isin(['persistence_observed','HistGB_N_residual','NGBoostLaplace_N_residual','Chronos2_N_native','Chronos2_N_joint_rolling','XLinear_N_direct'])]
    ae=ens[ens.target=='h1'];cal=pd.concat([pd.read_csv(RUN/f'EPS_V2_CALIBRATION_{p}_RESULTS.csv') for p in ['CONFIRMATION','MONITOR']])
    best_a=scores[(scores.surface=='confirmation')&(scores.target=='h1')&scores.model_id.isin(ids)]
    best_foundation=best_a[best_a.model_track=='RETROSPECTIVE_FOUNDATION'].sort_values('MAE').iloc[0]
    best_local=best_a[best_a.model_track=='LOCAL_CAUSAL_RESEARCH'].sort_values('MAE').iloc[0]
    best_robust=best_a[best_a.family.isin(['HistGB','NGBoostLaplace','CatBoost'])].sort_values('MAE').iloc[0]
    best_ensemble=ae[ae.surface=='confirmation'].sort_values('MAE').iloc[0]
    unit_scope='PASS(격리 정책) / 원문 단위 전체 검증 미완료'
    text=f'''# EPS Model Lab V2 최종 연구 보고서

작성: {utcnow()}
Output identity: `{RUN.name}`

## 먼저 볼 결론

- DATA V2 unit audit: **{unit_scope}**. verified 회계 cell {source['verified_accounting_cells']}개, 격리 {source['quarantined_accounting_cells']:,}개. pre-2019 verified 회계 cell 0 → Track A DATA_BLOCKED.
- Duration audit: PASS(정확한 fiscal context 선택/불명확 값 제외). PIT mutation: PASS, 276조건 중 실제 재구축 265건·과거 origin 없는 11조건, violation 0.
- Lane A best foundation(확인 성적 기술): `{best_foundation.model_id}`, MAE {best_foundation.MAE:.6f}. best local(확인 성적 기술): `{best_local.model_id}`, MAE {best_local.MAE:.6f}. 이것으로 개발 당시 대표를 교체하지 않았다.
- 사전 개발 기준 local core는 HistGB direct, robust distributional은 NGBoost Laplace direct, temporal diversity는 LSTM residual recency다. CatBoost direct도 독립 연구 통과 후보다.
- Best robust/tabular(확인 성적 기술): `{best_robust.model_id}`, MAE {best_robust.MAE:.6f}. 분포 예측 역할은 NGBoost로 별도 유지한다.
- Best observed frozen ensemble: `{best_ensemble.model_id}`, MAE {best_ensemble.MAE:.6f}; 확인 표본 persistence 0.893223. 이 성적을 보고 가중치를 다시 맞추지 않았다.
- Lane C: **{freeze['LANE_C_STATUS']}**. development를 통과한 HistGB/NGBoost residual은 confirmation에서 persistence를 이기지 못했다. direct/native/sequence 확인 성적이 상대적으로 좋아도 개발 gate 실패 모델을 구제하지 않는다.
- Coverage: 모든 선언한 원본 모델의 전체 연도 origin에 유한 예측 100%; 점수는 동일한 공개완료 truth mask. 미공개 미래 정답은 coverage 분모에 끼워 넣거나 0으로 채우지 않는다. 앙상블 점수 coverage는 고정 평가 표본 기준이다.
- Saved replay PASS, 실제 추가 재학습 **{retrain['actual_additional_fit_units']} fit units**, zero-shot 재추론 {retrain['zero_shot_inference_units']} units. 기존 main 외 repeat1/repeat2 두 번, 2019~2026 모든 8 annual cutoff/5 targets. 예측 exact PASS. checkpoint byte mismatch {retrain['checkpoint_byte_mismatches']}건은 아래 설명과 원본 audit를 참조한다.
- Final individual research survivors: {len(survivors)}개 Lane A model/role. Lane C 0개. 정식 인증·production 승격·인증된 PE 가격 결합 모두 **0**.
- PE readiness: 스키마/불일치 차단 구현, 실제 export {len(export):,}개 검사. 모두 quarterly EPS이므로 annual PE 곱셈은 차단. PE runtime 호출/수정 없음.

## 1. 데이터 의미와 아직 풀리지 않은 한계

69개 기업, 3,753 sample origins, 4,458 fiscal rows, 62,412 source ledger rows를 V2에서 새로 만들었다.
고정된 69개 회사 연구 cohort이므로 전 시장·상장폐지 포함 역사적 투자 가능 universe의 대표성을 인증한 것은 아니다.
V1과 공통 origin 3,751개 중 h1/h2/h3/h4/TTM label 변경은 각각 15/26/24/26/20개다.
가격/action evidence는 과거 시점 원본 vintage가 완전히 인증되지 않은 retrospective cache다.
Native EPS/TTM도 `PARTIALLY_VERIFIED_NATIVE`이며 완전한 historical source 원문 보장으로 부르지 않는다.

Share scale는 원문 context/unitRef/scale 우선, 같은 accession statement 표시 단위 차선이다.
MCD 2023 732.3 million, 2024 721.9 million 두 statement만 732,300,000 / 721,900,000으로 명시 수리했다.
작은 수치 threshold 배율, EPS≈NI/shares 역추론은 하지 않았다.
SEC 원문 HTTP 403 두 건, 공식 IR PDF download timeout 두 건을 보존했다. 접근 제한 우회는 하지 않았다.
무작위 100개는 API fact의 accession/기간/값 대조이며 original XBRL 원문 100개 감사 완료가 아니다.

정확한 native fiscal start와 이전 공개 분기 end+1을 사용하고 13/14/17주 context를 허용한다.
YTD/quarter/annual 혼합의 earliest-start 선택은 금지했다. IBM 2019 Q3 filed-date 모순은 격리했다.
TTM의 annual/direct/bridge approximation method를 보존했고 `EPS_V2_TTM_METHOD_DIAGNOSTICS.csv`에 분리 성적을 넣었다.
네 분기 예측 합계는 native TTM이라고 부르지 않는다. 역사적 동일 share basis 미인증이므로 path-derived 추가 estimator는 실행하지 않았다.

## 2. 시간축과 사전 고정

2015~2018 개발 → 2019~2021 이미 본 확인 → 2022~2026 관찰 전용이다.
Annual Jan1 fit에서 origin과 각 target 공개 시각이 cutoff보다 앞서야 한다. Label이 늦은 행은 horizon별 purge했다.
2018 origin의 C truth는 2019 이전에 공개되지 않아 C 개발 점수는 2015~2017 756개다. A 개발은 889개, 확인 A 648개/C 512개다.
46개 실제 변형을 11개 family 안에서 비교하고 family별 최대5, seed1729로 고정했다.
Ridge/HistGB/CatBoost/NGBoost는 direct/asinh/residual/rolling/recency, 시계열은 direct/residual/rolling/recency.
Chronos는 native/joint/residual/rolling/recency 5개와 별도 zero-shot control이다. Foundation family는 앙상블에 한 번만 포함한다.
Track A 7개 등록 entry는 data blocked이고 Track M은 꺼져 있다. 53개 등록을 53개 학습 성공으로 세지 않는다.
Foundation은 `RETROSPECTIVE_FOUNDATION`: 원래 weights vintage상 2019에 배포할 수 있었다는 주장이 아니다.
Local causal는 forward source/label purge 의미이며 데이터 historical vintage 자체가 완전 인증됐다는 뜻은 아니다.

## 3. Lane A 고정 후보 성적

{md(a,['surface','model_id','truth_rows','MAE','baseline_MAE','MAE_gain_fraction','MedianAE','p99_AE','worst_year_harm_ratio'])}

Monitor는 기록만 한다. 낮아진 성적을 이유로 recipe/seed/member/weight를 다시 고르지 않는다.
Confirmation에서 CatBoost/HistGB residual처럼 더 좋아 보이는 변형도 다른 lane 대표였으면 A winner로 교체하지 않았다.
모든 tail/subgroup/year gate는 pre-score protocol의 3% gain, MedianAE non-worse, p99 ratio≤1.25,
worst-year MAE ratio≤1.30, negative/transition ratio≤1.50(지원≥20행)을 그대로 적용했다.
소수 연도의 p-value나 subgroup 승자를 실시간 router로 과장하지 않았다.

## 4. Lane C: 실패를 유지하는 이유

{md(c,['surface','model_id','truth_rows','MAE','baseline_MAE','MAE_gain_fraction','MedianAE','p99_AE'])}

개발 HistGB residual gain5.55%, NGBoost residual3.55%가 확인에서 각각 -2.11%, -1.54%로 사라졌다.
C direct는 별도로 실제 학습했다. C delta는 current TTM이 관측된 origin에서 persistence residual과 정확히 같은 수학적 후보라서 별도 다양성으로 세지 않는다.
current TTM missing에서는 last observed native TTM, 완전 부재 시 4×last EPS라는 명시 예측 fallback만 사용한다. truth 생성은 아니다.
XLinear C는 확인 구간에서 약3.09% 개선되지만 개발 실패 모델이므로 승격하지 않는다.
다음 연구의 우선순위는 원문 회계·share-basis source 확보와 사전에 새로 잠근 C 가설이다. 이미 본 TEST를 대상으로 추가 미세조정하지 않는다.

## 5. 고정 앙상블·다양성·확률 구간

{md(ae,['surface','model_id','MAE','MAE_gain_fraction','MedianAE','p99_AE'])}

Local A는 HistGB/NGBoost/CatBoost/LSTM 4개, mixed A는 Chronos rolling을 더한 5개다.
Local 4개에서 floor(20%×4)=0이므로 trimmed mean=mean이다. C는 2개뿐이라 mean/median/trimmed가 같고 local=mixed다.
이를 독립 후보 다양성으로 세지 않는다. 모든 convex weight는 개발 residual만으로 fit, 확인/관찰에서는 적용만 했다.
앙상블에는 unbiased development-stack score가 없으므로 개별 모델의 양구간 gate 인증과 구분해 frozen research ensemble candidate로 둔다.
상관계수·equal-weight gain·global leader 대비 gain은 `EPS_V2_ERROR_CORRELATION.csv`에 있다. weak+weak 개선만으로 core를 승격하지 않았다.

{md(cal[cal.target=='h1'],['surface','model_id','interval80_additive_expansion','interval80_coverage','interval80_mean_width','mean_pinball_loss'])}

위 interval은 2018 A/2017 C 개발 residual의 고정 80% additive expansion이다. 보장된 exchangeable coverage가 아니다.
확인 A에서 Chronos rolling의 실제 coverage는 보정 전70.99%→보정 후74.07%로, 명목80%에 여전히 못 미친다.
NGBoost direct는72.84%→82.10%다. Chronos의 좋은 점 예측을 충분한 불확실성 보정으로 오해하지 않는다.
원래 probabilistic interval은 ALL_SURFACE_RESULTS, 보정 후는 CALIBRATION 파일로 구분한다.
Point-only 모델의 interval=0 표기 결함은 normalized reporting에서 missing으로 정정했다. 봉인 개발 CSV/점 선택 수치는 그대로다.

## 6. 재현성·유지보수·실행 최적화

V2 geometry 56개 HistGB fit 벤치마크: 16/20/24/28 workers = 5.756/6.663/7.762/8.919초. 16 outer×1 inner를 실제 사용했다.
GPU heavy train은 동시에1개, CPU 표 모델/감사는 병렬 실행했다. 메모리를 채우려고 불필요하게 모델을 키우지 않았다.
관측 telemetry {len(telemetry)}개는 중간부터 수집했다. RAM max {resource['metrics']['RAM_used_GiB']['max']:.2f}GiB,
VRAM max {resource['metrics']['VRAM_used_MiB']['max']/1024:.2f}GiB. 상세 median/p95/max는 RESOURCE_SUMMARY_V2.json.
Chronos 원본 checkpoint는 revision 29ec3766d36d6f73f0696f85560a422f50e8498c, 원본 weights hash 검증 후 fold마다 새로 시작했다.
Torch 결정성·cuDNN profile·TF32 off·CUBLAS :4096:8을 사용했다. Transformers helper의 workspace 변경은 본 학습 전에 제거했다.
Pandas rolling read-only 배열은 optimizer 전 실패하고 writable copy로 고쳤다. 기존 완료 checkpoint는 hash 확인 후 skip했다.
CatBoost는 prob 환경에 없어 import 전 실패; 이미 설치된 core 환경에서 최초 fit했다. 패키지를 덮어 설치하지 않았다.
Source SHA 반복 I/O는 회사당1회, native tensor는 frozen cache, outer oversubscription은 benchmark로 줄였다.
모든 실패 artifact는 `failures/`에 보존했다. 예측 clipping/NaN 정답 채움/완료 identity 덮어쓰기 없음.

새 프로세스 saved replay는 표 모델720 heads, 로컬 신경망120 annual, Chronos48 annual outputs를 검증했다.
실제 fresh retraining은 최종 후보 2019~2026 전체에 repeat1/repeat2를 수행했다. 파일 byte hash와 learned state/예측 일치를 별도로 기록한다.
직렬화 metadata 차이의 checkpoint byte mismatch는 예측 exact와 구분한다. 다른 GPU/driver의 universal reproducibility를 주장하지 않는다.
CatBoost 80회 native model JSON 대조에서 트리·feature 등 모든 학습 상태가 같았고 model_guid/train_finish_time만 달랐다. 원래 checkpoint 파일은 수정하지 않았다.
회귀 테스트35개 PASS. V1 3,840파일 inventory와 data/작은 파일 내용 hash, PE371파일 내용 hash 보존 PASS.
대형 V1 학습 weights는 size/mtime 검사만 했으며 full cryptographic rehash라고 부르지 않는다.

## 7. 완료도: 명시적 체크리스트

{' ; '.join(f'{name}: {r["passed"]}/{r["total"]} = {r["percent"]:.0f}%' for name,r in completion.items())}

각 항목의 true/false 근거는 `EPS_V2_COMPLETION_CHECKLIST.json`. 위 %는 engineering/research deliverable 비율일 뿐 예측 성공 확률·생산 준비도·통계 신뢰도가 아니다.
Formal certification readiness 0%는 untouched evidence, 원문 source vintage, historical share basis, formal joint gate가 모두 미완료라는 뜻이다.
이번 가능한 Track N 연구 실행은 끝났지만 Track A와 Lane C를 완성한 것으로 포장하지 않는다.

## 8. 인계·다음 작업

1. `EPS_V2_FROZEN_RESEARCH_SURVIVORS.json`과 이 보고서부터 읽는다. Lane A 6개 연구 후보, Lane C NOT_READY, Track A DATA_BLOCKED를 유지한다.
2. 새 회계 source wave는 공개 원문/허용된 원문 데이터 공급자로 exact accession/context/unitRef/scale/accepted를 확보해야 한다. 무단 접근·단위 heuristic 금지.
3. 신규 untouched 구간/사전 봉인 계획 없이는 formal 인증을 주장하지 않는다. 이번 confirmation/monitor를 추가튜닝에 쓰지 않는다.
4. Lane C는 source repair 후 predeclared direct/delta/seasonal residual 가설을 새 identity로 검증한다. 이번 C 실패 checkpoint를 조용히 refit하지 않는다.
5. PE source/C4 gates는 그대로 두고 현재 v04 production/C4-R2 research frozen 상태를 유지한다. EPS quarterly를 annual PE와 곱하지 않는다.
6. 재실행은 `EPS_V2_ENVIRONMENT_LOCK.json`, `EPS_V2_SOURCE_MANIFEST.json`, `CHECKSUMS.sha256`를 확인한다. 원래 RUN의 one-shot scripts 재실행은 existing identity guard로 거절되는 것이 정상이다.

`PROGRESS_AND_DECISIONS.md`는 근거·결정·해결점 기록이며 비공개 내부 사고의 원문 로그가 아니다.
이 출력 폴더에는 실제 모델/checkpoint/예측이 있다. 별도의 모든 외부 환경/원천 데이터가 포함된 standalone 배포판은 아니다.
'''
    (RUN/'EPS_MODEL_LAB_V2_FINAL_REPORT.md').write_text(text,encoding='utf-8')
    shutil_source=LAB/'PROGRESS_AND_DECISIONS.md'
    (RUN/'PROGRESS_AND_DECISIONS.md').write_text(shutil_source.read_text(encoding='utf-8'),encoding='utf-8')
    print('V2_FINAL_REPORT_WRITTEN',len(export),'survivor forecast rows',flush=True)


if __name__=='__main__':run()
