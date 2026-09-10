"""Research result summary; no selection on test results."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, ORIGINAL_RUN, LAB, save_json, sha
from research.eps_model_lab_v1.ensemble_models import EXCLUDED

START = datetime.fromisoformat('2026-09-07T16:23:26+00:00')
DEADLINE = datetime.fromisoformat('2026-09-08T02:23:26+00:00')


def read(name): return json.loads((RUN/name).read_text(encoding='utf-8'))


def smoke_pass(receipt):
    """Accept explicit successful smoke records, never infer success from scoring."""
    if isinstance(receipt, dict):
        return receipt.get('status') == 'PASS'
    if isinstance(receipt, list):
        return bool(receipt) and all(isinstance(r, dict) and r.get('status') == 'PASS' for r in receipt)
    return False


def table(frame, columns):
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return '—' if not np.isfinite(v) else f'{v:.6f}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    lines.extend('| '+' | '.join(cell(row[c]) for c in columns)+' |' for _, row in frame.iterrows())
    return '\n'.join(lines)


def run():
    now = datetime.now(timezone.utc); elapsed = (now-START).total_seconds()/3600
    registry = read('EPS_MODEL_REGISTRY_V1.json')['models']; status = Counter(r['status'] for r in registry)
    lock = read('ENSEMBLE_FREEZE_V1.json'); integrity = read('EPS_RESULT_INTEGRITY_AUDIT.json')
    if not integrity['all_pass']: raise RuntimeError('Unresolved prediction integrity failure')
    shortlist = read('EPS_RESEARCH_SHORTLIST_V1.json'); combo = read('EPS_PE_COMBINATION_RECEIPT.json')
    share_quality = read('SOURCE_SHARE_UNIT_QUALITY_AUDIT.json')
    duration_quality = read('SOURCE_ACCOUNTING_DURATION_QUALITY_AUDIT.json')
    late = read('EPS_LATE_FRONTIER_REVIEW.json')
    seed_review = read('CHRONOS_TRAINING_SEED_REVIEW.json')
    score = pd.read_csv(RUN/'EPS_SELECTION_LEADERBOARD_V1.csv')
    score = score[score.portfolio_eligible & ~score.eps_model_id.isin(EXCLUDED) & ~score.eps_model_id.str.startswith('ens_')]
    broad = score[score.coverage.ge(.999) & score.MAE.notna()]
    test = pd.read_csv(RUN/'EPS_MODEL_ZOO_LEADERBOARD_V1.csv')
    test = test[(test.split == 'RESEARCH_TEST') & (test['mask'] == 'COMPLETE_COVERAGE') & test.portfolio_eligible]
    receipts = [json.loads(p.read_text(encoding='utf-8')) for p in (RUN/'model_receipts').glob('*.json')]
    scored = [r for r in receipts if (RUN/'predictions'/f"{r.get('model_id')}.parquet").exists()]
    diagnostics = [r for r in scored if r['status'].startswith('FULL_DIAGNOSTIC_')]
    bases = [r for r in scored if not r['model_id'].startswith('ens_')]
    ensembles = [r for r in scored if r['model_id'].startswith('ens_')]
    base_ids = {r['model_id'] for r in bases}
    adapter_count = sum(r['model_id'] in base_ids and bool(r.get('adapter_path')) for r in registry)
    smoke_ids = {r['model_id'] for r in bases if str(r.get('smoke_status', '')).startswith('PASS')}
    for path in (RUN/'smoke_receipts').glob('*.json'):
        receipt = json.loads(path.read_text(encoding='utf-8')); name = path.stem.removesuffix('_summary')
        if name in base_ids and smoke_pass(receipt): smoke_ids.add(name)
    smoke_ids -= {r['model_id'] for r in diagnostics if 'FAILED_SMOKE' in r['status']}
    blocked_count = sum(r['status'] in {'BROKEN', 'DATA_BLOCKED', 'PLATFORM_BLOCKED', 'DATA_GEOMETRY_BLOCKED'} for r in registry)
    summaries = {'created_utc': now.isoformat(), 'requested_hours': 10, 'elapsed_hours_at_report': elapsed,
        'start_utc': START.isoformat(), 'deadline_utc': DEADLINE.isoformat(), 'registered_identities': len(registry),
        'registry_status_counts': dict(status), 'actual_base_prediction_files_including_diagnostics': len(bases),
        'qualified_full_base_outputs': sum(r['status'] == 'FULL_RESEARCH_SCORED' for r in bases),
        'diagnostic_only_base_outputs': len(diagnostics), 'frozen_ensemble_outputs': len(ensembles),
        'actual_prediction_files_total': len(scored), 'fixed_ensemble_pool_identities': len(lock['candidate_prediction_hashes']),
        'origin_isolation_corrections_not_independent_architectures': sum(r['model_id'].endswith('_origin_isolated') for r in bases),
        'imports_verified_by_actual_base_execution': len(bases), 'adapters_with_executed_source_path': adapter_count,
        'explicit_initial_smoke_pass_identities': len(smoke_ids), 'currently_blocked_registry_entries': blocked_count,
        'formal_certified': 0, 'PE_integration_status': combo['status']}
    summaries.update(source_share_unit_exposed_base_ids=share_quality['directly_exposed_scored_base_ids'],
        source_share_unit_exposed_ensemble_ids=share_quality['indirectly_exposed_ensemble_ids'],
        separate_post_freeze_no_share_feature_diagnostic_recipes=3,
        separate_post_freeze_additional_architectures=late['additional_prediction_files'],
        separate_fixed_training_seed_annual_models=15,
        separate_EPS_TTM_only_channel_ablation_recipes=2,
        accounting_duration_exposed_base_ids=duration_quality['direct_base_ids'],
        accounting_duration_defect_unresolved=True,
        clean_accounting_feature_certification=False)
    save_json(RUN/'EPS_FINAL_SCOREBOARD.json', summaries)
    columns = ['eps_model_id', 'predicted_rows', 'MAE', 'MedianAE', 'price_scaled_MAE', 'coverage']
    a_score = broad[broad.target_key == 'h1']
    top_point = a_score.sort_values(['MAE', 'eps_model_id']).iloc[0].eps_model_id
    top_robust = a_score.sort_values(['MedianAE', 'eps_model_id']).iloc[0].eps_model_id
    top_foundation = a_score[a_score.checkpoint_pretraining_overlap_unresolved].sort_values(['MAE', 'eps_model_id']).iloc[0].eps_model_id
    top_deep = a_score[a_score.eps_model_id.str.startswith(('nf_', 'epspredict_'))].sort_values(['MAE', 'eps_model_id']).iloc[0].eps_model_id
    diversity_ids = [r['model_id'] for r in shortlist['models'] if 'EPS_DIVERSITY_COMPONENT' in r['roles']]
    best_ensembles = {t: test[(test.target_key == t) & test.eps_model_id.str.startswith('ens_')]
        .sort_values(['MAE', 'eps_model_id']).iloc[0].eps_model_id for t in ['h1', 'ttm']}
    lines = ['# EPS Model Lab V1 — 10시간 예산 연구 결과', '',
        '**중요한 미해결 데이터 오류:** 원문 공시와 다른 주식 수 단위가 확인됐습니다. '
        '17개 기본 모델과 8개 앙상블이 직접·간접 영향을 받으므로 아래 성적을 깨끗한 회계 feature 성능이나 배포 인증으로 읽지 마십시오. '
        '`SOURCE_SHARE_UNIT_WARNING_KO.md`를 먼저 읽으십시오. 기존 성적은 삭제·보정하지 않았습니다.', '',
        '**추가 의미 오류:** YTD 회계 열에 최근 12개월·단독 분기 context가 혼합됐습니다. '
        '기본 26종(다변량 9종 포함)과 앙상블 8종에 영향이 있습니다. '
        '`SOURCE_ACCOUNTING_DURATION_WARNING_KO.md`도 먼저 읽으십시오. 주식 수를 쓰지 않는 TimeXer/SOFTS나 '
        'no-share ablation도 깨끗한 YTD 모델이라는 뜻은 아닙니다.', '',
        '## Scoreboard', '',
        f"- 등록 ID: {len(registry)}개. 별칭·미적용 클래스·미실행 후보를 포함하며 독립 모델 수가 아닙니다.",
        f"- 실제 기본 실행으로 import 확인: {len(bases)}개. 실행된 공통 adapter 경로: {adapter_count}개. 명시적 초기 smoke PASS 기록: {len(smoke_ids)}개(후속 QA 탈락은 별도).",
        f"- 현재 BLOCKED/BROKEN 등록 항목: {blocked_count}개. 단순 미실행·중복 클래스와 구분했습니다.",
        f"- 실제 기본 예측: {len(bases)}종 = 연구 점수 유지 {summaries['qualified_full_base_outputs']}종 + 진단 전용 {len(diagnostics)}종.",
        f"- 고정 앙상블: {len(ensembles)}종. 전체 예측 파일 {len(scored)}개, 정합성 검사 {'PASS' if integrity['all_pass'] else 'FAIL'}.",
        '- 별도 source-quality 진단 3종: 동일 Ridge/HistGB/CatBoost에서 raw 주식 수 열만 제거해 120개 head 학습·120개 fresh 재현 완료. 새 portfolio 멤버나 독립 구조로 세지 않습니다.',
        f"- 포트폴리오 고정 이후 별도 구조 {late['additional_prediction_files']}종: {late['additional_annual_models']}개 annual 모델, {late['fresh_saved_replay_checks']}개 saved 재현. 기존 117개 main 예측에 합치지 않습니다.",
        '- Chronos OOF 학습 난수 진단: 고정 5 seed × 3년 = 15개 실제 모델, 75개 point/quantile 재현. 기준 seed는 원래 예측과 bit-exact 일치. best seed로 교체하지 않았습니다.',
        '- NI/매출 채널 제거 진단: TimeXer/SOFTS를 EPS·TTM 두 채널로만 16개 annual 학습·80개 저장 재현. 별도 진단이며 117개 main 수에 합치지 않습니다.',
        f"- 최종 앙상블의 기본 후보 풀: {len(lock['candidate_prediction_hashes'])}개. 기존 보간 기준선 5종과 진단 전용 출력은 제외.",
        '- 정식 인증·운영 승격: **0개**. Foundation 사전학습 시점, SEC/주식수 데이터 빈티지 등 미해결.',
        f"- Lane A OOF: point/MAE `{top_point}`; robust/MedianAE `{top_robust}`; foundation `{top_foundation}`; local deep `{top_deep}`.",
        f"- OOF 실제 조합 근거가 있는 diversity 후보: {', '.join(diversity_ids) or '별도 확정 근거 없음'}.",
        f"- 고정 앙상블의 설명용 research-test raw-MAE 최소: A `{best_ensembles['h1']}`, C `{best_ensembles['ttm']}`. 이를 근거로 가중치 재조정·승격은 하지 않았습니다.",
        f"- PE 연결: {combo['status']}; 연구 시나리오 {combo['research_valid']}/{combo['rows']}, 엄격 인증 {combo['strict_certified_valid']}.",
        '', '사용자의 최신 지시에 따라 원래 8시간 프롬프트를 10시간 예산으로 변경했습니다.',
        f"시작: 2026-09-08 01:23:26 KST. 마감: 11:23:26 KST. 이 보고서 작성: {now.astimezone(timezone(timedelta(hours=9))).isoformat()} (경과 {elapsed:.3f}시간).", '',
        '## 먼저 알아야 할 결과 해석', '',
        '아래 후보 선정은 2019–2021 origin 중 해당 정답이 2022-01-01 전에 공개된 OOF만 사용합니다. '
        '2022년 이후 research-test 성적은 설명용이며, 이를 보고 모델·가중치를 다시 고르지 않았습니다. '
        '이전 데이터 오류를 수정한 재실행이므로 이미 본 기간을 fresh heldout이라 부르지 않습니다.', '',
        '현재 데이터: 69개 회사, 전체 3,775 origin, 연구 예측 2,101 origin. '
        'GAAP diluted signed EPS, forecast-origin share basis, 유효 거래일 종가 시점. '
        'A/B는 직접 보고된 분기 EPS, C는 Q+4 시점의 native TTM입니다. '
        '누락된 직접 Q4를 연간−9개월 값으로 꾸며 넣지 않았습니다.', '']
    for target, title in [('h1', 'Lane A: 다음 분기 EPS'), ('ttm', 'Lane C: 미래 native TTM EPS')]:
        g = broad[broad.target_key == target].sort_values(['MAE', 'eps_model_id'])
        lines += ['## '+title+' — full-coverage OOF', '', table(g.head(7), columns), '',
            '같은 정답 행에서의 비교입니다. 원화 수익률이나 주가 예측 정확도가 아닌 USD EPS 오차입니다. '
            'MAE만 아니라 중앙오차·가격으로 나눈 오차를 함께 확인하십시오.', '']
        local = g[~g.checkpoint_pretraining_overlap_unresolved]
        lines += [f"OOF raw-MAE 선두: `{g.iloc[0].eps_model_id}`. 사전학습 체크포인트를 쓰지 않는 로컬 트랙 선두: `{local.iloc[0].eps_model_id}`.", '']
    path_scope = read('EPS_LANE_B_SCOPE_RECEIPT.json')
    path_scores = pd.read_csv(RUN/'EPS_LANE_B_PURGED_PATH_REVIEW.csv')
    path_ids = {r['model_id'] for r in shortlist['models']} | {'persistence_observed', 'seasonal_observed'}
    path_scores = path_scores[path_scores.model_id.isin(path_ids)]
    lines += ['## Lane B: 네 분기 경로', '',
        '각 horizon별 OOF 정답은 h1 649개, h2 582개, h3 520개, h4 458개입니다. '
        '네 개 정답이 모두 실제 존재하고 2022년 전에 공개된 경로는 265개·43개 기업입니다. '
        'research-test의 완전한 네 분기 경로는 **34개·4개 기업**뿐입니다. '
        '따라서 h1 한 단계 성적을 네 분기 전체 경로 성능으로 확대하지 않습니다. '
        '네 직접 분기 EPS의 합과 Lane C native TTM도 서로 다른 target입니다.', '',
        table(path_scores, ['model_id', 'surface', 'complete_truth_paths', 'complete_predicted_paths',
            'path_mean_MAE', 'four_direct_quarter_sum_MAE_NOT_NATIVE_TTM']), '',
        '이 표로 B winner를 추가 선택하지 않았습니다. 전체 행은 `EPS_LANE_B_PURGED_PATH_REVIEW.csv`, '
        '연도·기업 수는 `EPS_LANE_B_SCOPE_RECEIPT.json`에 있습니다.', '']
    lines += ['## TOP EPS CORE CANDIDATES', '',
        '최대 10개 연구 후보입니다. 역할은 OOF 근거이며 독립적인 specialist 성능 인증이 아닙니다.', '']
    role_rows = [{'model_id': r['model_id'], 'roles': ', '.join(r['roles']),
                  'h1_OOF_MAE': r['metrics_by_target'].get('h1', {}).get('MAE', np.nan),
                  'ttm_OOF_MAE': r['metrics_by_target'].get('ttm', {}).get('MAE', np.nan)} for r in shortlist['models']]
    role_frame = pd.DataFrame(role_rows)
    core = role_frame.roles.str.contains('EPS_CORE|EPS_ENSEMBLE_COMPONENT', regex=True)
    lines += [table(role_frame[core], ['model_id', 'roles', 'h1_OOF_MAE', 'ttm_OOF_MAE']), '',
        '## TOP EPS DIVERSITY / SPECIALIST CANDIDATES', '',
        table(role_frame[~core], ['model_id', 'roles', 'h1_OOF_MAE', 'ttm_OOF_MAE']), '',
        '`EPS_RESEARCH_SHORTLIST_V1.json`에 각 역할의 실제 수치·표본 수를 보관했습니다. '
        'Diversity는 oracle이 아니라 실제 50:50 조합의 OOF 오차 개선을 확인합니다. '
        '적자·성장 subgroup은 최소 30개 관측을 요구하며, 작은 표본이나 다중 비교 문제는 남습니다.', '']
    lines += ['상세 해석은 `EPS_OUTCOME_INTERPRETATION_KO.md`를 읽으십시오. '
        'Diversity 두 모델은 약한 단독 모델끼리의 상대 개선이지 전체 선두 조합의 개선 증거는 아닙니다. '
        '적자·성장 역할은 미래 실제 EPS로 구분한 사후 집단입니다. 이를 origin에서 아는 상태처럼 사용한 '
        '실시간 router는 없으며 그렇게 사용하면 누수입니다.', '']
    for note in shortlist['notes']: lines.append('- '+str(note))
    lines += ['', '## 고정 앙상블: research-test 설명용 성적', '',
        '선택/가중치 cutoff는 2022-01-01. 각 target·트랙에서 실제 5개 구성원으로 '
        'mean, median, 20% trimmed mean, nonnegative simplex L1 stack을 고정했습니다. '
        '최종 앙상블의 OOF 학습 표면은 검증 성능으로 표시하지 않고 NaN으로 남겼습니다.', '']
    for target in ['h1', 'ttm']:
        g = test[(test.target_key == target) & test.eps_model_id.str.startswith('ens_')].sort_values('MAE')
        lines += ['### '+target, '', table(g, columns), '']
    lines += ['평균보다 학습형 stack이 나쁘게 나와도 가중치를 다시 맞추지 않습니다. '
        '이 표의 순위는 새로운 운영 모델 선택이나 formal promotion이 아닙니다.', '',
        '## OOF 선정 단일 후보의 research-test 추적', '']
    chosen = {r['model_id'] for r in shortlist['models']}
    selected_test = test[test.eps_model_id.isin(chosen) & test.target_key.isin(['h1', 'ttm'])]
    lines += [table(selected_test.sort_values(['target_key', 'eps_model_id']), ['eps_model_id', 'target_key', *columns[1:]]), '',
        '## 확률 예측과 안정성', '',
        '분위수 모델은 pinball loss, P10/P50/P90 경험적 CDF, 80% 구간 포함률·폭을 별도 평가합니다. '
        '점 예측 모델에 분위수를 만들어 붙이지 않았습니다. Normal/Laplace native 4종에만 정확한 CRPS를 계산했고 '
        '독립 CDF 적분 대조 테스트 12건을 통과했습니다.', '',
        '`EPS_NATIVE_ANALYTIC_CRPS_V1.csv`, `EPS_PROBABILISTIC_LEADERBOARD_V1.csv`, '
        '`sampling_stability/`를 참고하십시오. 시드 민감도는 OOF 진단이며 추가 후보/앙상블 가중치로 사용하지 않습니다.', '',
        '중요한 반례: C의 OOF 선두 Ridge는 research-test에서 관측 persistence보다 약 13.65% 나빴고, '
        'C L1 stack도 persistence보다 약 2.03% 나빴습니다. 고정 절사평균은 약 7.11% 개선했습니다. '
        'C OOF의 독립 origin-year는 2개뿐이므로 순위·가중치 안정성을 과신하지 않습니다.', '',
        '## 검증 및 실패 처리', '',
        '- 원래 101개 출력의 성적은 SEC fiscal metadata 미래 의존 결함으로 전체 폐기했습니다. '
        'V1.3 native-context calendar로 교체하고 69×4 snapshot truncation 검사를 통과한 뒤 재실행했습니다.',
        '- Foster/Griffin–Watts는 smoke 수렴 실패 후 진행된 출력이라 진단 전용입니다. '
        '배치 RNG 결합이 확인된 원래 Lag-Llama/Moirai 2종도 진단 전용이며, 분리 호출 수정 버전은 별도 ID입니다.',
        '- Windows 수집기 잠금은 이미 학습된 440개 head를 재로딩해 복구했습니다. '
        '재학습으로 숨기지 않았고 유실된 학습 시간은 null로 남겼습니다.',
        '- 실제 저장 가중치 재현, 미래 정답/다른 origin 입력 변경, cache-off 추론을 검사했습니다. '
        '신경망 280개 연도별 모델의 별도 엄격 FP32 입력 검사 840건이 통과했습니다.',
        '- 기존 BiTCN/Informer/TimeMixer의 일부 singleton 수치 실패는 CuDNN TF32/FP32 별도 진단과 함께 보존했습니다. '
        '기존 예측을 바꾸거나 허용오차를 완화하지 않았습니다.',
        '- TabPFN의 40-head 재현 및 다른 입력 변경은 통과했으나 singleton 크기 검사 2건은 실패했습니다. '
        'FP64도 첫 실패를 해결하지 못했고 다음 head는 OOM으로 종료됐습니다. 추가 재시도 없이 미해결 수치 제한으로 남겼습니다.',
        '- EPS 데이터 빈티지, 생존편향, 사전학습 overlap, 일부 긴 Q4 coverage 한계는 남습니다. '
        '실제 인과 입력 검사가 strict historical certification을 뜻하지 않습니다.', '',
        '- MCD 공시의 주식 수 백만 단위를 full shares로 정규화하지 않은 실제 데이터 결함을 추가 확인했습니다. '
        'HVZ 연간 모델의 research-test MAE는 약 129,960으로 폭증했습니다. 순수 EPS target은 이 주식 수로 만들지 않지만 '
        'wide 회계 feature 16종과 HVZ, 이를 쓰는 고정 앙상블은 별도 경고 대상입니다. '
        '해당 열 제거 3종은 test 관찰 후 진단이며 기존 데이터·가중치·성적을 대체하지 않습니다.', '',
        '- 별도 NumPy 계산기로 2,047개 성적 행·18,104개 지표를 대조했습니다. 초기 94개 OOF 구간 폭 차이는 '
        'native FP32와 승격된 FP64 평균의 차이였으며 원래 dtype으로 전부 재현했습니다. '
        '원본 성적·실패 감사 기록과 허용오차는 변경하지 않았습니다.', '',
        '전체 근거: `FINAL_VERIFICATION_SUMMARY.json`, `EPS_MODEL_FAILURES_AND_BLOCKERS.md`, '
        '`research/eps_model_lab_v1/RESEARCH_DECISIONS_AND_LIMITATIONS.md`.', '',
        '## PE 연결 결과', '',
        'v04 및 C4는 read-only입니다. EPS calendar 수정 전후 실제 소비한 5개 회사 입력이 동일함을 확인해 '
        '기존 실제 v04 출력과 C4 거절 증거만 재사용했습니다. EPS 조합은 새로 고정한 C-lane 구성원으로 재생성했습니다.', '',
        f"- 실제 연구 시나리오: {combo['research_valid']}/{combo['rows']}; 엄격 certified 조합: {combo['strict_certified_valid']}.",
        '- C4는 정확한 frozen runtime 분리 후에도 real-input prefix/fallback 조건을 거절했습니다. '
        'NaN 행을 꾸며 넣거나 IRLS cap80, fallback, 0.50 shrink 또는 gate를 바꾸지 않았습니다.',
        '- `STATIC_PE_HOLD_ASSUMPTION`, `HORIZON_MISMATCH_LIMITATION`, source-basis-vintage 미인증을 명시했습니다. '
        '음수 EPS는 정상 예측이지만 일반 PE 가격 결합에는 부적격입니다. 주가 예측 정확도 주장은 없습니다.', '',
        '기존 PE 조합의 research-valid는 정의·기하 조건만 뜻합니다. 주식 수 feature 오류의 '
        '직접·간접 영향은 `EPS_PE_SHARE_UNIT_QUALITY_FLAGS.csv`를 함께 읽어야 합니다. '
        '이 경고를 단순히 연구용이라는 말로 무시하지 마십시오.', '',
        '## 다음 wave 제안', '',
        '1. 먼저 원문 accession/context별 주식 수 단위를 수리하거나 검증되지 않은 열을 제외하는 새 데이터 wave가 필요합니다. '
        'PE 연결의 핵심인 Lane C를 우선하되 Ridge·native TTM 모델·간단 평균은 품질 수리 후 다시 확인할 비교 기준입니다. '
        '새로운 unseen 기간과 historical-vintage 증거 없이 formal tournament를 만들지 마십시오.',
        '2. 위 OOF shortlist 5–10개를 대상으로 별도 확정된 실험을 설계합니다. '
        'A의 Chronos 미세조정은 연구 신호가 있지만 사전학습 시점 문제를 해결하거나 별도 retrospective 트랙에 둡니다.',
        '3. 필요한 데이터는 PIT analyst history, 원래 filing/share-basis vintage, 과거 시점 universe, '
        '긴 Q4와 결측 직접 분기의 원문 context 증거입니다. 현재 consensus나 미래 split을 과거 feature로 넣지 마십시오.',
        '4. PE의 future-horizon 추정량과 EPS Q+4 target을 맞추는 별도 계약이 필요합니다. '
        'C4 real-input 허용 범위 검토는 새 권한의 PE 작업이며 이번 freeze를 수정하지 않습니다.',
        '5. 배포 전 TabPFN arbitrary-batch 안정성, stochastic sampling 민감도, 라이선스와 clean-environment 재현을 다시 검증합니다.', '',
        '## 인계 및 재현 범위', '',
        '7개 EPS 전용 환경을 사용했고 기존 PE venv, 시스템 Python, CUDA driver를 변경하지 않았습니다. '
        '추가로 검증 전용 여덟 번째 환경에 잠긴 core 패키지 80개를 새로 설치해 인계판 평가와 '
        '앙상블 재현을 실제 통과했습니다. 같은 Windows 장비에서의 clean-env 검증이며 다른 장비 인증은 아닙니다. '
        'CPU 사전 벤치마크는 outer 20 / inner 1을 선택했습니다. GPU 모델은 순차 실행했습니다. '
        '실측 리소스 요약은 `RESOURCE_TELEMETRY_SUMMARY.json`에 있고 모델별 GPU active time으로 오해하면 안 됩니다.', '',
        '추가 Lag-Llama 병목 실험에서는 20개 독립 CPU 프로세스×각 1스레드로 2,101개 origin의 '
        '4,202회 native 호출을 532.05초에 재현했고 원래 점 예측·분위수와 모두 정확히 일치했습니다. '
        '기존 4스레드 직렬 재현은 약 43분이었으나, 동시 작업 조건이 달라 전용 장비의 통제된 속도 비교는 아닙니다. '
        '모델·시드·샘플 수·기존 성적은 바꾸지 않았습니다.', '',
        'Moirai 1.1 및 Moirai-MoE도 4/12/20 worker 실측 후 4×1을 선택했습니다. '
        '각각 전체 2,101개 origin·4,202개 native 호출을 약 49.75초/60.94초에 원래 분위수와 정확히 일치하게 재현했습니다. '
        'Lag-Llama의 20 workers를 모든 모델에 일률 적용하지 않았습니다.', '',
        'ZIP은 코드·실제 예측·작은 frozen 평가 데이터·계약·레지스트리·감사/실패 기록을 전달하는 연구 인계판입니다. '
        '대용량 학습 가중치와 제한된 외부 코드는 포함하지 않습니다. '
        '`LOCAL_TRAINED_ARTIFACT_INDEX.json`, `weight_receipts/`, 환경 lock으로 원본을 찾습니다. '
        '완전한 배포판이나 다른 컴퓨터에서 즉시 실행되는 전체 모델 바이너리 모음은 아닙니다.', '',
        '상세 진행 과정·문제·해결 근거는 별도 decision record에 압축했습니다. 비공개 내부 사고 과정은 포함하지 않습니다.', '']
    lines += ['', '## 후반 추가 검증과 차기 데이터 수리 경계', '',
        '`EPS_LATE_FRONTIER_REPORT_KO.md`의 추가 구조는 별도 post-freeze 진단입니다. '
        '`CHRONOS_TRAINING_SEED_REPORT_KO.md`는 동일 OOF의 학습 난수 민감도이며 독립 확인시험이 아닙니다.', '',
        '`EPS_ONLY_CHANNEL_REPORT_KO.md`는 NI/매출을 제거한 두 구조의 사후 데이터 품질 진단입니다. '
        '채널 제거는 파라미터와 supervision도 바꾸므로 기간 오류의 순수 인과효과로 해석하지 않습니다. '
        '`EPS_FEW_YEAR_UNCERTAINTY_KO.md`는 A 3년/C 2년의 적은 연도 수와 선택 불확실성을 설명합니다. '
        '부호 조합 열거의 꼬리 비율을 검증된 p-value나 운영 승격 기준으로 사용하지 않습니다.', '',
        '`EPS_CHANNEL_RETRAINING_NUMERICAL_KO.md`는 저장 모델 replay와 새 학습 재현을 구분합니다. '
        '두 채널 TimeXer의 기본 새 학습은 달랐지만 입력 재구성 차이는 0이었습니다. '
        '별도 deterministic 설정에서는 TimeXer/SOFTS 각각 2019년 독립 프로세스 학습의 체크포인트 bytes가 '
        '일치했습니다. 모든 모델·연도·장비의 재학습 인증으로 확대하거나 원래 성적을 대체하지 않습니다.', '',
        '43,412개 회계 값은 raw accession 값과 일치했지만 147개 cell은 fiscal YTD 시작일과 다른 기간입니다. '
        'AMZN 원문에서도 3개월과 12개월 값이 함께 확인됐습니다. exact-context 차기 선택기는 구현·테스트하고 '
        '53,520개 panel cell에서 읽기 전용 비교를 수행했으나 frozen V1.3에 적용하지 않았습니다. '
        '`NEXT_WAVE_CONTEXT_SELECTOR_REVIEW.json`은 기간 선택만 다루며 주식 수 scale·역사적 빈티지는 미해결입니다.', '',
        'xLSTM용 추가 격리 training 환경은 원래 GPU lock을 유지합니다. 사용하지 않는 sLSTM compiler 초기화만 '
        '별도 복사본에서 지연시켰습니다. 기본 7 training 환경 + frontier 1 + 검증 전용 1이며 기존 PE 환경은 변경하지 않았습니다.', '']
    report = RUN/'EPS_MODEL_LAB_V1_10H_REPORT.md'; report.write_text('\n'.join(lines), encoding='utf-8')
    (RUN/'EPS_MODEL_LAB_V1_8H_REPORT.md').write_text('# Compatibility filename\n\nRead [the 10-hour report](EPS_MODEL_LAB_V1_10H_REPORT.md).\n', encoding='utf-8')
    for source, target in [(LAB/'PE_READONLY_INTEGRATION_CONTRACT.md', RUN/'EPS_PE_INTEGRATION_CONTRACT_V1.md'),
                           (RUN/'EPS_PE_COMBINATION_SMOKE.csv', RUN/'EPS_PE_COMBINATION_SMOKE_V1.csv')]:
        target.write_bytes(source.read_bytes())
    save_json(RUN/'REPORT_REQUIRED_NAME_ALIASES.json', {'ten_hour_report': True,
              'report_sha256': sha(report), 'eight_hour_filename_is_redirect_only': True,
              'PE_contract_alias_sha256': sha(RUN/'EPS_PE_INTEGRATION_CONTRACT_V1.md'),
              'PE_smoke_alias_sha256': sha(RUN/'EPS_PE_COMBINATION_SMOKE_V1.csv')})
    print('FINAL_REPORTS_WRITTEN', len(scored), 'prediction identities', flush=True)


if __name__ == '__main__': run()
