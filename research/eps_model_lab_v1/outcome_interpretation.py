"""Evidence-only outcome interpretation; never changes shortlist, weights or scores."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import atomic_csv
from research.eps_model_lab_v1.ensemble_models import EXCLUDED, family
from research.eps_model_lab_v1.final_reports import table


def run():
    protected = [RUN/'ENSEMBLE_FREEZE_V1.json', RUN/'EPS_RESEARCH_SHORTLIST_V1.json',
                 RUN/'EPS_SELECTION_LEADERBOARD_V1.csv', RUN/'EPS_MODEL_ZOO_LEADERBOARD_V1.csv',
                 RUN/'NESTED_META_DIAGNOSTIC_FREEZE.json', RUN/'nested_meta_diagnostic/predictions.parquet']
    before = {str(p.relative_to(RUN)): sha(p) for p in protected}
    shortlist = json.loads((RUN/'EPS_RESEARCH_SHORTLIST_V1.json').read_text(encoding='utf-8'))
    paired = pd.read_csv(RUN/'EPS_PAIRED_BASELINE_COMPARISON.csv')
    selection = pd.read_csv(RUN/'EPS_SELECTION_LEADERBOARD_V1.csv')
    eligible = selection[selection.portfolio_eligible & selection.coverage.ge(.999) & selection.MAE.notna()
        & ~selection.eps_model_id.isin(EXCLUDED) & ~selection.eps_model_id.str.startswith('ens_')].copy()
    eligible['family'] = eligible.eps_model_id.map(family)
    oof_pairs = paired[paired.surface == 'SELECTION_OOF']
    merged = eligible.merge(oof_pairs[['eps_model_id', 'target_key', 'baseline', 'relative_MAE_gain']], on=['eps_model_id', 'target_key'], validate='one_to_one')
    lock = json.loads((RUN/'ENSEMBLE_FREEZE_V1.json').read_text(encoding='utf-8'))
    members = {m for p in lock['plans'] for m in p['member_ids']}; groups = []
    for (fam, target), g in merged.groupby(['family', 'target_key']):
        best = g.sort_values(['MAE', 'eps_model_id']).iloc[0]
        # Reporting-only stopping rule, not a new exclusion gate or changed portfolio.
        weak = len(g) >= 3 and g.relative_MAE_gain.le(-.10).all() and not set(g.eps_model_id).intersection(members)
        groups.append({'family': fam, 'target': target, 'qualified_full_coverage_ids': len(g),
            'best_raw_MAE_model': best.eps_model_id, 'best_raw_MAE': best.MAE,
            'positive_observed_baseline_gain_ids': int(g.relative_MAE_gain.gt(0).sum()),
            'status': 'SATURATED_IN_V1_RAW_MAE_SCREEN' if weak else 'NO_BROAD_SATURATION_CLAIM',
            'rule': 'At least 3 tested IDs, every ID at least 10% worse raw MAE than OOF-selected observed control, no frozen ensemble member',
            'does_not_override_other_targets_robustness_or_diversity': True})
    atomic_csv(pd.DataFrame(groups), RUN/'EPS_FAMILY_SATURATION_REVIEW.csv')

    # Resolve a metadata-denominator ambiguity without overwriting the nested scores.
    nested = pd.read_parquet(RUN/'nested_meta_diagnostic/predictions.parquet'); counts = []
    original = pd.read_csv(RUN/'EPS_NESTED_META_DIAGNOSTIC_SCORES.csv')
    for (name, target, year), q in nested.groupby(['eps_model_id', 'target_key', nested.asof_date.dt.year]):
        mature = np.isfinite(q.actual_eps); scored = mature & q.prediction_valid & np.isfinite(q.predicted_eps)
        before2022 = q.label_asof < pd.Timestamp('2022-01-01', tz='UTC')
        old = original[(original.model_id == name) & (original.target_key == target) & (original.forecast_year == year)].iloc[0]
        assert int(scored.sum()) == int(old.predicted_rows)
        assert int(before2022.sum()) == int(old.evaluation_labels_available_before_2022)
        counts.append({'model_id': name, 'target_key': target, 'forecast_year': year, 'forecast_origins': len(q),
            'all_future_filing_timestamps_before_2022': int(before2022.sum()),
            'finite_native_truths_before_2022': int((mature & before2022).sum()),
            'scored_native_truths_before_2022': int((scored & before2022).sum()),
            'scored_native_truths_all_availability_dates': int(scored.sum()),
            'original_ambiguous_field_counts_filings_even_when_direct_EPS_missing': True})
    atomic_csv(pd.DataFrame(counts), RUN/'NESTED_META_LABEL_COUNT_CLARIFICATION.csv')

    tracked = paired[paired.eps_model_id.isin([r['model_id'] for r in shortlist['models']] +
        ['ens_mixed_retrospective_convex_l1', 'ens_local_causal_trimmed_mean']) & paired.target_key.isin(['h1', 'ttm'])]
    atomic_csv(tracked, RUN/'EPS_FROZEN_CANDIDATE_TRACKING.csv')
    special = []
    for r in shortlist['models']:
        for ev in r['evidence']:
            if 'equal_pair_actual_MAE' not in ev: continue
            target = ev['target']; best = eligible[eligible.target_key == target].sort_values('MAE').iloc[0]
            special.append({'model_id': r['model_id'], 'target': target, 'pair_MAE': ev['equal_pair_actual_MAE'],
                'better_single_gain': ev['gain_over_better_single'], 'global_best_OOF_MAE': best.MAE,
                'pair_gap_to_global_best': ev['equal_pair_actual_MAE']-best.MAE,
                'scope': 'Relative complementarity of two weak singles does not prove value in the leading ensemble'})
    save_json(RUN/'EPS_DIVERSITY_SCOPE_REVIEW.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'records': special, 'frozen_shortlist_changed': False})
    crps = pd.read_csv(RUN/'EPS_NATIVE_ANALYTIC_CRPS_V1.csv')
    crps = crps[crps.target_key.isin(['h1', 'ttm'])]
    prob = pd.read_csv(RUN/'EPS_PROBABILISTIC_LEADERBOARD_V1.csv')
    prob = prob[(prob['mask'] == 'COMPLETE_COVERAGE') & (prob.split == 'RESEARCH_TEST') &
                prob.eps_model_id.isin(['ngboost_normal', 'ngboost_laplace', 'bayesian_ridge_distribution', 'gaussian_process_matern_distribution']) & prob.target_key.isin(['h1', 'ttm'])]
    seeds = pd.read_csv(RUN/'EPS_NATIVE_SAMPLING_STABILITY_SUMMARY.csv')
    lines = ['# 고정 모델 결과 해석과 다음 연구의 범위', '',
        '이 문서는 성적을 해석할 뿐 데이터·모델·shortlist·앙상블 가중치를 변경하지 않는다. '
        'OOF는 선택에 사용한 연구 표면, research-test는 이미 관찰된 설명용 기간이다. 정식 인증은 0이다.', '',
        '## 핵심 성과와 반례', '',
        '- Lane A: Chronos 미세조정의 OOF MAE 0.745324는 관측 persistence 0.878626 대비 약 15.17% 낮다. '
        'research-test에서도 0.771477 대 0.906402로 약 14.89% 낮았다. 단 사전학습 자료의 역사적 겹침은 미확인이다.',
        '- 로컬 TimeXer의 A 개선은 OOF 약 9.48%, research-test 약 5.04%다. 두 표면 모두 ticker-cluster 95% 탐색 구간이 0을 포함한다. '
        '따라서 평균 개선을 확정된 안정적 우월성으로 읽지 않는다.',
        '- Lane C의 OOF 1위 Ridge는 3.411144 대 persistence 3.550347로 약 3.92% 개선했지만, '
        'research-test에서는 3.356356 대 2.953302로 약 13.65% 나빠졌다. OOF 순위의 지속성이 보장되지 않는 직접적인 반례다.',
        '- 고정 C 절사평균은 research-test MAE 2.743288로 persistence보다 약 7.11% 낮았다. '
        '같은 멤버의 L1 stack은 3.013230으로 persistence보다 약 2.03% 높았다. test를 보고 stack 가중치를 재학습하지 않았다.',
        '- C에서 local/mixed 두 트랙의 멤버와 가중치는 동일하다. 따라서 C 성적 두 행을 독립적인 두 성공으로 세지 않는다. '
        '가중치 0인 멤버도 mean/median에는 실제 사용되며, 5명 선정과 양의 stack 가중치 수는 다른 개념이다.', '',
        '## 통계적 한계', '',
        'C 선택 표면은 515개 행이지만 독립 origin-year는 2개뿐이다. A는 649개 행·3개 연도다. '
        'ticker 또는 year 한 축씩 시행한 2,000회 bootstrap은 이중 군집·다중 비교 보정이 아니며, '
        '겹치는 미래 TTM 정답과 많은 상관 후보의 선택 편향을 제거하지 않는다. CI가 0을 배제해도 formal certification이 아니다.', '',
        table(tracked, ['eps_model_id', 'target_key', 'surface', 'paired_rows', 'relative_MAE_gain',
            'ticker_cluster_ci95_low', 'ticker_cluster_ci95_high', 'independent_year_clusters']), '',
        '## Diversity와 specialist의 제한', '',
        table(pd.DataFrame(special), ['model_id', 'target', 'pair_MAE', 'better_single_gain', 'global_best_OOF_MAE', 'pair_gap_to_global_best']), '',
        'LSTM/DLinear 조합의 약 10.8–11.0% 개선은 Lag-Llama와 각 단독 모델 사이의 상대적 개선이다. '
        '조합 MAE 약 0.897은 전체 선두 0.745보다 여전히 높다. 선두 포트폴리오의 증분 가치가 입증된 것은 아니므로 '
        '다음 wave에서 별도 OOF incremental test가 필요한 낮은 우선순위 가설이다.', '',
        'negative/high-growth 역할은 **미래 실제 EPS로 나눈 사후 하위집단** 결과다. '
        'origin에서 그 상태를 안다고 가정해 모델을 라우팅하면 누수다. 현재 자료에는 사전 예측 가능한 '
        'specialist router가 없으며, 표본 30개 이상이라는 조건만으로 전문 모델 인증이 되지 않는다.', '',
        '## Native 확률 예측', '',
        table(crps, ['eps_model_id', 'target_key', 'surface', 'rows', 'CRPS', 'price_scaled_CRPS']), '',
        'CRPS의 낮음과 구간 calibration은 별개다. 아래는 설명용 research-test 구간 결과이며 '
        '이 표를 보고 calibration 보정이나 후보 선택을 추가하지 않았다.', '',
        table(prob, ['eps_model_id', 'target_key', 'quantile_rows', 'interval80_coverage', 'interval80_width', 'pinball_loss']), '',
        '## Nested meta와 확률 샘플링', '',
        '2020/2021별로 멤버와 가중치를 다시 과거 자료에서만 선택한 별도 진단은 14개 계획·56개 성적 행이다. '
        '최소 meta-fit 정답 100개 미달 6개 계획은 건너뛰었다. 예를 들어 local C의 2021년 mean 2.885603과 '
        'stack 3.075511은 단순 조합을 유지할 이유를 보여 주지만 최종 2022 가중치를 재선택하는 데 사용하지 않았다.', '',
        '`NESTED_META_LABEL_COUNT_CLARIFICATION.csv`는 원본 nested 표의 availability count가 '
        'EPS 값이 빠진 미래 filing도 셌다는 분모 차이를 명시한다. 실제 점수의 finite-truth 행 수와 '
        '확정된 예측은 변경하지 않았다. Availability timestamp가 있다고 EPS truth가 존재하는 것은 아니다.', '',
        table(seeds, list(seeds.columns)), '',
        '시드 1729와 추가 고정 시드 4개를 비교했다. Moirai 1.1 A의 MAE 범위는 약 8.62%이며 원래 시드가 '
        '가장 유리했다. 이 결과는 100회 샘플 추론의 변동성 경고이지, 좋은 시드를 골라 성적을 개선하라는 뜻이 아니다.', '',
        '## Family 중단 판단', '',
        '`EPS_FAMILY_SATURATION_REVIEW.csv`는 target별 OOF raw-MAE 중단 근거를 기록한다. '
        '같은 계열 3개 이상이 모두 관측 기준선보다 10% 이상 나쁘고 고정 앙상블 멤버도 없는 경우에만 '
        '해당 target을 SATURATED_IN_V1_RAW_MAE_SCREEN으로 표시한다. 다른 target·robustness·조합 가치는 배제하지 않는다. '
        '이 사후 보고 규칙은 이번 freeze를 바꾸는 탈락 gate가 아니다.', '',
        '## 다음 wave 우선순위', '',
        'Lane C의 truth 정의와 역사적 share-basis를 먼저 보강한다. Ridge·관측 persistence·SOFTS 계열과 '
        '고정 C 단순 조합을 비교 기준으로 유지하고, A의 Chronos FT/TimeXer/HistGB/NGBoost는 별도 target별 연구 대상으로 둔다. '
        'SOFTS는 현재 10개 역할 shortlist와 별개로 이미 고정 C 앙상블에 포함된 비교 기준이며, test를 보고 새로 승격한 모델이 아니다. '
        '사용자가 새 wave를 승인하면 새로운 output identity와 미관측 평가 설계를 만든다.', '']
    (RUN/'EPS_OUTCOME_INTERPRETATION_KO.md').write_text('\n'.join(lines), encoding='utf-8')
    unchanged = all(sha(RUN/p) == digest for p, digest in before.items())
    save_json(RUN/'OUTCOME_INTERPRETATION_RECEIPT.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
        'all_frozen_artifacts_unchanged': unchanged, 'artifact_hashes': before, 'new_model_count': 0,
        'family_target_review_rows': len(groups), 'nested_metadata_rows': len(counts), 'formal_certified': False})
    if not unchanged: raise RuntimeError('Outcome report modified frozen evidence')
    print('OUTCOME_INTERPRETATION_COMPLETE', len(groups), 'family-targets', len(counts), 'nested metadata rows', flush=True)


if __name__ == '__main__': run()
