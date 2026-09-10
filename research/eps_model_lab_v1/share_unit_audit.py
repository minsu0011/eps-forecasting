"""Late source-unit defect disclosure and exact consumer scope; no silent data repair."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN, save_json, sha
from research.eps_model_lab_v1.common import dataset, atomic_csv
from research.eps_model_lab_v1.core_models import CORE_SPECS
from research.eps_model_lab_v1.final_reports import table


def run():
    protected = [RUN/'data/samples.parquet', RUN/'data/fiscal_panel.parquet', RUN/'ENSEMBLE_FREEZE_V1.json',
                 RUN/'EPS_RESEARCH_SHORTLIST_V1.json', *sorted((RUN/'predictions').glob('*.parquet'))]
    before = {str(p.relative_to(RUN)): sha(p) for p in protected}
    frame = dataset(); panel = pd.read_parquet(RUN/'data/fiscal_panel.parquet')
    # Suspicion screen only. Small counts can be legitimate elsewhere; never auto-rescale.
    suspect = frame.account_diluted_shares.gt(0) & frame.account_diluted_shares.lt(1e7)
    rows = frame.loc[suspect, ['sample_id', 'ticker', 'asof_date', 'origin_period_end', 'origin_accession',
        'account_diluted_shares', 'account_net_income', 'current_ttm']].copy()
    confirmed = {'0000063908-24-000072': {'end': '2023-12-31', 'raw': 732.3, 'statement_shares': 732300000.,
        'url': 'https://www.sec.gov/Archives/edgar/data/63908/000006390824000072/mcd-20231231.htm', 'table_page': 38},
        '0000063908-25-000012': {'end': '2024-12-31', 'raw': 721.9, 'statement_shares': 721900000.,
        'url': 'https://www.sec.gov/Archives/edgar/data/63908/000006390825000012/mcd-20241231.htm', 'table_page': 40}}
    evidence = []
    for accn, spec in confirmed.items():
        rawpath = RUN/'data/raw/sec/MCD/companyfacts.json'
        raw = json.loads(rawpath.read_bytes())['facts']['us-gaap']['WeightedAverageNumberOfDilutedSharesOutstanding']['units']['shares']
        matches = [r for r in raw if r.get('accn') == accn and r.get('end') == spec['end'] and r.get('start') == spec['end'][:4]+'-01-01']
        assert matches and all(r['val'] == spec['raw'] for r in matches)
        evidence.append({'ticker': 'MCD', 'accession': accn, **spec, 'raw_companyfacts_records': matches,
            'raw_companyfacts_sha256': sha(rawpath), 'statement_display_unit': 'millions, except per-share data',
            'confirmed_factor_discrepancy': 1e6, 'factor_applied_to_frozen_data': False,
            'evidence_method': 'Official SEC original HTML table read via browser; original HTML not locally downloaded',
            'inference': 'Frozen shares field stores the table display number rather than full share units'})
    rows['assessment'] = np.where(rows.origin_accession.isin(confirmed), 'CONFIRMED_ORIGINAL_STATEMENT_UNIT_DISCREPANCY',
        'SUSPICIOUS_SCALE_REQUIRES_ORIGINAL_CONTEXT_AUDIT_NOT_AUTOMATIC_ERROR')
    atomic_csv(rows, RUN/'SOURCE_SHARE_UNIT_SUSPECT_ORIGINS.csv')
    direct = set(CORE_SPECS) | {'ngboost_normal', 'ngboost_laplace', 'bayesian_ridge_distribution',
        'gaussian_process_matern_distribution', 'tabpfn_v2_panel', 'hvz_gaap_annual_eps_originshares'}
    lock = json.loads((RUN/'ENSEMBLE_FREEZE_V1.json').read_text(encoding='utf-8'))
    indirect = {}
    for path in (RUN/'predictions').glob('ens_*.parquet'):
        track = 'mixed_retrospective' if 'mixed_retrospective' in path.stem else 'local_causal'
        indirect[path.stem] = {p['target']: [n for n in p['member_ids'] if n in direct] for p in lock['plans'] if p['track'] == track}
    consumer_records = []
    for path in sorted((RUN/'predictions').glob('*.parquet')):
        name = path.stem
        kind = 'ANNUAL_EPS_DENOMINATOR' if name == 'hvz_gaap_annual_eps_originshares' else (
            'RAW_SHARE_COUNT_WIDE_FEATURE' if name in direct else ('INDIRECT_ENSEMBLE_MEMBER_EXPOSURE' if name in indirect else 'NO_DIRECT_RAW_SHARE_COUNT_CONSUMPTION_IDENTIFIED'))
        consumer_records.append({'model_id': name, 'exposure': kind, 'source_unit_warning': name in direct or name in indirect,
            'clean_share_feature_claim_allowed': False if name in direct or name in indirect else None,
            'indirect_members': indirect.get(name), 'scored_status_not_overwritten': True,
            'no_direct_exposure_is_not_whole_data_certification': True})
    hvz = pd.read_parquet(RUN/'predictions/hvz_gaap_annual_eps_originshares.parquet')
    hvz['absolute_error'] = np.abs(hvz.predicted_eps-hvz.actual_eps)
    hvz_top = hvz.nlargest(5, 'absolute_error')[['ticker', 'asof_date', 'predicted_eps', 'actual_eps', 'absolute_error']]
    atomic_csv(hvz_top, RUN/'HVZ_SOURCE_UNIT_FAILURE_TOP_ROWS.csv')
    combinations = pd.read_parquet(RUN/'EPS_PE_STATIC_COMBINATION_V1.parquet')
    quality_flags = combinations[['sample_id', 'eps_model_id', 'pe_model_id', 'research_scenario_valid', 'price_combo_valid']].copy()
    quality_flags['raw_share_unit_warning'] = quality_flags.eps_model_id.isin(direct)
    quality_flags['not_affected_by_this_field_is_not_whole_data_certification'] = True
    atomic_csv(quality_flags, RUN/'EPS_PE_SHARE_UNIT_QUALITY_FLAGS.csv')
    diagnostic = json.loads((RUN/'share_unit_diagnostics/COMPLETION.json').read_text(encoding='utf-8')) if (RUN/'share_unit_diagnostics/COMPLETION.json').exists() else None
    unchanged = all(sha(RUN/p) == digest for p, digest in before.items())
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'CONFIRMED_SOURCE_UNIT_DEFECT_UNRESOLVED_IN_FROZEN_V1_3',
        'confirmed_original_statement_cases': evidence, 'suspicion_screen_total_sample_origins': len(rows),
        'suspicion_screen_research_origins': int(rows.asof_date.dt.year.ge(2019).sum()),
        'suspicion_screen_tickers': sorted(rows.ticker.unique().tolist()),
        'panel_suspicion_count_including_short_history_origins': int((panel.diluted_shares.gt(0) & panel.diluted_shares.lt(1e7)).sum()),
        'directly_exposed_scored_base_ids': sorted(direct), 'indirectly_exposed_ensemble_ids': sorted(indirect),
        'PE_combination_flag_sidecar': 'EPS_PE_SHARE_UNIT_QUALITY_FLAGS.csv',
        'PE_combinations_with_known_share_feature_exposure': int(quality_flags.raw_share_unit_warning.sum()),
        'PE_geometry_valid_but_share_feature_exposed': int((quality_flags.raw_share_unit_warning & quality_flags.research_scenario_valid).sum()),
        'consumers': consumer_records, 'all_frozen_artifacts_unchanged': unchanged, 'artifact_hashes': before,
        'future_leakage_demonstrated_by_this_defect': False,
        'native_EPS_truth_derived_from_diluted_shares': False,
        'not_a_reason_to_claim_clean_data': 'Prefix/time-causality PASS does not test dimensional scale accuracy',
        'diagnostic_ablation_completion': diagnostic and diagnostic['status'],
        'resolution': 'No score-based rescaling, clipping or row deletion. Three fixed no-share-feature ablations stored separately; no final-pool changes. Full filing-context unit repair requires a new data wave.',
        'formal_certified': False}
    save_json(RUN/'SOURCE_SHARE_UNIT_QUALITY_AUDIT.json', result)
    lines = ['# 중요: 주식 수 단위 오류 — V1.3 미해결', '',
        '최종 성적 검토에서 발견한 실제 원본 데이터 단위 문제다. 단순 수치 반올림이나 미래누수 문제와 다르다. '
        '고정 V1.3 데이터, 기존 117개 예측, shortlist, 앙상블 가중치를 수정하지 않았다. '
        '**17개 기본 모델과 8개 앙상블은 이 열의 직접·간접 영향이 있으므로 깨끗한 회계 feature 성능으로 해석하거나 배포하지 않는다.**', '',
        '추가 필수 경고: `SOURCE_ACCOUNTING_DURATION_WARNING_KO.md`도 읽는다. YTD 회계 입력에 '
        '다른 기간이 혼합된 별도 계약 오류가 기본 26종과 앙상블 8종에 영향을 준다. '
        '주식 수 비노출이나 no-share ablation이 전체 회계 데이터 품질을 보장하지 않는다.', '',
        '## 원문으로 확인한 사례', '',
        'MCD 2023년 연차보고서 손익계산서는 희석 가중평균 주식 수 732.3을 백만 단위로 표시한다. '
        '동일 accession의 로컬 companyfacts `shares` 값과 frozen panel에는 732.3이 저장되어 있다. '
        '[MCD 2023 원문, 재무제표 p.38](https://www.sec.gov/Archives/edgar/data/63908/000006390824000072/mcd-20231231.htm).', '',
        '2024년에도 721.9 백만 주와 로컬 값 721.9의 같은 차이가 있다. '
        '[MCD 2024 원문, 재무제표 p.40](https://www.sec.gov/Archives/edgar/data/63908/000006390825000012/mcd-20241231.htm).', '',
        'COP 2021년 1,328,151 등 다른 작은 값도 의심 대상이다. COP 원문 URL은 확인했지만 브라우저의 '
        '4 MiB 제한으로 본문 검증은 완료하지 못했다. 따라서 MCD에서 확인한 배율을 COP나 다른 ticker에 자동 적용하지 않았다.', '',
        f"전체 sample 중 규모 의심 {len(rows)}개, 연구 origin {int(rows.asof_date.dt.year.ge(2019).sum())}개, ticker {', '.join(sorted(rows.ticker.unique()))}. "
        '이 임계값은 조사 목록만 만들며 모든 작은 주식 수가 오류라는 규칙은 아니다.', '',
        '## 실제 영향', '',
        'HVZ는 예측한 달러 순이익을 이 주식 수로 나누므로 MCD의 EPS 예측이 약 1,200만 달러로 폭증했다. '
        '직접 EPS 모델들의 target은 native EPS와 corporate-action basis에서 오며 이 주식 수로 생성하지 않는다. '
        '그러나 wide tabular 11종, native distribution 4종, TabPFN은 `account_diluted_shares`를 feature로 소비한다. '
        '학습 입력과 시험 입력 모두의 영향이 가능하다. 고정 앙상블 8종에도 해당 구성원을 통한 간접 영향이 있다.', '',
        'PE 조합의 기존 research_scenario_valid는 기하·정의 조건 통과만 뜻한다. '
        '`EPS_PE_SHARE_UNIT_QUALITY_FLAGS.csv`를 키로 함께 읽어 주식 수 feature 경고를 확인해야 한다. '
        '기존 940개 계산 결과와 정식 인증 0개는 변경하지 않았다.', '',
        table(hvz_top, list(hvz_top.columns)), '',
        'EPS-only/지정된 NI·CFO·assets context 모델에 이 열을 직접 사용한 경로는 찾지 못했다. '
        '이것은 다른 데이터 품질이나 사전학습/PIT 전체를 인증한다는 의미가 아니다.', '',
        '## 수행한 진단과 다음 단계', '',
        '같은 설정의 Ridge, HistGradientBoosting, CatBoost에서 이 열만 제외하는 3개 진단을 사전에 고정했다. '
        '120개 annual-target head를 학습하고 다른 새 프로세스에서 재로딩했다. '
        '`share_unit_diagnostics/`에 따로 보관하며 이미 test를 본 뒤의 source-quality 진단임을 표시한다. '
        '이 세 결과로 shortlist/weights/기존 모델 성적을 교체하지 않는다.', '',
        '다음 data wave에서는 (1) 원문 context와 표시 단위를 accession별로 검증하고, '
        '(2) 실제 당시 공개된 증거만으로 scale을 정규화하거나 검증되지 않은 열을 제거하고, '
        '(3) 영향받은 모델을 새로운 identity에서 재학습하며, (4) 새로운 미관측 평가를 별도로 설계해야 한다. '
        '순이익/EPS 비율을 이용한 임의 배율 추정은 numerator·기간 차이 때문에 자동 정답이 아니다.', '']
    (RUN/'SOURCE_SHARE_UNIT_WARNING_KO.md').write_text('\n'.join(lines), encoding='utf-8')
    if not unchanged: raise RuntimeError('Source-unit audit changed frozen artifacts')
    print('SOURCE_UNIT_AUDIT_CONFIRMED_UNRESOLVED', len(rows), 'suspect origins', len(direct), 'direct model consumers', flush=True)


if __name__ == '__main__': run()
