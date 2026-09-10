"""Disclose proven mixed-duration feature semantics, without editing frozen research."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha


def run():
    audit=json.loads((RUN/'ACCOUNTING_CONTEXT_AND_UNIT_REVIEW.json').read_text(encoding='utf-8'))
    share=json.loads((RUN/'SOURCE_SHARE_UNIT_QUALITY_AUDIT.json').read_text(encoding='utf-8'))
    multivariate={p.stem for p in (RUN/'predictions').glob('nf_multivar_*.parquet')}
    direct=set(share['directly_exposed_scored_base_ids'])|multivariate
    assert len(multivariate)==9 and len(direct)==26
    rawpath=RUN/'data/raw/sec/AMZN/companyfacts.json'
    facts=json.loads(rawpath.read_bytes())['facts']['us-gaap'];accn='0001018724-25-000036';end='2025-03-31'
    evidence={tag:[v for v in facts[tag]['units']['USD'] if v.get('accn')==accn and v.get('end')==end]
              for tag in ['NetIncomeLoss','NetCashProvidedByUsedInOperatingActivities']}
    assert {r['start']:r['val'] for r in evidence['NetIncomeLoss']}=={'2024-04-01':65944000000,'2025-01-01':17127000000}
    cases=pd.read_csv(RUN/'ACCOUNTING_DURATION_START_REVIEW.csv')
    research=cases[cases['asof'].str[:4].ge('2019')]
    consumers=[]
    for path in sorted((RUN/'predictions').glob('*.parquet')):
        consumers.append({'model_id':path.stem,'duration_semantics_warning':path.stem in direct or path.stem.startswith('ens_'),
            'exposure':'MULTIVARIATE_AUXILIARY_CHANNELS' if path.stem in multivariate else ('WIDE_ACCOUNTING_FEATURES_OR_ANNUAL_NI' if path.stem in direct else ('INDIRECT_ENSEMBLE' if path.stem.startswith('ens_') else 'NO_DIRECT_DURATION_FIELD_CONSUMPTION_IDENTIFIED')),
            'not_affected_by_this_field_is_not_whole_data_certification':True})
    registry_path=RUN/'EPS_MODEL_REGISTRY_V1.json';registry=json.loads(registry_path.read_text(encoding='utf-8'))
    for model in registry['models']:
        if model['model_id'] in direct:
            model.update(accounting_duration_semantics_warning=True,
                accounting_duration_warning='SOURCE_ACCOUNTING_DURATION_WARNING_KO.md',
                scored_status_does_not_certify_ytd_feature_contract=True)
    save_json(registry_path,registry)
    combos=pd.read_parquet(RUN/'EPS_PE_STATIC_COMBINATION_V1.parquet')
    flags=combos[['sample_id','eps_model_id','pe_model_id','research_scenario_valid','price_combo_valid']].copy()
    flags['accounting_duration_warning']=flags.eps_model_id.isin(direct)
    flags['clean_ytd_feature_certified']=False
    flags.to_csv(RUN/'EPS_PE_ACCOUNTING_DURATION_FLAGS.csv',index=False)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'CONFIRMED_MIXED_DURATION_FEATURE_CONTRACT_DEFECT_UNRESOLVED',
        'source_cell_count':audit['nonmissing_accounting_cells'],'duration_mismatch_cells':len(cases),
        'research_origin_mismatch_cells':len(research),'research_origin_mismatch_accessions':int(research.accn.nunique()),
        'known_mismatch_tickers':sorted(cases.ticker.unique()),'direct_base_ids':sorted(direct),'multivariate_ids':sorted(multivariate),
        'indirect_main_ensemble_ids':sorted(p.stem for p in (RUN/'predictions').glob('ens_*.parquet')),
        'late_affected_models':['MLPMultivariate','SOFTSSharp'],'no_share_feature_ablations_still_duration_exposed':3,
        'AMZN_original_accession':accn,'AMZN_original_document':'https://www.sec.gov/Archives/edgar/data/1018724/000101872425000036/amzn-20250331.htm',
        'AMZN_raw_source_sha256':sha(rawpath),'AMZN_native_context_records':evidence,
        'mechanism':'native_accounting chooses earliest start within first available tag, mixing trailing 12-month and standalone quarters into claimed YTD fields',
        'future_leakage_demonstrated_by_this_defect':False,'EPS_labels_recomputed_from_NI':False,
        'frozen_dataset_or_predictions_changed':False,'portfolio_reselected':False,'formal_certified':False,
        'consumers':consumers,'PE_combo_rows_duration_exposed':int(flags.accounting_duration_warning.sum()),
        'repair_boundary':'Implement exact native_fiscal_start context selector in a NEW data wave, preserve missingness if no matching context. Do not rewrite this run or rank repaired scores on previously inspected research-test.'}
    save_json(RUN/'SOURCE_ACCOUNTING_DURATION_QUALITY_AUDIT.json',result)
    lines=['# 중요: YTD 회계 입력에 다른 기간 값이 혼합됨', '',
        '43,412개 비결측 회계 입력은 캐시된 공시 값과 정확히 일치했지만, 그 사실만으로 데이터 의미가 맞는 것은 아니다. '
        f'147개 cell에서 선택된 기간 시작일이 fiscal YTD 시작일과 다르다. 연구 origin에 해당하는 것은 {len(research)}개 cell, '
        f'{research.accn.nunique()}개 accession이다. 시작일 차이 자체가 언제나 원문 오류인 것은 아니지만, 이를 모두 YTD라고 '
        '표시한 현재 feature 계약은 잘못됐다.', '',
        'AMZN 2025-03-31 공시의 NetIncomeLoss에는 2024-04-01부터 12개월 값 65,944백만 달러와 '
        '2025-01-01부터 3개월 값 17,127백만 달러가 함께 있다. current selector의 earliest-start 규칙은 전자를 골랐다. '
        'CFO 역시 113,903백만 달러와 17,015백만 달러 중 12개월 값을 골랐다. '
        '[공식 AMZN 원문](https://www.sec.gov/Archives/edgar/data/1018724/000101872425000036/amzn-20250331.htm).', '',
        'GE의 일부 연차 revenue, JNJ/F의 일부 net income, 신규 배당 등에도 기간 차이가 있다. '
        '모든 사례를 같은 오류로 자동 보정하지 않으며 `ACCOUNTING_DURATION_START_REVIEW.csv`의 원문 context별로 검토해야 한다.', '',
        '노출 범위: 기존 wide 회계/연간 NI 소비자 17종 + 다변량 보조 채널 모델 9종 = 기본 26종, '
        '간접 앙상블 8종. 추가 MLPMultivariate·SOFTSSharp와 no-share-feature ablation 3종도 해당한다. '
        '따라서 raw shares 열을 제거했거나 주식 수를 입력하지 않는다는 이유만으로 깨끗한 회계 feature 모델로 볼 수 없다. '
        'TimeXer/SOFTS 등 local multivariate 선두의 성적에도 이 해석 제한이 적용된다.', '',
        'EPS-only 모델과 Chronos EPS/TTM 모델은 이 NI/revenue 기간 열을 직접 쓰지 않는다. 다만 이 특정 입력에 '
        '비노출이라는 뜻일 뿐, 역사적 PIT·데이터 빈티지·사전학습·전체 데이터 품질 인증은 아니다.', '',
        '기존 117개 예측, dataset, ensemble/shortlist는 수정하지 않았다. `EPS_PE_ACCOUNTING_DURATION_FLAGS.csv`도 '
        '기존 조합값을 바꾸지 않는 sidecar다. 현재 결과는 연구 진단으로만 보존하고 formal 인증 수는 0으로 유지한다.', '',
        '차기 wave는 tag 우선순위보다 정확한 accession+end+native_fiscal_start+공개시간을 먼저 만족시키고, '
        '해당 context가 없으면 결측으로 남기는 선택기를 써야 한다. 주식 수 원문 scale 문제도 별도로 해결해야 한다. '
        '기존 research-test를 확인한 뒤 같은 identity를 조용히 교체해서는 안 된다.']
    (RUN/'SOURCE_ACCOUNTING_DURATION_WARNING_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('DURATION_DEFECT_DISCLOSED',len(direct),'main bases,8 ensembles,2 late models; frozen values unchanged',flush=True)


if __name__=='__main__':run()
