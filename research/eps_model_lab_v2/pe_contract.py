"""EPS-only PE handoff validation. Never imports or changes a PE runtime."""
from pathlib import Path
import sys
import math
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from research.eps_model_lab_v2.common import RUN,save_json,utcnow

REQUIRED=['asof_date','ticker','forecast_horizon','eps_model_id','target_definition','forecast_eps',
    'forecast_share_basis','currency','GAAP_flag','diluted_flag','TTM_flag','prediction_valid','data_quality_tier','model_track']


def compatibility(eps,pe):
    errors=[]
    if any(k not in eps for k in REQUIRED):return {'PE_READY':False,'errors':['MISSING_EPS_CONTRACT_FIELDS']}
    if not math.isfinite(float(eps['forecast_eps'])) or not eps['prediction_valid']:errors.append('INVALID_EPS')
    for key in ['ticker','asof_date','forecast_horizon','currency','forecast_share_basis','GAAP_flag','diluted_flag','TTM_flag']:
        if eps.get(key)!=pe.get(key):errors.append('MISMATCH_'+key.upper())
    if not eps['TTM_flag']:errors.append('QUARTER_EPS_NOT_ANNUAL_PE_DENOMINATOR')
    if not eps['GAAP_flag'] or not eps['diluted_flag']:errors.append('WRONG_EARNINGS_DEFINITION')
    if float(eps['forecast_eps'])<=0:errors.append('NONPOSITIVE_EPS_VALID_FORECAST_BUT_NOT_STANDARD_PE_PRICE')
    expected_pe=pe.get('expected_pe')
    if expected_pe is None or not math.isfinite(float(expected_pe)) or float(expected_pe)<=0:errors.append('INVALID_EXPECTED_PE')
    structural=not errors
    strict=structural and bool(eps.get('basis_vintage_certified')) and bool(pe.get('basis_vintage_certified')) and bool(eps.get('formal_certified')) and bool(pe.get('formal_certified'))
    return {'PE_READY':structural,'strict_certified_combination':strict,'errors':errors,
        'scope':'Matched interface research compatibility only; no price-performance claim',
        'implied_price_research_only':float(eps['forecast_eps'])*float(expected_pe) if structural else None}


def write():
    schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'EPS Model Lab V2 PE-ready record',
        'type':'object','required':REQUIRED,'additionalProperties':True,
        'properties':{k:{'type':('boolean' if k.endswith('_flag') or k=='prediction_valid' else 'number' if k in ['forecast_eps','forecast_horizon'] else 'string')} for k in REQUIRED},
        'description':'Schema readiness does not certify source/basis vintage, PE horizon compatibility or production use'}
    save_json(RUN/'EPS_PE_READY_OUTPUT_SCHEMA.json',schema,immutable=True)
    (RUN/'EPS_PE_COMPATIBILITY_CONTRACT_V2.md').write_text('''# EPS V2 → PE: 읽기 전용 연결 계약

PE source/runtime/IRLS80/fallback allowlist/0.50 shrink는 변경하지 않는다.
현재 v04는 Production Champion, C4-R2는 Frozen Research Survivor라는 기존 상태를 유지한다.

EPS 출력의 ticker, 정확한 origin close, horizon, currency, GAAP diluted,
share basis, native TTM/quarter 정의를 PE 입력과 모두 대조한다. 하나라도 다르면
fail closed: 가격 값을 만들지 않는다. C4 real-input prefix 거절을 우회하지 않는다.

signed EPS ≤ 0은 정상적인 EPS 예측이다. 다만 일반적인 양수 PE 가격 곱셈에는
부적격이다. 음수 EPS 예측을 삭제하거나 정답을 0으로 자르지 않는다.

Lane C는 native ledger TTM이며 annual/direct/bridge approximation flags를 보존한다.
분기 합계와 native TTM을 같은 truth라고 표시하지 않는다.

PE_READY는 인터페이스 정의 일치에 한정한다. 엄격 가격 결합에는 별도의 historical
share/data vintage와 모델 정식 인증이 모두 필요하며 이번 V2에서 엄격 인증은 0이다.
현재 PE를 미래 Q+4 EPS에 고정 적용하는 선택적 별도 연구 시나리오에는
STATIC_PE_HOLD_ASSUMPTION 및 HORIZON_MISMATCH_LIMITATION을 반드시 표시해야 한다.
이 가정은 기본 compatibility gate를 우회하는 운영 권한이 아니며 주가 예측 성능도 아니다.
이번 source-quality 제한 상태에서 actual PE 재실행은 필수 작업으로 간주하지 않는다.
''',encoding='utf-8')
    save_json(RUN/'audit/PE_INTERFACE_SCOPE_V2.json',{'created_utc':utcnow(),'PE_source_modified':False,
        'schema_written':True,'actual_PE_runtime_invoked':False,'C4_gates_relaxed':False,'strict_certified_combinations':0},immutable=True)


if __name__=='__main__':write()
