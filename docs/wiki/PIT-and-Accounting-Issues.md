# PIT와 회계 의미는 다른 검사다

## fiscal metadata

SEC의 `fy/fp`는 항상 불변인 fiscal key가 아니었다. 후속 filing의 중복 제거가 과거 기록을 없애면 모델이 과거에 볼 수 있었던 입력 자체가 바뀐다. 같은 accession의 native context로 기간을 식별하고 첫 가용 identity를 유지하도록 바꿨다.

UPS의 Jan 1/Jan 2처럼 모순된 context는 격리했다. Costco·Pepsi의 긴 Q4처럼 실제 회계 관행인 사례는 날짜 오류와 구별했다. V2는 이전에 가용한 quarter-end와 연결되는지를 확인한다.

## 주식 수 표시 단위

MCD의 732.3 같은 값은 실제 주식 수가 아니라 표에서 백만 단위로 표시한 값이었다. EPS label은 native EPS라서 이 주식 수로 NI를 나눠 만든 정답은 아니지만, raw share feature를 쓰는 여러 모델의 입력은 영향을 받았다.

숫자가 작다는 이유로 모든 기업에 백만 배를 적용하면 새 오류를 만든다. V2는 원문 단위가 확인된 context만 수리하고 unknown을 격리했다. API context 100개 대응 검사는 원문 XBRL 100개 검증 완료와 다르다.

## 기간 의미

값이 원천 cache와 같아도 YTD라는 이름의 feature에 TTM이 들어가면 의미가 틀렸다. fiscal-start와 context-start를 정확히 맞추는 선택기가 필요하다. V1의 동결된 예측은 조용히 덮어쓰지 않고 문제와 노출 범위를 남겼다.

## 검사 범위

V2의 69×4 prefix 조건 중 실제 비어 있지 않은 rebuild는 265개, 과거 표본이 없어 비어 있는 조건은 11개였다. 둘을 모두 의미 있는 데이터 검증으로 세지 않는다. prefix 통과는 현재 수집본의 미래 제거에 대한 성질이지 과거 원문 vintage의 완전한 복원은 아니다.

[회계 코드](../../research/eps_model_lab_v2/clean_data.py) · [회귀 테스트](../../research/eps_model_lab_v2/test_data_contracts.py)
