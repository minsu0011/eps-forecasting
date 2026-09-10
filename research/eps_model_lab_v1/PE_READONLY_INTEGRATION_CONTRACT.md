# EPS와 PE의 연결 조건

이 문서는 V1 연구의 실제 PE 연결 조건이다. v04 코드·설정과 동결된 C4 파일 59개의 해시를 실행 전후 대조하며 PE 환경은 읽기 전용으로 둔다. EPS adapter·입력·출력은 별도 경로를 쓴다.

CPU 처리량 측정에 따라 v04의 외부 worker 수만 32에서 20으로 조정했다. 통계 매개변수·fold·seed는 그대로다. C4는 IRLS 80회 상한, 허용된 block fallback과 log shrink 0.5를 사용하며 다른 estimator로 바꾸지 않는다.

upstream의 regime·EPS·valuation 계산과 최초 native GAAP diluted ledger를 사용한다. 없는 diluted TTM을 basic·adjusted EPS로 대체하지 않는다. XOM은 유한한 native TTM이 없어 데이터 단계에서 제외됐고 에너지 부문 연결 검사에는 데이터가 있는 CVX를 사용했다. 나머지 사전 지정 종목은 AAPL·MSFT·JPM·KO다.

C4 입력은 최근 1,800개 연속 시장 세션이다. 이익 부호나 PE 유효성으로 행을 먼저 제거하지 않는다. 계약 거부도 결과이며 동결 gate를 완화하지 않는다. 일반 fold 구성기로 실제 시장 입력을 연결하며 별도 자격 평가·승격 절차를 실행하는 경로는 아니다.

정적 조합은 현재 Expected PE × Q+4 native TTM EPS다. `STATIC_PE_HOLD_ASSUMPTION`과 `HORIZON_MISMATCH_LIMITATION`을 표시한다. 분기 EPS에 연간 PE를 바로 곱하지 않는다. 통화·정의·정확한 origin 종가일·양의 PE·양의 TTM·origin 주식 수 기준을 확인하고 맞지 않으면 계산을 거부한다. 음수·0 EPS는 예측 목표로 유효하지만 일반 PE 가격 조합에는 적격하지 않다.

현재 SEC snapshot과 Yahoo 기업행동 이력은 과거 공개 빈티지의 완전한 복원이 아니다. 연구 시나리오와 엄격한 조합 유효성을 별도로 표시한다. 독립적으로 정렬한 미래 가격 정답 없이 가격 예측 성과를 주장하지 않는다. EPS 구성원은 2022년 선택 cutoff 전에 가용했던 label만으로 고른다.
