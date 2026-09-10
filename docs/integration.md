# EPS × P/E

EPS는 earnings, P/E는 valuation multiple을 담당한다. 분기 EPS를 annual P/E와 그대로 곱하지 않는다. target horizon, fiscal basis, share basis, 공개 시각, prefix와 fallback 계약을 모두 맞춰야 한다. V2의 strict price combination은 인증되지 않았다. `research/eps_model_lab_v2/pe_contract.py`가 불일치를 차단한다.

V1의 C4 bridge 및 동결 run 재생은 별도 PE 연구 artifact와 당시 입력을 요구한다. 이 저장소는 해당 원천이나 weights를 재배포하지 않는다.
