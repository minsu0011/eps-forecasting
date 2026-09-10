# EPS 목표와 데이터

## 세 lane을 나눈 이유

Lane A는 바로 다음 native fiscal quarter의 signed USD GAAP diluted EPS다. 음수는 정상적인 손실 관측이며 제거 대상이 아니다. basic·adjusted EPS를 빈 diluted EPS 대신 넣지 않는다.

Lane B는 Q+1~Q+4 경로다. Q4가 없다고 annual EPS에서 9개월 EPS를 빼 “정확한 분기 EPS”로 만들지 않는다. native quarter label과 모델이 근사한 경로는 다른 개념이다. V1의 완전한 네 분기 연구-test 경로는 4개 회사 34개뿐이어서 단기 목표의 성적을 장기 경로 성적으로 대신할 수 없었다.

Lane C는 fiscal Q+4 시점의 native ledger TTM이다. 네 분기 예측의 합은 모델 쪽 근사일 수 있지만 share basis·기간이 다르면 native TTM 정답과 같지 않다.

## origin과 주식 수

공시 가용 시각을 반영한 XNYS session close가 origin이다. 장중 acceptance 즉시 매매를 예측한 실험은 아니다. 미래 기업행동은 미래 label을 origin share basis로 환산하는 데 쓰일 수 있지만 과거 feature에 들어가면 안 된다.

NI/CFO context는 fiscal 기간을 식별하는 날짜 근거로 쓸 수 있다. 그 금액 자체를 native EPS로 바꾸는 것은 별도 문제다.

## V2 모집단과 구간

69개 기업, 3,753개 origin, panel 4,458행과 ledger 62,412행으로 구성했다. 71개 캐시 전체가 아니라 원래 V1의 실제 69개 cohort를 유지했다. V와 XOM의 native 지원 부족을 성능을 본 뒤의 종목 제거로 해석하지 않는다.

개발 2015–2018, 확인 2019–2021, 관찰 2022 이후다. 확인·관찰은 이미 본 기간이고, 현재 수집본을 썼다는 한계가 있다. 여기서 회계 원문 검증을 요구하는 Track A는 목표 구분인 Lane A와 다르다. Track A의 데이터 부족과 native EPS 기반 Lane A 후보 생존을 혼동하지 않는다. [데이터 계약 코드](../../research/eps_model_lab_v2/clean_data.py)와 [PIT 문제](PIT-and-Accounting-Issues.md)를 함께 읽어야 한다.
