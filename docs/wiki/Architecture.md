# 예측 전에 고정하는 것

EPS 파이프라인의 중심은 모델이 아니라 origin별 데이터 계약이다. 같은 기업이어도 서로 다른 공시 시점의 예측은 독립된 입력 묶음으로 취급한다.

1. `fiscal_builder.py`가 공시 context에서 기간을 식별한다.
2. `clean_data.py`와 `statement_evidence.py`가 단위·기간·원문 근거를 점검한다.
3. `freeze_inputs.py`가 데이터와 recipe의 경계를 고정한다.
4. 표·sequence·Chronos 모듈이 같은 lane의 목표를 예측한다.
5. `evaluation.py`가 공통 truth mask로 점·분포 오차를 계산한다.
6. `ensemble.py`가 개발 구간에서만 대표와 결합 가중치를 정한다.
7. `replay.py`, `retraining.py`는 저장 모델 재추론과 새 학습 반복을 구분한다.

[해당 모듈](../../research/eps_model_lab_v2)을 읽을 때 데이터 생성과 freeze 이후의 선택 코드를 나눠 보면 된다. training row는 origin과 해당 horizon의 label이 모두 cutoff 전에 공개되어야 한다. 보간해서 채운 미래 target은 학습 정답으로 쓰지 않는다.

company-origin은 batch의 독립 표본이다. 여러 기업의 미래 시점을 한 multivariate forecast의 채널처럼 결합하지 않는다. 이 설계 때문에 다른 origin의 값을 바꿨을 때 첫 origin의 예측이 유지되는지 별도 검사했다.
