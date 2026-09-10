# 모델 선택이 달라진 이유

V1에서는 넓은 model zoo를 같은 EPS 목표에 올려 가정 차이를 보았다. V1.3 수리 후에도 share-unit·duration 노출이 남았기 때문에 기존 점수 순서를 깨끗한 데이터의 순위로 이어받을 수 없었다. V2는 단순히 새 모델을 추가한 버전이 아니라 데이터와 선택 절차를 다시 고정한 실험이다.

## direct, residual, rolling, recency

direct는 EPS 수준을 바로 예측하고 residual은 기준값과의 차이를 학습한다. asinh 변형은 부호를 유지한 크기 변환을 비교한다. rolling과 recency는 과거를 얼마나 오래, 얼마나 강하게 반영할지에 대한 가설이다. 확인 성적을 보고 이 조합을 다시 고르지 않는다.

V2 Lane A 대표는 CatBoost direct, HistGB direct, NGBoost Laplace direct, LSTM residual recency, Chronos joint rolling, Chronos zero-shot이다. 같은 모델이 모든 lane을 맡는다고 가정하지 않았고, Lane C 대표는 HistGB와 NGBoost의 residual 변형이었다.

## local과 mixed ensemble

개발 구간에서 coarse family·checkpoint family 중복을 제한하고 동일한 예측 벡터를 제거한다. local 계열과 foundation을 섞는 retrospective 계열을 나눠 전이 성능과 과거 가용성을 혼동하지 않는다.

평균·중앙값·절사 평균·convex 결합은 고정된 후보 위에서 비교한다. 어느 두 모델의 oracle 오차가 좋아 보인다는 사실은 실행 가능한 ensemble의 성과가 아니다.

## 남긴 것과 남기지 않은 것

Lane A 연구 대표는 유지했다. Lane C는 confirmation gain gate에서 실패했으므로 더 좋은 confirmation 숫자를 가진 XLinear로 바꾸지 않았다. Track A는 검증 회계 값 부족으로 막혀 있어 그 위의 모델 우열을 논할 수 없다.

[선택 구현](../../research/eps_model_lab_v2/ensemble.py) · [결과](Validation-and-Results.md)
