# 결과와 해석

## V2 확인 구간

2019–2021 confirmation, Lane A native h1 EPS, truth 648개에 대한 보존된 결과다. 표의 수치는 MAE이며 작을수록 좋다.

| 모델 | MAE |
| --- | ---: |
| CatBoost direct | 0.838665 |
| Chronos zero-shot | 0.790284 |
| Chronos joint rolling | 0.749959 |
| HistGB direct | 0.828726 |
| LSTM residual recency | 0.816004 |
| NGBoost Laplace direct | 0.829899 |

[보존한 집계값](../results/confirmation.json)과 [집계 출처](../results/provenance.json)를 함께 볼 수 있다. 당시 선택 기록에는 persistence 0.893223, fixed mixed convex stack 0.746329가 기록돼 있다. 이 문서에서는 sample ID·전처리·recipe의 전체 비교 계약을 새로 맞춰 계산한 delta를 제시하지 않는다.

Lane C는 다른 목표다. confirmation truth 512개에서 persistence 3.549880, HistGB residual 3.624759, NGBoost residual 3.604710이었다. 두 후보 모두 사전 MAE gain gate를 통과하지 못했다. Lane A 결과를 가져와 장기 TTM 성과로 대신하지 않는다.

## 분포와 재현

Chronos 80% interval은 보정 후 confirmation coverage 74.07%, NGBoost direct는 82.10%였다. point error와 calibration의 결론을 분리한다. 분포를 내지 않는 모델에는 interval coverage를 만들지 않는다.

당시 별도 프로세스의 저장 모델 재추론과 repeat 학습 기록이 있다. CatBoost는 metadata bytes와 learned state 차이를 구분했다. 구체적인 실행·보존 범위는 [테스트 범위](../testing-notes.md)에 둔다.

## 일반화 제한

confirmation과 monitor는 이미 본 기간이다. 69개 고정 기업의 생존 편향, 현재 수집본의 수정치, foundation 사전학습 시점과 회계 원문 vintage는 남아 있다. V1과 V2는 평가 조건이 달라 직접적인 개선치로 비교하지 않았다.

엄격한 EPS×P/E 가격 예측 인증은 없다. 음의 EPS 자체는 유효하지만 양의 P/E 연결 조건에는 맞지 않을 수 있다.
