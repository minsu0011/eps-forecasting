# Model zoo를 읽는 법

V1은 다양한 가정의 비교가 목적이었다. 외부 패키지 이름 수나 checkpoint 수를 새로운 모델 구조 수로 세지 않는다. V2는 비교 대상과 선택 규칙을 더 좁혀 따로 고정했다.

| 계열 | 무엇을 추가하는가 | 주의할 점 |
| --- | --- | --- |
| persistence·seasonal persistence | 가장 단순한 EPS 관측 패턴 | random-walk 별칭을 별개 모델로 세지 않음 |
| Ridge | 선형 관계와 규제 | 복잡한 모델의 필요성을 확인하는 기준 |
| HistGB·CatBoost | 비선형 표 관계 | direct/asinh/residual과 rolling·recency는 변형 |
| 로컬 sequence | 시간 순서의 표현 학습 | 학습 recipe와 batch 독립성을 함께 확인 |
| Chronos zero-shot | 사전학습 지식의 직접 전이 | 2026 가중치를 과거 이용 가능 모델로 보지 않음 |
| Chronos joint fine-tuning | EPS 목표에 대한 적응 | zero-shot과 훈련 단위를 구분 |
| NGBoost Normal/Laplace 등 | 예측 분포 | point-only 모델에 가짜 interval 점수를 주지 않음 |
| ensemble | 오차의 상보성 | checkpoint·동일 벡터 중복 제거와 family 제한 |

V1에는 StatsForecast, NeuralForecast, AutoGluon, Darts, TimesFM, Moirai 등도 비교 경로가 있었다. 수렴 실패나 입력 독립성 실패가 있으면 모델을 실행했다는 사실만으로 최종 portfolio에 넣지 않았다.

V2의 53개 등록 항목에는 막힌 항목도 포함하며 실제 변형은 46개다. 다섯 로컬 시계열 구조의 네 변형, 네 개발 cutoff는 80개 학습 단위이지 80개의 독립 구조가 아니다.

[표 모델](../../research/eps_model_lab_v2/tabular.py) · [sequence](../../research/eps_model_lab_v2/sequence.py) · [Chronos 적응](../../research/eps_model_lab_v2/chronos_specialization.py) · [외부 기여](../../ATTRIBUTION.md)
