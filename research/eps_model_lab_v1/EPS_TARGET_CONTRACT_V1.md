# EPS 목표의 초기 정의: V1

모델 점수 확인 전에 정한 초기 목표다. 아래 Lane C의 초기 합계 정의는 [V1.1 보완](DATA_CONTRACT_PRESCORE_ADDENDUM.md)에서 Q+4 native TTM으로 구체화했으며 실제 점수에는 보완된 정의를 사용한다.

기본 target은 GAAP diluted per-share EPS이며 음수와 0을 보존한다. Lane A는 다음 fiscal-quarter,
Lane B는 다음 4개 fiscal-quarter path, Lane C는 공통 share basis의 미래 4분기 EPS 합계다.
Basic/adjusted EPS와 혼합하지 않는다. 희석주식수의 기간별 변화 때문에 분기 EPS 합계가 연간 공시 EPS와
항상 같지는 않으므로 Lane C는 합산 방식과 approximation flag를 명시한다.

SEC fact는 period start/end, accession, form, filed/accepted timestamp 및 version을 유지한다.
나중 filing의 restatement를 과거 origin feature로 소급하지 않는다. 최초 보고 EPS를 평가 truth로 고정한다.
미래 filing/split/analyst revision은 feature에 사용하지 않는다. 원천 정합성이 없는 target은 DATA_BLOCKED다.
Q4가 직접 관측되지 않으면 연간 diluted EPS에서 9개월 diluted EPS를 빼서 정확한 Q4라고 주장하지 않는다.
근사치 기반 별도 표면이 필요한 경우 score 전에 별도 identity와 제한을 고정한다.

가용시점은 기존 PIT calendar를 재사용한다. precise acceptance가 없으면 다음 거래일 같은 보수적인 규칙을
명시한다. 신뢰할 수 없는 timestamp를 정밀한 실제 시각으로 가장하지 않는다.

분할 조정은 forecast-origin share basis로 한다. 미래 split은 평가 target을 origin basis로 환산할 때만
사용할 수 있다. 역사적 feature는 origin 이전에 알려진 split만 적용한다. 조정 증거가 없으면 PE 결합을 차단한다.

각 origin의 학습에는 fit cutoff보다 먼저 실제 공개된 label만 포함한다. Multi-horizon에서는 각 horizon의
label availability를 따로 검사한다. TRAIN → VAL → RESEARCH TEST 경계 및 dataset manifest는 scoring 전 동결한다.
OOF ensemble은 실제 chronological OOF/VAL 예측으로만 가중치를 학습한다. Test 결과에 맞춘 재튜닝은 하지 않는다.

Foundation 모델의 inference causality와 사전학습 데이터 오염은 별개다. cutoff/코퍼스가 불명확하면 점수는
retrospective exploratory로 표시하고 엄밀한 PIT-certified 성능으로 간주하지 않는다.

PE 결합은 TTM EPS와 TTM PE 정의·currency·asof·share basis를 확인한 후에만 계산한다.
예측 EPS <= 0은 정상 EPS 예측이지만 일반 PE 결합은 `PE_COMBINATION_INELIGIBLE_NEGATIVE_EPS`다.
현재 PE를 미래 PE처럼 해석하지 않고 `STATIC_PE_HOLD_ASSUMPTION` 및 `HORIZON_MISMATCH_LIMITATION`을 명시한다.
