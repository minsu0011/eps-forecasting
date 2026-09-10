# EPS Forecasting

기업이 다음 분기에 얼마를 벌지와 시장이 그 이익에 몇 배를 지불할지는 다른 문제다. 이 프로젝트에서는 가격이나 P/E를 목표에 섞지 않고, 공시된 회계 정보로 분기 EPS와 장기 TTM EPS를 예측한다. 여러 모델을 비교하면서 가장 크게 부딪힌 문제는 모델 크기가 아니라 **당시 알 수 있었던 회계 기간과 주식 수 단위를 정확히 복원하는 일**이었다.

## 전체 구조

SEC 공시 → native fiscal ledger → 시점별 feature·성숙한 label → 모델 family별 예측 → 개발 구간에서 고정한 ensemble → 확인·관찰 구간 평가

- [V1 연구 코드](research/eps_model_lab_v1): 넓은 model zoo, PIT 수리와 입력 독립성 실험
- [V2 연구 코드](research/eps_model_lab_v2): 데이터와 선택 규칙을 새로 고정한 독립 실험
- [PE 연계 모듈](src): EPS 예측과 valuation multiple의 연결 조건

사용 기술은 Python, pandas·NumPy, scikit-learn, CatBoost, NGBoost, PyTorch, Chronos다. V1의 외부 시계열 라이브러리는 더 넓은 비교군이며, 모두 V2 최종 후보로 채택한 것은 아니다.

## 무엇을 예측하는가

| 구분 | 목표 | 구분해야 할 점 |
| --- | --- | --- |
| Lane A | 다음 native 분기의 GAAP diluted EPS | 음수도 유효한 목표값 |
| Lane B | Q+1~Q+4의 개별 분기 경로 | 없는 Q4를 연간 값에서 억지로 만들지 않음 |
| Lane C | Q+4 시점의 native TTM EPS | 분기 예측 네 개의 합과 동일한 정답이 아님 |

공시 가용 시각을 거래 세션에 맞춘 origin을 사용한다. 학습에는 origin뿐 아니라 목표값의 공개 시점도 cutoff 이전인 행만 넣는다. V2는 69개 기업, 3,753개 origin으로 구성했고 개발 2015–2018, 확인 2019–2021, 관찰 2022 이후를 분리했다. 확인·관찰 기간은 앞선 연구에서 이미 본 구간이다.

## 개발 과정

1. **먼저 비교 기준을 만들었다.** 직전 EPS 유지와 계절 반복을 기준으로 표 모델, 시계열 모델, 사전학습 모델을 같은 목표 위에 올렸다. 이름이 다른 별칭이나 같은 예측 벡터는 독립적인 다양성으로 세지 않았다.
2. **나중 공시가 과거 기록을 지우는 문제를 발견했다.** SEC의 가변 `fy/fp`를 fiscal identity로 쓰면서 HON의 후속 공시가 이전 기간과 충돌했다. 특정 기업만 고치는 대신 기존 점수 전체를 폐기 대상에 두고 기간 식별 규칙을 다시 설계했다.
3. **V1.3에서 native context로 달력을 복원했다.** 같은 accession의 시작·종료일과 기간 길이를 사용하고, 먼저 알려진 기간은 후속 공시 때문에 사라지지 않게 했다. 날짜가 모순되는 context는 추정하지 않고 격리했다.
4. **PIT 수리만으로 충분하지 않았다.** MCD의 주식 수가 실제 주식 수가 아닌 표의 백만 단위로 들어간 사례, YTD 채널에 TTM·단일 분기 금액이 섞인 사례를 찾았다. 재현 가능한 점수라고 해서 회계 feature가 올바른 것은 아니었다.
5. **V2는 별도 데이터와 규칙으로 시작했다.** 원문에서 확인된 단위만 수리하고 불명확한 회계 값은 격리했다. 2019년 이전에 검증된 회계 셀이 부족한 Track A는 모델 선택 전에 막았다.
6. **개발 구간에서만 대표와 가중치를 골랐다.** 46개 실제 변형을 평가한 뒤 lane·family 대표를 고정했다. 확인 결과를 보고 탈락 모델 대신 다른 모델을 끼워 넣지 않았다.
7. **목표마다 결론을 나눴다.** Lane A의 6개 대표는 사전 기준을 통과했지만 Lane C의 두 후보는 확인 구간 MAE 기준을 넘지 못했다. 이를 하나의 “EPS 모델 성공”으로 합치지 않았다.

## 모델의 역할

| 모델군 | 이 연구에서 맡긴 역할 | 선택할 때 본 점 |
| --- | --- | --- |
| 관측 baseline | 마지막 값·계절 반복만으로 가능한 수준 | 복잡한 모델이 필요한가 |
| Ridge·HistGB·CatBoost | 회계 시계열을 표 형태로 풀어 비선형 관계 비교 | direct/residual, rolling, 최근 표본 가중의 차이 |
| LSTM 등 로컬 시계열 | EPS 경로의 시간 의존성 학습 | 잔차 학습과 최근성 변형의 안정성 |
| Chronos zero-shot·미세조정 | 사전학습 표현의 전이와 EPS 적응 비교 | 체크포인트 시점 및 origin 간 입력 독립성 |
| NGBoost | 점 예측과 예측 분포를 함께 추정 | MAE와 구간 보정은 별도 평가 |
| 고정 ensemble | 서로 다른 오차를 결합 | 개발 구간에서만 구성원·비음수 가중치 선택 |

외부 모델 자체를 새로 구현한 프로젝트는 아니다. 직접 구성한 부분은 fiscal ledger, 시점·단위 계약, EPS용 adapter, 공통 평가와 선택·재현 절차다. 라이브러리와 사전학습 가중치의 출처는 [출처](ATTRIBUTION.md)에 구분했다.

## 결과와 남은 문제

보존된 [확인 구간 결과](docs/results/confirmation.json)에서 Lane A는 동일한 648개 truth에 대해 Chronos rolling MAE 0.7500, CatBoost 0.8387, LSTM 0.8160이었다. 이는 USD EPS 오차이며 투자 수익률이 아니다. 세부 비교와 다른 모델 값은 [결과 문서](docs/wiki/Validation-and-Results.md)에 정리했다.

Lane C는 512개 truth에서 HistGB 3.6248, NGBoost 3.6047이었다. 당시 기준선 MAE 3.5499를 넘지 못해 준비 완료로 판단하지 않았다. Chronos의 80% 구간도 확인 구간에서 보정 후 coverage 74.07%에 그쳐 점 예측과 확률 보정의 결론이 다르다.

V1과 V2는 평가 조건이 달라 직접적인 개선치로 비교하지 않았다. 현재 수집본의 과거 수정치, 고정 기업군의 생존 편향, foundation model의 과거 이용 가능 시점도 남아 있다. EPS×P/E는 단위·기간을 맞추는 통합 실험이며 엄격한 가격 예측 인증 조합은 없다.

## 시작하기

환경 구성과 데이터 계약은 [실행 안내](docs/wiki/How-to-Run.md), 작은 계약 테스트는 다음 명령에서 시작한다.

```powershell
python -m pytest research/eps_model_lab_v1/test_checksum_stream.py -q
```

[데이터 준비](data/README.md)에는 별도 준비가 필요한 공시와 가중치 범위를 설명했다.

## 상세 문서

[연구 안내](docs/wiki/Home.md) · [전체 개발 과정](docs/wiki/Development-Journey.md) · [모델 발전](docs/wiki/Model-Evolution.md) · [회계·PIT 병목](docs/wiki/PIT-and-Accounting-Issues.md) · [실험 결정](docs/wiki/Experiments-and-Decisions.md) · [결과](docs/wiki/Validation-and-Results.md)
