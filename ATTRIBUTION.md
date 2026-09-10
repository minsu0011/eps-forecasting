# 출처와 이용 범위

사용 패키지: Python, numpy, pandas, scipy, sklearn, torch, lightgbm, xgboost, catboost, requests, chronos, ngboost.

외부 라이브러리와 모델의 구현은 각 프로젝트의 기여다. 데이터 처리·연결·평가에서 직접 구성한 범위는 [프로젝트 설명](README.md)에 구분했다.

- `src\pe_regime_v04\model_lab\probabilistic\references.py`

`LICENSE`와 `SOURCE_ATTRIBUTION.md`는 함께 포함한 PE overlay source에서 보존한 고지다. 이것만으로 EPS 전체와 외부 pretrained 모델에 일괄 MIT 권리가 확인되었다고 보지 않는다.

외부 모델/구현 참조: Amazon Science Chronos, Nixtla StatsForecast/NeuralForecast, AutoGluon, Unit8 Darts, Google TimesFM, Salesforce Uni2TS/Moirai. 이들의 구현과 가중치는 각 배포자의 권리를 따른다.


기존 라이선스·저작권 고지는 해당 범위에 그대로 적용된다. 별도 명시가 없는 코드에 포괄적인 재사용 허가를 추가하지 않는다. 원천 데이터와 공급자 응답은 코드와 별개의 이용 조건을 따른다. 외부 사전학습 가중치는 포함하지 않으며 사용할 모델의 model card와 가중치 이용 조건을 따로 확인해야 한다.
