# 외부 모델과 패키지 출처

이 저장소는 외부 알고리즘 구현을 복사하지 않고 설치 패키지 API를 호출하거나 v0.3이 이미 생성한 출력을 소비한다.

- LightGBM: https://github.com/microsoft/LightGBM — MIT License
- scikit-learn: https://github.com/scikit-learn/scikit-learn — BSD-3-Clause
- pandas: https://github.com/pandas-dev/pandas — BSD-3-Clause
- NumPy: https://github.com/numpy/numpy — BSD-3-Clause
- PyYAML: https://github.com/yaml/pyyaml — MIT License

v0.3 upstream에서 생성된 다음 열을 입력으로 재사용한다.

- `sjm3_p_*`, `sjm2_gate_p_*`: jumpmodels/Sparse Jump Model 계열
- `hmm3_p_*`: hmmlearn Gaussian HMM 계열
- `rule_p_*`: DEAM-inspired 독립 rule implementation
- `forecast_p_*`: v0.3 regime forecast model

이 overlay는 SJM, HMM, DEAM Pine code를 재구현하거나 소스 복사하지 않는다.
