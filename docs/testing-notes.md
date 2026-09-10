# 테스트와 실행 범위

저장소 루트에서 의존성을 준비한 뒤 다음 범위를 확인할 수 있다.

```powershell
python -m pytest research/eps_model_lab_v1/test_checksum_stream.py research/eps_model_lab_v1/test_atomic_io.py research/eps_model_lab_v1/test_native_crps.py -q
```

checksum·atomic I/O·native CRPS의 16개 테스트와 연구 모델 성과 재현은 다른 검사다. V2 confirmation 집계는 docs/results의 실험 집계를 따른다. V1 점수는 단위·기간 문제에 노출된 연구 기록이며 올바른 회계 feature 검증이 아니다.

V1에는 주식 수 단위와 회계 기간 문제가 남아 있어 깨끗한 feature의 성과로 해석할 수 없다. V2 Track A는 회계 근거 부족으로 막혔고 Lane C는 확인 기준에 미달했다. Lane A의 연구 후보도 실배포 인증은 아니다. 확인·관찰 기간은 이미 본 구간이다.

테스트 실행과 전체 원천 수집·학습은 별개다. 연구 결과는 [README](../README.md)의 당시 기록과 입력 조건을 함께 읽는다.
