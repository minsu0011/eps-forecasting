# 코드와 실험 진입점

가볍게 확인할 부분과 실제 학습을 분리한다. 저장소 루트에서 Python 의존성을 준비한 뒤 checksum 계약 테스트부터 볼 수 있다.

```powershell
python -m pytest research/eps_model_lab_v1/test_checksum_stream.py -q
```

[테스트의 확인 범위](../testing-notes.md)는 학습 성과 재현과 다르다. V2의 전체 실행에는 SEC context cache, 동결된 데이터·recipe, 외부 모델 환경과 가중치가 필요하다. [데이터 안내](../../data/README.md)를 먼저 읽는다.

읽는 순서는 `fiscal_builder` → `clean_data` → `freeze_inputs` → `model_common` → 모델별 runner → `evaluation` → `ensemble`이다. [V2 모듈](../../research/eps_model_lab_v2)에서 경로·입력 계약을 확인하고 새 output identity를 사용한다.

원래의 로컬 경로·receipt에 묶인 연구 runner도 있다. 이를 모든 환경에서 한 명령으로 재현 가능한 배포 패키지라고 보지 않는다. 데이터·checkpoint를 임의로 대체하면 보존된 결과와 같은 실험이 아니다.
