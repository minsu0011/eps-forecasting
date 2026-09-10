# 실험 중 내린 결정

| 관측 | 시도·대응 | 결정 |
| --- | --- | --- |
| 후속 공시가 이전 fiscal record 삭제 | accession 하나의 예외가 아닌 기간 식별 재설계 | 기존 점수 wave를 폐기 대상으로 보존 |
| 주식 수·기간 의미 노출 | no-share 진단과 exact-context selector | V1의 동결 점수 대체 금지, V2 별도 identity |
| 통계 모델 smoke 수렴 실패 | 본 학습 진입 gate 강화 | 실패 출력을 최종 portfolio에서 제외 |
| batch를 바꾸면 한 origin 예측도 변경 | origin-isolated inference, 순서·다른 행·singleton 검사 | 동일 batch replay만으로 승인하지 않음 |
| CatBoost pickle bytes가 반복 학습마다 다름 | native model JSON·학습 state 비교 | GUID·완료 시각 차이와 학습 차이를 구분 |
| point-only 모델에 quantile 열이 생김 | interval 지표를 missing 처리 | 없는 분포 성능을 0으로 보고하지 않음 |
| Lane C confirmation 기준 미달 | 사전 고정 대표만 평가 | 확인 후 winner 교체 없이 보류 |

Windows atomic replace 실패 때 이미 훈련된 모델을 새 프로세스로 다시 읽어 예측을 복구한 사례도 있었다. 훈련 시간 기록을 잃었으면 추정값을 채우지 않았다. 파일 저장·재추론·새 학습 결정성은 각각 다른 검사다.

V1 foundation adapter의 origin 간 독립성 문제는 모델이 미래를 학습했다는 직접 증거와 다르다. 다만 요구한 입력 독립성을 위반했으므로 diagnostic으로 제한할 근거는 된다. 원인 불명인 singleton 수치 차이의 허용오차를 넓혀 통과시키지 않았다.

[재현 구현](../../research/eps_model_lab_v2/replay.py) · [새 학습 반복](../../research/eps_model_lab_v2/retraining.py)
