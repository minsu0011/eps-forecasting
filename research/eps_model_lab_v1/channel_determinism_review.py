"""Keep initial fresh-fit mismatch and distinguish saved replay from retraining."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha


if __name__=='__main__':
    root=RUN/'eps_only_channel_diagnostics';checks=[]
    diagnosis=json.loads((root/'PUBLIC_API_INITIAL_FAILURE_DIAGNOSIS.json').read_text(encoding='utf-8'))
    for name in ['TimeXer','SOFTS']:
        left=root/'deterministic_training/a';right=root/'deterministic_training/b'
        l=np.load(left/f'{name}.npy',allow_pickle=False);r=np.load(right/f'{name}.npy',allow_pickle=False)
        np.testing.assert_array_equal(l,r)
        if sha(left/f'{name}.pt')!=sha(right/f'{name}.pt'):raise RuntimeError('Deterministic checkpoint bytes differ')
        checks.append({'name':name,'year':2019,'independent_processes':True,'prediction_rows':len(l),
            'prediction_bit_exact':True,'entire_checkpoint_bytes_exact':True,'checkpoint_sha256':sha(left/f'{name}.pt')})
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_TWO_MODEL_2019_DETERMINISTIC_RETRAINING_PROFILE',
        'checks':checks,'initial_input_arrays_exact':all(r['equal'] for r in diagnosis['input_arrays']),
        'initial_cached_vs_rebuilt_prediction_difference':diagnosis['cached_vs_rebuilt_input_prediction_max_difference'],
        'initial_new_fit_prediction_difference':diagnosis['original_vs_new_weights_cached_prediction_max_difference'],
        'initial_failure_preserved':True,'saved_replay_still_exact':True,'original_scores_replaced':False,
        'exact_nondeterministic_kernel_not_individually_localized':True,
        'scope':'Two independent process fits for each of TimeXer/SOFTS, 2019 only, same machine. Not all-year, all-model, or cross-hardware retraining certification.',
        'concurrency_note':'Replicate processes briefly overlapped on the GPU; this is an output-identity check, not a dedicated-process runtime benchmark.',
        'settings':{'CUBLAS_WORKSPACE_CONFIG':':4096:8','torch_deterministic_algorithms':True,'cudnn_deterministic':True,'cudnn_benchmark':False,'TF32':False},
        'source':'https://docs.pytorch.org/docs/2.11/notes/randomness.html','formal_certified':False}
    save_json(RUN/'EPS_CHANNEL_RETRAINING_NUMERICAL_REVIEW.json',result)
    lines=['# 저장 모델 재현과 새 학습 재현은 다르다', '',
        'EPS·TTM 두 채널 TimeXer의 공통 API 재학습은 원래 진단 모델과 달랐다. 캐시와 재구성 입력은 '
        'x·mask·calendar·label·availability·scale 모두 정확히 일치했고, 원래 가중치로 예측하면 입력 경로 간 차이는 '
        '0이었다. 새로 학습한 가중치에서는 전체 2019 예측 경로 최대 차이 약 0.2984 EPS가 관측됐다. '
        '초기 실패 checkpoint와 상세 차이는 그대로 보존했다.', '',
        '별도 재현 설정으로 cuBLAS workspace, 결정론적 연산, cuDNN deterministic을 명시하고 benchmark와 TF32를 '
        '끄자 TimeXer와 SOFTS 각각 독립 프로세스 두 번의 2019 학습에서 예측뿐 아니라 전체 checkpoint 파일 bytes가 '
        '동일했다. 특정 비결정론 커널 하나까지 분리해 원인을 확정한 것은 아니므로 해당 설정 묶음의 효과로 기록한다. '
        '[실행 버전에 맞춘 PyTorch 재현성 문서](https://docs.pytorch.org/docs/2.11/notes/randomness.html).', '',
        '범위는 두 모델·2019·동일 장비다. 모든 연도·모델·다른 GPU에서의 완전한 재학습 재현을 인증하지 않는다. '
        '새 설정은 원래 학습 결과를 바꿀 수 있으므로 기존 main/ablation 예측이나 가중치를 덮어쓰지 않았다. '
        '기존 저장 모델의 80개 연도×target 재실행 PASS도 이 별도 새 학습 문제와 구분한다.']
    (RUN/'EPS_CHANNEL_RETRAINING_NUMERICAL_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('CHANNEL_DETERMINISTIC_PROFILE_REVIEW_PASS',len(checks),'model pairs; original failure preserved',flush=True)
