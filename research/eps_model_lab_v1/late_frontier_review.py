"""Audit and describe added breadth without reopening the already frozen portfolio."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.final_reports import table


def run():
    root=RUN/'late_frontier';records=[];scores=[];artifacts=[]
    names=['MLPMultivariate','SOFTSSharp','xLSTM']
    if (root/'TimeLLM/COMPLETION.json').exists():names.append('TimeLLM')
    for name in names:
        folder=root/name
        plan=json.loads((folder/'PLAN.json').read_text(encoding='utf-8'))
        complete=json.loads((folder/'COMPLETION.json').read_text(encoding='utf-8'))
        replay=json.loads((folder/'REPLAY.json').read_text(encoding='utf-8'))
        frame=pd.read_parquet(folder/'predictions.parquet');score=pd.read_csv(folder/'SCORES.csv')
        assert complete['prediction_sha256']==sha(folder/'predictions.parquet')
        assert len(frame)==10505 and not frame.duplicated(['sample_id','target_key']).any()
        assert np.isfinite(frame.predicted_eps).all()
        assert replay['status']=='PASS' and replay['fresh_process'] and len(replay['checks'])==40
        assert all(c['status']=='EXACT_SAVED_REPLAY_PASS' for c in replay['checks'])
        assert all(sha(RUN/path)==value for path,value in plan['frozen_main_hashes'].items())
        for r in score.to_dict('records'):
            g=frame[(frame.target_key==r['target'])&(frame.split==r['split'])]
            if r['split']=='VALIDATION_OOF':g=g[g.label_asof<pd.Timestamp('2022-01-01',tz='UTC')]
            valid=g[np.isfinite(g.actual_eps)&np.isfinite(g.predicted_eps)]
            errors=valid.predicted_eps.to_numpy()-valid.actual_eps.to_numpy()
            independent={'MAE':np.mean(np.abs(errors)),'MedianAE':np.median(np.abs(errors)),
                'RMSE':np.sqrt(np.mean(errors**2)),'bias':np.mean(errors)}
            assert len(valid)==r['predicted_rows']
            for metric,value in independent.items():np.testing.assert_allclose(value,r[metric],rtol=1e-12,atol=1e-12)
        scores.append(score)
        for path in folder.rglob('*'):
            if path.is_file() and ('fitted' in path.parts or 'smoke_fitted' in path.parts):
                artifacts.append({'path':path.relative_to(RUN).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
        records.append({'name':name,'status':complete['status'],'origins':2101,'prediction_rows':len(frame),
            'annual_models':8,'fresh_replay_checks':40,'independent_metric_checks':40,
            'main_unchanged':True,'post_test_inspection_diagnostic':True,'new_portfolio_members':0,
            'runtime_seconds':complete['runtime_seconds'],'no_direct_raw_share_count_feature':True})
    all_scores=pd.concat(scores,ignore_index=True)
    all_scores.to_csv(RUN/'EPS_LATE_FRONTIER_SCORES.csv',index=False)
    save_json(root/'TRAINED_ARTIFACT_INDEX.json',{'files':artifacts,'full_weights_in_compact_zip':False})
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_SEPARATE_POST_FREEZE_DIAGNOSTICS',
        'models':records,'additional_prediction_files':len(names),'additional_annual_models':8*len(names),
        'fresh_saved_replay_checks':40*len(names),'independent_scalar_metric_checks':40*len(names),
        'frozen_main_prediction_files':117,'main_pool_identities':99,'main_portfolio_unchanged':True,
        'formal_certified':False,'new_portfolio_members':0,'environment_recovery':'late_frontier/environment/RECOVERY.json',
        'extra_models_not_215_registry_status_rewrite':True,
        'residual_scope_note':'DeepAR exists in already executed AutoGluon ecosystem; no new-algorithm claim for another wrapper. HINT lacks a defined company hierarchy. TimeLLM execution status is in its separate late-frontier receipts.'}
    save_json(RUN/'EPS_LATE_FRONTIER_REVIEW.json',summary)
    selected=all_scores[all_scores.target.isin(['h1','ttm'])]
    lines=[f'# 포트폴리오 고정 이후 추가 구조 {len(names)}종', '',
        '사용자 10시간 실행 창의 후반에 실시한 별도 진단이다. 이미 본 research-test와 독립적인 확인시험이 아니며, '
        '117개 main 예측·99개 후보 풀·10개 shortlist·8개 앙상블은 바꾸지 않았다.', '',
        'MLPMultivariate와 SOFTSSharp는 한 기업의 EPS·native TTM·native YTD 순이익·native YTD 매출 4개 채널을 '
        '공식 network와 고정 masked trainer로 학습했다. xLSTM은 mLSTM backbone의 공식 NeuralForecast 학습 경로를 썼다. '
        'xLSTM의 C 출력은 분기 예측 합이므로 native TTM 예측과 share-weighting 의미가 완전히 같지는 않다.', '',
        table(selected,['model_id','target','split','predicted_rows','MAE','MedianAE','price_scaled_MAE']), '',
        '각 구조 8개 annual cutoff, 2,101 origin × 5 target를 실제 예측했다. 저장 artifact를 새 프로세스에서 불러 '
        f'{40*len(names)}개 연도×target 재현이 정확히 일치했다. 각 연도에서 singleton·순서 변경·다른 origin 변조·미래 정답 변조를 '
        '별도 점검했다. 이 입력 독립성 검사는 역사적 원자료 빈티지나 사전학습 누수를 인증하는 검사가 아니다.', '',
        'MLPMultivariate·SOFTSSharp·xLSTM은 이 설정에서 A/C OOF 기존 선두보다 MAE가 컸다. 이를 근거로 family 전체가 무가치하다고 '
        '단정하지 않지만, 이번 실행에서는 반복 튜닝하거나 기존 포트폴리오를 교체하지 않았다.', '',
        'xLSTM 추가 환경은 기존 GPU package lock을 그대로 유지했다. 기본 import는 사용하지 않는 sLSTM의 CUDA '
        '개발 도구 초기화 때문에 실패했다. 설치본을 건드리지 않은 독립 복사본에서 해당 초기화만 실제 sLSTM load '
        '시점으로 옮겼다. 54개 원본 파일 중 이 1개만 달라졌고 mLSTM 계산 코드는 동일하다. '
        '`late_frontier/environment/RECOVERY.json`과 `XLSTM_OPTIONAL_COMPILER_PATCH.txt`에 정확한 변경과 재현 방법이 있다.', '',
        '직접 raw 주식 수 입력은 없지만 전체 데이터 품질 인증을 뜻하지 않는다. '
        '특히 MLPMultivariate·SOFTSSharp의 회계 채널은 뒤늦게 확인된 YTD/12개월 기간 혼합 문제에 노출된다. '
        '`SOURCE_ACCOUNTING_DURATION_WARNING_KO.md`와 `SOURCE_SHARE_UNIT_WARNING_KO.md`를 함께 읽어야 하며 정식 인증 수는 0이다.']
    if 'TimeLLM' in names:
        lines += ['', 'TimeLLM은 고정 GPT-2 12개 층과 학습되는 reprogramming/head를 쓴다. '
            '공식 구현의 prompt padding에는 attention mask가 없어 batch의 다른 prompt 길이가 patch 위치를 바꿀 수 있다. '
            '따라서 평가 전에 inference window를 1개로 고정하고 실제 입력 독립성을 검사했다. '
            '언어 모델의 사전학습 overlap은 미해결이며 정식 역사적 인증이 아니다.']
    (RUN/'EPS_LATE_FRONTIER_REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('LATE_FRONTIER_AUDIT_PASS',len(records),'models,',40*len(names),'fresh replay and independent metrics',flush=True)


if __name__=='__main__':run()
