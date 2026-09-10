"""Summarize all prespecified training seeds; no best-seed selection."""
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
    root=RUN/'training_seed_stability/chronos2_eps_joint_finetuned'
    plan=json.loads((root/'PLAN.json').read_text(encoding='utf-8'))
    complete=json.loads((root/'COMPLETION.json').read_text(encoding='utf-8'))
    replay=json.loads((root/'REPLAY.json').read_text(encoding='utf-8'))
    assert complete['trained_annual_models']==15 and complete['seed1729_original_bit_exact']
    assert replay['status']=='PASS' and replay['fresh_process'] and len(replay['checks'])==75
    assert all(c['status']=='EXACT_SAVED_REPLAY_PASS' for c in replay['checks'])
    assert all(sha(RUN/path)==value for path,value in plan['frozen_main_hashes'].items())
    scores=pd.read_csv(root/'OOF_SCORES.csv');assert set(scores.seed)==set(plan['seeds'])
    comparisons=0
    for seed in plan['seeds']:
        frame=pd.read_parquet(root/f'seed_{seed}.parquet')
        assert frame.asof_date.dt.year.isin([2019,2020,2021]).all()
        for target,g in frame.groupby('target_key'):
            valid=g[(g.label_asof<pd.Timestamp('2022-01-01',tz='UTC'))&np.isfinite(g.actual_eps)]
            row=scores[(scores.seed==seed)&(scores.target_key==target)].iloc[0]
            assert len(valid)==row.predicted_rows
            np.testing.assert_allclose(np.abs(valid.predicted_eps-valid.actual_eps).mean(),row.MAE,rtol=1e-12,atol=1e-12)
            comparisons+=1
    ranges=[]
    for target,g in scores.groupby('target_key'):
        ranges.append({'target':target,'seeds':len(g),'truth_rows':int(g.predicted_rows.iloc[0]),
            'MAE_min':float(g.MAE.min()),'MAE_max':float(g.MAE.max()),'MAE_mean':float(g.MAE.mean()),
            'MAE_seed_sd_ddof1':float(g.MAE.std(ddof=1)),'relative_range_percent':float(100*(g.MAE.max()-g.MAE.min())/g.MAE.mean()),
            'medianAE_min':float(g.MedianAE.min()),'medianAE_max':float(g.MedianAE.max()),
            'interval80_coverage_min':float(g.interval80_coverage.min()),'interval80_coverage_max':float(g.interval80_coverage.max())})
    summary={'status':'PASS_FIVE_SEED_OOF_TRAINING_SENSITIVITY','created_utc':datetime.now(timezone.utc).isoformat(),
        'seeds':plan['seeds'],'ranges':ranges,'actual_saved_replay_checks':75,'independent_MAE_checks':comparisons,
        'original_seed_reproduced_bit_exact':True,'new_portfolio_members':0,'main_artifacts_unchanged':True,
        'best_seed_selected':False,'research_test_used_for_this_diagnostic':False,'formal_certified':False,
        'interpretation':'Training-seed sensitivity on the same selected OOF surface, not an independent generalization test or a confidence interval over new data'}
    save_json(RUN/'CHRONOS_TRAINING_SEED_REVIEW.json',summary)
    pd.DataFrame(ranges).to_csv(RUN/'CHRONOS_TRAINING_SEED_RANGES.csv',index=False)
    lines=['# Chronos EPS 미세조정: 고정 5-seed OOF 학습 안정성', '',
        '학습 seed와 data_seed만 미리 정한 1729, 2027, 3407, 31415, 27182로 바꿨다. '
        '각 seed에서 2019·2020·2021을 각각 원본 체크포인트부터 300 step 학습했다. 총 15개 모델을 '
        '실제로 저장했고, 별도 프로세스의 75개 연도×target point/quantile 재현이 정확히 일치했다.', '',
        '기준 1729의 point와 분위수는 원래 고정 예측과 bit-exact 일치했다. 다른 seed의 성적은 모두 '
        '보고하며 가장 좋은 seed를 선택하거나 원래 예측·앙상블을 교체하지 않았다.', '',
        table(pd.DataFrame(ranges),['target','truth_rows','MAE_min','MAE_max','MAE_mean','relative_range_percent',
            'interval80_coverage_min','interval80_coverage_max']), '',
        '값은 동일 OOF 표면에서 학습 난수의 민감도를 측정한 것이다. 새 데이터의 신뢰구간, '
        '독립적인 winner 확인시험 또는 pretraining-overlap/PIT 인증이 아니다. 예측구간의 목표 포함률은 '
        '80%지만 실제 포함률은 표처럼 낮을 수 있으므로 점 예측 우위를 확률 calibration 인증으로 확대하지 않는다.']
    (RUN/'CHRONOS_TRAINING_SEED_REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('CHRONOS_TRAINING_SEED_REVIEW_PASS',[(r['target'],r['MAE_min'],r['MAE_max']) for r in ranges],flush=True)


if __name__=='__main__':run()
