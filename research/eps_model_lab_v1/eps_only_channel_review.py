"""Evidence-based interpretation of two separate accounting-channel ablations."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.common import metric_row
from research.eps_model_lab_v1.final_reports import table


def run():
    root=RUN/'eps_only_channel_diagnostics';plan=json.loads((root/'PLAN.json').read_text(encoding='utf-8'))
    replay=json.loads((root/'REPLAY.json').read_text(encoding='utf-8'))
    assert replay['status']=='PASS' and replay['fresh_process'] and len(replay['checks'])==80
    assert all(r['status']=='EXACT_SAVED_REPLAY_PASS' for r in replay['checks'])
    assert all(sha(RUN/path)==digest for path,digest in plan['frozen_main_hashes'].items())
    comparisons=[];files=[]
    for name in ['TimeXer','SOFTS']:
        original=pd.read_parquet(RUN/'predictions'/f'nf_multivar_{name}.parquet')
        modified=pd.read_parquet(root/f'{name}.parquet')
        assert len(modified)==10505 and not modified.duplicated(['sample_id','target_key']).any()
        for (target,split),g in modified.groupby(['target_key','split']):
            if split=='VALIDATION_OOF':g=g[g.label_asof<pd.Timestamp('2022-01-01',tz='UTC')]
            old=original[original.target_key==target].set_index('sample_id').loc[g.sample_id]
            newscore=metric_row(g);oldscore=metric_row(old)
            valid=np.isfinite(g.actual_eps)&np.isfinite(g.predicted_eps)
            independent=np.abs(g.loc[valid,'predicted_eps']-g.loc[valid,'actual_eps']).mean()
            np.testing.assert_allclose(independent,newscore['MAE'],rtol=1e-12,atol=1e-12)
            comparisons.append({'base':name,'target':target,'split':split,'truth_rows':newscore['predicted_rows'],
                'original_4channel_MAE':oldscore['MAE'],'EPS_TTM_only_MAE':newscore['MAE'],
                'EPS_TTM_only_MedianAE':newscore['MedianAE'],'used_for_selection':False})
        for path in (root/'fitted'/name).glob('*.pt'):
            files.append({'path':path.relative_to(RUN).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    pd.DataFrame(comparisons).to_csv(RUN/'EPS_ONLY_CHANNEL_COMPARISON.csv',index=False)
    save_json(root/'TRAINED_ARTIFACT_INDEX.json',{'files':files,'weights_in_compact_zip':False})
    save_json(RUN/'EPS_ONLY_CHANNEL_REVIEW.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_TWO_FIXED_ABLATIONS_FRESH_REPLAY','trained_annual_models':16,'fresh_replay_checks':80,
        'prediction_files':2,'main_artifacts_unchanged':True,'new_portfolio_members':0,'formal_certified':False,
        'known_NI_revenue_share_fields_not_consumed':True,'EPS_TTM_source_vintage_and_share_basis_certified':False,
        'not_pure_causal_identification':'Removing channels also changes auxiliary training supervision and model parameter dimensions',
        'post_test_inspection_diagnostic':True})
    selected=pd.DataFrame(comparisons);selected=selected[selected.target.isin(['h1','ttm'])]
    lines=['# NI·매출 보조 채널 제거 진단', '',
        '고정 TimeXer/SOFTS에서 기간 혼합이 확인된 NI·매출 채널을 제거하고 native EPS·TTM 두 채널만 '
        '남겼다. 300 step·seed 1729·연도 cutoff·EPS/TTM 값과 scale은 유지했다. 각각 8개 연도 모델을 '
        '실제 학습했고, 새 프로세스의 80개 연도×target 재현이 정확히 일치했다.', '',
        table(selected,['base','target','split','truth_rows','original_4channel_MAE','EPS_TTM_only_MAE','EPS_TTM_only_MedianAE']), '',
        '이는 post-test 데이터 품질 진단이며 원래 117개 예측·shortlist·99개 풀·앙상블을 바꾸지 않았다. '
        '두 채널 삭제는 보조 supervision과 파라미터 차원도 바꾸므로 성적 차이를 기간 오류의 순수 인과효과라고 '
        '해석하지 않는다. 알려진 회계 열을 쓰지 않지만 EPS/TTM 원자료 빈티지·split basis·전체 PIT가 인증된 것도 아니다.']
    (RUN/'EPS_ONLY_CHANNEL_REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('EPS_ONLY_CHANNEL_REVIEW_PASS',len(files),'annual models,80 saved replay checks',flush=True)


if __name__=='__main__':run()
