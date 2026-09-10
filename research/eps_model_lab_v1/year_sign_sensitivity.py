"""Exact finite sign enumeration illustrates few-year resolution, not certification."""
from datetime import datetime,timezone
from itertools import product
import json
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN,save_json,sha
from research.eps_model_lab_v1.final_reports import table


def enumerate_signs(values):
    v=np.asarray(values,dtype=float)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('Finite nonempty annual means required')
    if len(v)>15:raise ValueError('Bounded exact enumeration only')
    signs=np.asarray(list(product([-1.,1.],repeat=len(v))))
    distribution=(signs*v).mean(axis=1);observed=v.mean()
    return {'years':len(v),'enumerated_patterns':len(signs),'equal_year_mean_loss_difference':float(observed),
        'conditional_one_sided_tail_fraction':float(np.mean(distribution<=observed+1e-14)),
        'conditional_two_sided_tail_fraction':float(np.mean(np.abs(distribution)>=abs(observed)-1e-14)),
        'minimum_nonzero_one_sided_resolution':1/len(signs),'formal_p_value_claim':False}


def run():
    pairs=[('chronos2_eps_joint_finetuned','h1'),('nf_multivar_TimeXer','h1'),('ridge_raw','ttm'),('nf_multivar_SOFTS','ttm')]
    baseline=pd.read_parquet(RUN/'predictions/persistence_observed.parquet');rows=[];annual=[];hashes={}
    for name,target in pairs:
        path=RUN/'predictions'/f'{name}.parquet';hashes[name]=sha(path);frame=pd.read_parquet(path)
        q=frame[(frame.target_key==target)&(frame.split=='VALIDATION_OOF')&(frame.label_asof<pd.Timestamp('2022-01-01',tz='UTC'))]
        b=baseline[baseline.target_key==target].set_index('sample_id').loc[q.sample_id]
        valid=np.isfinite(q.actual_eps.to_numpy())&np.isfinite(q.predicted_eps.to_numpy())&np.isfinite(b.predicted_eps.to_numpy())
        q=q.loc[valid].copy();b=b.loc[valid]
        q['loss_difference']=np.abs(q.predicted_eps.to_numpy()-q.actual_eps.to_numpy())-np.abs(b.predicted_eps.to_numpy()-q.actual_eps.to_numpy())
        grouped=q.groupby(q.asof_date.dt.year).loss_difference.agg(['mean','size'])
        rows.append({'model':name,'target':target,'matched_rows':len(q),**enumerate_signs(grouped['mean'].to_numpy())})
        for year,r in grouped.iterrows():annual.append({'model':name,'target':target,'year':int(year),'rows':int(r['size']),'mean_AE_difference_vs_observed_persistence':float(r['mean'])})
    pd.DataFrame(rows).to_csv(RUN/'EPS_FEW_YEAR_SIGN_ENUMERATION.csv',index=False)
    pd.DataFrame(annual).to_csv(RUN/'EPS_FEW_YEAR_PAIRED_LOSS.csv',index=False)
    save_json(RUN/'EPS_FEW_YEAR_UNCERTAINTY_REVIEW.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'EXACT_DIAGNOSTIC_ENUMERATION_COMPLETE','comparisons':rows,'model_hashes':hashes,
        'formal_p_values_claimed':False,'multiplicity_adjusted':False,'selected_models_independent_of_viewed_OOF':False,
        'required_but_unverified_assumptions':['Independent year clusters','Joint sign symmetry under the null','No model-selection conditioning'],
        'interpretation':'Finite sign-pattern resolution is 1/8 with three years and 1/4 with two years. Thousands of bootstrap draws cannot create additional independent years.',
        'main_portfolio_changed':False,'formal_certified':False})
    lines=['# 적은 OOF 연도 수와 선택 불확실성', '',
        'OOF 선두 4개 역할을 persistence와 동일 정답 행에서 비교하고, 연도별 평균 절대오차 차이에 '
        '가능한 ± 부호 조합을 전부 열거했다. 음수 차이는 모델 오차가 더 작다는 뜻이다. '
        '연도별 동일 가중이므로 전체 행 가중 MAE 차이와 숫자가 다를 수 있다.', '',
        table(pd.DataFrame(rows),['model','target','matched_rows','years','enumerated_patterns',
            'equal_year_mean_loss_difference','conditional_one_sided_tail_fraction','conditional_two_sided_tail_fraction']), '',
        'A는 3개 연도이므로 8개 부호 조합, C는 2개 연도이므로 4개 조합뿐이다. '
        '이 설정의 한쪽 꼬리 분해능은 각각 1/8, 1/4이다. 2,000회나 100,000회 bootstrap을 '
        '실행하더라도 독립 연도가 늘어나는 것은 아니다.', '',
        '독립 연도·귀무가설 아래 부호 대칭성·선정 과정 조건부 문제를 검증하지 않았으므로 '
        '표의 꼬리 비율을 정식 p-value나 승격 기준으로 사용하지 않는다. 모델은 이미 OOF로 골랐고 '
        '99개 풀과 다수 사후 subgroup 비교가 있다. 회계 입력 단위·기간 문제도 별도로 남는다. '
        '따라서 재현 가능한 낮은 오차라는 사실을 통계적 우월성 인증으로 바꾸지 않는다.']
    (RUN/'EPS_FEW_YEAR_UNCERTAINTY_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FEW_YEAR_SIGN_ENUMERATION_COMPLETE',[(r['target'],r['years'],r['enumerated_patterns']) for r in rows],flush=True)


if __name__=='__main__':run()
