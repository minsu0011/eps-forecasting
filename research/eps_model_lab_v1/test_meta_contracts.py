import copy
import numpy as np
import pandas as pd
import pytest
from research.eps_model_lab_v1.common import dataset,metric_row
from research.eps_model_lab_v1.ensemble_models import aggregate,convex_l1,select,load_pool


def test_convex_stack_constraints_and_real_predictions():
    rng=np.random.default_rng(1729);x=rng.normal(size=(160,5));truth=x@np.array([.4,.3,.2,.1,0.])
    w,receipt=convex_l1(x,truth)
    assert receipt['success'] and w.min()>=0 and np.isclose(w.sum(),1)
    np.testing.assert_allclose(x@w,truth,atol=1e-8)
    assert np.mean(np.abs(x@w-truth))<=np.mean(np.abs(x.mean(axis=1)-truth))


@pytest.mark.parametrize('method',['mean','median','trimmed_mean','convex_l1'])
def test_ensemble_missing_member_fails_closed_negative_eps_allowed(method):
    x=np.array([[-5.,-4.,-3.,-2.,-1.],[1.,2.,3.,np.nan,5.]])
    result=aggregate(x,[.2]*5,method)
    assert result[0]==pytest.approx(-3.) and np.isnan(result[1])
    with pytest.raises(RuntimeError): aggregate(x[:,:4],[.25]*4,method)


def test_all_selection_labels_available_before_cutoff():
    f=dataset();cutoff=pd.Timestamp('2022-01-01',tz='UTC')
    # Real late-released validation-origin labels exist; origin-only splitting is unsafe.
    candidates=f[(f.asof_date.dt.year>=2019)&(f.asof_date<cutoff)]
    assert ((candidates.label_asof_ttm>=cutoff)&candidates.y_ttm.notna()).any()
    pool=load_pool();plan,_=select(pool,f,'ttm','local_causal',cutoff)
    chosen=f.set_index('sample_id').loc[plan['selection_sample_ids']]
    assert (chosen.label_asof_ttm<cutoff).all() and (chosen.asof_date<cutoff).all()


def test_selection_unaffected_by_test_or_late_released_truth_mutation():
    f=dataset();pool=load_pool();cutoff=pd.Timestamp('2022-01-01',tz='UTC')
    before,_=select(pool,f,'ttm','local_causal',cutoff)
    changed=f.copy();future=(f.asof_date>=cutoff)|(f.label_asof_ttm>=cutoff)
    changed.loc[future,'y_ttm']=1e12
    altered={}
    for name,p in pool.items():
        q=p.copy();outside=(q.asof_date>=cutoff)|(q.label_asof>=cutoff)
        q.loc[outside,'actual_eps']=1e12;q.loc[outside,'predicted_eps']=-1e12
        altered[name]=q
    after,_=select(altered,changed,'ttm','local_causal',cutoff)
    assert before==after


def test_probability_metrics_not_invented_for_point_models():
    p=next(iter(load_pool().values())).copy()
    p[['p10_eps','p50_eps','p90_eps']]=np.nan
    score=metric_row(p[p.target_key=='h1'])
    assert 'pinball_loss' not in score and 'interval80_coverage' not in score
