from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import pytest
from research.eps_model_lab_v2.model_common import dataset,native_features,baseline,training_rows,surface
from research.eps_model_lab_v2.pe_contract import compatibility
from research.eps_model_lab_v2.evaluation import gate


def test_native_features_ignore_all_future_and_accounting():
    frame=dataset().iloc[:20].copy();before=native_features(frame)
    for c in frame:
        if c.startswith(('y_','label_','target_','account_','quality_account_')):frame[c]=1e12
    np.testing.assert_array_equal(before,native_features(frame))


@pytest.mark.parametrize('year',[2015,2016,2017,2018,2019,2022])
def test_training_purges_origin_and_label(year):
    frame=dataset()
    for target in ['h1','ttm']:
        train,weights,cutoff=training_rows(frame,year,target,'rolling')
        assert (train.asof_date<cutoff).all() and (train['label_asof_'+target]<cutoff).all()
        assert (train.asof_date>=cutoff-pd.DateOffset(years=5)).all() and len(weights)==len(train)


def test_baseline_aliases_not_independent():
    f=dataset().iloc[:100]
    for t in ['h1','ttm']:
        np.testing.assert_array_equal(baseline(f,t,'random_walk'),baseline(f,t,'persistence_observed'))
        np.testing.assert_array_equal(baseline(f,t,'seasonal_random_walk'),baseline(f,t,'seasonal_observed'))


def records():
    eps={'asof_date':'2020-01-02T21:00:00Z','ticker':'X','forecast_horizon':4,'eps_model_id':'test',
        'target_definition':'NATIVE_TTM','forecast_eps':2.,'forecast_share_basis':'ORIGIN','currency':'USD',
        'GAAP_flag':True,'diluted_flag':True,'TTM_flag':True,'prediction_valid':True,
        'data_quality_tier':'PARTIALLY_VERIFIED','model_track':'LOCAL_CAUSAL_RESEARCH'}
    return eps,{**eps,'expected_pe':10.}


@pytest.mark.parametrize('key,value',[('currency','KRW'),('forecast_horizon',0),('forecast_share_basis','FUTURE'),
    ('ticker','Y'),('asof_date','2020-01-03T21:00:00Z'),('TTM_flag',False),('GAAP_flag',False),('diluted_flag',False)])
def test_pe_mismatch_fails_closed(key,value):
    eps,pe=records();pe[key]=value
    result=compatibility(eps,pe);assert not result['PE_READY'] and result['implied_price_research_only'] is None


def test_positive_match_not_formal_certification():
    eps,pe=records();result=compatibility(eps,pe)
    assert result['PE_READY'] and result['implied_price_research_only']==20 and not result['strict_certified_combination']


def test_negative_eps_remains_forecast_but_price_blocked():
    eps,pe=records();eps['forecast_eps']=-2
    result=compatibility(eps,pe);assert eps['prediction_valid'] and not result['PE_READY']


def test_gate_does_not_rescue_bad_tail_with_good_mean():
    r={'coverage':1,'MAE_gain_fraction':.5,'MedianAE':1,'baseline_MedianAE':1,'p99_AE':100,
        'baseline_p99_AE':2,'worst_year_harm_ratio':1,'negative_transition_max_harm_ratio':1}
    assert 'P99_TAIL_HARM' in gate(r)
