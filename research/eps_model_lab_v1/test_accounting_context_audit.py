import numpy as np
from research.eps_model_lab_v1.accounting_context_audit import unit_ratio


def test_ratio_is_dimensionless_triage_not_rescaling():
    assert unit_ratio(1000,100,10)==1
    assert unit_ratio(1e9,100,10)==1e6


def test_missing_zero_small_eps_do_not_generate_infinite_identity():
    for args in [(100,0,2),(100,-2,2),(100,2,0),(np.nan,2,1),(100,2,.01)]:
        assert np.isnan(unit_ratio(*args))


def test_negative_eps_and_income_are_retained():
    assert unit_ratio(-1000,100,-10)==1
    assert unit_ratio(-1000,100,10)==-1
