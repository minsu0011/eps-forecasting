import pytest
from research.eps_model_lab_v1.year_sign_sensitivity import enumerate_signs


def test_three_year_exact_resolution():
    r=enumerate_signs([-1,-2,-3]);assert r['enumerated_patterns']==8
    assert r['conditional_one_sided_tail_fraction']==.125
    assert r['conditional_two_sided_tail_fraction']==.25


def test_two_year_exact_resolution():
    r=enumerate_signs([-1,-2]);assert r['conditional_one_sided_tail_fraction']==.25
    assert r['conditional_two_sided_tail_fraction']==.5


def test_zero_differences_do_not_claim_improvement():
    r=enumerate_signs([0,0,0]);assert r['conditional_one_sided_tail_fraction']==1


def test_invalid_and_unbounded_enumeration_rejected():
    for values in [[],[float('nan')],[1]*16]:
        with pytest.raises(ValueError):enumerate_signs(values)
