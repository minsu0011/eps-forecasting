import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, laplace
from research.eps_model_lab_v1.native_crps import crps


@pytest.mark.parametrize('family,distribution', [('normal', norm), ('laplace', laplace)])
@pytest.mark.parametrize('truth', [-20., -1.7, 0., 2.1, 30.])
def test_analytic_crps_matches_independent_cdf_integral(family, distribution, truth):
    location = -1.7; scale = 2.3
    cdf = distribution(loc=location, scale=scale).cdf
    expected = quad(lambda x: cdf(x)**2, -np.inf, truth, epsabs=1e-9)[0]
    expected += quad(lambda x: (1-cdf(x))**2, truth, np.inf, epsabs=1e-9)[0]
    np.testing.assert_allclose(crps(truth, location, scale, family), expected, rtol=1e-8, atol=1e-8)


def test_scale_and_location_equivariance():
    for family in ['normal', 'laplace']:
        a = crps(np.array([-4., 0., 3.]), 1., 2., family)
        b = crps(np.array([-4., 0., 3.])*7-5, 2., 14., family)
        np.testing.assert_allclose(b, a*7)


def test_unknown_family_and_nonpositive_scale_rejected():
    with pytest.raises(ValueError): crps(0., 0., 1., 'three_quantiles_unknown')
    with pytest.raises(ValueError): crps(0., 0., 0., 'normal')
