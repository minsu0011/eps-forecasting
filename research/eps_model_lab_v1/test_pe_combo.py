import numpy as np
import pytest
from research.eps_model_lab_v1.static_pe_combo import gate


def valid():
    return dict(target_key='ttm',forecast_horizon=4,currency='USD',pe_currency='USD',eps_definition='GAAP_DILUTED',
       pe_eps_definition='GAAP_DILUTED_TTM',share_basis='FORECAST_ORIGIN',pe_price_basis='contemporaneous',
       pe_date='2024-01-01',asof_date='2024-01-01T21:00:00Z',predicted_eps=4.,expected_pe=20.,current_ttm=3.,pe_current_ttm=3.,
       pe_contract_failure='',basis_vintage_certified=False,eps_pretraining_pit_valid=True)


def test_research_scenario_does_not_claim_certification():
    result=gate(valid());assert result['research_scenario_valid'] and not result['price_combo_valid']
    assert result['implied_price_origin_share_basis']==80.
    assert 'UNCERTIFIED' in result['strict_invalid_reason']


@pytest.mark.parametrize('field,value',[('target_key','h1'),('forecast_horizon',1),('currency','EUR'),
 ('eps_definition','BASIC'),('share_basis','CURRENT_2026'),('pe_date','2024-01-02'),('predicted_eps',0.),
 ('predicted_eps',-1.),('predicted_eps',np.nan),('expected_pe',np.nan),('expected_pe',-20.),('pe_current_ttm',1.5),
 ('pe_contract_failure','fixed frozen policy rejected'),('prediction_valid',False)])
def test_incompatible_combinations_fail_closed(field,value):
    row=valid();row[field]=value;result=gate(row)
    assert not result['research_scenario_valid'] and not result['price_combo_valid']
    assert np.isnan(result['implied_price_origin_share_basis']) and result['research_invalid_reason']
