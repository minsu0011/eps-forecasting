from copy import deepcopy
import pytest
from research.eps_model_lab_v1.next_wave_accounting_selector import exact_context_fact


def choose(data,accepted='2025-05-02T18:00:00Z'):
    return exact_context_fact(data,['Primary','Fallback'],'USD','a','2025-03-31','2025-01-01',
        accepted,'2025-05-02T20:00:00Z',True)


def fact(start='2025-01-01',value=17,filed='2025-05-02'):
    return {'start':start,'end':'2025-03-31','accn':'a','val':value,'filed':filed}


def data(rows):return {'Primary':{'units':{'USD':rows}}}


def test_trailing_twelve_month_is_not_ytd():
    assert choose(data([fact('2024-04-01',66),fact()]))['value']==17


def test_no_matching_period_stays_missing():
    assert choose(data([fact('2024-04-01',66)]))['status']=='NO_EXACT_CONTEXT_KEEP_MISSING'


def test_fallback_tag_only_after_matching_context_filter():
    d=data([fact('2024-04-01',66)]);d['Fallback']={'units':{'USD':[fact(value=18)]}}
    assert choose(d)['value']==18


def test_conflicting_primary_does_not_cherry_pick_fallback():
    d=data([fact(),fact(value=19)]);d['Fallback']={'units':{'USD':[fact()]}}
    assert choose(d)['status']=='CONFLICTING_EXACT_CONTEXT_FAIL_CLOSED'


def test_duplicate_identical_records_do_not_create_conflict():
    assert choose(data([fact(),fact()]))['value']==17


def test_future_record_mutation_cannot_change_origin_result():
    original=data([fact()]);changed=deepcopy(original)
    changed['Primary']['units']['USD'].append(fact(value=1e12,filed='2026-01-01'))
    assert choose(original)==choose(changed)


def test_unknown_accession_is_not_available():
    assert choose(data([fact()]),accepted='2025-05-03T18:00:00Z')['value'] is None


def test_no_unit_or_historical_vintage_certification_from_context_match():
    result=choose(data([fact()]));assert not result['unit_scale_verified'] and not result['source_vintage_verified']


def test_naive_acceptance_timestamp_rejected():
    with pytest.raises(ValueError):choose(data([fact()]),accepted='2025-05-02')
