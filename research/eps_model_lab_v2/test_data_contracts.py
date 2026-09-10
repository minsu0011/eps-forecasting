"""Prescore regression tests for source semantics and immutable boundaries."""
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
import pytest
from research.eps_model_lab_v2.source_authority import parse_ixbrl
from research.eps_model_lab_v2.clean_data import exact_candidate
from research.eps_model_lab_v2.common import V1, require_output


def document(value='732.3',scale='6',sign='',dimension='',unit='shares',format_name='ixt:num-dot-decimal'):
    return f'''<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>
    <xbrli:context id="c"><xbrli:entity><xbrli:identifier>0000063908</xbrli:identifier>{dimension}</xbrli:entity>
    <xbrli:period><xbrli:startDate>2023-01-01</xbrli:startDate><xbrli:endDate>2023-12-31</xbrli:endDate></xbrli:period></xbrli:context>
    <xbrli:unit id="u"><xbrli:measure>xbrli:{unit}</xbrli:measure></xbrli:unit>
    <ix:nonFraction name="us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding" contextRef="c" unitRef="u" scale="{scale}" sign="{sign}" decimals="-5" format="{format_name}">{value}</ix:nonFraction>
    </body></html>'''.encode()


def parse(payload):return parse_ixbrl(payload,'a','MCD',63908,'https://example.invalid/synthetic-test')[0]


def test_scale_from_source_not_size():
    assert parse(document())['canonical_value']==732300000
    assert parse(document(scale='0'))['canonical_value']==732.3


def test_negative_kept():assert parse(document('1.5',sign='-'))['canonical_value']==-1500000


def test_decimal_and_unit_preserved():
    r=parse(document());assert r['decimals']=='-5' and r['unitRef']=='u' and r['unit']=='shares'


def test_dimension_fails_closed():
    r=parse(document(dimension='<xbrli:segment><xbrldi:explicitMember>ClassB</xbrldi:explicitMember></xbrli:segment>'))
    assert r['status']=='QUARANTINED' and r['canonical_value'] is None


def test_unimplemented_transform_quarantined():
    assert parse(document(format_name='unknown:magic'))['status']=='QUARANTINED'


def test_missing_unit_quarantined():
    assert parse(document().replace(b'unitRef="u"',b'unitRef="missing"'))['status']=='QUARANTINED'


def test_context_entity_not_inferred():
    assert parse(document().replace(b'0000063908',b'999999'))['status']=='QUARANTINED'


def fact(start='2025-01-01',value=17127000000,filed='2025-05-02',accn='a'):
    return {'start':start,'end':'2025-03-31','val':value,'accn':accn,'filed':filed}


def select(facts,tags=('NI',)):
    us={'NI':{'units':{'USD':facts}}}
    return exact_candidate(us,list(tags),'USD','a','2025-03-31','2025-01-01','2025-05-02',True)


def test_exact_ytd_beats_earlier_ttm():
    assert select([fact('2024-04-01',65944000000),fact()])['value']==17127000000


def test_missing_exact_is_not_nearest():assert select([fact('2024-04-01')])['value'] is None


def test_conflicting_exact_not_first_value():
    assert select([fact(),fact(value=99)])['status']=='CONFLICT_QUARANTINED'


def test_future_amendment_not_selected():
    assert select([fact(),fact(value=99,filed='2026-05-02',accn='future')])['value']==17127000000


def test_same_accession_bad_future_filed_rejected():
    assert select([fact(filed='2026-05-02')])['value'] is None


def test_duplicate_same_value_allowed():assert select([fact(),fact()])['records']==2


def test_v1_write_rejected():
    with pytest.raises(RuntimeError):require_output(V1/'RUN_STATE.json')


def test_instant_no_duration():
    us={'Assets':{'units':{'USD':[{'accn':'a','end':'2025-03-31','filed':'2025-05-02','val':10}]}}}
    assert exact_candidate(us,['Assets'],'USD','a','2025-03-31',None,'2025-05-02',False)['value']==10


def test_53week_exact_duration_not_rigid_days():
    us={'NI':{'units':{'USD':[{'accn':'a','start':'2022-08-29','end':'2023-09-03','filed':'2023-10-11','val':1}]}}}
    assert exact_candidate(us,['NI'],'USD','a','2023-09-03','2022-08-29','2023-10-11',True)['value']==1
