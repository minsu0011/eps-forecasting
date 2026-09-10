import pytest
from research.eps_model_lab_v1.resource_summary import summarize


def test_gap_is_not_invented_idle_time():
    rows = [{'utc': '2026-01-01T00:00:00+00:00', 'cpu_percent': 10},
            {'utc': '2026-01-01T00:00:30+00:00', 'cpu_percent': 40},
            {'utc': '2026-01-01T00:03:30+00:00', 'cpu_percent': 90}]
    out = summarize(rows)
    assert out['long_gaps_over_60s'] == 1
    assert out['observed_interval_hours_capped_60s'] == pytest.approx(90/3600)
    assert out['metrics']['cpu_percent']['interval_weighted_mean'] == 30
    assert out['metrics']['cpu_percent']['max'] == 90


def test_time_order_fails_closed():
    row = {'utc': '2026-01-01T00:00:00+00:00', 'cpu_percent': 10}
    with pytest.raises(ValueError): summarize([row, row])
