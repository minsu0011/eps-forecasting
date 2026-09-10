"""Regression tests for real SEC fy/fp mislabels and prefix calendar repair."""
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN
from research.eps_model_lab_v1.dataset import native_period_context,calendar_index


def test_reported_metadata_cannot_change_context():
    facts=[dict(start='2021-01-01',end='2021-09-30',val=4.,fy=2020,fp='Q2'),
           dict(start='2021-07-01',end='2021-09-30',val=1.,fy=2020,fp='Q2')]
    assert native_period_context(facts,'10-Q')==(3,'2021-01-01',4.)
    for fact in facts: fact.update(fy=2099,fp='FY')
    assert native_period_context(facts,'10-Q')==(3,'2021-01-01',4.)


def test_fifty_three_week_calendar_and_incompatible_transition():
    assert calendar_index('2020-10-04','2018-09-30',2018,1)==2020*4
    assert calendar_index('2020-07-01','2018-09-30',2018,1) is None


def test_annual_does_not_turn_quarter_amount_into_annual_eps():
    direct=[dict(start='2020-10-01',end='2020-12-31',val=.89)]
    assert native_period_context(direct,'10-K') is None
    context=native_period_context(direct+[dict(start='2020-01-01',end='2020-12-31',val=1e9)],'10-K')
    assert context[:2]==(4,'2020-01-01')
    assert not any(v['start']==context[1] for v in direct)


def test_real_native_calendar_and_horizon_spacing():
    p=pd.read_parquet(RUN/'data/fiscal_panel.parquet')
    assert not p.duplicated(['ticker','fiscal_index']).any()
    assert ((p.fiscal_quarter==4)==p.form.isin(['10-K','10-KT'])).all()
    for ticker,end,quarter in [('HON','2021-09-30',3),('HON','2021-12-31',4),
                               ('DIS','2022-01-01',1),('DIS','2022-04-02',2),('DIS','2022-07-02',3),
                               ('DIS','2022-10-01',4),('GE','2020-03-31',1),('T','2021-09-30',3)]:
        selected=p[(p.ticker==ticker)&(p.end==end)]
        assert len(selected)==1 and selected.iloc[0].fiscal_quarter==quarter
    f=pd.read_parquet(RUN/'data/samples.parquet')
    for h in range(1,5):
        days=(pd.to_datetime(f[f'target_period_end_h{h}'])-pd.to_datetime(f.origin_period_end)).dt.days
        # Some fiscal calendars use 12/12/12/16 weeks, or 17 weeks in 53-week FY.
        # A 119-day Q4 is legitimate, not a missing-quarter compression.
        allowed=days.between(h*91-25,h*92+25)
        if h==1: allowed|=(f.fiscal_quarter==3)&days.between(112,119)
        assert allowed[days.notna()].all()
