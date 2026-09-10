import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.eps_model_lab_v1.bootstrap import RUN
from research.eps_model_lab_v1.common import features,split_train,dataset
from research.eps_model_lab_v1.dataset import causal_fill,normalize_eps,one

def test_causal_imputation_retains_prefix():
    old=np.array([1.,2.,np.nan,4.,np.nan,6.,np.nan,8.])
    changed=old.copy();changed[5:]=1e9
    np.testing.assert_array_equal(causal_fill(old)[:5],causal_fill(changed)[:5])

def test_split_future_not_feature_and_label_reverses_basis():
    splits=pd.Series([4.],index=pd.to_datetime(['2020-08-31']))
    assert normalize_eps(8.,pd.Timestamp('2020-01-01'),pd.Timestamp('2020-08-01'),splits)==8.
    assert normalize_eps(8.,pd.Timestamp('2020-01-01'),pd.Timestamp('2020-09-01'),splits)==2.
    assert normalize_eps(2.,pd.Timestamp('2020-09-01'),pd.Timestamp('2020-08-01'),splits)==8.
    assert normalize_eps(-2.,pd.Timestamp('2020-09-01'),pd.Timestamp('2020-08-01'),splits)==-8.

def test_all_folds_purge_label_availability():
    frame=dataset()
    for year in range(2019,2027):
        for target in ['h1','h2','h3','h4','ttm']:
            train,_=split_train(frame,year,target)
            cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC')
            assert (train.asof_date<cutoff).all()
            assert (train['label_asof_'+target]<cutoff).all()

def test_allowlist_excludes_all_future_and_price_data():
    frame=dataset();a=features(frame)
    changed=frame.copy()
    for c in changed:
        if c.startswith(('y_','target_','label_')) or c=='origin_price': changed[c]=0
    pd.testing.assert_frame_equal(a,features(changed))
    assert not any(c.startswith(('y_','target_','label_')) for c in a)
    assert 'origin_price' not in a

def test_native_missing_quarter_never_becomes_label():
    frame=dataset();panel=pd.read_parquet(RUN/'data/fiscal_panel.parquet')
    lookup={(r.ticker,r.accn):r.eps for r in panel.itertuples()}
    for h in range(1,5):
        for r in frame.itertuples():
            acc=getattr(r,f'target_accession_h{h}')
            if acc is not None and not np.isfinite(lookup.get((r.ticker,acc),np.nan)):
                assert not np.isfinite(getattr(r,f'y_h{h}'))
    assert (frame.y_h1<0).any()

def test_future_sec_mutation_does_not_change_past_features():
    original=json.loads((RUN/'data/raw/sec/AAPL/companyfacts.json').read_bytes())
    mutated=copy.deepcopy(original)
    for namespace in mutated['facts'].values():
        for tag in namespace.values():
            for values in tag.get('units',{}).values():
                for fact in values:
                    if fact.get('filed','')>='2022-01-01': fact['val']=float(fact['val'])*37-123
    before,_,_=one('AAPL',original);after,_,_=one('AAPL',mutated)
    before=before[before.asof_date<pd.Timestamp('2022-01-01',tz='UTC')].reset_index(drop=True)
    after=after[after.asof_date<pd.Timestamp('2022-01-01',tz='UTC')].reset_index(drop=True)
    pd.testing.assert_series_equal(before.sample_id,after.sample_id)
    pd.testing.assert_frame_equal(features(before),features(after),check_exact=True)
    pd.testing.assert_series_equal(before.current_ttm,after.current_ttm,check_exact=True)
