"""Frozen V2 dataset, causal feature construction and prediction contracts."""
from pathlib import Path
import numpy as np
import pandas as pd
from research.eps_model_lab_v2.common import RUN,sha,read_json,check_stop

TARGETS=['h1','h2','h3','h4','ttm']


def dataset():
    check_stop()
    manifest=read_json(RUN/'EPS_DATASET_MANIFEST_V2.json')
    path=RUN/'data/samples_v2.parquet'
    hashes={k.replace('\\','/'):v for k,v in manifest['files'].items()}
    if sha(path)!=hashes['data/samples_v2.parquet']:raise RuntimeError('Frozen V2 data changed')
    return pd.read_parquet(path)


def baseline(frame,target,kind='persistence_observed'):
    history=frame[[f'ttm_lag_{j}' for j in range(7,-1,-1)]].to_numpy() if target=='ttm' else frame[[f'eps_lag_{j}' for j in range(31,-1,-1)]].to_numpy()
    horizon=4 if target=='ttm' else int(target[1:]);out=[]
    for i,row in enumerate(history):
        positions=np.flatnonzero(np.isfinite(row))
        fallback=float(frame.iloc[i].last_observed_eps)*(4 if target=='ttm' else 1)
        last=float(row[positions[-1]]) if len(positions) else fallback
        seasonal=row[-4+(horizon-1)%4]
        if kind in ['seasonal_observed','seasonal_random_walk']:
            p=float(seasonal) if np.isfinite(seasonal) else last
        elif kind=='simple_drift' and len(positions)>1:
            p=last+horizon*(last-row[positions[0]])/max(positions[-1]-positions[0],1)
        else:p=last
        out.append(p)
    return np.asarray(out)


def native_features(frame):
    eps=frame[[f'eps_lag_{j}' for j in range(32)]].to_numpy(dtype=float)
    ttm=frame[[f'ttm_lag_{j}' for j in range(8)]].to_numpy(dtype=float)
    scale=frame.scale.to_numpy(dtype=float)
    raw=np.column_stack([eps/scale[:,None],ttm/scale[:,None]])
    observed=np.isfinite(raw).astype(float)
    ancillary=frame[['eps_growth','eps_acceleration','eps_volatility']].to_numpy()/scale[:,None]
    fixed=frame[['eps_negative','fiscal_sin','fiscal_cos','history_observed','eps_staleness_quarters']].to_numpy()
    return np.column_stack([raw,observed,ancillary,fixed])


def training_rows(frame,year,target,mode):
    cutoff=pd.Timestamp(f'{year}-01-01',tz='UTC')
    mask=(frame.asof_date<cutoff)&frame['y_'+target].notna()&(frame['label_asof_'+target]<cutoff)
    if mode=='rolling':mask &= frame.asof_date>=cutoff-pd.DateOffset(years=5)
    selected=frame.loc[mask]
    weights=np.ones(len(selected))
    if mode=='recency_weighted':weights=np.exp2(-(cutoff-selected.asof_date).dt.total_seconds().to_numpy()/(2*365.2425*86400))
    return selected,weights,cutoff


def predictions(frame,target,model_id,point,quantiles=None,track='LOCAL_CAUSAL_RESEARCH',feature_track='N'):
    if len(point)!=len(frame) or not np.isfinite(point).all():raise ValueError('Full finite prediction coverage required')
    out=frame[['sample_id','ticker','asof_date','fiscal_index','origin_price','last_observed_eps','current_ttm',
        'data_quality_tier','current_ttm_method','current_ttm_approximate','ttm_target_method','ttm_target_approximate']].copy()
    out['target']=target;out['eps_model_id']=model_id;out['forecast_eps']=point
    out['actual_eps']=frame['y_'+target].to_numpy();out['label_asof']=frame['label_asof_'+target].to_numpy()
    out['forecast_horizon']=4 if target=='ttm' else int(target[1:])
    out['target_definition']='NATIVE_LEDGER_TTM_AT_Q_PLUS_4_WITH_METHOD_FLAGS' if target=='ttm' else 'DIRECT_NATIVE_GAAP_DILUTED_QUARTER_EPS'
    out['forecast_share_basis']='FORECAST_ORIGIN_RETROSPECTIVE_ACTION_EVIDENCE';out['currency']='USD'
    out['GAAP_flag']=True;out['diluted_flag']=True;out['TTM_flag']=target=='ttm'
    out['prediction_valid']=True;out['feature_track']=feature_track;out['model_track']=track
    out['formal_certified']=False;out['baseline_persistence']=baseline(frame,target)
    out['point_baseline_fallback_used']=~np.isfinite(frame.current_ttm.to_numpy()) if target=='ttm' else False
    if quantiles is not None:
        if quantiles.shape!=(len(out),3) or not np.isfinite(quantiles).all():raise ValueError('Quantile geometry invalid')
        for j,q in enumerate(['q10','q50','q90']):out[q]=quantiles[:,j]
    return out


def surface(table,name):
    year=table.asof_date.dt.year
    if name=='development':return table[(year>=2015)&(year<=2018)&(table.label_asof<pd.Timestamp('2019-01-01',tz='UTC'))]
    if name=='confirmation':return table[(year>=2019)&(year<=2021)&(table.label_asof<pd.Timestamp('2022-01-01',tz='UTC'))]
    if name=='monitor':return table[year>=2022]
    raise KeyError(name)


def point_metrics(table):
    observed=table.actual_eps.notna()
    t=table[observed];valid=t.forecast_eps.notna()&np.isfinite(t.forecast_eps)
    t=t[valid]
    if not len(t):return {'truth_rows':int(observed.sum()),'predicted_rows':0,'coverage':0.0}
    error=t.forecast_eps.to_numpy()-t.actual_eps.to_numpy();absolute=np.abs(error)
    base_error=np.abs(t.baseline_persistence.to_numpy()-t.actual_eps.to_numpy())
    current=np.where(t.target.eq('ttm'),t.current_ttm,t.last_observed_eps)
    direction=np.sign(t.forecast_eps.to_numpy()-current)==np.sign(t.actual_eps.to_numpy()-current)
    prices=t.origin_price.to_numpy();pvalid=np.isfinite(prices)&(prices>0)
    result={'truth_rows':int(observed.sum()),'predicted_rows':len(t),'coverage':float(len(t)/observed.sum()),
        'MAE':float(absolute.mean()),'RMSE':float(np.sqrt((error**2).mean())),
        'MedianAE':float(np.median(absolute)),'bias':float(error.mean()),
        'price_scaled_MAE':float((absolute[pvalid]/prices[pvalid]).mean()) if pvalid.any() else None,
        'direction_accuracy':float(direction[np.isfinite(current)].mean()) if np.isfinite(current).any() else None,
        'p90_AE':float(np.quantile(absolute,.9)),'p95_AE':float(np.quantile(absolute,.95)),'p99_AE':float(np.quantile(absolute,.99)),
        'baseline_MAE':float(base_error.mean()),'baseline_MedianAE':float(np.median(base_error)),
        'baseline_p99_AE':float(np.quantile(base_error,.99)),
        'MAE_gain_fraction':float(1-absolute.mean()/base_error.mean()) if base_error.mean()>0 else None,
        'ticker_macro_MAE':float(pd.DataFrame({'ticker':t.ticker.to_numpy(),'absolute':absolute}).groupby('ticker').absolute.mean().mean())}
    if {'q10','q50','q90'}<=set(t):
        result['interval80_coverage']=float(((t.actual_eps>=t.q10)&(t.actual_eps<=t.q90)).mean())
        result['interval80_mean_width']=float((t.q90-t.q10).mean())
        losses=[]
        for q,column in [(.1,'q10'),(.5,'q50'),(.9,'q90')]:
            residual=t.actual_eps.to_numpy()-t[column].to_numpy();losses.append(np.maximum(q*residual,(q-1)*residual).mean())
        result['mean_pinball_loss']=float(np.mean(losses))
    return result
