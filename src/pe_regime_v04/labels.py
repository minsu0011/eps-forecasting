from __future__ import annotations

import numpy as np
import pandas as pd


def future_return_proxy_labels(
    benchmark_close: pd.Series,
    *,
    horizon: int,
    bull_threshold: float,
    bear_threshold: float,
) -> pd.Series:
    close = pd.to_numeric(benchmark_close, errors="coerce")
    forward_return = close.shift(-int(horizon)) / close - 1.0
    labels = pd.Series(np.nan, index=close.index, dtype=float)
    labels.loc[forward_return >= float(bull_threshold)] = 2
    labels.loc[forward_return <= float(bear_threshold)] = 0
    sideways = (forward_return > float(bear_threshold)) & (
        forward_return < float(bull_threshold)
    )
    labels.loc[sideways] = 1
    return labels


def probability_log_loss_rows(
    probabilities: pd.DataFrame,
    labels: pd.Series,
) -> pd.Series:
    p = probabilities[["p_bear", "p_sideways", "p_bull"]].to_numpy(dtype=float)
    y = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=float)
    output = np.full(len(probabilities), np.nan, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p).all(axis=1) & (p.sum(axis=1) > 0)
    if valid.any():
        pp = np.clip(p[valid], 1e-8, np.inf)
        pp = pp / pp.sum(axis=1, keepdims=True)
        yy = y[valid].astype(int)
        output[valid] = -np.log(pp[np.arange(len(yy)), yy])
    return pd.Series(output, index=probabilities.index)
