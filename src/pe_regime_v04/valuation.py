from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _causal_rank_and_robust_z(
    values: pd.Series,
    *,
    window: int,
    min_history: int,
    low_quantile: float,
    high_quantile: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    percentile = np.full(len(array), np.nan, dtype=float)
    robust_z = np.full(len(array), np.nan, dtype=float)
    low_threshold = np.full(len(array), np.nan, dtype=float)
    high_threshold = np.full(len(array), np.nan, dtype=float)
    for i, current in enumerate(array):
        if not np.isfinite(current):
            continue
        start = max(0, i - int(window))
        history = array[start:i]  # strict past only
        history = history[np.isfinite(history)]
        if len(history) < int(min_history):
            continue
        percentile[i] = 100.0 * np.mean(history <= current)
        median = float(np.median(history))
        mad = float(np.median(np.abs(history - median)))
        robust_z[i] = (current - median) / max(1.4826 * mad, 1e-8)
        low_threshold[i] = float(np.quantile(history, low_quantile))
        high_threshold[i] = float(np.quantile(history, high_quantile))
    index = values.index
    return (
        pd.Series(percentile, index=index),
        pd.Series(robust_z, index=index),
        pd.Series(low_threshold, index=index),
        pd.Series(high_threshold, index=index),
    )


def _series(output: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in output.columns:
        numeric = pd.to_numeric(output[name], errors="coerce")
        return pd.Series(
            numeric.to_numpy(dtype=float, na_value=np.nan),
            index=output.index,
            name=name,
        )
    return pd.Series(default, index=output.index, dtype=float)


def add_adaptive_valuation(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    output = frame.copy()
    observed = _series(output, "observed_pe")
    expected = _series(output, "v04_expected_pe")
    negative = _series(output, "negative_earnings_flag", 0.0).fillna(0).astype(bool)
    meaningful = (
        np.isfinite(observed)
        & np.isfinite(expected)
        & (observed > 0)
        & (expected > 0)
        & ~negative
    )
    output["v04_expected_pe"] = expected.where(meaningful)
    gap_log = pd.Series(np.nan, index=output.index, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gap_log.loc[meaningful] = (
            np.log(observed.loc[meaningful]) - np.log(expected.loc[meaningful])
        )
    output["v04_pe_gap_log"] = gap_log
    output["v04_pe_gap_percent"] = np.expm1(gap_log)
    cheap_fraction = float(config.get("cheap_quantile", 0.15))
    expensive_fraction = float(config.get("expensive_quantile", 0.85))
    if not 0.0 < cheap_fraction < expensive_fraction < 1.0:
        raise ValueError(
            "valuation quantiles must satisfy 0 < cheap < expensive < 1"
        )
    percentile, robust_z, low_threshold, high_threshold = _causal_rank_and_robust_z(
        gap_log,
        window=int(config.get("rolling_window", 756)),
        min_history=int(config.get("min_history", 126)),
        low_quantile=cheap_fraction,
        high_quantile=expensive_fraction,
    )
    output["v04_gap_history_percentile"] = percentile
    output["v04_gap_robust_z"] = robust_z
    output["v04_gap_low_history_threshold"] = low_threshold
    output["v04_gap_high_history_threshold"] = high_threshold
    output["v04_valuation_strength"] = -robust_z

    minimum_gap = float(config.get("minimum_absolute_gap", 0.03))
    cheap_quantile = 100.0 * cheap_fraction
    expensive_quantile = 100.0 * expensive_fraction
    adaptive = np.full(len(output), "INSUFFICIENT_DATA", dtype=object)
    valid = percentile.notna() & gap_log.notna()
    cheap = valid & (percentile <= cheap_quantile) & (gap_log <= -minimum_gap)
    expensive = valid & (percentile >= expensive_quantile) & (gap_log >= minimum_gap)
    adaptive[valid] = "NORMAL_FOR_CONDITIONS"
    adaptive[cheap] = "CHEAP_TAIL_FOR_CONDITIONS"
    adaptive[expensive] = "EXPENSIVE_TAIL_FOR_CONDITIONS"

    absolute = np.full(len(output), "INSUFFICIENT_DATA", dtype=object)
    gap_percent = pd.to_numeric(output["v04_pe_gap_percent"], errors="coerce")
    valid_absolute = gap_percent.notna()
    absolute[valid_absolute] = "NORMAL_FOR_CONDITIONS"
    absolute[
        valid_absolute
        & (gap_percent <= float(config.get("absolute_cheap_gap", -0.10)))
    ] = "CHEAP_FOR_CONDITIONS"
    absolute[
        valid_absolute
        & (gap_percent >= float(config.get("absolute_expensive_gap", 0.10)))
    ] = "EXPENSIVE_FOR_CONDITIONS"
    adaptive[negative] = "NO_MEANINGFUL_PE"
    absolute[negative] = "NO_MEANINGFUL_PE"
    output["v04_valuation_state_adaptive"] = adaptive
    output["v04_valuation_state_absolute"] = absolute

    # A second state path retains precision.  The adaptive-tail state above is a
    # screening signal (higher recall); the confirmed state requires a more extreme
    # historical rank, robust deviation and absolute gap.  Downstream code can use
    # the continuous gap regardless of either categorical threshold.
    confirmed_cheap_quantile = 100.0 * float(
        config.get("confirmed_cheap_quantile", 0.10)
    )
    confirmed_expensive_quantile = 100.0 * float(
        config.get("confirmed_expensive_quantile", 0.90)
    )
    confirmed_minimum_gap = float(
        config.get("confirmed_minimum_absolute_gap", 0.05)
    )
    confirmed_robust_z = float(config.get("confirmed_robust_z", 1.0))
    confirmed = np.full(len(output), "INSUFFICIENT_DATA", dtype=object)
    confirmed[valid] = "NORMAL_FOR_CONDITIONS"
    confirmed[
        valid
        & (percentile <= confirmed_cheap_quantile)
        & (gap_log <= -confirmed_minimum_gap)
        & (robust_z <= -confirmed_robust_z)
    ] = "CHEAP_CONFIRMED_FOR_CONDITIONS"
    confirmed[
        valid
        & (percentile >= confirmed_expensive_quantile)
        & (gap_log >= confirmed_minimum_gap)
        & (robust_z >= confirmed_robust_z)
    ] = "EXPENSIVE_CONFIRMED_FOR_CONDITIONS"
    confirmed[negative] = "NO_MEANINGFUL_PE"
    output["v04_valuation_state_confirmed"] = confirmed

    eps_conf = _series(output, "eps_confidence", 0.0).fillna(0) / 100.0
    history_conf = np.minimum(
        _series(output, "pe_history_count", 0.0).fillna(0) / 504.0,
        1.0,
    )
    candidate_gain = _series(output, "v04_regime_features_incremental_oos_gain", 0.0).fillna(0)
    candidate_certainty = np.tanh(np.abs(candidate_gain) / 0.02)
    candidate_agreement = np.exp(
        -np.abs(
            np.log(
                _series(output, "v04_ml_expected_pe_no_regime")
                / _series(output, "v04_ml_expected_pe_with_regime")
            )
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    )
    confidence = (
        0.45 * eps_conf
        + 0.25 * history_conf
        + 0.15 * candidate_certainty
        + 0.15 * candidate_agreement
    ).clip(0.0, 1.0)
    confidence = confidence.where(meaningful, 0.0)
    output["v04_valuation_confidence"] = confidence.replace(
        [np.inf, -np.inf], 0.0
    )
    numeric_columns = [
        "v04_pe_gap_log",
        "v04_pe_gap_percent",
        "v04_gap_history_percentile",
        "v04_gap_robust_z",
        "v04_gap_low_history_threshold",
        "v04_gap_high_history_threshold",
        "v04_valuation_strength",
    ]
    output[numeric_columns] = output[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return output
