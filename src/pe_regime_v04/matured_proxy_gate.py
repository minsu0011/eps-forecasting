from __future__ import annotations

from numbers import Real
from typing import Any, Mapping

import numpy as np
import pandas as pd


LOCKED_MATURED_PROXY_GATE_CONFIG: dict[str, Any] = {
    "target_horizon_sessions": 21,
    "target_method": "forward_log_median",
    "maturity_shift_sessions": 1,
    "loss_window": 252,
    "min_history": 63,
    "improvement_margin": 0.0,
    "temperature": 0.004,
    "max_challenger_weight": 0.50,
    "blend_method": "log_geometric",
}

MATURED_PROXY_GATE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "v04_matured_proxy_regularized_expected_pe",
    "v04_matured_proxy_baseline_oos_log_mae",
    "v04_matured_proxy_challenger_oos_log_mae",
    "v04_matured_proxy_paired_oos_count",
    "v04_matured_proxy_oos_gain",
    "v04_matured_proxy_challenger_weight",
    "v04_matured_proxy_accepted",
    "v04_matured_proxy_fallback_used",
)


def _numeric_series(values: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    return pd.Series(array, index=values.index, name=values.name, dtype=float)


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("matured_proxy_gate config must be a mapping")
    expected_keys = {"enabled", *LOCKED_MATURED_PROXY_GATE_CONFIG}
    unknown = sorted(set(config) - expected_keys)
    missing = sorted(expected_keys - set(config))
    if unknown or missing:
        raise ValueError(
            f"matured_proxy_gate config keys must be exact: missing={missing}, unknown={unknown}"
        )
    if not isinstance(config["enabled"], bool):
        raise ValueError("matured_proxy_gate.enabled must be boolean")
    for key, locked in LOCKED_MATURED_PROXY_GATE_CONFIG.items():
        actual = config[key]
        if isinstance(locked, int):
            valid = not isinstance(actual, bool) and isinstance(actual, int) and actual == locked
        elif isinstance(locked, float):
            valid = (
                not isinstance(actual, bool)
                and isinstance(actual, Real)
                and np.isfinite(float(actual))
                and float(actual) == locked
            )
        else:
            valid = isinstance(actual, str) and actual == locked
        if not valid:
            raise ValueError(f"matured_proxy_gate.{key} must equal locked value {locked!r}")
    return dict(config)


def _disabled_output(index: pd.Index) -> tuple[pd.DataFrame, dict[str, Any]]:
    size = len(index)
    output = pd.DataFrame(index=index)
    output[MATURED_PROXY_GATE_OUTPUT_COLUMNS[0]] = np.full(size, np.nan, dtype=float)
    output[MATURED_PROXY_GATE_OUTPUT_COLUMNS[1]] = np.full(size, np.nan, dtype=float)
    output[MATURED_PROXY_GATE_OUTPUT_COLUMNS[2]] = np.full(size, np.nan, dtype=float)
    output[MATURED_PROXY_GATE_OUTPUT_COLUMNS[3]] = np.zeros(size, dtype=np.int64)
    output[MATURED_PROXY_GATE_OUTPUT_COLUMNS[4]] = np.full(size, np.nan, dtype=float)
    output[MATURED_PROXY_GATE_OUTPUT_COLUMNS[5]] = np.zeros(size, dtype=float)
    output[MATURED_PROXY_GATE_OUTPUT_COLUMNS[6]] = np.zeros(size, dtype=bool)
    output[MATURED_PROXY_GATE_OUTPUT_COLUMNS[7]] = np.zeros(size, dtype=bool)
    return output, {
        "status": "unavailable_disabled",
        "enabled": False,
        "target_horizon_sessions": 21,
        "routing_estimand": ("fully_matured_21_session_forward_median_of_observed_log_pe"),
        "promotion_estimand_not_claimed": "contemporaneous_latent_or_intrinsic_fair_pe",
        "fundamental_or_intrinsic_fair_pe_claim_allowed": False,
        "latest_target_observation_lag_sessions": 1,
        "current_row_target_consumed": False,
        "true_fair_pe_consumed": False,
        "main_expected_pe_path_connected": False,
    }


def _validate_single_symbol(symbols: pd.Series | None, index: pd.Index) -> str:
    if not isinstance(symbols, pd.Series):
        raise ValueError("enabled matured_proxy_gate requires a symbol Series")
    if not symbols.index.equals(index):
        raise ValueError("matured_proxy_gate symbol index must exactly match inputs")
    normalized = symbols.astype("string").str.strip()
    if normalized.isna().any() or normalized.eq("").any():
        raise ValueError("enabled matured_proxy_gate requires non-empty symbols")
    unique = normalized.unique()
    if len(unique) != 1:
        raise ValueError("enabled matured_proxy_gate requires exactly one symbol timeline")
    return str(unique[0])


def _forward_log_median_target(target: np.ndarray, horizon: int) -> np.ndarray:
    """Return an origin-indexed future median that is never used before maturity."""

    result = np.full(len(target), np.nan, dtype=np.float64)
    if len(target) <= horizon:
        return result
    windows = np.lib.stride_tricks.sliding_window_view(target, horizon)
    future_windows = windows[1:]
    complete = np.isfinite(future_windows).all(axis=1)
    if complete.any():
        medians = np.full(len(future_windows), np.nan, dtype=np.float64)
        medians[complete] = np.median(future_windows[complete], axis=1)
        result[: len(medians)] = medians
    return result


def causal_matured_forward_median_gate(
    observed_pe: pd.Series,
    baseline_pe: pd.Series,
    challenger_pe: pd.Series,
    config: Mapping[str, Any],
    *,
    symbols: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Blend toward a challenger using only fully matured forward-multiple losses.

    An error belonging to origin ``s`` compares its origin-time predictions with
    the median of ``log(observed_pe[s+1:s+22])``.  At decision row ``t`` the most
    recent admissible origin is ``s=t-22``; consequently the latest observation
    consumed by the weight is row ``t-1``.  The current row and all future rows are
    excluded from the routing decision.
    """

    locked = _validate_config(config)
    observed = _numeric_series(observed_pe, name="observed_pe")
    baseline = _numeric_series(baseline_pe, name="baseline_pe")
    challenger = _numeric_series(challenger_pe, name="challenger_pe")
    if not observed.index.equals(baseline.index) or not observed.index.equals(challenger.index):
        raise ValueError("matured_proxy_gate inputs must have exactly equal indices")
    if not locked["enabled"]:
        return _disabled_output(observed.index)
    symbol = _validate_single_symbol(symbols, observed.index)

    observed_values = observed.to_numpy(dtype=np.float64)
    baseline_values = baseline.to_numpy(dtype=np.float64)
    challenger_values = challenger.to_numpy(dtype=np.float64)
    observed_valid = np.isfinite(observed_values) & (observed_values > 0.0)
    baseline_valid = np.isfinite(baseline_values) & (baseline_values > 0.0)
    challenger_valid = np.isfinite(challenger_values) & (challenger_values > 0.0)

    observed_log = np.full(len(observed), np.nan, dtype=np.float64)
    baseline_log = np.full(len(observed), np.nan, dtype=np.float64)
    challenger_log = np.full(len(observed), np.nan, dtype=np.float64)
    observed_log[observed_valid] = np.log(observed_values[observed_valid])
    baseline_log[baseline_valid] = np.log(baseline_values[baseline_valid])
    challenger_log[challenger_valid] = np.log(challenger_values[challenger_valid])

    horizon = int(locked["target_horizon_sessions"])
    maturity_shift = int(locked["maturity_shift_sessions"])
    availability_shift = horizon + maturity_shift
    target_log = _forward_log_median_target(observed_log, horizon)
    paired = observed_valid & np.isfinite(target_log) & baseline_valid & challenger_valid
    baseline_error = np.where(paired, np.abs(target_log - baseline_log), np.nan)
    challenger_error = np.where(paired, np.abs(target_log - challenger_log), np.nan)

    index = observed.index
    shifted_pair = pd.Series(paired.astype(np.float64), index=index).shift(
        availability_shift, fill_value=0.0
    )
    shifted_baseline_error = pd.Series(baseline_error, index=index).shift(availability_shift)
    shifted_challenger_error = pd.Series(challenger_error, index=index).shift(availability_shift)
    window = int(locked["loss_window"])
    min_history = int(locked["min_history"])
    paired_count = shifted_pair.rolling(window, min_periods=1).sum().astype(np.int64)
    mature = paired_count.ge(min_history)
    baseline_loss = shifted_baseline_error.rolling(window, min_periods=1).mean().where(mature)
    challenger_loss = shifted_challenger_error.rolling(window, min_periods=1).mean().where(mature)
    gain = (baseline_loss - challenger_loss).where(
        np.isfinite(baseline_loss.to_numpy(dtype=np.float64))
        & np.isfinite(challenger_loss.to_numpy(dtype=np.float64))
    )

    margin = float(locked["improvement_margin"])
    temperature = float(locked["temperature"])
    max_weight = float(locked["max_challenger_weight"])
    current_pair = observed_valid & baseline_valid & challenger_valid
    accepted = (
        mature.to_numpy(dtype=bool)
        & np.isfinite(gain.to_numpy(dtype=np.float64))
        & (gain.to_numpy(dtype=np.float64) > margin)
        & current_pair
    )
    excess = np.where(accepted, gain.to_numpy(dtype=np.float64) - margin, 0.0)
    weight = max_weight * (-np.expm1(-excess / temperature))
    weight = np.where(accepted & np.isfinite(weight), np.clip(weight, 0.0, max_weight), 0.0)
    accepted = weight > 0.0

    blended = np.full(len(observed), np.nan, dtype=np.float64)
    valid_anchor = observed_valid & baseline_valid
    blended[valid_anchor] = baseline_values[valid_anchor]
    both = observed_valid & baseline_valid & challenger_valid & accepted
    blended[both] = np.exp(
        (1.0 - weight[both]) * baseline_log[both] + weight[both] * challenger_log[both]
    )
    blended[~np.isfinite(blended) | (blended <= 0.0)] = np.nan
    fallback_used = observed_valid & baseline_valid & ~challenger_valid

    output = pd.DataFrame(
        {
            MATURED_PROXY_GATE_OUTPUT_COLUMNS[0]: blended,
            MATURED_PROXY_GATE_OUTPUT_COLUMNS[1]: baseline_loss,
            MATURED_PROXY_GATE_OUTPUT_COLUMNS[2]: challenger_loss,
            MATURED_PROXY_GATE_OUTPUT_COLUMNS[3]: paired_count.to_numpy(dtype=np.int64),
            MATURED_PROXY_GATE_OUTPUT_COLUMNS[4]: gain,
            MATURED_PROXY_GATE_OUTPUT_COLUMNS[5]: weight,
            MATURED_PROXY_GATE_OUTPUT_COLUMNS[6]: accepted,
            MATURED_PROXY_GATE_OUTPUT_COLUMNS[7]: fallback_used,
        },
        index=index,
    )
    diagnostics = {
        "status": "ok",
        "enabled": True,
        "target_definition": "median(log(observed_pe[s+1:s+22]))",
        "routing_estimand": ("fully_matured_21_session_forward_median_of_observed_log_pe"),
        "promotion_estimand_not_claimed": "contemporaneous_latent_or_intrinsic_fair_pe",
        "fundamental_or_intrinsic_fair_pe_claim_allowed": False,
        "target_horizon_sessions": horizon,
        "maturity_shift_sessions": maturity_shift,
        "availability_shift_sessions": availability_shift,
        "latest_target_observation_lag_sessions": 1,
        "loss_window": window,
        "min_history": min_history,
        "temperature": temperature,
        "max_challenger_weight": max_weight,
        "paired_error_origins": int(paired.sum()),
        "mature_decision_rows": int(mature.sum()),
        "accepted_rows": int(accepted.sum()),
        "fallback_rows": int(fallback_used.sum()),
        "current_row_target_consumed": False,
        "true_fair_pe_consumed": False,
        "baseline_anchor_required": True,
        "main_expected_pe_path_connected": False,
        "valuation_path_connected": False,
        "publication_timing": "parallel_eod_research_only",
        "single_symbol_timeline_validated": True,
        "symbol": symbol,
    }
    return output, diagnostics
