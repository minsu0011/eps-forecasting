from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np
import pandas as pd


LOCKED_MARKET_CONDITIONED_DIAGNOSTIC: dict[str, float | int] = {
    "q_over_r": 0.01,
    "clip_sigma": 3.0,
    "scale_window": 252,
    "scale_min_history": 126,
    "scale_floor": 0.01,
    "latent_weight": 0.25,
}


def _validate_locked_config(config: Mapping[str, Any]) -> None:
    expected_keys = {"enabled", *LOCKED_MARKET_CONDITIONED_DIAGNOSTIC}
    unknown = sorted(set(config) - expected_keys)
    missing = sorted(expected_keys - set(config))
    if unknown or missing:
        raise ValueError(
            "market_conditioned_diagnostic keys must match the locked candidate; "
            f"unknown={unknown}, missing={missing}"
        )
    if not isinstance(config["enabled"], bool):
        raise ValueError("market_conditioned_diagnostic.enabled must be boolean")
    for field, locked_value in LOCKED_MARKET_CONDITIONED_DIAGNOSTIC.items():
        value = config[field]
        if isinstance(locked_value, int):
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, Real) and not isinstance(value, bool)
        if not valid_type or not math.isfinite(float(value)) or float(value) != float(locked_value):
            raise ValueError(
                f"market_conditioned_diagnostic.{field} must equal locked value {locked_value}"
            )


@dataclass(frozen=True)
class MarketConditionedKalmanResult:
    """Causal scalar-filter state and diagnostics in log-P/E space."""

    state_log_pe: np.ndarray
    past_innovation_scale: np.ndarray
    kalman_gain: np.ndarray
    raw_innovation: np.ndarray
    used_innovation: np.ndarray
    innovation_was_clipped: np.ndarray


def _positive_finite_log(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    valid = np.isfinite(numeric) & (numeric > 0.0)
    output = np.full(len(numeric), np.nan, dtype=np.float64)
    output[valid] = np.log(numeric[valid])
    return output


def _past_robust_scale(
    history: list[float],
    *,
    window: int,
    min_history: int,
    floor: float,
) -> float:
    if len(history) < min_history:
        return np.nan
    values = np.asarray(history[-window:], dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return max(1.4826 * mad, floor)


def filter_market_conditioned_log_pe(
    observed_log_pe: np.ndarray,
    *,
    q_over_r: float,
    clip_sigma: float,
    scale_window: int,
    scale_min_history: int,
    scale_floor: float,
) -> MarketConditionedKalmanResult:
    """Filter current observed log-P/E with the sealed local-level recurrence.

    The current observation intentionally updates the current state.  The robust
    clipping scale is estimated only from raw innovations available through t-1.
    Missing observations emit a missing current output while the latent state and
    covariance continue causally for later rows.
    """

    raw_measurements = np.asarray(observed_log_pe)
    if raw_measurements.ndim != 1:
        raise ValueError("observed_log_pe must be one-dimensional")
    if (
        not np.issubdtype(raw_measurements.dtype, np.number)
        or np.issubdtype(raw_measurements.dtype, np.bool_)
        or np.issubdtype(raw_measurements.dtype, np.complexfloating)
    ):
        raise TypeError("observed_log_pe must have a real numeric dtype")
    measurements = raw_measurements.astype(np.float64, copy=False)
    if np.isinf(measurements).any():
        raise ValueError("observed_log_pe may contain finite values or NaN, not infinity")

    for name, value in (
        ("q_over_r", q_over_r),
        ("clip_sigma", clip_sigma),
        ("scale_floor", scale_floor),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be positive and finite")
    for name, value in (
        ("scale_window", scale_window),
        ("scale_min_history", scale_min_history),
    ):
        if (
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if int(scale_min_history) > int(scale_window):
        raise ValueError("scale_min_history must be <= scale_window")

    state_output = np.full(len(measurements), np.nan, dtype=np.float64)
    scale_output = np.full(len(measurements), np.nan, dtype=np.float64)
    gain_output = np.full(len(measurements), np.nan, dtype=np.float64)
    innovation_output = np.full(len(measurements), np.nan, dtype=np.float64)
    used_innovation_output = np.full(len(measurements), np.nan, dtype=np.float64)
    clipped_output = np.zeros(len(measurements), dtype=bool)

    state = np.nan
    covariance = 1.0
    history: list[float] = []
    for position, measurement in enumerate(measurements):
        if not np.isfinite(measurement):
            if np.isfinite(state):
                covariance += float(q_over_r)
            continue
        if not np.isfinite(state):
            state = float(measurement)
            state_output[position] = state
            gain_output[position] = 1.0
            continue

        innovation = float(measurement - state)
        scale = _past_robust_scale(
            history,
            window=int(scale_window),
            min_history=int(scale_min_history),
            floor=float(scale_floor),
        )
        used_innovation = innovation
        if np.isfinite(scale):
            limit = float(clip_sigma) * scale
            used_innovation = float(np.clip(innovation, -limit, limit))
            clipped_output[position] = used_innovation != innovation

        prior_covariance = covariance + float(q_over_r)
        gain = prior_covariance / (prior_covariance + 1.0)
        covariance = (1.0 - gain) * prior_covariance
        state = float(state + gain * used_innovation)

        state_output[position] = state
        scale_output[position] = scale
        gain_output[position] = gain
        innovation_output[position] = innovation
        used_innovation_output[position] = used_innovation
        # The current raw innovation becomes scale history only for t+1 onward.
        history.append(innovation)

    return MarketConditionedKalmanResult(
        state_log_pe=state_output,
        past_innovation_scale=scale_output,
        kalman_gain=gain_output,
        raw_innovation=innovation_output,
        used_innovation=used_innovation_output,
        innovation_was_clipped=clipped_output,
    )


def market_conditioned_kalman_pe_diagnostic(
    observed_pe: pd.Series,
    incumbent_pe: pd.Series,
    config: Mapping[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    """Build the non-production ``kalman_qr0p010_clip3__blend025`` diagnostic."""

    _validate_locked_config(config)
    if not observed_pe.index.equals(incumbent_pe.index):
        raise ValueError("observed_pe and incumbent_pe indexes must match")
    enabled = bool(config["enabled"])
    output = np.full(len(observed_pe), np.nan, dtype=np.float64)
    if not enabled:
        return (
            pd.Series(output, index=observed_pe.index, dtype=float),
            {
                "enabled": False,
                "semantic_type": "market_conditioned_same_row_observed_pe_diagnostic",
                "same_row_observed_pe_consumed": False,
                "production_connected": False,
                "valid_rows": 0,
            },
        )

    observed_log = _positive_finite_log(observed_pe)
    incumbent_log = _positive_finite_log(incumbent_pe)
    filtered = filter_market_conditioned_log_pe(
        observed_log,
        q_over_r=float(config["q_over_r"]),
        clip_sigma=float(config["clip_sigma"]),
        scale_window=int(config["scale_window"]),
        scale_min_history=int(config["scale_min_history"]),
        scale_floor=float(config["scale_floor"]),
    )
    valid = np.isfinite(incumbent_log) & np.isfinite(filtered.state_log_pe)
    candidate_log = np.full(len(observed_pe), np.nan, dtype=np.float64)
    latent_weight = float(config["latent_weight"])
    candidate_log[valid] = (1.0 - latent_weight) * incumbent_log[
        valid
    ] + latent_weight * filtered.state_log_pe[valid]
    with np.errstate(over="ignore", invalid="ignore"):
        candidate_pe = np.exp(candidate_log[valid])
    candidate_valid = np.isfinite(candidate_pe) & (candidate_pe > 0.0)
    valid_positions = np.flatnonzero(valid)
    output[valid_positions[candidate_valid]] = candidate_pe[candidate_valid]

    return (
        pd.Series(output, index=observed_pe.index, dtype=float),
        {
            "enabled": True,
            "candidate_id": "kalman_qr0p010_clip3__blend025",
            "semantic_type": "market_conditioned_same_row_observed_pe_diagnostic",
            "same_row_observed_pe_consumed": True,
            "clip_scale_information_set": "raw_innovations_through_t_minus_1",
            "production_connected": False,
            "fundamental_fair_pe_claim_allowed": False,
            "q_over_r": float(config["q_over_r"]),
            "clip_sigma": float(config["clip_sigma"]),
            "scale_window": int(config["scale_window"]),
            "scale_min_history": int(config["scale_min_history"]),
            "scale_floor": float(config["scale_floor"]),
            "latent_weight": latent_weight,
            "incumbent_weight": 1.0 - latent_weight,
            "valid_rows": int(np.isfinite(output).sum()),
            "clipped_rows": int(filtered.innovation_was_clipped.sum()),
        },
    )
