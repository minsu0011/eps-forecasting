from __future__ import annotations

from collections import deque
import math
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .utility_gate import shifted_candidate_gate


LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS: tuple[str, ...] = (
    "v04_lagged_market_conditioned_raw_expected_pe",
    "v04_gated_lagged_market_conditioned_expected_pe",
    "v04_gated_lagged_market_conditioned_paired_oos_count",
    "v04_gated_lagged_market_conditioned_challenger_weight",
    "v04_gated_lagged_market_conditioned_accepted",
    "v04_gated_lagged_market_conditioned_fallback_used",
)

LOCKED_LAGGED_MARKET_CONDITIONED_CONFIG: dict[str, float | int | str] = {
    "q_over_r": 0.01,
    "clip_sigma": 3.0,
    "scale_window": 252,
    "scale_min_history": 126,
    "scale_floor": 0.01,
    "latent_weight": 0.25,
    "incumbent_weight": 0.75,
    "loss_window": 252,
    "min_history": 63,
    "improvement_margin": 0.001,
    "decisive_margin": 0.002,
    "temperature": 0.004,
    "max_challenger_weight": 1.0,
    "decision_mode": "winner_take_most",
    "availability_shift": 1,
}


def _validate_locked_config(config: Mapping[str, Any]) -> None:
    expected_keys = {"enabled", *LOCKED_LAGGED_MARKET_CONDITIONED_CONFIG}
    unknown = sorted(set(config) - expected_keys)
    missing = sorted(expected_keys - set(config))
    if unknown or missing:
        raise ValueError(
            "lagged_market_conditioned keys must match the locked candidate; "
            f"unknown={unknown}, missing={missing}"
        )
    if not isinstance(config["enabled"], bool):
        raise ValueError("lagged_market_conditioned.enabled must be boolean")

    for field, locked_value in LOCKED_LAGGED_MARKET_CONDITIONED_CONFIG.items():
        value = config[field]
        if isinstance(locked_value, int):
            valid_type = isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        elif isinstance(locked_value, float):
            valid_type = isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        else:
            valid_type = isinstance(value, str)
        if not valid_type:
            raise ValueError(
                f"lagged_market_conditioned.{field} must equal locked value {locked_value!r}"
            )
        if isinstance(locked_value, (int, float)):
            valid_value = math.isfinite(float(value)) and float(value) == float(locked_value)
        else:
            valid_value = value == locked_value
        if not valid_value:
            raise ValueError(
                f"lagged_market_conditioned.{field} must equal locked value {locked_value!r}"
            )


def _numeric_float_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    return pd.Series(array, index=values.index, name=values.name, dtype=float)


def _empty_result(index: pd.Index) -> pd.DataFrame:
    size = len(index)
    return pd.DataFrame(
        {
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[0]: np.full(size, np.nan, dtype=np.float64),
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[1]: np.full(size, np.nan, dtype=np.float64),
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[2]: np.zeros(size, dtype=np.int64),
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[3]: np.zeros(size, dtype=np.float64),
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[4]: np.zeros(size, dtype=bool),
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[5]: np.zeros(size, dtype=bool),
        },
        index=index,
    )


def _robust_scale(history: deque[float], config: Mapping[str, Any]) -> float:
    if len(history) < int(config["scale_min_history"]):
        return np.nan
    values = np.asarray(history, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return max(1.4826 * mad, float(config["scale_floor"]))


def _raw_predict_before_update(
    observed_pe: pd.Series,
    incumbent_pe: pd.Series,
    config: Mapping[str, Any],
) -> tuple[pd.Series, dict[str, int]]:
    observed = _numeric_float_series(observed_pe).to_numpy(dtype=np.float64)
    incumbent = _numeric_float_series(incumbent_pe).to_numpy(dtype=np.float64)
    raw = np.full(len(observed), np.nan, dtype=np.float64)

    state = np.nan
    covariance = 1.0
    history: deque[float] = deque(maxlen=int(config["scale_window"]))
    initialized_rows = 0
    warmup_fallback_rows = 0
    skipped_update_rows = 0
    clipped_update_rows = 0
    measurement_update_rows = 0

    q_over_r = float(config["q_over_r"])
    latent_weight = float(config["latent_weight"])
    incumbent_weight = float(config["incumbent_weight"])
    clip_sigma = float(config["clip_sigma"])

    for position, (observation, incumbent_value) in enumerate(
        zip(observed, incumbent, strict=True)
    ):
        prior_state = float(state) if math.isfinite(state) else np.nan
        incumbent_valid = math.isfinite(incumbent_value) and incumbent_value > 0.0
        if incumbent_valid:
            if math.isfinite(prior_state):
                candidate_log = (
                    incumbent_weight * math.log(incumbent_value) + latent_weight * prior_state
                )
                try:
                    candidate = math.exp(candidate_log)
                except OverflowError:
                    candidate = np.nan
                if math.isfinite(candidate) and candidate > 0.0:
                    raw[position] = candidate
            else:
                raw[position] = float(incumbent_value)
                warmup_fallback_rows += 1

        observed_valid = math.isfinite(observation) and observation > 0.0
        if not observed_valid:
            skipped_update_rows += 1
            if math.isfinite(state):
                covariance += q_over_r
            continue

        measurement = math.log(observation)
        if not math.isfinite(state):
            state = measurement
            initialized_rows += 1
            continue

        innovation = measurement - state
        scale = _robust_scale(history, config)
        used_innovation = innovation
        if math.isfinite(scale):
            limit = clip_sigma * scale
            used_innovation = min(max(innovation, -limit), limit)
            if used_innovation != innovation:
                clipped_update_rows += 1
        prior_covariance = covariance + q_over_r
        gain = prior_covariance / (prior_covariance + 1.0)
        covariance = (1.0 - gain) * prior_covariance
        state += gain * used_innovation
        history.append(innovation)
        measurement_update_rows += 1

    return pd.Series(raw, index=observed_pe.index, dtype=float), {
        "state_initialization_rows": initialized_rows,
        "warmup_incumbent_fallback_rows": warmup_fallback_rows,
        "measurement_update_rows": measurement_update_rows,
        "skipped_measurement_update_rows": skipped_update_rows,
        "clipped_measurement_update_rows": clipped_update_rows,
        "final_innovation_history_count": len(history),
    }


def lagged_market_conditioned_expected_pe(
    observed_pe: pd.Series,
    incumbent_pe: pd.Series,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a causal EOD market anchor emitted before the row observation update.

    The local-level component at row ``t`` contains observations only through ``t-1``.
    The row prediction still consumes the row-t incumbent, whose upstream features may
    be price-derived; this is deliberately not a price-free or fundamental estimator.
    """

    _validate_locked_config(config)
    if not isinstance(observed_pe, pd.Series) or not isinstance(incumbent_pe, pd.Series):
        raise TypeError("lagged market-conditioned inputs must be pandas Series")
    if len(observed_pe) != len(incumbent_pe) or not observed_pe.index.equals(incumbent_pe.index):
        raise ValueError("lagged market-conditioned inputs must share index and length")
    if not bool(config["enabled"]):
        return _empty_result(observed_pe.index), {
            "enabled": False,
            "status": "disabled",
            "production_connected": False,
            "valuation_connected": False,
            "same_row_observed_pe_consumed_for_prediction": False,
            "current_price_independence_claim_allowed": False,
        }

    raw, recurrence_diagnostics = _raw_predict_before_update(
        observed_pe,
        incumbent_pe,
        config,
    )
    gate_config = {
        key: config[key]
        for key in (
            "loss_window",
            "min_history",
            "improvement_margin",
            "decisive_margin",
            "temperature",
            "max_challenger_weight",
            "decision_mode",
        )
    }
    prefix = "v04_gated_lagged_market_conditioned"
    gate = shifted_candidate_gate(
        observed_pe,
        incumbent_pe,
        raw,
        gate_config,
        availability_shift=int(config["availability_shift"]),
        prefix=prefix,
    )

    result = pd.DataFrame(
        {
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[0]: raw,
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[1]: gate[f"{prefix}_blended"],
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[2]: gate[f"{prefix}_paired_oos_count"].astype(
                "int64"
            ),
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[3]: gate[f"{prefix}_challenger_weight"].astype(
                float
            ),
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[4]: gate[f"{prefix}_accepted"].astype(bool),
            LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[5]: gate[f"{prefix}_fallback_used"].astype(
                bool
            ),
        },
        index=observed_pe.index,
    )
    raw_values = result[LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[0]].to_numpy(dtype=np.float64)
    final_values = result[LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[1]].to_numpy(dtype=np.float64)
    if (
        np.isinf(raw_values).any()
        or np.isinf(final_values).any()
        or np.any(raw_values[np.isfinite(raw_values)] <= 0.0)
        or np.any(final_values[np.isfinite(final_values)] <= 0.0)
    ):
        raise RuntimeError("lagged market-conditioned P/E must be positive finite or NaN")

    return result, {
        "enabled": True,
        "status": "ok",
        "semantic_type": "gated_market_conditioned_one_step_prior_observed_pe_anchor",
        "eod_only": True,
        "incumbent_column": "ml_expected_pe",
        "gate_control_target": "observed_pe",
        "gate_control_target_role": "causal_online_routing_only_not_promotion_truth",
        "same_row_observed_pe_consumed_for_prediction": False,
        "observed_pe_state_update_first_available": "next_row",
        "incumbent_may_consume_same_row_price_derived_features": True,
        "current_price_independence_claim_allowed": False,
        "fundamental_or_intrinsic_fair_pe_claim_allowed": False,
        "production_connected": False,
        "valuation_connected": False,
        "accepted_rows": int(result[LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[4]].sum()),
        "fallback_rows": int(result[LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[5]].sum()),
        "soft_weight_rows": int(
            (
                result[LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[3]].gt(0.0)
                & result[LAGGED_MARKET_CONDITIONED_OUTPUT_COLUMNS[3]].lt(1.0)
            ).sum()
        ),
        "locked_parameters": {key: config[key] for key in LOCKED_LAGGED_MARKET_CONDITIONED_CONFIG},
        **recurrence_diagnostics,
    }
