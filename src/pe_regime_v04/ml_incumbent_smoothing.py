from __future__ import annotations

from collections import deque
import math
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np
import pandas as pd


ML_INCUMBENT_WEEKLY_MEDIAN_SHRINKAGE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "v04_ml_weekly_median_shrinkage_expected_pe",
    "v04_ml_weekly_median_shrinkage_window_count",
    "v04_ml_weekly_median_shrinkage_fallback_used",
)

LOCKED_ML_INCUMBENT_WEEKLY_MEDIAN_SHRINKAGE_CONFIG: dict[str, float | int | str] = {
    "window_sessions": 5,
    "incumbent_weight": 0.75,
    "anchor_weight": 0.25,
    "method": "log_median",
}


def _validate_locked_config(config: Mapping[str, Any]) -> None:
    expected_keys = {"enabled", *LOCKED_ML_INCUMBENT_WEEKLY_MEDIAN_SHRINKAGE_CONFIG}
    unknown = sorted(set(config) - expected_keys)
    missing = sorted(expected_keys - set(config))
    if unknown or missing:
        raise ValueError(
            "ml_incumbent_weekly_median_shrinkage keys must match the locked candidate; "
            f"unknown={unknown}, missing={missing}"
        )
    if not isinstance(config["enabled"], bool):
        raise ValueError("ml_incumbent_weekly_median_shrinkage.enabled must be boolean")

    for field, locked_value in LOCKED_ML_INCUMBENT_WEEKLY_MEDIAN_SHRINKAGE_CONFIG.items():
        value = config[field]
        if isinstance(locked_value, int):
            valid_type = isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        elif isinstance(locked_value, float):
            valid_type = isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        else:
            valid_type = isinstance(value, str)
        if not valid_type:
            raise ValueError(
                "ml_incumbent_weekly_median_shrinkage."
                f"{field} must equal locked value {locked_value!r}"
            )
        if isinstance(locked_value, (int, float)):
            valid_value = math.isfinite(float(value)) and float(value) == float(locked_value)
        else:
            valid_value = value == locked_value
        if not valid_value:
            raise ValueError(
                "ml_incumbent_weekly_median_shrinkage."
                f"{field} must equal locked value {locked_value!r}"
            )


def _require_aligned_series(
    values: pd.Series,
    reference: pd.Series,
    *,
    name: str,
) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not values.index.equals(reference.index):
        raise ValueError(f"{name} index must exactly match incumbent_pe")


def _empty_result(index: pd.Index) -> pd.DataFrame:
    size = len(index)
    return pd.DataFrame(
        {
            ML_INCUMBENT_WEEKLY_MEDIAN_SHRINKAGE_OUTPUT_COLUMNS[0]: np.full(
                size, np.nan, dtype=np.float64
            ),
            ML_INCUMBENT_WEEKLY_MEDIAN_SHRINKAGE_OUTPUT_COLUMNS[1]: np.zeros(size, dtype=np.int64),
            ML_INCUMBENT_WEEKLY_MEDIAN_SHRINKAGE_OUTPUT_COLUMNS[2]: np.zeros(size, dtype=bool),
        },
        index=index,
    )


def causal_ml_incumbent_weekly_median_shrinkage(
    incumbent_pe: pd.Series,
    dates: pd.Series,
    symbols: pd.Series | None,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the fixed causal log-median shrinkage operator to a v0.3 ML incumbent.

    A session means one adjacent input row, not one calendar day. The current row
    is included, so this is an end-of-day stabilizer after ``ml_expected_pe_t`` is
    available rather than a pre-close forecast. A symbol transition defensively
    resets the local window; the overlay's globally unique/increasing date contract
    is still a single-timeline API and does not claim general panel support.
    """

    _validate_locked_config(config)
    if not isinstance(incumbent_pe, pd.Series):
        raise TypeError("incumbent_pe must be a pandas Series")
    _require_aligned_series(dates, incumbent_pe, name="dates")
    if not incumbent_pe.index.is_unique:
        raise ValueError("incumbent_pe index must be unique")

    result = _empty_result(incumbent_pe.index)
    if not bool(config["enabled"]):
        return result, {
            "enabled": False,
            "status": "disabled",
            "production_connected": False,
            "valuation_connected": False,
        }

    if symbols is None:
        raise ValueError("enabled ml incumbent smoothing requires a symbol column")
    _require_aligned_series(symbols, incumbent_pe, name="symbols")

    parsed_dates = pd.to_datetime(dates, errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("ml incumbent smoothing dates contain missing or invalid values")
    if parsed_dates.duplicated().any():
        raise ValueError("ml incumbent smoothing dates must be globally unique")
    if not parsed_dates.is_monotonic_increasing:
        raise ValueError("ml incumbent smoothing dates must be monotonically increasing")

    symbol_missing = symbols.isna().to_numpy(dtype=bool)
    symbol_values = symbols.astype("string")
    symbol_text = symbol_values.str.strip()
    symbol_invalid = symbol_missing | symbol_text.eq("").fillna(True).to_numpy(dtype=bool)
    if symbol_invalid.any():
        positions = np.flatnonzero(symbol_invalid)[:5].tolist()
        raise ValueError(
            "ml incumbent smoothing symbols must be non-empty and non-missing; "
            f"invalid positions={positions}"
        )

    numeric = pd.to_numeric(incumbent_pe, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    valid = np.isfinite(values) & (values > 0.0)
    output = np.full(len(values), np.nan, dtype=np.float64)
    counts = np.zeros(len(values), dtype=np.int64)
    fallback = np.zeros(len(values), dtype=bool)
    history: deque[float] = deque(maxlen=int(config["window_sessions"]))
    previous_symbol: str | None = None
    symbol_reset_rows = 0
    invalid_break_rows = 0
    nonfinite_candidate_rows = 0

    incumbent_weight = float(config["incumbent_weight"])
    anchor_weight = float(config["anchor_weight"])
    window_sessions = int(config["window_sessions"])
    for position, (incumbent_value, symbol) in enumerate(
        zip(values, symbol_text.to_numpy(dtype=str), strict=True)
    ):
        if previous_symbol is not None and symbol != previous_symbol:
            history.clear()
            symbol_reset_rows += 1
        previous_symbol = symbol

        if not valid[position]:
            history.clear()
            invalid_break_rows += 1
            continue

        current_log = math.log(float(incumbent_value))
        history.append(current_log)
        counts[position] = len(history)
        if len(history) < window_sessions:
            output[position] = float(incumbent_value)
            fallback[position] = True
            continue

        anchor = float(np.median(np.asarray(history, dtype=np.float64)))
        candidate_log = incumbent_weight * current_log + anchor_weight * anchor
        try:
            candidate = math.exp(candidate_log)
        except OverflowError:
            candidate = np.nan
        if math.isfinite(candidate) and candidate > 0.0:
            output[position] = candidate
        else:
            nonfinite_candidate_rows += 1

    expected_column, count_column, fallback_column = (
        ML_INCUMBENT_WEEKLY_MEDIAN_SHRINKAGE_OUTPUT_COLUMNS
    )
    result[expected_column] = pd.Series(output, index=incumbent_pe.index, dtype=float)
    result[count_column] = pd.Series(counts, index=incumbent_pe.index, dtype="int64")
    result[fallback_column] = pd.Series(fallback, index=incumbent_pe.index, dtype=bool)
    return result, {
        "enabled": True,
        "status": "ok",
        "candidate_id": "causal_ml_incumbent_weekly_median_shrinkage_v1",
        "semantic_type": "code_owned_incumbent_expected_pe_forecast_stabilizer",
        "formula": ("exp(0.75*log(ml_expected_pe_t) + 0.25*median(log(ml_expected_pe_[t-4:t])))"),
        "input_column": "ml_expected_pe",
        "window_sessions": window_sessions,
        "window_definition": (
            "current row plus four preceding adjacent input rows for the same symbol; "
            "calendar-day gaps do not imply missing sessions"
        ),
        "window_includes_current_incumbent": True,
        "decision_timestamp": "end_of_day_after_ml_expected_pe_t_is_final",
        "pre_close_forecast_claim_allowed": False,
        "direct_observed_pe_consumed": False,
        "true_fair_pe_consumed": False,
        "future_rows_consumed": False,
        "same_row_incumbent_consumed": True,
        "symbol_transition_behavior": "defensive_reset_not_general_panel_support",
        "invalid_current_incumbent_behavior": "nan_and_reset_contiguous_window",
        "warmup_fallback": "exact_positive_finite_current_incumbent",
        "production_connected": False,
        "valuation_connected": False,
        "valid_output_rows_before_pipeline_valuation_mask": int(np.isfinite(output).sum()),
        "warmup_fallback_rows_before_pipeline_valuation_mask": int(fallback.sum()),
        "invalid_incumbent_window_break_rows": invalid_break_rows,
        "symbol_transition_reset_rows": symbol_reset_rows,
        "nonfinite_candidate_rows": nonfinite_candidate_rows,
        "locked_parameters": {
            key: config[key] for key in LOCKED_ML_INCUMBENT_WEEKLY_MEDIAN_SHRINKAGE_CONFIG
        },
    }
