from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor


FUNDAMENTAL_VINTAGE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "v04_fundamental_vintage_expected_pe",
    "v04_fundamental_vintage_mode",
    "v04_fundamental_vintage_usable_train_vintages",
    "v04_fundamental_vintage_fallback_used",
)

LOCKED_FUNDAMENTAL_VINTAGE_CONFIG: dict[str, float | int | str] = {
    "min_completed_target_vintages": 4,
    "min_usable_vintages": 8,
    "max_train_vintages": 20,
    "quantile": 0.50,
    "alpha": 0.05,
    "solver": "highs",
    "growth_winsor_lower": 0.05,
    "growth_winsor_upper": 0.95,
    "growth_iqr_floor": 0.01,
    "target_clip_lower": 0.10,
    "target_clip_upper": 0.90,
    "confidence_floor": 25.0,
    "confidence_ceiling": 100.0,
    "approximation_weight": 0.75,
}

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "available_at",
    "effective_date",
    "period_end",
    "eps_ttm_raw",
    "eps_ttm",
    "eps_ttm_growth_252",
    "eps_primary_method",
    "eps_definition",
    "eps_source_tag",
    "event_source",
    "timestamp_exact",
    "eps_confidence",
    "eps_disagreement",
    "eps_approximation_flag",
    "eps_reconstruction_approximate",
    "negative_earnings_flag",
    "observed_pe",
)

_OPTIONAL_IDENTITY_COLUMNS: tuple[str, ...] = (
    "accession",
    "source_tag",
    "shares_source_tag",
)

_LEADING_WARMUP_NULL_COLUMNS: tuple[str, ...] = (
    "available_at",
    "effective_date",
    "period_end",
    "eps_ttm_raw",
    "eps_ttm",
    "eps_ttm_growth_252",
    "eps_primary_method",
    "eps_definition",
    "eps_source_tag",
    "event_source",
    "timestamp_exact",
    "eps_confidence",
    "eps_disagreement",
    "eps_approximation_flag",
    "eps_reconstruction_approximate",
)


@dataclass(frozen=True)
class _CompletedVintage:
    growth: float
    target_log_pe: float
    weight: float


def _empty_result(
    frame: pd.DataFrame,
    *,
    mode: str,
    usable_count: float = np.nan,
) -> pd.DataFrame:
    size = len(frame)
    return pd.DataFrame(
        {
            FUNDAMENTAL_VINTAGE_OUTPUT_COLUMNS[0]: np.full(size, np.nan, dtype=np.float64),
            FUNDAMENTAL_VINTAGE_OUTPUT_COLUMNS[1]: pd.Series(
                np.full(size, mode, dtype=object), index=frame.index, dtype="string"
            ),
            FUNDAMENTAL_VINTAGE_OUTPUT_COLUMNS[2]: np.full(size, usable_count, dtype=np.float64),
            FUNDAMENTAL_VINTAGE_OUTPUT_COLUMNS[3]: np.zeros(size, dtype=bool),
        },
        index=frame.index,
    )


def _validate_locked_config(config: Mapping[str, Any]) -> None:
    expected_keys = {"enabled", *LOCKED_FUNDAMENTAL_VINTAGE_CONFIG}
    unknown = sorted(set(config) - expected_keys)
    missing = sorted(expected_keys - set(config))
    if unknown or missing:
        raise ValueError(
            "fundamental_vintage keys must match the locked candidate; "
            f"unknown={unknown}, missing={missing}"
        )
    if not isinstance(config["enabled"], bool):
        raise ValueError("fundamental_vintage.enabled must be boolean")

    for field, locked_value in LOCKED_FUNDAMENTAL_VINTAGE_CONFIG.items():
        value = config[field]
        if isinstance(locked_value, int):
            valid = isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        elif isinstance(locked_value, float):
            valid = isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(
                f"fundamental_vintage.{field} must equal locked value {locked_value!r}"
            )
        if isinstance(locked_value, (int, float)):
            valid = math.isfinite(float(value)) and float(value) == float(locked_value)
        else:
            valid = value == locked_value
        if not valid:
            raise ValueError(
                f"fundamental_vintage.{field} must equal locked value {locked_value!r}"
            )


def _numeric_scalar(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if math.isfinite(numeric) else np.nan


def _boolean_scalar(value: Any) -> bool | None:
    if value is pd.NA or value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        numeric = float(value)
        return bool(int(numeric)) if math.isfinite(numeric) and numeric in {0.0, 1.0} else None
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _identity_token(value: Any) -> str:
    if value is pd.NA or value is None:
        return "<NA>"
    try:
        if bool(pd.isna(value)):
            return "<NA>"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, np.floating)):
        return np.float64(value).hex()
    return str(value).strip()


def _is_genuine_null(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _is_leading_warmup_row(row: pd.Series) -> bool:
    symbol = _identity_token(row["symbol"])
    if symbol in {"", "<NA>"}:
        return False
    null_columns = (*_LEADING_WARMUP_NULL_COLUMNS, *_OPTIONAL_IDENTITY_COLUMNS)
    return all(_is_genuine_null(row.get(column)) for column in null_columns)


def _calendar_day(value: Any, *, utc: bool = False) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=utc)
    if pd.isna(parsed):
        return None
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None) if utc else timestamp.tz_localize(None)
    return timestamp.normalize()


def _vintage_key(row: pd.Series) -> tuple[str, ...] | None:
    effective = pd.to_datetime(row["effective_date"], errors="coerce")
    available = pd.to_datetime(row["available_at"], errors="coerce", utc=True)
    period_end = pd.to_datetime(row["period_end"], errors="coerce")
    if pd.isna(effective) or pd.isna(available) or pd.isna(period_end):
        return None

    exact = _boolean_scalar(row["timestamp_exact"])
    symbol = _identity_token(row["symbol"])
    if exact is None or symbol in {"", "<NA>"}:
        return None

    identity_columns = (
        "eps_ttm_raw",
        "eps_primary_method",
        "eps_definition",
        "eps_source_tag",
        "event_source",
        "eps_confidence",
        "eps_disagreement",
        "eps_approximation_flag",
        "eps_reconstruction_approximate",
        *_OPTIONAL_IDENTITY_COLUMNS,
    )
    return (
        symbol,
        pd.Timestamp(effective).isoformat(),
        pd.Timestamp(available).isoformat(),
        pd.Timestamp(period_end).isoformat(),
        str(int(exact)),
        *(_identity_token(row.get(column)) for column in identity_columns),
    )


def _vintage_weight(row: pd.Series, config: Mapping[str, Any]) -> float:
    confidence = _numeric_scalar(row["eps_confidence"])
    disagreement_raw = row["eps_disagreement"]
    disagreement_missing = disagreement_raw is pd.NA or disagreement_raw is None
    if not disagreement_missing:
        try:
            disagreement_missing = bool(pd.isna(disagreement_raw))
        except (TypeError, ValueError):
            disagreement_missing = False
    disagreement = 0.0 if disagreement_missing else _numeric_scalar(disagreement_raw)
    approximate = _boolean_scalar(row["eps_approximation_flag"])
    reconstruction_approximate = _boolean_scalar(row["eps_reconstruction_approximate"])
    if (
        not math.isfinite(confidence)
        or not math.isfinite(disagreement)
        or approximate is None
        or reconstruction_approximate is None
    ):
        return np.nan

    confidence = float(
        np.clip(confidence, config["confidence_floor"], config["confidence_ceiling"])
    )
    weight = confidence / float(config["confidence_ceiling"])
    if approximate or reconstruction_approximate:
        weight *= float(config["approximation_weight"])
    weight /= 1.0 + max(disagreement, 0.0)
    return weight if math.isfinite(weight) and weight > 0.0 else np.nan


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    total = float(ordered_weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        return np.nan
    position = int(np.searchsorted(np.cumsum(ordered_weights), 0.5 * total, side="left"))
    return float(ordered_values[min(position, len(ordered_values) - 1)])


def _predict_for_vintage(
    completed: list[_CompletedVintage],
    current_growth: float,
    config: Mapping[str, Any],
) -> tuple[float, str, int, bool]:
    max_train = int(config["max_train_vintages"])
    target_records = [
        record
        for record in completed
        if math.isfinite(record.target_log_pe)
        and math.isfinite(record.weight)
        and record.weight > 0.0
    ][-max_train:]
    usable_records = [record for record in target_records if math.isfinite(record.growth)]
    usable_count = len(usable_records)
    if len(target_records) < int(config["min_completed_target_vintages"]):
        return np.nan, "unavailable_insufficient_completed_targets", usable_count, False

    intercept_targets = np.asarray(
        [record.target_log_pe for record in target_records], dtype=np.float64
    )
    intercept_weights = np.asarray([record.weight for record in target_records], dtype=np.float64)
    intercept = _weighted_median(intercept_targets, intercept_weights)
    if not math.isfinite(intercept):
        return np.nan, "unavailable_invalid_intercept", usable_count, False

    fallback_reason = "intercept_only_insufficient_usable"
    prediction = intercept
    clip_targets = intercept_targets
    if usable_count >= int(config["min_usable_vintages"]):
        if math.isfinite(current_growth):
            growth = np.asarray([record.growth for record in usable_records], dtype=np.float64)
            target = np.asarray(
                [record.target_log_pe for record in usable_records], dtype=np.float64
            )
            weight = np.asarray([record.weight for record in usable_records], dtype=np.float64)
            lower, upper = np.quantile(
                growth,
                [config["growth_winsor_lower"], config["growth_winsor_upper"]],
            )
            winsorized = np.clip(growth, lower, upper)
            center = float(np.median(winsorized))
            q25, q75 = np.quantile(winsorized, [0.25, 0.75])
            scale = max(float(q75 - q25), float(config["growth_iqr_floor"]))
            x_train = ((winsorized - center) / scale).reshape(-1, 1)
            x_current = np.asarray(
                [[(float(np.clip(current_growth, lower, upper)) - center) / scale]],
                dtype=np.float64,
            )
            try:
                model = QuantileRegressor(
                    quantile=float(config["quantile"]),
                    alpha=float(config["alpha"]),
                    fit_intercept=True,
                    solver=str(config["solver"]),
                )
                model.fit(x_train, target, sample_weight=weight)
                slope = float(np.asarray(model.coef_, dtype=np.float64)[0])
                fitted = float(np.asarray(model.predict(x_current), dtype=np.float64)[0])
                if math.isfinite(slope) and slope > 0.0 and math.isfinite(fitted):
                    prediction = fitted
                    clip_targets = target
                    fallback_reason = "conditional_q50"
                else:
                    fallback_reason = "intercept_only_nonpositive_slope"
            except (ArithmeticError, RuntimeError, ValueError):
                fallback_reason = "intercept_only_fit_failure"
        else:
            fallback_reason = "intercept_only_missing_current_growth"

    clip_lower, clip_upper = np.quantile(
        clip_targets,
        [config["target_clip_lower"], config["target_clip_upper"]],
    )
    prediction = float(np.clip(prediction, clip_lower, clip_upper))
    with np.errstate(over="ignore", invalid="ignore"):
        expected_pe = float(np.exp(prediction))
    if not math.isfinite(expected_pe) or expected_pe <= 0.0:
        return np.nan, "unavailable_invalid_prediction", usable_count, False
    return expected_pe, fallback_reason, usable_count, fallback_reason != "conditional_q50"


def _current_eps_is_valid(row: pd.Series) -> bool:
    eps = _numeric_scalar(row["eps_ttm"])
    negative = _boolean_scalar(row["negative_earnings_flag"])
    return math.isfinite(eps) and eps > 0.0 and negative is False


def walk_forward_fundamental_vintage_pe(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Estimate a current-price-independent multiple from completed EPS vintages.

    A block becomes a training label only when the next information vintage begins.
    The current block prediction is computed once at its first row and then held fixed;
    its observed P/E values can affect only predictions beginning with a later block.
    """

    _validate_locked_config(config)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("walk_forward_fundamental_vintage_pe requires a DataFrame")
    if not bool(config["enabled"]):
        return _empty_result(frame, mode="unavailable_disabled"), {
            "enabled": False,
            "status": "disabled",
            "production_connected": False,
            "current_price_consumed_as_feature": False,
        }

    missing = sorted(set(_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        return _empty_result(frame, mode="unavailable_missing_provenance"), {
            "enabled": True,
            "status": "unavailable_missing_provenance",
            "missing_columns": missing,
            "production_connected": False,
            "current_price_consumed_as_feature": False,
        }

    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    if (
        dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
        or not frame.index.is_unique
    ):
        return _empty_result(frame, mode="unavailable_noncanonical_order"), {
            "enabled": True,
            "status": "unavailable_noncanonical_order",
            "production_connected": False,
            "current_price_consumed_as_feature": False,
        }

    result = _empty_result(frame, mode="unavailable_no_vintage", usable_count=0.0)
    expected_column, mode_column, count_column, fallback_column = FUNDAMENTAL_VINTAGE_OUTPUT_COLUMNS
    completed: list[_CompletedVintage] = []
    current_key: tuple[str, ...] | None = None
    current_effective: pd.Timestamp | None = None
    current_prediction = np.nan
    current_mode = "unavailable_no_vintage"
    current_usable_count = 0
    current_fallback = False
    current_growth = np.nan
    current_weight = np.nan
    current_target_eligible = False
    current_observed_logs: list[float] = []
    stopped = False
    completed_blocks = 0

    def finalize_current() -> None:
        nonlocal completed_blocks
        if current_key is None:
            return
        completed_blocks += 1
        if not current_target_eligible or not current_observed_logs:
            return
        target = float(np.median(np.asarray(current_observed_logs, dtype=np.float64)))
        if math.isfinite(target) and math.isfinite(current_weight) and current_weight > 0.0:
            completed.append(
                _CompletedVintage(
                    growth=float(current_growth),
                    target_log_pe=target,
                    weight=float(current_weight),
                )
            )

    for position, (_, row) in enumerate(frame.iterrows()):
        if stopped:
            result.iloc[position, result.columns.get_loc(mode_column)] = (
                "unavailable_noncanonical_provenance"
            )
            continue

        key = _vintage_key(row)
        if key is None:
            if current_key is None and _is_leading_warmup_row(row):
                continue
            stopped = True
            result.iloc[position, result.columns.get_loc(mode_column)] = (
                "unavailable_noncanonical_provenance"
            )
            continue

        effective = pd.to_datetime(row["effective_date"], errors="coerce")
        available_day = _calendar_day(row["available_at"], utc=True)
        effective_day = _calendar_day(row["effective_date"])
        period_end_day = _calendar_day(row["period_end"])
        row_day = _calendar_day(dates.iloc[position], utc=True)
        if (
            available_day is None
            or effective_day is None
            or period_end_day is None
            or row_day is None
            or not period_end_day <= available_day <= effective_day <= row_day
        ):
            stopped = True
            result.iloc[position, result.columns.get_loc(mode_column)] = (
                "unavailable_noncanonical_provenance"
            )
            continue

        if current_key is None or key != current_key:
            if current_key is not None:
                if (
                    pd.Timestamp(effective) <= pd.Timestamp(current_effective)
                    or key[1] == current_key[1]
                ):
                    stopped = True
                    result.iloc[position, result.columns.get_loc(mode_column)] = (
                        "unavailable_noncanonical_provenance"
                    )
                    continue
                finalize_current()

            current_key = key
            current_effective = pd.Timestamp(effective)
            current_growth = _numeric_scalar(row["eps_ttm_growth_252"])
            current_weight = _vintage_weight(row, config)
            current_target_eligible = _current_eps_is_valid(row)
            current_observed_logs = []
            if current_target_eligible:
                (
                    current_prediction,
                    current_mode,
                    current_usable_count,
                    current_fallback,
                ) = _predict_for_vintage(completed, current_growth, config)
            else:
                usable_records = [
                    record
                    for record in completed[-int(config["max_train_vintages"]) :]
                    if math.isfinite(record.growth)
                ]
                current_prediction = np.nan
                current_mode = "unavailable_invalid_current_eps"
                current_usable_count = len(usable_records)
                current_fallback = False

        row_valid_eps = _current_eps_is_valid(row)
        output_prediction = current_prediction if row_valid_eps else np.nan
        output_mode = current_mode if row_valid_eps else "unavailable_invalid_current_eps"
        output_fallback = current_fallback if row_valid_eps else False
        result.iloc[position, result.columns.get_loc(expected_column)] = output_prediction
        result.iloc[position, result.columns.get_loc(mode_column)] = output_mode
        result.iloc[position, result.columns.get_loc(count_column)] = float(current_usable_count)
        result.iloc[position, result.columns.get_loc(fallback_column)] = output_fallback
        observed = _numeric_scalar(row["observed_pe"])
        if current_target_eligible and row_valid_eps and math.isfinite(observed) and observed > 0.0:
            current_observed_logs.append(float(math.log(observed)))

    expected_values = pd.to_numeric(result[expected_column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if stopped:
        failed = _empty_result(
            frame,
            mode="unavailable_noncanonical_provenance",
        )
        return failed, {
            "enabled": True,
            "status": "unavailable_noncanonical_provenance",
            "semantic_type": (
                "historical_price_trained_current_price_independent_"
                "completed_vintage_median_multiple"
            ),
            "prediction_feature_allowlist": ["eps_ttm_growth_252"],
            "current_price_consumed_as_feature": False,
            "current_observed_pe_consumed_as_feature": False,
            "observed_pe_role": "completed_prior_vintage_target_only",
            "incumbent_fallback_used": False,
            "production_connected": False,
            "valid_expected_rows": 0,
            "conditional_rows": 0,
            "intercept_fallback_rows": 0,
            "state_machine_stopped_on_noncanonical_provenance": True,
        }
    result[expected_column] = pd.Series(expected_values, index=frame.index, dtype=float)
    result[count_column] = pd.to_numeric(result[count_column], errors="coerce").astype(float)
    result[fallback_column] = result[fallback_column].fillna(False).astype(bool)
    result[mode_column] = result[mode_column].astype("string")
    valid_expected = np.isfinite(expected_values) & (expected_values > 0.0)
    conditional_rows = int((result[mode_column] == "conditional_q50").sum())
    fallback_rows = int(result[fallback_column].sum())
    mode_counts = {
        str(mode): int(count)
        for mode, count in result[mode_column].value_counts(dropna=False).items()
    }
    return result, {
        "enabled": True,
        "status": "causal_fail_closed" if stopped else "ok",
        "semantic_type": (
            "historical_price_trained_current_price_independent_completed_vintage_median_multiple"
        ),
        "estimand": (
            "exp(Q0.5(completed_vintage_median_log_observed_pe | "
            "first_row_252_session_pit_ttm_eps_growth))"
        ),
        "prediction_feature_allowlist": ["eps_ttm_growth_252"],
        "current_price_consumed_as_feature": False,
        "current_observed_pe_consumed_as_feature": False,
        "observed_pe_role": "completed_prior_vintage_target_only",
        "incumbent_fallback_used": False,
        "production_connected": False,
        "valid_expected_rows": int(valid_expected.sum()),
        "conditional_rows": conditional_rows,
        "intercept_fallback_rows": fallback_rows,
        "completed_information_vintages": int(completed_blocks),
        "completed_target_vintages": int(len(completed)),
        "mode_counts": dict(sorted(mode_counts.items())),
        "state_machine_stopped_on_noncanonical_provenance": stopped,
        "locked_parameters": {key: config[key] for key in LOCKED_FUNDAMENTAL_VINTAGE_CONFIG},
    }
