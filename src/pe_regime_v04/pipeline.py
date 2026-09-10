from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .ablation import build_ablation_report
from .expected_pe import (
    global_statistical_expected_pe,
    walk_forward_expected_pe_pair,
)
from .fundamental_vintage import walk_forward_fundamental_vintage_pe
from .lagged_market_conditioned import lagged_market_conditioned_expected_pe
from .market_conditioned import market_conditioned_kalman_pe_diagnostic
from .matured_proxy_gate import causal_matured_forward_median_gate
from .ml_incumbent_smoothing import causal_ml_incumbent_weekly_median_shrinkage
from .regime_stacker import walk_forward_hierarchical_stacker
from .utility_gate import shifted_candidate_gate, shifted_two_candidate_prior_blend
from .utils import REGIME_ORDER, ensure_columns, normalized_entropy, write_json
from .valuation import add_adaptive_valuation

_REQUIRED_COLUMNS = [
    "date",
    "benchmark_close",
    "observed_pe",
    "p_bear",
    "p_sideways",
    "p_bull",
    "market_regime",
]

V04_APPEND_COLUMNS: tuple[str, ...] = (
    "v04_return_forecast_p_bear",
    "v04_return_forecast_p_sideways",
    "v04_return_forecast_p_bull",
    "v04_return_forecast_market_regime",
    "v04_return_forecast_entropy",
    "v04_return_forecast_confidence",
    "v04_current_p_bear",
    "v04_current_p_sideways",
    "v04_current_p_bull",
    "v04_current_market_regime",
    "v04_global_statistical_expected_pe",
    "v04_global_baseline_expected_pe",
    "v04_stat_regime_baseline_oos_log_mae",
    "v04_stat_regime_challenger_oos_log_mae",
    "v04_stat_regime_paired_oos_count",
    "v04_stat_regime_oos_gain",
    "v04_stat_regime_challenger_weight",
    "v04_stat_regime_accepted",
    "v04_stat_regime_fallback_used",
    "v04_stat_regime_blended",
    "v04_guarded_statistical_expected_pe",
    "v04_ml_expected_pe_no_regime",
    "v04_ml_no_regime_shrinkage_expected_pe",
    "v04_market_conditioned_kalman_diagnostic_expected_pe",
    "v04_ml_expected_pe_with_regime",
    "v04_ml_matched_regime_baseline_oos_log_mae",
    "v04_ml_matched_regime_challenger_oos_log_mae",
    "v04_ml_matched_regime_paired_oos_count",
    "v04_ml_matched_regime_oos_gain",
    "v04_ml_matched_regime_challenger_weight",
    "v04_ml_matched_regime_accepted",
    "v04_ml_matched_regime_fallback_used",
    "v04_ml_matched_regime_blended",
    "v04_matched_best_ml_expected_pe",
    "v04_regime_features_incremental_oos_gain",
    "v04_regime_features_accepted",
    "v04_ml_incumbent_expected_pe",
    "v04_ml_no_regime_baseline_oos_log_mae",
    "v04_ml_no_regime_challenger_oos_log_mae",
    "v04_ml_no_regime_paired_oos_count",
    "v04_ml_no_regime_oos_gain",
    "v04_ml_no_regime_challenger_weight",
    "v04_ml_no_regime_accepted",
    "v04_ml_no_regime_fallback_used",
    "v04_ml_no_regime_blended",
    "v04_ml_incumbent_baseline_oos_log_mae",
    "v04_ml_incumbent_challenger_oos_log_mae",
    "v04_ml_incumbent_paired_oos_count",
    "v04_ml_incumbent_oos_gain",
    "v04_ml_incumbent_challenger_weight",
    "v04_ml_incumbent_accepted",
    "v04_ml_incumbent_fallback_used",
    "v04_ml_incumbent_blended",
    "v04_guarded_ml_expected_pe",
    "v04_final_stat_oos_log_mae",
    "v04_final_ml_oos_log_mae",
    "v04_final_paired_oos_count",
    "v04_final_stat_weight",
    "v04_final_ml_weight",
    "v04_final_fallback_used",
    "v04_final_expected_pe",
    "v04_dynamic_blend_expected_pe",
    "v04_final_stat_challenger_baseline_oos_log_mae",
    "v04_final_stat_challenger_challenger_oos_log_mae",
    "v04_final_stat_challenger_paired_oos_count",
    "v04_final_stat_challenger_oos_gain",
    "v04_final_stat_challenger_challenger_weight",
    "v04_final_stat_challenger_accepted",
    "v04_final_stat_challenger_fallback_used",
    "v04_final_stat_challenger_blended",
    "v04_expected_pe",
    "v04_pe_gap_log",
    "v04_pe_gap_percent",
    "v04_gap_history_percentile",
    "v04_gap_robust_z",
    "v04_gap_low_history_threshold",
    "v04_gap_high_history_threshold",
    "v04_valuation_strength",
    "v04_valuation_state_adaptive",
    "v04_valuation_state_absolute",
    "v04_valuation_state_confirmed",
    "v04_valuation_confidence",
    # Heldout-promoted EOD surface. Keep this tail-appended so the preceding
    # 82-column v0.4 contract remains an exact positional prefix.
    "v04_market_conditioned_expected_pe",
    # Price-free completed-information-vintage challenger. Preserve all prior 83
    # append columns as an exact positional prefix and add audit surfaces at tail.
    "v04_fundamental_vintage_expected_pe",
    "v04_fundamental_vintage_mode",
    "v04_fundamental_vintage_usable_train_vintages",
    "v04_fundamental_vintage_fallback_used",
    # Default-off, EOD-only one-step-prior candidate. Preserve the preceding 87
    # append columns exactly and keep every research/audit surface at the tail.
    "v04_lagged_market_conditioned_raw_expected_pe",
    "v04_gated_lagged_market_conditioned_expected_pe",
    "v04_gated_lagged_market_conditioned_paired_oos_count",
    "v04_gated_lagged_market_conditioned_challenger_weight",
    "v04_gated_lagged_market_conditioned_accepted",
    "v04_gated_lagged_market_conditioned_fallback_used",
    # Default-off fixed EOD stabilizer of the code-owned v0.3 ML incumbent.
    # Preserve every preceding 93-column append surface as an exact prefix.
    "v04_ml_weekly_median_shrinkage_expected_pe",
    "v04_ml_weekly_median_shrinkage_window_count",
    "v04_ml_weekly_median_shrinkage_fallback_used",
    # Default-off gate-target alignment candidate.  Preserve the preceding 96
    # append columns exactly and publish the matured-loss gate at the tail only.
    "v04_matured_proxy_regularized_expected_pe",
    "v04_matured_proxy_baseline_oos_log_mae",
    "v04_matured_proxy_challenger_oos_log_mae",
    "v04_matured_proxy_paired_oos_count",
    "v04_matured_proxy_oos_gain",
    "v04_matured_proxy_challenger_weight",
    "v04_matured_proxy_accepted",
    "v04_matured_proxy_fallback_used",
)

if len(V04_APPEND_COLUMNS) != 104 or len(set(V04_APPEND_COLUMNS)) != 104:
    raise RuntimeError("v0.4 append schema must contain exactly 104 unique columns")
if any(not column.startswith("v04_") for column in V04_APPEND_COLUMNS):
    raise RuntimeError("Every public v0.4 append column must use the v04_ prefix")


def _numeric_float_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    return pd.Series(array, index=values.index, name=values.name, dtype=float)


def _geometric_blend_with_one_sided_fallback(
    incumbent: pd.Series,
    challenger: pd.Series,
    *,
    weight: float,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    if isinstance(weight, bool):
        raise ValueError("no_regime_shrinkage.weight must be finite and in [0, 1]")
    try:
        weight = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("no_regime_shrinkage.weight must be finite and in [0, 1]") from exc
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("no_regime_shrinkage.weight must be finite and in [0, 1]")

    incumbent_numeric = _numeric_float_series(incumbent)
    challenger_numeric = _numeric_float_series(challenger)
    incumbent_values = incumbent_numeric.to_numpy(dtype=np.float64)
    challenger_values = challenger_numeric.to_numpy(dtype=np.float64)
    incumbent_available = np.isfinite(incumbent_values) & (incumbent_values > 0.0)
    challenger_available = np.isfinite(challenger_values) & (challenger_values > 0.0)
    both_available = incumbent_available & challenger_available
    incumbent_only = incumbent_available & ~challenger_available
    challenger_only = challenger_available & ~incumbent_available
    neither_available = ~incumbent_available & ~challenger_available

    blended = np.full(len(incumbent_numeric), np.nan, dtype=np.float64)
    blended[incumbent_only] = incumbent_values[incumbent_only]
    blended[challenger_only] = challenger_values[challenger_only]
    if weight == 0.0:
        blended[both_available] = incumbent_values[both_available]
    elif weight == 1.0:
        blended[both_available] = challenger_values[both_available]
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            blended[both_available] = np.exp(
                (1.0 - weight) * np.log(incumbent_values[both_available])
                + weight * np.log(challenger_values[both_available])
            )
    blended[~np.isfinite(blended) | (blended <= 0.0)] = np.nan

    index = incumbent_numeric.index
    availability = {
        "both_available": pd.Series(both_available, index=index, dtype=bool),
        "incumbent_only": pd.Series(incumbent_only, index=index, dtype=bool),
        "challenger_only": pd.Series(challenger_only, index=index, dtype=bool),
        "neither_available": pd.Series(neither_available, index=index, dtype=bool),
    }
    return pd.Series(blended, index=index, dtype=float), availability


def _assert_probability_simplex(frame: pd.DataFrame, columns: list[str]) -> None:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    complete = np.isfinite(values).all(axis=1)
    partial = np.isfinite(values).any(axis=1) & ~complete
    if partial.any():
        raise RuntimeError(f"partial probability rows found for {columns}")
    if complete.any():
        sums = values[complete].sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-8, rtol=0.0):
            raise RuntimeError(f"probabilities do not sum to one for {columns}")
        if np.any(values[complete] < -1e-12) or np.any(values[complete] > 1.0 + 1e-12):
            raise RuntimeError(f"probabilities outside [0,1] for {columns}")


def _negative_earnings_mask(frame: pd.DataFrame) -> pd.Series:
    if "negative_earnings_flag" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    source = frame["negative_earnings_flag"]
    numeric = _numeric_float_series(source)
    source_present = source.notna().to_numpy(dtype=bool)
    numeric_values = numeric.to_numpy(dtype=np.float64)
    invalid = pd.Series(
        source_present & (~np.isfinite(numeric_values) | ~np.isin(numeric_values, [0.0, 1.0])),
        index=frame.index,
        dtype=bool,
    )
    if invalid.any():
        positions = np.flatnonzero(invalid.to_numpy(dtype=bool))[:5].tolist()
        raise ValueError(
            "negative_earnings_flag must contain only 0/1 or missing values; "
            f"invalid positions={positions}"
        )
    return numeric.fillna(0.0).astype(bool)


def apply_v04_layers(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append v0.4 diagnostics without mutating, sorting or retyping input columns."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("apply_v04_layers requires a pandas DataFrame")
    duplicate_columns = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        raise ValueError(f"input contains duplicate columns: {duplicate_columns}")
    ensure_columns(frame, _REQUIRED_COLUMNS, context="v0.3 canonical output")
    existing_v04 = [column for column in frame.columns if column.startswith("v04_")]
    legacy_unprefixed = [
        column
        for column in ("stacker_p_bear", "stacker_p_sideways", "stacker_p_bull")
        if column in frame.columns
    ]
    if existing_v04 or legacy_unprefixed:
        raise ValueError(
            "input already contains v0.4 overlay columns; use the original v0.3 frame: "
            f"{existing_v04 + legacy_unprefixed}"
        )
    parsed_dates = pd.to_datetime(frame["date"], errors="raise")
    if parsed_dates.isna().any():
        raise ValueError("date contains missing or invalid values")
    if parsed_dates.duplicated().any():
        raise ValueError("date must be unique for in-memory v0.4 integration")
    if not parsed_dates.is_monotonic_increasing:
        raise ValueError("date must already be monotonically increasing")
    if not frame.index.is_unique:
        raise ValueError("input index must be unique")

    original = frame.copy(deep=True)
    working = frame.copy(deep=True)
    input_negative = _negative_earnings_mask(working)
    input_observed = _numeric_float_series(working["observed_pe"])
    contradictory = input_negative & input_observed.notna()
    if contradictory.any():
        positions = np.flatnonzero(contradictory.to_numpy(dtype=bool))[:5].tolist()
        raise ValueError(
            "negative_earnings_flag=1 requires observed_pe to be missing; "
            f"contradictory positions={positions}"
        )
    seed = int(config.get("project", {}).get("random_seed", 42))
    frame = working
    fundamental_outputs, fundamental_diagnostics = walk_forward_fundamental_vintage_pe(
        frame,
        config.get("fundamental_vintage", {}),
    )

    stacker, stacker_diagnostics, proxy_labels = walk_forward_hierarchical_stacker(
        frame,
        config.get("regime_stacker", {}),
        seed=seed,
    )
    return_probabilities = stacker[
        ["stacker_p_bear", "stacker_p_sideways", "stacker_p_bull"]
    ].to_numpy(dtype=float)
    return_valid = np.isfinite(return_probabilities).all(axis=1)
    return_hard = np.full(len(frame), "UNKNOWN", dtype=object)
    if return_valid.any():
        return_hard[return_valid] = np.asarray(REGIME_ORDER, dtype=object)[
            np.argmax(return_probabilities[return_valid], axis=1)
        ]
    return_entropy = normalized_entropy(return_probabilities)
    frame["v04_return_forecast_p_bear"] = return_probabilities[:, 0]
    frame["v04_return_forecast_p_sideways"] = return_probabilities[:, 1]
    frame["v04_return_forecast_p_bull"] = return_probabilities[:, 2]
    frame["v04_return_forecast_market_regime"] = return_hard
    frame["v04_return_forecast_entropy"] = return_entropy
    frame["v04_return_forecast_confidence"] = 1.0 - return_entropy

    # Current-state and forward-return probabilities remain separate. There is no
    # semantically matched forward-return baseline in v0.3, so the raw stacker is
    # exposed for diagnostics/features and is never averaged with p_*.
    frame["v04_current_p_bear"] = pd.to_numeric(frame["p_bear"], errors="coerce")
    frame["v04_current_p_sideways"] = pd.to_numeric(frame["p_sideways"], errors="coerce")
    frame["v04_current_p_bull"] = pd.to_numeric(frame["p_bull"], errors="coerce")
    frame["v04_current_market_regime"] = frame["market_regime"].copy()

    frame["v04_global_statistical_expected_pe"] = global_statistical_expected_pe(
        _numeric_float_series(frame["observed_pe"]),
        config.get("statistics", {}),
    )
    historical_fallback = (
        pd.to_numeric(frame["pe_median_756_lag"], errors="coerce")
        if "pe_median_756_lag" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="Float64")
    )
    frame["v04_global_baseline_expected_pe"] = pd.to_numeric(
        frame["v04_global_statistical_expected_pe"], errors="coerce"
    ).combine_first(historical_fallback)
    v03_stat = (
        pd.to_numeric(frame["statistical_expected_pe"], errors="coerce")
        if "statistical_expected_pe" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="Float64")
    )
    stat_gate = shifted_candidate_gate(
        frame["observed_pe"],
        frame["v04_global_baseline_expected_pe"],
        v03_stat,
        config.get("statistical_gate", config.get("candidate_gate", {})),
        prefix="v04_stat_regime",
    )
    for column in stat_gate.columns:
        frame[column] = stat_gate[column]
    frame["v04_guarded_statistical_expected_pe"] = frame["v04_stat_regime_blended"]

    expected_config = config.get("expected_pe", {})
    (
        no_regime,
        with_regime,
        matched_pair_diagnostics,
        no_regime_features,
        with_regime_features,
    ) = walk_forward_expected_pe_pair(
        frame,
        expected_config,
        # Match the v0.3 Expected-P/E seed namespace as well as its high-profile
        # scheduler and model parameters.
        seed=seed + 400_000,
    )
    frame["v04_ml_expected_pe_no_regime"] = pd.to_numeric(no_regime, errors="coerce")
    frame["v04_ml_expected_pe_with_regime"] = pd.to_numeric(with_regime, errors="coerce")
    v03_ml_incumbent = (
        _numeric_float_series(frame["ml_expected_pe"])
        if "ml_expected_pe" in frame.columns
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    lagged_market_outputs, lagged_market_diagnostics = lagged_market_conditioned_expected_pe(
        frame["observed_pe"],
        v03_ml_incumbent,
        config.get("lagged_market_conditioned", {}),
    )
    shrinkage_config = config.get("no_regime_shrinkage", {})
    shrinkage_weight = shrinkage_config.get("weight", 0.50)
    shrinkage_expected_pe, shrinkage_availability = _geometric_blend_with_one_sided_fallback(
        v03_ml_incumbent,
        frame["v04_ml_expected_pe_no_regime"],
        weight=shrinkage_weight,
    )
    frame["v04_ml_no_regime_shrinkage_expected_pe"] = shrinkage_expected_pe
    (
        market_conditioned_diagnostic,
        market_conditioned_diagnostics,
    ) = market_conditioned_kalman_pe_diagnostic(
        frame["observed_pe"],
        v03_ml_incumbent,
        config.get("market_conditioned_diagnostic", {}),
    )
    frame["v04_market_conditioned_kalman_diagnostic_expected_pe"] = market_conditioned_diagnostic

    # Matched-pair ablation.  Both candidates above share the same training target,
    # walk-forward boundaries and hyper-parameters; only the regime feature block
    # differs.  This is the primary answer to "did regime information add value?".
    matched_regime_gate = shifted_candidate_gate(
        frame["observed_pe"],
        frame["v04_ml_expected_pe_no_regime"],
        frame["v04_ml_expected_pe_with_regime"],
        config.get("ml_matched_regime_gate", config.get("candidate_gate", {})),
        prefix="v04_ml_matched_regime",
    )
    for column in matched_regime_gate.columns:
        frame[column] = matched_regime_gate[column]
    frame["v04_matched_best_ml_expected_pe"] = frame["v04_ml_matched_regime_blended"]
    frame["v04_regime_features_incremental_oos_gain"] = pd.to_numeric(
        frame["v04_ml_matched_regime_oos_gain"], errors="coerce"
    )
    frame["v04_regime_features_accepted"] = frame["v04_ml_matched_regime_accepted"].astype(bool)

    # Preserve the v0.3 high model as a production incumbent.  The matched v0.4
    # candidate may replace it only after a decisive, shifted OOS improvement.
    reuse_v03 = bool(expected_config.get("reuse_v03_ml_with_regime", True))
    if reuse_v03 and "ml_expected_pe" in frame.columns:
        incumbent = v03_ml_incumbent
        incumbent_source = "v03_ml_expected_pe"
    else:
        incumbent = pd.to_numeric(frame["v04_matched_best_ml_expected_pe"], errors="coerce")
        incumbent_source = "v04_matched_pair"
    frame["v04_ml_incumbent_expected_pe"] = incumbent

    # Retain the old direct no-regime-vs-incumbent diagnostic for compatibility,
    # but do not use it to claim regime utility because the two models are not a
    # matched pair when the incumbent came from v0.3.
    direct_no_regime_gate = shifted_candidate_gate(
        frame["observed_pe"],
        frame["v04_ml_incumbent_expected_pe"],
        frame["v04_ml_expected_pe_no_regime"],
        config.get("ml_no_regime_gate", config.get("candidate_gate", {})),
        prefix="v04_ml_no_regime",
    )
    for column in direct_no_regime_gate.columns:
        frame[column] = direct_no_regime_gate[column]

    incumbent_gate = shifted_candidate_gate(
        frame["observed_pe"],
        frame["v04_ml_incumbent_expected_pe"],
        frame["v04_matched_best_ml_expected_pe"],
        config.get("ml_incumbent_gate", config.get("candidate_gate", {})),
        prefix="v04_ml_incumbent",
    )
    for column in incumbent_gate.columns:
        frame[column] = incumbent_gate[column]
    frame["v04_guarded_ml_expected_pe"] = frame["v04_ml_incumbent_blended"]

    final_blend = shifted_two_candidate_prior_blend(
        frame["observed_pe"],
        frame["v04_guarded_statistical_expected_pe"],
        frame["v04_guarded_ml_expected_pe"],
        config.get("final_blend", {}),
        prefix="v04_final",
    )
    for column in final_blend.columns:
        frame[column] = final_blend[column]
    frame["v04_dynamic_blend_expected_pe"] = frame["v04_final_expected_pe"]

    # Production-safe final promotion.  ML is the incumbent; the statistical
    # candidate receives weight only after it proves a decisive shifted-OOS gain.
    # This closes the v0.3/v0.4 sample failure where a tiny statistical prior made
    # the final estimate slightly worse than ML alone.
    final_guard = shifted_candidate_gate(
        frame["observed_pe"],
        frame["v04_guarded_ml_expected_pe"],
        frame["v04_guarded_statistical_expected_pe"],
        config.get("final_guard", config.get("candidate_gate", {})),
        prefix="v04_final_stat_challenger",
    )
    for column in final_guard.columns:
        frame[column] = final_guard[column]
    frame["v04_expected_pe"] = frame["v04_final_stat_challenger_blended"]
    frame = add_adaptive_valuation(frame, config.get("valuation", {}))

    # The heldout-promoted market-conditioned surface is deliberately published
    # only after valuation has finished.  It is an exact compatibility alias of
    # the evaluated diagnostic formula, not an input to v04_expected_pe, gates,
    # valuation gaps or states.
    frame["v04_market_conditioned_expected_pe"] = frame[
        "v04_market_conditioned_kalman_diagnostic_expected_pe"
    ].copy(deep=True)
    for column in fundamental_outputs.columns:
        frame[column] = fundamental_outputs[column]
    for column in lagged_market_outputs.columns:
        frame[column] = lagged_market_outputs[column]
    smoothing_symbols = frame["symbol"] if "symbol" in frame.columns else None
    smoothing_outputs, smoothing_diagnostics = causal_ml_incumbent_weekly_median_shrinkage(
        v03_ml_incumbent,
        frame["date"],
        smoothing_symbols,
        config.get("ml_incumbent_weekly_median_shrinkage", {}),
    )
    for column in smoothing_outputs.columns:
        frame[column] = smoothing_outputs[column]
    matured_proxy_outputs, matured_proxy_diagnostics = causal_matured_forward_median_gate(
        frame["observed_pe"],
        v03_ml_incumbent,
        frame["v04_ml_expected_pe_with_regime"],
        config.get("matured_proxy_gate", {}),
        symbols=frame["symbol"] if "symbol" in frame.columns else None,
    )
    for column in matured_proxy_outputs.columns:
        frame[column] = matured_proxy_outputs[column]

    observed = _numeric_float_series(frame["observed_pe"])
    negative = _negative_earnings_mask(frame)
    observed_values = observed.to_numpy(dtype=np.float64)
    invalid_valuation = pd.Series(
        negative.to_numpy(dtype=bool) | ~np.isfinite(observed_values) | (observed_values <= 0.0),
        index=frame.index,
        dtype=bool,
    )

    # Sanitize every public numeric append surface before schema enforcement.
    for column in V04_APPEND_COLUMNS:
        if column not in frame.columns or pd.api.types.is_bool_dtype(frame[column]):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric = _numeric_float_series(frame[column])
            frame[column] = numeric.where(np.isfinite(numeric.to_numpy(dtype=float)))

    expected_columns = [column for column in V04_APPEND_COLUMNS if "expected_pe" in column]
    positive_pe_columns = expected_columns
    for column in positive_pe_columns:
        numeric = _numeric_float_series(frame[column])
        values = numeric.to_numpy(dtype=np.float64)
        frame[column] = numeric.where(np.isfinite(values) & (values > 0.0))

    gate_nan_columns = [
        column for column in V04_APPEND_COLUMNS if "_oos_" in column or column.endswith("_blended")
    ]
    gap_columns = [
        column
        for column in V04_APPEND_COLUMNS
        if column.startswith("v04_gap_")
        or column.startswith("v04_pe_gap_")
        or column == "v04_valuation_strength"
    ]
    weight_columns = [column for column in V04_APPEND_COLUMNS if column.endswith("_weight")]
    accepted_columns = [column for column in V04_APPEND_COLUMNS if column.endswith("_accepted")]
    fallback_columns = [
        column for column in V04_APPEND_COLUMNS if column.endswith("_fallback_used")
    ]
    fundamental_expected_column = "v04_fundamental_vintage_expected_pe"
    observed_dependent_positive_columns = [
        column for column in positive_pe_columns if column != fundamental_expected_column
    ]
    for column in set(observed_dependent_positive_columns + gate_nan_columns + gap_columns):
        frame.loc[invalid_valuation, column] = np.nan
    for column in weight_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").clip(0.0, 1.0)
        frame[column] = numeric
        frame.loc[invalid_valuation, column] = 0.0
    for column in accepted_columns + fallback_columns:
        frame[column] = frame[column].fillna(False).astype(bool)
        if column != "v04_fundamental_vintage_fallback_used":
            frame.loc[invalid_valuation, column] = False
    smoothing_count_column = "v04_ml_weekly_median_shrinkage_window_count"
    smoothing_count = pd.to_numeric(frame[smoothing_count_column], errors="coerce")
    smoothing_count_values = smoothing_count.to_numpy(dtype=np.float64)
    valid_smoothing_count = (
        np.isfinite(smoothing_count_values)
        & (smoothing_count_values >= 0.0)
        & (smoothing_count_values <= 5.0)
        & (smoothing_count_values == np.floor(smoothing_count_values))
    )
    if not valid_smoothing_count.all():
        raise RuntimeError("ml weekly median window count must be an integer in [0, 5]")
    frame[smoothing_count_column] = smoothing_count.astype(np.int64)
    frame.loc[invalid_valuation, "v04_valuation_confidence"] = 0.0

    pd.testing.assert_series_equal(
        frame["v04_market_conditioned_expected_pe"],
        frame["v04_market_conditioned_kalman_diagnostic_expected_pe"].rename(
            "v04_market_conditioned_expected_pe"
        ),
        check_exact=True,
        check_dtype=True,
        check_names=True,
    )

    for probability_columns in (
        ["v04_current_p_bear", "v04_current_p_sideways", "v04_current_p_bull"],
        [
            "v04_return_forecast_p_bear",
            "v04_return_forecast_p_sideways",
            "v04_return_forecast_p_bull",
        ],
    ):
        _assert_probability_simplex(frame, probability_columns)
    expected = _numeric_float_series(frame["v04_expected_pe"])
    expected_values = expected.to_numpy(dtype=np.float64)
    finite_expected = np.isfinite(expected_values)
    if np.isinf(expected_values).any() or (expected_values[finite_expected] <= 0.0).any():
        raise RuntimeError("v04_expected_pe must be positive finite or NaN")

    pd.testing.assert_frame_equal(
        frame.loc[:, original.columns],
        original,
        check_exact=True,
        check_dtype=True,
        check_names=True,
        check_column_type=True,
        check_index_type=True,
    )
    generated_columns = [column for column in frame.columns if column not in original.columns]
    missing_append = [column for column in V04_APPEND_COLUMNS if column not in generated_columns]
    unknown_append = [column for column in generated_columns if column not in V04_APPEND_COLUMNS]
    if missing_append or unknown_append:
        raise RuntimeError(
            f"v0.4 append schema mismatch: missing={missing_append}, unknown={unknown_append}"
        )
    append = frame.loc[:, V04_APPEND_COLUMNS].copy(deep=True)
    output = pd.concat([original, append], axis=1)
    expected_columns_order = tuple(original.columns) + V04_APPEND_COLUMNS
    if tuple(output.columns) != expected_columns_order or output.columns.duplicated().any():
        raise RuntimeError("v0.4 output column order/uniqueness contract failed")
    pd.testing.assert_frame_equal(
        output.loc[:, original.columns],
        original,
        check_exact=True,
        check_dtype=True,
        check_names=True,
        check_column_type=True,
        check_index_type=True,
    )

    diagnostics = {
        "invariants": {
            "all_input_columns_exact": True,
            "input_column_count": int(len(original.columns)),
            "append_column_count": int(len(V04_APPEND_COLUMNS)),
            "append_schema_exact": True,
            "date_was_retyped_or_reordered": False,
            "current_and_forward_return_probabilities_mixed": False,
            "negative_or_invalid_observed_pe_fail_closed_rows": int(invalid_valuation.sum()),
        },
        "semantics": {
            "v04_current_p_*": "v0.3 contemporaneous regime detector",
            "v04_return_forecast_p_*": (
                "raw matured forward-return proxy stacker; no direct probability "
                "mixture because no same-target v0.3 baseline exists"
            ),
            "return_forecast_gate_status": "not_gated_no_semantically_matched_baseline",
            "ml_incumbent_source": incumbent_source,
        },
        "no_regime_shrinkage": {
            "method": "causal_fixed_weight_geometric_blend",
            "formula": "exp((1-weight)*log(incumbent)+weight*log(no_regime))",
            "weight": float(shrinkage_weight),
            "incumbent_column": "ml_expected_pe",
            "challenger_column": "v04_ml_expected_pe_no_regime",
            "production_output_unchanged": True,
            "one_sided_fallback": "use_positive_finite_available_side",
            "both_available_rows": int(
                (shrinkage_availability["both_available"] & ~invalid_valuation).sum()
            ),
            "incumbent_only_fallback_rows": int(
                (shrinkage_availability["incumbent_only"] & ~invalid_valuation).sum()
            ),
            "no_regime_only_fallback_rows": int(
                (shrinkage_availability["challenger_only"] & ~invalid_valuation).sum()
            ),
            "neither_available_rows": int(
                (shrinkage_availability["neither_available"] & ~invalid_valuation).sum()
            ),
            "valuation_fail_closed_rows": int(invalid_valuation.sum()),
        },
        "market_conditioned_diagnostic": {
            **market_conditioned_diagnostics,
            "output_column": "v04_market_conditioned_kalman_diagnostic_expected_pe",
            "production_output_column": "v04_expected_pe",
            "production_path_byte_exact_isolation_required": True,
            "legacy_diagnostic_output_column": (
                "v04_market_conditioned_kalman_diagnostic_expected_pe"
            ),
            "promoted_parallel_output_column": "v04_market_conditioned_expected_pe",
            "publication_status": "benchmark_promoted_parallel_surface",
            "formula_consumes_same_row_observed_pe": True,
            "decision_timestamp": "after_current_row_observed_pe_is_final_eod_only",
            "eod_only": True,
            "main_expected_pe_path_connected": False,
            "valuation_path_connected": False,
            "fundamental_fair_pe_claim_allowed": False,
            "promoted_alias_bit_exact": True,
            "valuation_fail_closed_rows": int(invalid_valuation.sum()),
        },
        "fundamental_vintage": {
            **fundamental_diagnostics,
            "output_column": "v04_fundamental_vintage_expected_pe",
            "mode_column": "v04_fundamental_vintage_mode",
            "usable_train_vintages_column": ("v04_fundamental_vintage_usable_train_vintages"),
            "fallback_column": "v04_fundamental_vintage_fallback_used",
            "production_output_column": "v04_expected_pe",
            "production_path_byte_exact_isolation_required": True,
            "main_expected_pe_path_connected": False,
            "valuation_path_connected": False,
            "current_price_intervention_exact_required": True,
            "historical_price_target_role": "completed_prior_information_vintage_only",
        },
        "lagged_market_conditioned": {
            **lagged_market_diagnostics,
            "raw_output_column": "v04_lagged_market_conditioned_raw_expected_pe",
            "output_column": "v04_gated_lagged_market_conditioned_expected_pe",
            "paired_count_column": ("v04_gated_lagged_market_conditioned_paired_oos_count"),
            "challenger_weight_column": ("v04_gated_lagged_market_conditioned_challenger_weight"),
            "accepted_column": "v04_gated_lagged_market_conditioned_accepted",
            "fallback_column": "v04_gated_lagged_market_conditioned_fallback_used",
            "production_output_column": "v04_expected_pe",
            "production_path_byte_exact_isolation_required": True,
            "main_expected_pe_path_connected": False,
            "valuation_path_connected": False,
            "publication_position": "after_valuation_tail_only",
        },
        "ml_incumbent_weekly_median_shrinkage": {
            **smoothing_diagnostics,
            "output_column": "v04_ml_weekly_median_shrinkage_expected_pe",
            "window_count_column": "v04_ml_weekly_median_shrinkage_window_count",
            "fallback_column": "v04_ml_weekly_median_shrinkage_fallback_used",
            "production_output_column": "v04_expected_pe",
            "production_path_byte_exact_isolation_required": True,
            "main_expected_pe_path_connected": False,
            "valuation_path_connected": False,
            "publication_position": "after_valuation_tail_only",
            "pipeline_symbol_column_present": "symbol" in original.columns,
            "pipeline_timeline_contract": "globally_unique_monotone_date_single_timeline",
            "public_output_valuation_fail_closed_rows": int(invalid_valuation.sum()),
            "public_window_count_dtype": str(frame[smoothing_count_column].dtype),
            "invalid_valuation_window_count_behavior": (
                "retain_price_independent_operator_audit_count"
            ),
        },
        "matured_proxy_gate": {
            **matured_proxy_diagnostics,
            "output_column": "v04_matured_proxy_regularized_expected_pe",
            "baseline_column": "ml_expected_pe",
            "challenger_column": "v04_ml_expected_pe_with_regime",
            "production_output_column": "v04_expected_pe",
            "production_path_byte_exact_isolation_required": True,
            "main_expected_pe_path_connected": False,
            "valuation_path_connected": False,
            "publication_position": "after_valuation_tail_only",
            "current_row_observed_target_consumed": False,
            "synthetic_true_fair_pe_consumed": False,
            "semantic_role": "market_conditioned_delayed_feedback_eod_research",
        },
        "features": {
            "no_regime": no_regime_features,
            "with_regime": with_regime_features,
        },
        "regime_stacker_folds": stacker_diagnostics,
        "regime_proxy_mature_label_rows": int(proxy_labels.notna().sum()),
        "expected_pe_matched_pair_folds": matched_pair_diagnostics,
    }
    return output, diagnostics


def run_overlay(
    input_csv: str | Path,
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    truth_csv: str | Path | None = None,
) -> dict[str, Any]:
    """CSV adapter around the authoritative in-memory integration API."""

    input_path = Path(input_csv)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_frame = pd.read_csv(input_path, float_precision="round_trip")
    frame, layer_diagnostics = apply_v04_layers(input_frame, config)
    truth = pd.read_csv(truth_csv, float_precision="round_trip") if truth_csv else None
    regime_stacker_config = config.get("regime_stacker", {})
    ablation = build_ablation_report(
        frame,
        truth,
        forecast_horizon=int(regime_stacker_config.get("horizon", 21)),
        bull_return_threshold=float(regime_stacker_config.get("bull_return_threshold", 0.03)),
        bear_return_threshold=float(regime_stacker_config.get("bear_return_threshold", -0.03)),
    )
    stem = input_path.stem
    output_csv = output_path / f"{stem}_v04_bottleneck.csv"
    report_json = output_path / f"{stem}_v04_bottleneck_report.json"
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    round_tripped = pd.read_csv(output_csv, float_precision="round_trip")
    try:
        pd.testing.assert_frame_equal(
            round_tripped.loc[:, input_frame.columns],
            input_frame,
            check_exact=True,
            check_dtype=True,
            check_names=True,
            check_column_type=True,
            check_index_type=True,
        )
    except (AssertionError, KeyError) as exc:
        raise RuntimeError("CSV adapter did not preserve the frozen input prefix exactly") from exc
    layer_diagnostics["invariants"]["csv_float_precision"] = "round_trip"
    layer_diagnostics["invariants"]["csv_frozen_prefix_round_trip_exact"] = True
    report = {
        "version": "0.4.0",
        "input_csv": str(input_path),
        "output_csv": str(output_csv),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        **layer_diagnostics,
        **ablation,
    }
    write_json(report_json, report)
    return {
        "output_csv": output_csv,
        "report_json": report_json,
        "report": report,
    }
