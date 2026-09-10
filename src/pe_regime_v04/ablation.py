from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .labels import future_return_proxy_labels
from .metrics import (
    candidate_metrics,
    paired_candidate_metrics,
    regime_metrics,
    valuation_direction_metrics,
    valuation_state_metrics,
)


def build_ablation_report(
    frame: pd.DataFrame,
    truth: pd.DataFrame | None = None,
    *,
    forecast_horizon: int = 21,
    bull_return_threshold: float = 0.03,
    bear_return_threshold: float = -0.03,
) -> dict[str, Any]:
    working = frame.copy()
    working_dates = pd.to_datetime(working["date"], errors="raise")
    if working_dates.duplicated().any() or not working_dates.is_monotonic_increasing:
        raise ValueError("ablation frame dates must be unique and monotonic increasing")
    fair_pe = None
    true_regime = None
    truth_alignment: dict[str, Any] = {"available": False}
    if truth is not None:
        truth_frame = truth.copy()
        truth_frame["date"] = pd.to_datetime(truth_frame["date"], errors="raise")
        if truth_frame["date"].duplicated().any():
            raise ValueError("truth dates must be unique")
        versions: list[str] = []
        if "demo_generator_version" in truth_frame.columns:
            versions = sorted(
                str(value)
                for value in truth_frame["demo_generator_version"].dropna().unique()
            )
            if len(versions) > 1:
                raise ValueError("truth contains mixed demo_generator_version values")
        aligned = pd.DataFrame({"date": working_dates}).merge(
            truth_frame,
            on="date",
            how="left",
            validate="one_to_one",
        )
        # ``merge`` creates a RangeIndex even when the authoritative in-memory
        # frame uses another unique index.  Metrics align Series by index, so put
        # the truth rows back on the exact working index before any arithmetic.
        aligned.index = working.index
        fair_column = "true_fair_pe" if "true_fair_pe" in aligned.columns else None
        if fair_column:
            fair_pe = aligned[fair_column]
        if "true_regime" in aligned.columns:
            true_regime = aligned["true_regime"]
        truth_alignment = {
            "available": True,
            "input_rows": int(len(truth_frame)),
            "matched_rows": int(aligned.drop(columns=["date"]).notna().any(axis=1).sum()),
            "demo_generator_versions": versions,
            "observed_pe_checked_rows": 0,
            "observed_pe_max_abs_error": None,
        }
        if "true_observed_pe" in aligned.columns:
            actual = pd.to_numeric(working["observed_pe"], errors="coerce")
            expected = pd.to_numeric(aligned["true_observed_pe"], errors="coerce")
            valid = np.isfinite(actual) & np.isfinite(expected)
            truth_alignment["observed_pe_checked_rows"] = int(valid.sum())
            if valid.any():
                error = np.abs(actual.loc[valid] - expected.loc[valid])
                max_error = float(error.max())
                truth_alignment["observed_pe_max_abs_error"] = max_error
                scale = np.maximum(np.abs(expected.loc[valid].to_numpy(dtype=float)), 1.0)
                if np.any(error.to_numpy(dtype=float) > 1e-10 * scale):
                    raise ValueError("truth true_observed_pe does not match output observed_pe")

    candidates = {
        "B0_observed_pe_identity": "observed_pe",
        "B1_pe_median_756_lag": "pe_median_756_lag",
        "B2_global_statistical": "v04_global_statistical_expected_pe",
        "B2b_global_full_coverage": "v04_global_baseline_expected_pe",
        "B3_v03_regime_statistical": "statistical_expected_pe",
        "B4_guarded_statistical": "v04_guarded_statistical_expected_pe",
        "M0_v03_expected_pe": "expected_pe",
        "M1_v03_ml_with_regime": "ml_expected_pe",
        "M2_v04_ml_no_regime": "v04_ml_expected_pe_no_regime",
        "M3_v04_ml_with_regime": "v04_ml_expected_pe_with_regime",
        "M3b_v04_matched_best": "v04_matched_best_ml_expected_pe",
        "M4_v04_ml_incumbent_safe": "v04_guarded_ml_expected_pe",
        "M5_v04_dynamic_blend_diagnostic": "v04_dynamic_blend_expected_pe",
        "M6_v04_final_safe": "v04_expected_pe",
    }
    candidate_results: dict[str, Any] = {}
    for name, column in candidates.items():
        if column not in working.columns:
            continue
        candidate_results[name] = {
            "column": column,
            **candidate_metrics(working[column], working["observed_pe"], fair_pe),
        }

    paired_definitions = {
        "M2_no_regime_vs_M3_with_regime": (
            "v04_ml_expected_pe_no_regime",
            "v04_ml_expected_pe_with_regime",
        ),
        "M0_v03_incumbent_vs_M3b_matched_best": (
            "ml_expected_pe",
            "v04_matched_best_ml_expected_pe",
        ),
        "M4_guarded_ml_vs_B4_guarded_statistical": (
            "v04_guarded_ml_expected_pe",
            "v04_guarded_statistical_expected_pe",
        ),
    }
    paired_results: dict[str, Any] = {}
    for name, (baseline_column, challenger_column) in paired_definitions.items():
        if baseline_column in working.columns and challenger_column in working.columns:
            paired_results[name] = {
                "baseline_column": baseline_column,
                "challenger_column": challenger_column,
                **paired_candidate_metrics(
                    working[baseline_column],
                    working[challenger_column],
                    working["observed_pe"],
                    fair_pe,
                ),
            }

    regime_results: dict[str, Any] = {}
    if true_regime is not None:
        probability_sets = {
            "v03_ensemble": ["p_bear", "p_sideways", "p_bull"],
            "v04_current_state": [
                "v04_current_p_bear",
                "v04_current_p_sideways",
                "v04_current_p_bull",
            ],
        }
        for name, columns in probability_sets.items():
            if all(column in working.columns for column in columns):
                probabilities = working[columns].copy()
                probabilities.columns = ["p_bear", "p_sideways", "p_bull"]
                regime_results[name] = regime_metrics(probabilities, true_regime)

    forecast_proxy = future_return_proxy_labels(
        working["benchmark_close"],
        horizon=int(forecast_horizon),
        bull_threshold=float(bull_return_threshold),
        bear_threshold=float(bear_return_threshold),
    )
    forecast_results: dict[str, Any] = {}
    forecast_probability_sets = {
        "v04_return_forecast_raw": [
            "v04_stacker_p_bear",
            "v04_stacker_p_sideways",
            "v04_stacker_p_bull",
        ],
        "v04_return_forecast": [
            "v04_return_forecast_p_bear",
            "v04_return_forecast_p_sideways",
            "v04_return_forecast_p_bull",
        ],
    }
    for name, columns in forecast_probability_sets.items():
        if all(column in working.columns for column in columns):
            probabilities = working[columns].copy()
            probabilities.columns = ["p_bear", "p_sideways", "p_bull"]
            forecast_results[name] = regime_metrics(probabilities, forecast_proxy)

    valuation_results = None
    valuation_state_results: dict[str, Any] = {}
    if fair_pe is not None and "v04_expected_pe" in working.columns:
        valuation_results = valuation_direction_metrics(
            working["observed_pe"],
            working["v04_expected_pe"],
            fair_pe,
        )
        state_columns = {
            "absolute_10pct": "v04_valuation_state_absolute",
            "adaptive_tail": "v04_valuation_state_adaptive",
            "confirmed_tail": "v04_valuation_state_confirmed",
        }
        for name, column in state_columns.items():
            if column in working.columns:
                valuation_state_results[name] = valuation_state_metrics(
                    working[column],
                    working["observed_pe"],
                    fair_pe,
                )

    return {
        "truth_alignment": truth_alignment,
        "candidate_ablation": candidate_results,
        "paired_candidate_ablation": paired_results,
        "regime_ablation": regime_results,
        "return_forecast_ablation": forecast_results,
        "valuation_gap": valuation_results,
        "valuation_state_ablation": valuation_state_results,
    }
