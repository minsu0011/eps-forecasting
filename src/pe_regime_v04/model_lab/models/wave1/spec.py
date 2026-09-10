"""Exact Wave-1 definitions transcribed from DESIGN_LOCK B3A190... ."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from ...contracts import FeatureMetadata


DESIGN_LOCK_SHA256 = "b3a190cbf80045b460551af4b9a53476d5c369e47f2cbf460a02d8779672c82d"
ORIGINAL_PRECOMMIT_SHA256 = "008e468b4b58ce13fd7d4c2791353b912841f7efc110ec851df50552c6ea849f"
EVALUATION_START = "2015-01-02"
EVIDENCE_SEEDS = (6301, 6421, 6521, 6607, 6701)

COMMON_FEATURES = (
    "benchmark_return_21",
    "benchmark_return_63",
    "benchmark_return_126",
    "benchmark_return_252",
    "benchmark_realized_vol_20",
    "benchmark_realized_vol_63",
    "benchmark_drawdown_252",
    "benchmark_price_vs_sma_50",
    "benchmark_price_vs_sma_200",
    "benchmark_sma_50_vs_200",
    "benchmark_sma_200_slope_20",
    "benchmark_trend_efficiency_63",
    "benchmark_volume_log_z_63",
    "stock_return_63",
    "stock_return_126",
    "eps_ttm_growth_126",
    "eps_ttm_growth_252",
    "eps_staleness_days",
    "eps_period_age_days",
    "eps_confidence",
    "eps_disagreement",
    "eps_approximation_flag",
    "pe_median_252_lag",
    "pe_median_756_lag",
)

REGIME_EXTENSION = (
    "v04_current_p_bear",
    "v04_current_p_sideways",
    "v04_current_p_bull",
    "forecast_p_bear",
    "forecast_p_sideways",
    "forecast_p_bull",
    "pe_bear_median",
    "pe_sideways_median",
    "pe_bull_median",
    "pe_bear_effective_history",
    "pe_sideways_effective_history",
    "pe_bull_effective_history",
)

SPLINE_COLUMNS = (
    "benchmark_return_63",
    "benchmark_return_252",
    "benchmark_realized_vol_20",
    "benchmark_realized_vol_63",
    "benchmark_drawdown_252",
    "benchmark_price_vs_sma_200",
    "stock_return_63",
    "stock_return_126",
    "eps_ttm_growth_126",
    "eps_ttm_growth_252",
    "eps_staleness_days",
    "eps_confidence",
    "pe_median_252_lag",
    "pe_median_756_lag",
)

BASELINE_MODEL_IDS = (
    "v04_expected_pe",
    "ml_expected_pe",
    "v04_ml_expected_pe_no_regime",
    "v04_ml_expected_pe_with_regime",
)
PRIMARY_BASELINE_IDS = ("v04_expected_pe", "ml_expected_pe")
DIRECT_REGIME_REFERENCE_ID = "v04_ml_expected_pe_with_regime"


@dataclass(frozen=True)
class Wave1ModelDefinition:
    model_id: str
    family: str
    variant: str
    feature_columns: tuple[str, ...]
    parameters: Mapping[str, Any]
    package: str
    license: str


_FAMILY_PARAMETERS: dict[str, dict[str, Any]] = {
    "ridge_svd": {"alpha": 10.0, "fit_intercept": True, "solver": "svd"},
    "elastic_net_cyclic": {
        "alpha": 0.001,
        "l1_ratio": 0.2,
        "max_iter": 10000,
        "selection": "cyclic",
        "tol": 1.0e-7,
    },
    "huber": {"epsilon": 1.35, "alpha": 0.0001, "max_iter": 1000, "tol": 1.0e-7},
    "spline_ridge": {
        "n_knots": 4,
        "degree": 3,
        "knots": "quantile",
        "extrapolation": "linear",
        "include_bias": False,
        "order": "C",
        "sparse_output": False,
        "ridge_alpha": 10.0,
        "ridge_solver": "svd",
    },
    "extra_trees": {
        "n_estimators": 256,
        "criterion": "squared_error",
        "min_samples_leaf": 8,
        "max_features": 0.75,
        "bootstrap": False,
        "n_jobs": 1,
    },
    "hist_gradient_boosting": {
        "loss": "absolute_error",
        "max_iter": 180,
        "learning_rate": 0.04,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.2,
        "early_stopping": False,
    },
    "catboost_cpu": {
        "loss_function": "MAE",
        "iterations": 300,
        "depth": 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 3.0,
        "thread_count": 1,
        "task_type": "CPU",
        "allow_writing_files": False,
        "verbose": False,
    },
    "xgboost_cpu": {
        "objective": "reg:absoluteerror",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 10.0,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": 1,
    },
    "state_space_local_linear_trend": {
        "level": "local linear trend",
        "irregular": True,
        "fit_method": "lbfgs",
        "maxiter": 100,
        "disp": False,
        "forecast": "fixed_origin_full_test_window",
        "within_test_updates": False,
    },
}

_PACKAGES = {
    "ridge_svd": ("scikit-learn==1.7.2", "BSD-3-Clause"),
    "elastic_net_cyclic": ("scikit-learn==1.7.2", "BSD-3-Clause"),
    "huber": ("scikit-learn==1.7.2", "BSD-3-Clause"),
    "spline_ridge": ("scikit-learn==1.7.2", "BSD-3-Clause"),
    "extra_trees": ("scikit-learn==1.7.2", "BSD-3-Clause"),
    "hist_gradient_boosting": ("scikit-learn==1.7.2", "BSD-3-Clause"),
    "catboost_cpu": ("catboost==1.2.10", "Apache-2.0"),
    "xgboost_cpu": ("xgboost==3.2.0", "Apache-2.0"),
    "state_space_local_linear_trend": ("statsmodels==0.14.6", "BSD-3-Clause"),
}


def _definitions() -> tuple[Wave1ModelDefinition, ...]:
    rows: list[Wave1ModelDefinition] = []
    for family in (
        "ridge_svd",
        "elastic_net_cyclic",
        "huber",
        "spline_ridge",
        "extra_trees",
        "hist_gradient_boosting",
        "catboost_cpu",
        "xgboost_cpu",
    ):
        package, license_name = _PACKAGES[family]
        for variant in ("common", "with_regime"):
            columns = COMMON_FEATURES if variant == "common" else (*COMMON_FEATURES, *REGIME_EXTENSION)
            rows.append(
                Wave1ModelDefinition(
                    model_id=f"{family}_{variant}",
                    family=family,
                    variant=variant,
                    feature_columns=tuple(columns),
                    parameters=dict(_FAMILY_PARAMETERS[family]),
                    package=package,
                    license=license_name,
                )
            )
    package, license_name = _PACKAGES["state_space_local_linear_trend"]
    rows.append(
        Wave1ModelDefinition(
            model_id="state_space_local_linear_trend_target_history_only",
            family="state_space_local_linear_trend",
            variant="target_history_only",
            feature_columns=(),
            parameters=dict(_FAMILY_PARAMETERS["state_space_local_linear_trend"]),
            package=package,
            license=license_name,
        )
    )
    return tuple(rows)


WAVE1_MODEL_DEFINITIONS = _definitions()
CANDIDATE_MODEL_IDS = tuple(definition.model_id for definition in WAVE1_MODEL_DEFINITIONS)
MODEL_BY_ID = {definition.model_id: definition for definition in WAVE1_MODEL_DEFINITIONS}

if len(MODEL_BY_ID) != 17 or len(MODEL_BY_ID) != len(WAVE1_MODEL_DEFINITIONS):
    raise RuntimeError("Wave1 definition table must contain exactly 17 unique candidates")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def logical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def feature_metadata(columns: tuple[str, ...]) -> tuple[FeatureMetadata, ...]:
    """Build immutable PIT declarations for the locked v0.4 feature columns."""

    rows: list[FeatureMetadata] = []
    for column in columns:
        if column.startswith("eps_"):
            family = "EPS"
            publication = "PASS"
            restatement = "PASS"
        elif column.startswith("pe_"):
            family = "VALUATION"
            publication = "NOT_APPLICABLE"
            restatement = "NOT_APPLICABLE"
        elif column.startswith("forecast_") or column.startswith("v04_current_p_"):
            family = "REGIME"
            publication = "NOT_APPLICABLE"
            restatement = "NOT_APPLICABLE"
        else:
            family = "MARKET"
            publication = "NOT_APPLICABLE"
            restatement = "NOT_APPLICABLE"
        lagged = column.startswith("pe_median_") and column.endswith("_lag")
        rows.append(
            FeatureMetadata(
                feature_id=column,
                column_name=column,
                description=f"Wave1 locked PIT feature {column}",
                source="v0.4 canonical150 or exact output254 PIT sidecar",
                dtype="float64",
                availability="lagged" if lagged else "point_in_time",
                availability_lag_sessions=1 if lagged else 0,
                lookahead_sessions=0,
                evaluation_only=False,
                allowed_for_fit=True,
                allowed_for_predict=True,
                point_in_time_safe=True,
                uses_revised_data=False,
                feature_family=family,
                provenance_reference="outputs/model_zoo_wave1_screen_20260819/DESIGN_LOCK.json",
                provenance_sha256=DESIGN_LOCK_SHA256,
                prefix_invariance_status="PASS",
                future_intervention_status="PASS",
                same_row_target_leakage_status="PASS",
                pit_availability_status="PASS",
                publication_date_status=publication,
                restatement_availability_status=restatement,
                audit_notes="Frozen Wave1 spent-seed screening feature; truth is never in fit/predict.",
            )
        )
    return tuple(rows)


GATE_POLICY = {
    "gain_formula": "(baseline_loss-candidate_loss)/baseline_loss",
    "primary_metric_min_relative_gain": 0.005,
    "primary_other_metric_max_relative_degradation": 0.005,
    "complementarity_max_relative_degradation_each_primary_metric": 0.01,
    "complementarity_min_oracle_relative_gain_mae": 0.01,
    "complementarity_max_pearson_absolute_error_correlation": 0.95,
    "max_worst_seed_relative_degradation_over_both_primary_metrics": 0.02,
    "coverage_required": 1.0,
    "max_variants_advanced_per_family": 1,
    "max_families_advanced": 3,
    "variant_and_family_order": ["fair_log_mae", "fair_log_rmse", "family", "model_id"],
}

RESOURCE_POLICY = {
    "outer_backend": "thread",
    "outer_workers_max": 32,
    "inner_threads": 1,
    "affinity_logical_processors": list(range(32)),
    "max_total_rss_bytes": 64 * 1024**3,
    "minimum_free_ram_bytes": 16 * 1024**3,
    "gpu": "sealed_off",
    "runtime_boundary": "sum of per-fold adapter preprocessing+fit+predict+validation wall seconds; excludes input I/O, queue wait, and artifact serialization",
}
