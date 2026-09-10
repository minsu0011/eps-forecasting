"""Exact score-free candidate definitions transcribed from the bound design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import FeatureMetadata
from .contracts import (
    DistributionModelMetadata,
    PROBABILISTIC_DESIGN_SHA256,
    QUANTILE_LEVELS,
    ProbabilisticContractError,
)


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

FEATURE_COLUMNS = (*COMMON_FEATURES, *REGIME_EXTENSION)
CANDIDATE_IDS = (
    "qlinear_l1_with_regime_v1",
    "qhistgb_with_regime_v1",
    "ngboost_normal_crps_with_regime_v1",
)


@dataclass(frozen=True)
class ProbabilisticModelDefinition:
    model_id: str
    family: str
    variant: str
    package: str
    license: str
    execution_status: str
    full_density: bool
    parameters: Mapping[str, Any]
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS

    def __post_init__(self) -> None:
        if self.model_id not in CANDIDATE_IDS:
            raise ProbabilisticContractError("unknown probabilistic candidate id")
        if self.family != "probabilistic_distributional":
            raise ProbabilisticContractError("probabilistic family differs from the design")
        if self.feature_columns != FEATURE_COLUMNS:
            raise ProbabilisticContractError("probabilistic feature surface differs from design")

    def metadata(self) -> DistributionModelMetadata:
        return DistributionModelMetadata(
            model_id=self.model_id,
            family=self.family,
            variant=self.variant,
            package=self.package,
            license=self.license,
            feature_ids=self.feature_columns,
            quantile_levels=QUANTILE_LEVELS,
            full_density=self.full_density,
        )


MODEL_DEFINITIONS = (
    ProbabilisticModelDefinition(
        model_id="qlinear_l1_with_regime_v1",
        family="probabilistic_distributional",
        variant="sparse_linear_quantiles",
        package="scikit-learn==1.7.2",
        license="BSD-3-Clause",
        execution_status="IMPLEMENTABLE_NOW",
        full_density=False,
        parameters={
            "quantiles": QUANTILE_LEVELS,
            "alpha": 0.001,
            "fit_intercept": True,
            "solver": "highs",
            "solver_options": None,
            "sample_weight": None,
            "imputation": "training_median",
            "scaling": "training_standard_scaler",
        },
    ),
    ProbabilisticModelDefinition(
        model_id="qhistgb_with_regime_v1",
        family="probabilistic_distributional",
        variant="nonlinear_histogram_tree_quantiles",
        package="scikit-learn==1.7.2",
        license="BSD-3-Clause",
        execution_status="IMPLEMENTABLE_NOW",
        full_density=False,
        parameters={
            "quantiles": QUANTILE_LEVELS,
            "loss": "quantile",
            "max_iter": 180,
            "learning_rate": 0.04,
            "max_leaf_nodes": 15,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 0.2,
            "max_bins": 255,
            "max_features": 1.0,
            "early_stopping": False,
            "warm_start": False,
            "sample_weight": None,
            "missing": "native",
        },
    ),
    ProbabilisticModelDefinition(
        model_id="ngboost_normal_crps_with_regime_v1",
        family="probabilistic_distributional",
        variant="conditional_normal_log_pe_crps",
        package="ngboost==0.5.11",
        license="Apache-2.0",
        execution_status="CONDITIONAL_GO_METADATA_FIT_CLEAN_ROOM_AND_RUNTIME_PENDING",
        full_density=True,
        parameters={
            "quantiles": QUANTILE_LEVELS,
            "Dist": "Normal",
            "Score": "CRPScore",
            "Base": {
                "criterion": "friedman_mse",
                "splitter": "best",
                "max_depth": 2,
                "min_samples_leaf": 20,
                "max_features": None,
            },
            "natural_gradient": True,
            "n_estimators": 300,
            "learning_rate": 0.03,
            "minibatch_frac": 1.0,
            "col_sample": 1.0,
            "tol": 0.0001,
            "verbose": False,
            "early_stopping_rounds": None,
            "sample_weight": None,
            "imputation": "training_median",
            "scaling": None,
        },
    ),
)

MODEL_BY_ID = {definition.model_id: definition for definition in MODEL_DEFINITIONS}
if tuple(MODEL_BY_ID) != CANDIDATE_IDS or len(MODEL_BY_ID) != 3:
    raise RuntimeError("probabilistic definition table must have three ordered candidates")


def feature_metadata() -> tuple[FeatureMetadata, ...]:
    """Return the exact 36 already-audited Track-C feature declarations."""

    output: list[FeatureMetadata] = []
    for column in FEATURE_COLUMNS:
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
        output.append(
            FeatureMetadata(
                feature_id=column,
                column_name=column,
                description=f"Probabilistic-wave locked Track-C feature {column}",
                source="exact verified Wave-1 with_regime feature sidecar",
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
                provenance_reference=(
                    "outputs/model_zoo_probabilistic_wave_design_20260819/DESIGN.json"
                ),
                provenance_sha256=PROBABILISTIC_DESIGN_SHA256,
                prefix_invariance_status="PASS",
                future_intervention_status="PASS",
                same_row_target_leakage_status="PASS",
                pit_availability_status="PASS",
                publication_date_status=publication,
                restatement_availability_status=restatement,
                audit_notes=(
                    "Same-session Track-C feature; same-row observed_pe and evaluation truth "
                    "are excluded from the model frame."
                ),
            )
        )
    return tuple(output)
