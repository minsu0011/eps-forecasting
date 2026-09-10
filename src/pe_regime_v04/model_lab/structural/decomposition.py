"""Locked block-Ridge plus fixed-origin AR(1) Expected-P/E kernel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .ar import FixedOriginAR1Fit, fit_zero_intercept_ar1
from .authorization import StructuralExecutionAuthorization
from .contracts import (
    KernelBinding,
    STRUCTURAL_DESIGN_SHA256,
    StructuralContractError,
    canonical_json_bytes,
    normalize_dates,
    require_positive_finite,
    sha256_bytes,
)
from .features import (
    FUNDAMENTAL_CURRENT_PIT,
    TRACK_A_FEATURE_COLUMNS,
    TRACK_C_FEATURE_COLUMNS,
    require_exact_feature_columns,
)
from .preprocessing import (
    NumericPreprocessorFit,
    fit_numeric_preprocessor,
    transform_numeric_preprocessor,
)


StructuralTrack = Literal["A", "C"]
RIDGE_ALPHA = 10.0
RIDGE_SOLVER = "svd"
RIDGE_FIT_INTERCEPT = True


def feature_columns_for_track(track: StructuralTrack) -> tuple[str, ...]:
    if track == "A":
        return TRACK_A_FEATURE_COLUMNS
    if track == "C":
        return TRACK_C_FEATURE_COLUMNS
    raise StructuralContractError(f"unsupported structural track: {track!r}")


@dataclass(frozen=True)
class DecompositionFit:
    candidate_id: str
    track: StructuralTrack
    binding: KernelBinding
    preprocessor: NumericPreprocessorFit
    coefficients: tuple[float, ...]
    centered_intercept: float
    fundamental_center: float
    market_center: float
    ar1: FixedOriginAR1Fit
    train_end_iso: str
    train_row_count: int
    ridge_alpha: float = RIDGE_ALPHA
    ridge_solver: str = RIDGE_SOLVER
    ridge_fit_intercept: bool = RIDGE_FIT_INTERCEPT
    design_sha256: str = STRUCTURAL_DESIGN_SHA256

    def __post_init__(self) -> None:
        expected_candidate = {
            "A": "decomp_block_ridge_ar1_lag1",
            "C": "decomp_block_ridge_ar1_current",
        }.get(self.track)
        if self.candidate_id != expected_candidate:
            raise StructuralContractError("decomposition fit candidate/track differs")
        if self.track not in ("A", "C"):
            raise StructuralContractError("decomposition track must be A or C")
        if self.preprocessor.requested_columns != feature_columns_for_track(self.track):
            raise StructuralContractError("decomposition feature set differs from the frozen track")
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if len(coefficients) != len(self.preprocessor.active_columns):
            raise StructuralContractError("decomposition coefficient length differs")
        if not np.isfinite(coefficients).all():
            raise StructuralContractError("decomposition coefficients must be finite")
        diagnostics = np.asarray(
            [self.centered_intercept, self.fundamental_center, self.market_center],
            dtype=np.float64,
        )
        if not np.isfinite(diagnostics).all():
            raise StructuralContractError("decomposition centers/intercept must be finite")
        if self.ridge_alpha != RIDGE_ALPHA or self.ridge_solver != RIDGE_SOLVER:
            raise StructuralContractError("decomposition Ridge parameters differ from the design")
        if self.ridge_fit_intercept is not True:
            raise StructuralContractError("decomposition Ridge must fit an intercept")
        if self.design_sha256 != STRUCTURAL_DESIGN_SHA256:
            raise StructuralContractError("decomposition fit is bound to a different design")
        if self.train_row_count < 2:
            raise StructuralContractError("decomposition fit requires at least two rows")
        dates = normalize_dates([self.train_end_iso], context="decomposition train_end")
        if len(dates) != 1:
            raise StructuralContractError("decomposition train_end is invalid")

    def to_process_payload(self) -> dict[str, Any]:
        """Return a JSON-only payload safe for spawn-based process workers."""

        payload = {
            "format_version": 1,
            "candidate_id": self.candidate_id,
            "track": self.track,
            "binding": self.binding.as_dict(),
            "preprocessor": {
                "requested_columns": list(self.preprocessor.requested_columns),
                "active_columns": list(self.preprocessor.active_columns),
                "dropped_all_missing_columns": list(self.preprocessor.dropped_all_missing_columns),
                "medians": list(self.preprocessor.medians),
                "scales": list(self.preprocessor.scales),
            },
            "coefficients": list(self.coefficients),
            "centered_intercept": self.centered_intercept,
            "fundamental_center": self.fundamental_center,
            "market_center": self.market_center,
            "ar1": {
                "rho": self.ar1.rho,
                "last_residual": self.ar1.last_residual,
                "consecutive_pair_count": self.ar1.consecutive_pair_count,
                "minimum_pair_count": self.ar1.minimum_pair_count,
            },
            "train_end_iso": self.train_end_iso,
            "train_row_count": self.train_row_count,
            "ridge_alpha": self.ridge_alpha,
            "ridge_solver": self.ridge_solver,
            "ridge_fit_intercept": self.ridge_fit_intercept,
            "design_sha256": self.design_sha256,
        }
        # Fails here if a future edit introduces arrays, NaN, or non-process-safe state.
        canonical_json_bytes(payload)
        return payload

    @property
    def fit_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_process_payload()))

    @classmethod
    def from_process_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        authorization: StructuralExecutionAuthorization,
    ) -> "DecompositionFit":
        if (
            set(payload)
            != {
                "format_version",
                "candidate_id",
                "track",
                "binding",
                "preprocessor",
                "coefficients",
                "centered_intercept",
                "fundamental_center",
                "market_center",
                "ar1",
                "train_end_iso",
                "train_row_count",
                "ridge_alpha",
                "ridge_solver",
                "ridge_fit_intercept",
                "design_sha256",
            }
            or payload.get("format_version") != 1
        ):
            raise StructuralContractError("decomposition process payload schema is invalid")
        binding_raw = payload["binding"]
        prep_raw = payload["preprocessor"]
        ar_raw = payload["ar1"]
        if not all(isinstance(value, Mapping) for value in (binding_raw, prep_raw, ar_raw)):
            raise StructuralContractError("decomposition process payload objects are invalid")
        return cls(
            candidate_id=str(payload["candidate_id"]),
            track=str(payload["track"]),  # type: ignore[arg-type]
            binding=KernelBinding.from_dict(dict(binding_raw), authorization=authorization),
            preprocessor=NumericPreprocessorFit(
                requested_columns=tuple(prep_raw["requested_columns"]),
                active_columns=tuple(prep_raw["active_columns"]),
                dropped_all_missing_columns=tuple(prep_raw["dropped_all_missing_columns"]),
                medians=tuple(float(value) for value in prep_raw["medians"]),
                scales=tuple(float(value) for value in prep_raw["scales"]),
            ),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            centered_intercept=float(payload["centered_intercept"]),
            fundamental_center=float(payload["fundamental_center"]),
            market_center=float(payload["market_center"]),
            ar1=FixedOriginAR1Fit(**dict(ar_raw)),
            train_end_iso=str(payload["train_end_iso"]),
            train_row_count=int(payload["train_row_count"]),
            ridge_alpha=float(payload["ridge_alpha"]),
            ridge_solver=str(payload["ridge_solver"]),
            ridge_fit_intercept=bool(payload["ridge_fit_intercept"]),
            design_sha256=str(payload["design_sha256"]),
        )


def _training_weights(frame: pd.DataFrame) -> np.ndarray:
    confidence = pd.to_numeric(frame["eps_confidence"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    confidence[~np.isfinite(confidence)] = 50.0
    return np.clip(confidence, 5.0, 100.0) / 100.0


def _validated_ordered_dates(values: pd.Series | np.ndarray, *, context: str) -> pd.Series:
    dates = normalize_dates(values, context=context)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise StructuralContractError(f"{context} must be strictly increasing and unique")
    return dates


def fit_decomposition_kernel(
    train_features: pd.DataFrame,
    observed_pe: pd.Series | np.ndarray,
    train_dates: pd.Series | np.ndarray,
    *,
    candidate_id: str,
    track: StructuralTrack,
    binding: KernelBinding,
    authorization: StructuralExecutionAuthorization,
) -> DecompositionFit:
    if not isinstance(authorization, StructuralExecutionAuthorization):
        raise StructuralContractError("decomposition fit requires sealed authorization")
    authorization.require_candidate(candidate_id)
    binding.verify(authorization, candidate_id=candidate_id, track=track)
    feature_columns = feature_columns_for_track(track)
    require_exact_feature_columns(
        train_features,
        feature_columns=feature_columns,
        context="decomposition train features",
    )
    if len(train_features) < 2:
        raise StructuralContractError("decomposition fit requires at least two rows")
    dates = _validated_ordered_dates(train_dates, context="decomposition train dates")
    if len(dates) != len(train_features):
        raise StructuralContractError("decomposition train date/feature row counts differ")
    target = require_positive_finite(observed_pe, context="decomposition observed_pe")
    if len(target) != len(train_features):
        raise StructuralContractError("decomposition target/feature row counts differ")

    preprocessor, standardized = fit_numeric_preprocessor(
        train_features,
        feature_columns,
        context="decomposition train preprocessing",
    )
    weights = _training_weights(train_features)
    estimator = Ridge(alpha=RIDGE_ALPHA, solver=RIDGE_SOLVER, fit_intercept=True)
    estimator.fit(standardized, np.log(target), sample_weight=weights)
    coefficients = np.asarray(estimator.coef_, dtype=np.float64)
    active = preprocessor.active_columns
    fundamental_indices = np.asarray(
        [index for index, column in enumerate(active) if column in FUNDAMENTAL_CURRENT_PIT],
        dtype=np.int64,
    )
    market_indices = np.asarray(
        [index for index, column in enumerate(active) if column not in FUNDAMENTAL_CURRENT_PIT],
        dtype=np.int64,
    )
    fundamental_raw = (
        standardized[:, fundamental_indices] @ coefficients[fundamental_indices]
        if fundamental_indices.size
        else np.zeros(len(train_features), dtype=np.float64)
    )
    market_raw = (
        standardized[:, market_indices] @ coefficients[market_indices]
        if market_indices.size
        else np.zeros(len(train_features), dtype=np.float64)
    )
    fundamental_center = float(np.average(fundamental_raw, weights=weights))
    market_center = float(np.average(market_raw, weights=weights))
    centered_intercept = float(estimator.intercept_) + fundamental_center + market_center
    fitted_log = (
        centered_intercept + (fundamental_raw - fundamental_center) + (market_raw - market_center)
    )
    residuals = np.log(target) - fitted_log
    ar1 = fit_zero_intercept_ar1(
        residuals,
        session_positions=np.arange(len(residuals), dtype=np.int64),
    )
    train_end = pd.Timestamp(dates.iloc[-1]).isoformat()
    return DecompositionFit(
        candidate_id=candidate_id,
        track=track,
        binding=binding,
        preprocessor=preprocessor,
        coefficients=tuple(float(value) for value in coefficients),
        centered_intercept=centered_intercept,
        fundamental_center=fundamental_center,
        market_center=market_center,
        ar1=ar1,
        train_end_iso=train_end,
        train_row_count=len(train_features),
    )


def predict_decomposition_kernel(
    fit: DecompositionFit,
    test_features: pd.DataFrame,
    test_dates: pd.Series | np.ndarray,
    *,
    authorization: StructuralExecutionAuthorization,
) -> pd.DataFrame:
    """Forecast from the fixed training origin; no test target argument exists."""

    if not isinstance(authorization, StructuralExecutionAuthorization):
        raise StructuralContractError("decomposition predict requires sealed authorization")
    authorization.require_candidate(fit.candidate_id)
    fit.binding.verify(
        authorization,
        candidate_id=fit.candidate_id,
        track=fit.track,
    )

    columns = feature_columns_for_track(fit.track)
    require_exact_feature_columns(
        test_features,
        feature_columns=columns,
        context="decomposition test features",
    )
    dates = _validated_ordered_dates(test_dates, context="decomposition test dates")
    if len(test_features) == 0 or len(dates) != len(test_features):
        raise StructuralContractError("decomposition test rows must be non-empty and aligned")
    train_end = normalize_dates([fit.train_end_iso], context="decomposition fit train_end").iloc[0]
    if not (dates > train_end).all():
        raise StructuralContractError("decomposition test dates must strictly follow train_end")

    standardized = transform_numeric_preprocessor(
        test_features,
        fit.preprocessor,
        context="decomposition test preprocessing",
    )
    coefficients = np.asarray(fit.coefficients, dtype=np.float64)
    active = fit.preprocessor.active_columns
    fundamental_indices = np.asarray(
        [index for index, column in enumerate(active) if column in FUNDAMENTAL_CURRENT_PIT],
        dtype=np.int64,
    )
    market_indices = np.asarray(
        [index for index, column in enumerate(active) if column not in FUNDAMENTAL_CURRENT_PIT],
        dtype=np.int64,
    )
    fundamental = (
        standardized[:, fundamental_indices] @ coefficients[fundamental_indices]
        - fit.fundamental_center
        if fundamental_indices.size
        else np.full(len(test_features), -fit.fundamental_center, dtype=np.float64)
    )
    market = (
        standardized[:, market_indices] @ coefficients[market_indices] - fit.market_center
        if market_indices.size
        else np.full(len(test_features), -fit.market_center, dtype=np.float64)
    )
    dynamic = fit.ar1.forecast(len(test_features))
    expected_log_pe = fit.centered_intercept + fundamental + market + dynamic
    expected_pe = np.exp(expected_log_pe)
    if not np.isfinite(expected_pe).all() or (expected_pe <= 0.0).any():
        raise StructuralContractError("decomposition expected P/E is non-positive or non-finite")
    return pd.DataFrame(
        {
            "date": dates.to_numpy(copy=True),
            "intercept": np.full(len(test_features), fit.centered_intercept, dtype=np.float64),
            "fundamental_component": fundamental,
            "market_component": market,
            "dynamic_residual_component": dynamic,
            "expected_log_pe": expected_log_pe,
            "expected_pe": expected_pe,
        }
    )
