"""Fail-closed all-five-quantile coverage over an authorized common mask."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .authorization import ExecutionAuthorization
from .contracts import (
    DENSITY_OUTPUT_COLUMNS,
    MODEL_OUTPUT_COLUMNS,
    NORMAL_QUANTILE_Z,
    PE_QUANTILE_COLUMNS,
    ProbabilisticContractError,
    VerifiedPredictionBatch,
    identity_sha256,
    logical_frame_sha256,
    normalize_identity_frame,
    require_no_evaluation_truth,
    require_unique_columns,
)
from .spec import CANDIDATE_IDS


def _validate_prediction_frame(
    predictions: pd.DataFrame,
    *,
    authorized_identity: pd.DataFrame,
    allowed_model_ids: tuple[str, ...] = CANDIDATE_IDS,
    context: str = "candidate",
) -> tuple[pd.DataFrame, str, str]:
    """Validate bytes parsed by the sole custody loader (not a public factory)."""

    require_unique_columns(predictions, context=f"{context} predictions")
    require_no_evaluation_truth(predictions.columns, context=f"{context} predictions")
    missing = [column for column in MODEL_OUTPUT_COLUMNS if column not in predictions]
    extra = [
        column
        for column in predictions
        if column not in (*MODEL_OUTPUT_COLUMNS, *DENSITY_OUTPUT_COLUMNS)
    ]
    density_present = [column for column in DENSITY_OUTPUT_COLUMNS if column in predictions]
    if missing or extra or len(density_present) not in {0, len(DENSITY_OUTPUT_COLUMNS)}:
        raise ProbabilisticContractError(
            f"candidate prediction schema mismatch; missing={missing}, extra={extra}"
        )
    output_columns = (*MODEL_OUTPUT_COLUMNS, *density_present)
    if tuple(predictions.columns) != output_columns:
        raise ProbabilisticContractError("candidate prediction column order differs")
    candidate_identity = normalize_identity_frame(
        predictions, context="candidate identity", sort=False
    )
    expected_identity = normalize_identity_frame(
        authorized_identity, context="authorized identity", sort=False
    )
    expected_hash = identity_sha256(expected_identity, sort=False)
    if (
        len(candidate_identity) != len(expected_identity)
        or identity_sha256(candidate_identity, sort=False) != expected_hash
    ):
        raise ProbabilisticContractError(
            "candidate identity/order differs from authorized common mask; mask shrink forbidden"
        )
    values = predictions.loc[:, list(PE_QUANTILE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ProbabilisticContractError("all five candidate quantiles must be positive and finite")
    if (np.diff(values, axis=1) <= 0.0).any():
        raise ProbabilisticContractError(
            "all five candidate quantiles must be unique and strictly ordered"
        )
    alias = predictions["expected_pe"].to_numpy(dtype=np.float64)
    if not np.array_equal(alias, values[:, 2]):
        raise ProbabilisticContractError("expected_pe must exactly alias predicted_pe_p50")
    model_ids = tuple(pd.unique(predictions["model_id"]))
    if len(model_ids) != 1 or model_ids[0] not in allowed_model_ids:
        raise ProbabilisticContractError(f"prediction batch must contain one locked {context} id")
    uncertainty = predictions.loc[
        :,
        [
            "uncertainty_log_iqr",
            "uncertainty_log_idr",
            "uncertainty_robust_sigma",
            "uncertainty_p90_p10_ratio",
        ],
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(uncertainty).all() or (uncertainty[:, :3] < 0.0).any():
        raise ProbabilisticContractError("uncertainty outputs are invalid")
    if (uncertainty[:, 3] <= 1.0).any():
        raise ProbabilisticContractError("strict quantiles require p90/p10 ratio above one")
    expected_uncertainty = np.column_stack(
        [
            np.log(values[:, 3]) - np.log(values[:, 1]),
            np.log(values[:, 4]) - np.log(values[:, 0]),
            (np.log(values[:, 3]) - np.log(values[:, 1])) / 1.3489795003921634,
            values[:, 4] / values[:, 0],
        ]
    )
    if not np.allclose(uncertainty, expected_uncertainty, rtol=0.0, atol=2e-14):
        raise ProbabilisticContractError("uncertainty outputs do not match the five quantiles")
    if density_present:
        if model_ids[0] != "ngboost_normal_crps_with_regime_v1":
            raise ProbabilisticContractError("only the locked NGBoost candidate may emit density")
        density = predictions.loc[:, list(DENSITY_OUTPUT_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(density).all() or (density[:, 1] <= 0.0).any():
            raise ProbabilisticContractError("density loc/scale are invalid")
        normal_log_quantiles = (
            density[:, 0, None] + density[:, 1, None] * NORMAL_QUANTILE_Z[None, :]
        )
        if not np.array_equal(np.exp(normal_log_quantiles), values):
            raise ProbabilisticContractError(
                "NGBoost quantiles are not generated from the published Normal loc/scale"
            )
    normalized = predictions.loc[:, list(output_columns)].copy()
    return normalized, expected_hash, logical_frame_sha256(normalized)


def verify_prediction_coverage(
    predictions: VerifiedPredictionBatch,
    *,
    authorization: ExecutionAuthorization,
) -> str:
    """Accept only a loader-created artifact over the authorization-owned mask."""

    if not isinstance(predictions, VerifiedPredictionBatch):
        raise ProbabilisticContractError(
            "coverage verification requires a content-addressed prediction artifact"
        )
    if not isinstance(authorization, ExecutionAuthorization):
        raise ProbabilisticContractError("coverage verification requires factory authorization")
    authorization.verify_integrity()
    predictions.verify_integrity()
    expected = identity_sha256(authorization.identity_frame(), sort=False)
    if predictions.identity_sha256 != expected:
        raise ProbabilisticContractError("prediction common-mask identity differs")
    return expected


def exact_common_identity(
    artifacts: tuple[VerifiedPredictionBatch, ...],
    *,
    authorization: ExecutionAuthorization,
) -> str:
    """Verify all candidates cover the exact authorization-owned row sequence."""

    if not artifacts:
        raise ProbabilisticContractError("common identity requires at least one participant")
    hashes = {
        verify_prediction_coverage(artifact, authorization=authorization) for artifact in artifacts
    }
    if len(hashes) != 1:
        raise ProbabilisticContractError("participant common masks differ")
    return next(iter(hashes))
