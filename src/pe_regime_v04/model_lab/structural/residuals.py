"""Authorized strictly cross-fitted residual correction entry points."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .ar import AR_RHO_MAX, AR_RHO_MIN, FixedOriginAR1Fit, fit_zero_intercept_ar1
from .authorization import (
    EXPECTED_RESIDUAL_BASE,
    StructuralExecutionAuthorization,
)
from .contracts import (
    FULL_META_IDENTITY_COLUMNS,
    STRUCTURAL_DESIGN_SHA256,
    StructuralContractError,
    canonical_json_bytes,
    require_exact_identity,
    require_no_evaluation_truth,
    require_positive_finite,
    require_sha256,
    sha256_bytes,
)
from .meta import GeneratedMetaFeatureArtifact


HUBER_EPSILON = 1.35
HUBER_ALPHA = 0.0001
HUBER_MAX_ITER = 1000
HUBER_TOL = 1e-7
CORRECTION_BOUND = math.log(1.5)
MINIMUM_META_ROWS = 252
RESIDUAL_AR1_CANDIDATE = "residual_ar1_nested_oof"
RESIDUAL_HUBER_CANDIDATE = "residual_huber_nested_oof"
_FIT_TOKEN = object()


@dataclass(frozen=True, init=False)
class BoundBaseFeatureContract:
    """Removed caller-authored contract retained as an explicit tombstone."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError(
            "generic residual base contracts are disabled; use sealed authorization"
        )


def _aligned_observed_pe(meta: pd.DataFrame, observed: pd.DataFrame) -> np.ndarray:
    expected = (*FULL_META_IDENTITY_COLUMNS, "observed_pe")
    if tuple(observed.columns) != expected:
        raise StructuralContractError(f"residual labels must have exact schema {expected}")
    require_no_evaluation_truth(observed.columns, context="residual labels")
    require_exact_identity(
        meta,
        observed,
        columns=FULL_META_IDENTITY_COLUMNS,
        context="residual label alignment",
    )
    return require_positive_finite(observed["observed_pe"], context="residual observed_pe")


@dataclass(frozen=True, init=False)
class ResidualAR1Fit:
    candidate_id: str
    base_model_id: str
    authorization_sha256: str
    plan_sha256: str
    inner_artifact_sha256: str
    inner_identity_sha256: str
    environment_sha256: str
    outer_fold_id: str
    ar1: FixedOriginAR1Fit
    meta_row_count: int
    correction_bound: float
    fit_sha256: str
    design_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError("ResidualAR1Fit is factory-only")

    @classmethod
    def _mint(cls, *, token: object, **values: Any) -> "ResidualAR1Fit":
        if token is not _FIT_TOKEN:
            raise StructuralContractError("invalid residual-fit mint token")
        output = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(output, key, value)
        return output

    def _unsigned(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "candidate_id": self.candidate_id,
            "base_model_id": self.base_model_id,
            "authorization_sha256": self.authorization_sha256,
            "plan_sha256": self.plan_sha256,
            "inner_artifact_sha256": self.inner_artifact_sha256,
            "inner_identity_sha256": self.inner_identity_sha256,
            "environment_sha256": self.environment_sha256,
            "outer_fold_id": self.outer_fold_id,
            "ar1": {
                "rho": self.ar1.rho,
                "last_residual": self.ar1.last_residual,
                "consecutive_pair_count": self.ar1.consecutive_pair_count,
                "minimum_pair_count": self.ar1.minimum_pair_count,
                "coefficient_lower": AR_RHO_MIN,
                "coefficient_upper": AR_RHO_MAX,
            },
            "meta_row_count": self.meta_row_count,
            "correction_bound": self.correction_bound,
            "design_sha256": self.design_sha256,
        }

    def verify(self, authorization: StructuralExecutionAuthorization) -> None:
        authorization.require_residual_base(self.candidate_id, self.base_model_id)
        for field in (
            "authorization_sha256",
            "plan_sha256",
            "inner_artifact_sha256",
            "inner_identity_sha256",
            "environment_sha256",
            "fit_sha256",
            "design_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        if self.authorization_sha256 != authorization.authorization_sha256:
            raise StructuralContractError("residual fit authorization changed")
        if self.candidate_id != RESIDUAL_AR1_CANDIDATE:
            raise StructuralContractError("residual fit candidate changed")
        if self.base_model_id != EXPECTED_RESIDUAL_BASE:
            raise StructuralContractError("residual fit base changed")
        if self.meta_row_count < MINIMUM_META_ROWS:
            raise StructuralContractError("AR correction has fewer than 252 inner OOS rows")
        if self.correction_bound != CORRECTION_BOUND:
            raise StructuralContractError("AR correction clip changed")
        if self.design_sha256 != STRUCTURAL_DESIGN_SHA256:
            raise StructuralContractError("AR correction design changed")
        if sha256_bytes(canonical_json_bytes(self._unsigned())) != self.fit_sha256:
            raise StructuralContractError("AR correction fit seal changed")


def fit_residual_ar1_kernel(
    inner_artifact: GeneratedMetaFeatureArtifact,
    observed: pd.DataFrame,
    *,
    authorization: StructuralExecutionAuthorization,
) -> ResidualAR1Fit:
    """Fit only the authorized v04_expected_pe nested-OOF residual stream."""

    if not isinstance(authorization, StructuralExecutionAuthorization):
        raise StructuralContractError("residual fit requires sealed authorization")
    authorization.require_residual_base(RESIDUAL_AR1_CANDIDATE, EXPECTED_RESIDUAL_BASE)
    if not isinstance(inner_artifact, GeneratedMetaFeatureArtifact):
        raise StructuralContractError("raw/in-sample predictions cannot impersonate OOF")
    meta = inner_artifact.verify(
        authorization,
        expected_role="INNER_OOS_FIT",
        expected_candidate_id=RESIDUAL_AR1_CANDIDATE,
    )
    if inner_artifact.base_model_id != EXPECTED_RESIDUAL_BASE:
        raise StructuralContractError("residual artifact base must be v04_expected_pe")
    if len(meta) < MINIMUM_META_ROWS:
        raise StructuralContractError("AR correction requires at least 252 inner OOS rows")
    target_pe = _aligned_observed_pe(meta, observed)
    base_prediction = require_positive_finite(
        meta["base_prediction"], context="AR inner OOS base prediction"
    )
    residual = np.log(target_pe) - np.log(base_prediction)
    ar1 = fit_zero_intercept_ar1(
        residual,
        session_positions=meta["session_position"].to_numpy(dtype=np.int64),
        segment_ids=np.repeat(
            f"{meta['seed'].iloc[0]}::{meta['outer_fold_id'].iloc[0]}", len(meta)
        ),
    )
    values: dict[str, Any] = {
        "candidate_id": RESIDUAL_AR1_CANDIDATE,
        "base_model_id": EXPECTED_RESIDUAL_BASE,
        "authorization_sha256": authorization.authorization_sha256,
        "plan_sha256": inner_artifact.plan.plan_sha256,
        "inner_artifact_sha256": inner_artifact.artifact_sha256,
        "inner_identity_sha256": inner_artifact.identity_sha256,
        "environment_sha256": inner_artifact.environment_sha256,
        "outer_fold_id": str(meta["outer_fold_id"].iloc[0]),
        "ar1": ar1,
        "meta_row_count": len(meta),
        "correction_bound": CORRECTION_BOUND,
        "design_sha256": STRUCTURAL_DESIGN_SHA256,
    }
    provisional = ResidualAR1Fit._mint(token=_FIT_TOKEN, **values, fit_sha256="0" * 64)
    values["fit_sha256"] = sha256_bytes(canonical_json_bytes(provisional._unsigned()))
    output = ResidualAR1Fit._mint(token=_FIT_TOKEN, **values)
    output.verify(authorization)
    return output


def predict_residual_ar1_kernel(
    fit: ResidualAR1Fit,
    outer_artifact: GeneratedMetaFeatureArtifact,
    *,
    authorization: StructuralExecutionAuthorization,
) -> pd.DataFrame:
    """Apply a fixed-origin correction; no outer target argument exists."""

    if not isinstance(fit, ResidualAR1Fit):
        raise StructuralContractError("residual prediction requires factory-created fit")
    fit.verify(authorization)
    if not isinstance(outer_artifact, GeneratedMetaFeatureArtifact):
        raise StructuralContractError("residual prediction requires verified outer artifact")
    meta = outer_artifact.verify(
        authorization,
        expected_role="OUTER_TEST_PREDICT",
        expected_candidate_id=RESIDUAL_AR1_CANDIDATE,
    )
    if outer_artifact.plan.plan_sha256 != fit.plan_sha256:
        raise StructuralContractError("residual fit/predict must use the same nested plan")
    if outer_artifact.environment_sha256 != fit.environment_sha256:
        raise StructuralContractError("residual predict environment changed")
    if str(meta["outer_fold_id"].iloc[0]) != fit.outer_fold_id:
        raise StructuralContractError("residual predict outer fold changed")
    positions = meta["session_position"].to_numpy(dtype=np.int64)
    if len(positions) > 1 and not (np.diff(positions) == 1).all():
        raise StructuralContractError("AR outer-test positions must be consecutive")
    base_prediction = require_positive_finite(
        meta["base_prediction"], context="AR outer-test base prediction"
    )
    correction = np.clip(fit.ar1.forecast(len(meta)), -fit.correction_bound, fit.correction_bound)
    expected_pe = np.exp(np.log(base_prediction) + correction)
    if not np.isfinite(expected_pe).all() or (expected_pe <= 0.0).any():
        raise StructuralContractError("AR-corrected expected P/E is invalid")
    return pd.DataFrame(
        {
            "seed": meta["seed"].to_numpy(copy=True),
            "date": meta["date"].to_numpy(copy=True),
            "outer_fold_id": meta["outer_fold_id"].to_numpy(copy=True),
            "correction": correction,
            "expected_pe": expected_pe,
        }
    )


def fit_residual_huber_kernel(
    *_args: object,
    authorization: StructuralExecutionAuthorization,
    **_kwargs: object,
) -> None:
    """The sealed T_STABLE_BIAS=false decision makes Huber unreachable."""

    if not isinstance(authorization, StructuralExecutionAuthorization):
        raise StructuralContractError("Huber entrypoint requires sealed authorization")
    authorization.require_candidate(RESIDUAL_HUBER_CANDIDATE)
    raise AssertionError("unreachable")


def predict_residual_huber_kernel(
    *_args: object,
    authorization: StructuralExecutionAuthorization,
    **_kwargs: object,
) -> None:
    if not isinstance(authorization, StructuralExecutionAuthorization):
        raise StructuralContractError("Huber entrypoint requires sealed authorization")
    authorization.require_candidate(RESIDUAL_HUBER_CANDIDATE)
    raise AssertionError("unreachable")
