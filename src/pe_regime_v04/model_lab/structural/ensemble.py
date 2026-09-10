"""Authorized exact-pair natural-log blends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .authorization import EXPECTED_PAIR, StructuralExecutionAuthorization
from .contracts import (
    STRUCTURAL_DESIGN_SHA256,
    StructuralContractError,
    canonical_json_bytes,
    require_exact_identity,
    require_no_evaluation_truth,
    require_positive_finite,
    require_sha256,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .meta import GeneratedMetaFeatureArtifact


PAIR_IDENTITY_COLUMNS = ("seed", "date", "outer_fold_id", "inner_fold_id")
SIMPLEX_DENOMINATOR_THRESHOLD = 1e-12
DISTINCT_WEIGHT_MIN = 0.05
DISTINCT_WEIGHT_MAX = 0.95
MINIMUM_INNER_OOS_META_ROWS = 252
EQUAL_CANDIDATE = "stack_geometric_equal_pair"
SIMPLEX_CANDIDATE = "stack_simplex_pair_frozen"
_WEIGHT_TOKEN = object()


@dataclass(frozen=True, init=False)
class BoundEnsembleBase:
    """Removed caller-authored pair contract retained as a tombstone."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError(
            "generic ensemble base contracts are disabled; use sealed authorization"
        )


def _require_pair(
    artifact0: GeneratedMetaFeatureArtifact,
    artifact1: GeneratedMetaFeatureArtifact,
    *,
    authorization: StructuralExecutionAuthorization,
    candidate_id: str,
    role: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(authorization, StructuralExecutionAuthorization):
        raise StructuralContractError("ensemble entrypoint requires sealed authorization")
    base0, base1 = authorization.require_pair(candidate_id, *EXPECTED_PAIR)
    if not isinstance(artifact0, GeneratedMetaFeatureArtifact) or not isinstance(
        artifact1, GeneratedMetaFeatureArtifact
    ):
        raise StructuralContractError("ensemble inputs must be verified generated artifacts")
    frame0 = artifact0.verify(authorization, expected_role=role, expected_candidate_id=candidate_id)
    frame1 = artifact1.verify(authorization, expected_role=role, expected_candidate_id=candidate_id)
    if (artifact0.base_model_id, artifact1.base_model_id) != EXPECTED_PAIR:
        raise StructuralContractError("ensemble artifact pair/order changed")
    if (
        artifact0.base_source_sha256,
        artifact0.base_config_sha256,
        artifact1.base_source_sha256,
        artifact1.base_config_sha256,
    ) != (
        base0.source_sha256,
        base0.config_sha256,
        base1.source_sha256,
        base1.config_sha256,
    ):
        raise StructuralContractError("ensemble exact source/config binding changed")
    if artifact0.environment_sha256 != artifact1.environment_sha256:
        raise StructuralContractError("ensemble bases use different environments")
    plan0 = artifact0.plan
    plan1 = artifact1.plan
    if (
        plan0.seed,
        plan0.outer_fold_id,
        plan0.outer_cutoff_iso,
        plan0.outer_train_identity_sha256,
        plan0.outer_test_identity_sha256,
        plan0.meta_session_positions,
    ) != (
        plan1.seed,
        plan1.outer_fold_id,
        plan1.outer_cutoff_iso,
        plan1.outer_train_identity_sha256,
        plan1.outer_test_identity_sha256,
        plan1.meta_session_positions,
    ):
        raise StructuralContractError("ensemble plans do not share the exact fold surface")
    require_exact_identity(frame0, frame1, columns=PAIR_IDENTITY_COLUMNS, context="ensemble pair")
    if not np.array_equal(
        frame0["session_position"].to_numpy(dtype=np.int64),
        frame1["session_position"].to_numpy(dtype=np.int64),
    ):
        raise StructuralContractError("ensemble pair session positions differ")
    return frame0, frame1


def _pair_output(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    if not np.isfinite(prediction).all() or (prediction <= 0.0).any():
        raise StructuralContractError("ensemble prediction is non-positive or non-finite")
    output = frame.loc[:, list(PAIR_IDENTITY_COLUMNS)].copy()
    output["expected_pe"] = prediction
    return output


def equal_geometric_blend(
    artifact0: GeneratedMetaFeatureArtifact,
    artifact1: GeneratedMetaFeatureArtifact,
    *,
    authorization: StructuralExecutionAuthorization,
) -> pd.DataFrame:
    """Apply exact 0.5/0.5 weights to the one authorized ordered pair."""

    frame0, frame1 = _require_pair(
        artifact0,
        artifact1,
        authorization=authorization,
        candidate_id=EQUAL_CANDIDATE,
        role="OUTER_TEST_PREDICT",
    )
    prediction0 = require_positive_finite(frame0["base_prediction"], context="equal base0")
    prediction1 = require_positive_finite(frame1["base_prediction"], context="equal base1")
    return _pair_output(frame0, np.exp(0.5 * np.log(prediction0) + 0.5 * np.log(prediction1)))


@dataclass(frozen=True, init=False)
class SimplexLogWeightFit:
    candidate_id: str
    base_model_ids: tuple[str, str]
    authorization_sha256: str
    plan_sha256s: tuple[str, str]
    weight0: float
    weight1: float
    denominator: float
    used_degenerate_fallback: bool
    meta_row_count: int
    base0_artifact_sha256: str
    base1_artifact_sha256: str
    training_identity_sha256: str
    environment_sha256: str
    design_sha256: str
    weight_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError("SimplexLogWeightFit is factory-only")

    @classmethod
    def _mint(cls, *, token: object, **values: Any) -> "SimplexLogWeightFit":
        if token is not _WEIGHT_TOKEN:
            raise StructuralContractError("invalid simplex-weight mint token")
        output = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(output, key, value)
        return output

    @property
    def genuinely_distinct(self) -> bool:
        return DISTINCT_WEIGHT_MIN <= self.weight0 <= DISTINCT_WEIGHT_MAX

    def _unsigned_payload(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "mode": "authorized_structural_simplex_log_weight",
            "candidate_id": self.candidate_id,
            "base_model_ids": list(self.base_model_ids),
            "authorization_sha256": self.authorization_sha256,
            "plan_sha256s": list(self.plan_sha256s),
            "weight0": self.weight0,
            "weight1": self.weight1,
            "denominator": self.denominator,
            "used_degenerate_fallback": self.used_degenerate_fallback,
            "meta_row_count": self.meta_row_count,
            "base0_artifact_sha256": self.base0_artifact_sha256,
            "base1_artifact_sha256": self.base1_artifact_sha256,
            "training_identity_sha256": self.training_identity_sha256,
            "environment_sha256": self.environment_sha256,
            "design_sha256": self.design_sha256,
        }

    def verify(self, authorization: StructuralExecutionAuthorization) -> None:
        authorization.require_pair(self.candidate_id, *self.base_model_ids)
        values = np.asarray([self.weight0, self.weight1, self.denominator], dtype=np.float64)
        if not np.isfinite(values).all() or self.denominator < 0.0:
            raise StructuralContractError("simplex fit values are invalid")
        if not 0.0 <= self.weight0 <= 1.0 or not 0.0 <= self.weight1 <= 1.0:
            raise StructuralContractError("simplex weights must be non-negative")
        if not np.isclose(self.weight0 + self.weight1, 1.0, rtol=0.0, atol=1e-15):
            raise StructuralContractError("simplex weights must sum to one")
        fallback = self.denominator <= SIMPLEX_DENOMINATOR_THRESHOLD
        if self.used_degenerate_fallback != fallback:
            raise StructuralContractError("simplex fallback flag changed")
        if fallback and (self.weight0, self.weight1) != (0.5, 0.5):
            raise StructuralContractError("degenerate simplex must use 0.5/0.5")
        if self.meta_row_count < MINIMUM_INNER_OOS_META_ROWS:
            raise StructuralContractError("simplex fit requires 252 inner OOS rows")
        if self.authorization_sha256 != authorization.authorization_sha256:
            raise StructuralContractError("simplex authorization changed")
        if self.base_model_ids != EXPECTED_PAIR or self.candidate_id != SIMPLEX_CANDIDATE:
            raise StructuralContractError("simplex candidate/pair changed")
        for field in (
            *self.plan_sha256s,
            self.base0_artifact_sha256,
            self.base1_artifact_sha256,
            self.training_identity_sha256,
            self.environment_sha256,
            self.design_sha256,
            self.weight_sha256,
        ):
            require_sha256(field, field="simplex hash")
        if self.design_sha256 != STRUCTURAL_DESIGN_SHA256:
            raise StructuralContractError("simplex design changed")
        if sha256_bytes(canonical_json_bytes(self._unsigned_payload())) != self.weight_sha256:
            raise StructuralContractError("simplex weight content hash changed")

    def to_sealed_payload(self) -> dict[str, Any]:
        payload = self._unsigned_payload()
        payload["weight_sha256"] = self.weight_sha256
        return seal_payload(payload)

    @classmethod
    def from_sealed_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        authorization: StructuralExecutionAuthorization,
    ) -> "SimplexLogWeightFit":
        verify_payload_seal(payload)
        required = set(
            cls._mint(
                token=_WEIGHT_TOKEN,
                candidate_id=SIMPLEX_CANDIDATE,
                base_model_ids=EXPECTED_PAIR,
                authorization_sha256="0" * 64,
                plan_sha256s=("0" * 64, "0" * 64),
                weight0=0.5,
                weight1=0.5,
                denominator=0.0,
                used_degenerate_fallback=True,
                meta_row_count=252,
                base0_artifact_sha256="0" * 64,
                base1_artifact_sha256="0" * 64,
                training_identity_sha256="0" * 64,
                environment_sha256="0" * 64,
                design_sha256=STRUCTURAL_DESIGN_SHA256,
                weight_sha256="0" * 64,
            )._unsigned_payload()
        ) | {"weight_sha256", "manifest_sha256"}
        if set(payload) != required:
            raise StructuralContractError("sealed simplex weight schema changed")
        values: dict[str, Any] = {
            "candidate_id": str(payload["candidate_id"]),
            "base_model_ids": tuple(str(item) for item in payload["base_model_ids"]),
            "authorization_sha256": str(payload["authorization_sha256"]),
            "plan_sha256s": tuple(str(item) for item in payload["plan_sha256s"]),
            "weight0": float(payload["weight0"]),
            "weight1": float(payload["weight1"]),
            "denominator": float(payload["denominator"]),
            "used_degenerate_fallback": bool(payload["used_degenerate_fallback"]),
            "meta_row_count": int(payload["meta_row_count"]),
            "base0_artifact_sha256": str(payload["base0_artifact_sha256"]),
            "base1_artifact_sha256": str(payload["base1_artifact_sha256"]),
            "training_identity_sha256": str(payload["training_identity_sha256"]),
            "environment_sha256": str(payload["environment_sha256"]),
            "design_sha256": str(payload["design_sha256"]),
            "weight_sha256": str(payload["weight_sha256"]),
        }
        if len(values["base_model_ids"]) != 2 or len(values["plan_sha256s"]) != 2:
            raise StructuralContractError("sealed simplex tuple lengths changed")
        output = cls._mint(token=_WEIGHT_TOKEN, **values)
        output.verify(authorization)
        return output


def fit_two_base_simplex_log_weights(
    artifact0: GeneratedMetaFeatureArtifact,
    artifact1: GeneratedMetaFeatureArtifact,
    observed: pd.DataFrame,
    *,
    authorization: StructuralExecutionAuthorization,
) -> SimplexLogWeightFit:
    frame0, frame1 = _require_pair(
        artifact0,
        artifact1,
        authorization=authorization,
        candidate_id=SIMPLEX_CANDIDATE,
        role="INNER_OOS_FIT",
    )
    if len(frame0) < MINIMUM_INNER_OOS_META_ROWS:
        raise StructuralContractError("simplex fit requires at least 252 inner OOS rows")
    expected = (*PAIR_IDENTITY_COLUMNS, "observed_pe")
    if tuple(observed.columns) != expected:
        raise StructuralContractError(f"simplex labels must have exact schema {expected}")
    require_no_evaluation_truth(observed.columns, context="simplex labels")
    require_exact_identity(
        frame0, observed, columns=PAIR_IDENTITY_COLUMNS, context="simplex label alignment"
    )
    prediction0 = require_positive_finite(frame0["base_prediction"], context="simplex base0")
    prediction1 = require_positive_finite(frame1["base_prediction"], context="simplex base1")
    target = require_positive_finite(observed["observed_pe"], context="simplex observed_pe")
    d = np.log(prediction0) - np.log(prediction1)
    q = np.log(target) - np.log(prediction1)
    denominator = float(np.dot(d, d))
    if denominator <= SIMPLEX_DENOMINATOR_THRESHOLD:
        weight0 = 0.5
        fallback = True
    else:
        weight0 = float(np.clip(float(np.dot(d, q)) / denominator, 0.0, 1.0))
        fallback = False
    values: dict[str, Any] = {
        "candidate_id": SIMPLEX_CANDIDATE,
        "base_model_ids": EXPECTED_PAIR,
        "authorization_sha256": authorization.authorization_sha256,
        "plan_sha256s": (artifact0.plan.plan_sha256, artifact1.plan.plan_sha256),
        "weight0": weight0,
        "weight1": 1.0 - weight0,
        "denominator": denominator,
        "used_degenerate_fallback": fallback,
        "meta_row_count": len(frame0),
        "base0_artifact_sha256": artifact0.artifact_sha256,
        "base1_artifact_sha256": artifact1.artifact_sha256,
        "training_identity_sha256": sha256_bytes(
            frame0.loc[:, list(PAIR_IDENTITY_COLUMNS)]
            .to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S.%f")
            .encode("utf-8")
        ),
        "environment_sha256": artifact0.environment_sha256,
        "design_sha256": STRUCTURAL_DESIGN_SHA256,
    }
    provisional = SimplexLogWeightFit._mint(token=_WEIGHT_TOKEN, **values, weight_sha256="0" * 64)
    values["weight_sha256"] = sha256_bytes(canonical_json_bytes(provisional._unsigned_payload()))
    output = SimplexLogWeightFit._mint(token=_WEIGHT_TOKEN, **values)
    output.verify(authorization)
    return output


def apply_two_base_simplex_log_weights(
    fit: SimplexLogWeightFit,
    artifact0: GeneratedMetaFeatureArtifact,
    artifact1: GeneratedMetaFeatureArtifact,
    *,
    authorization: StructuralExecutionAuthorization,
) -> pd.DataFrame:
    if not isinstance(fit, SimplexLogWeightFit):
        raise StructuralContractError("simplex apply requires factory-created weights")
    fit.verify(authorization)
    frame0, frame1 = _require_pair(
        artifact0,
        artifact1,
        authorization=authorization,
        candidate_id=SIMPLEX_CANDIDATE,
        role="OUTER_TEST_PREDICT",
    )
    if (artifact0.plan.plan_sha256, artifact1.plan.plan_sha256) != fit.plan_sha256s:
        raise StructuralContractError("simplex fit/predict plans differ")
    if artifact0.environment_sha256 != fit.environment_sha256:
        raise StructuralContractError("simplex predict environment differs")
    prediction0 = require_positive_finite(frame0["base_prediction"], context="simplex outer0")
    prediction1 = require_positive_finite(frame1["base_prediction"], context="simplex outer1")
    return _pair_output(
        frame0,
        np.exp(fit.weight0 * np.log(prediction0) + fit.weight1 * np.log(prediction1)),
    )
