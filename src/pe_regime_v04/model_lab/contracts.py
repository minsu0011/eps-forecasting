"""Typed, fail-closed contracts for isolated Model Lab experiments."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Protocol, Sequence, runtime_checkable

import pandas as pd


EVALUATION_ONLY_TRUTH_COLUMN = "true_fair_pe"
MODEL_CONTROL_COLUMNS = frozenset({"seed", "fold_id", "model_id", EVALUATION_ONLY_TRUTH_COLUMN})
DEFAULT_MODEL_IDENTITY_COLUMNS = ("date", "symbol", "entity_id")
PredictionPhase = Literal["fit", "predict"]
AvailabilityKind = Literal["point_in_time", "lagged", "evaluation_only"]
FeatureFamily = Literal[
    "VALUATION",
    "EPS",
    "MARKET",
    "REGIME",
    "SECTOR",
    "MACRO",
    "QUALITY",
    "TEMPORAL",
    "CROSS_SECTIONAL",
]
FeatureAuditStatus = Literal["PASS", "FAIL", "UNTESTED", "NOT_APPLICABLE", "BROKEN"]
FEATURE_FAMILIES = (
    "VALUATION",
    "EPS",
    "MARKET",
    "REGIME",
    "SECTOR",
    "MACRO",
    "QUALITY",
    "TEMPORAL",
    "CROSS_SECTIONAL",
)
FEATURE_AUDIT_STATUSES = ("PASS", "FAIL", "UNTESTED", "NOT_APPLICABLE", "BROKEN")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")


class ContractError(ValueError):
    """Raised when model or feature metadata could permit leakage."""


def _require_identifier(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ContractError(f"{field} must match {_IDENTIFIER.pattern!r}")


@dataclass(frozen=True)
class FeatureMetadata:
    """Availability metadata required before a column can enter a model.

    ``lookahead_sessions`` describes future information in the value itself.
    Such columns may be registered for diagnostics, but cannot be used by
    ``fit`` or ``predict``.  Revised/non-PIT values are treated the same way.
    """

    feature_id: str
    column_name: str
    description: str
    source: str
    dtype: str
    availability: AvailabilityKind
    availability_lag_sessions: int
    lookahead_sessions: int
    evaluation_only: bool
    allowed_for_fit: bool
    allowed_for_predict: bool
    point_in_time_safe: bool
    uses_revised_data: bool
    feature_family: FeatureFamily
    provenance_reference: str
    provenance_sha256: str | None
    prefix_invariance_status: FeatureAuditStatus
    future_intervention_status: FeatureAuditStatus
    same_row_target_leakage_status: FeatureAuditStatus
    pit_availability_status: FeatureAuditStatus
    publication_date_status: FeatureAuditStatus
    restatement_availability_status: FeatureAuditStatus
    audit_notes: str

    def __post_init__(self) -> None:
        _require_identifier(self.feature_id, field="feature_id")
        if not self.column_name or not isinstance(self.column_name, str):
            raise ContractError("column_name must be a non-empty string")
        forbidden_controls = MODEL_CONTROL_COLUMNS.difference({EVALUATION_ONLY_TRUTH_COLUMN})
        if self.feature_id in forbidden_controls or self.column_name in forbidden_controls:
            raise ContractError("experiment control columns cannot be registered as features")
        if not self.description or not self.source or not self.dtype:
            raise ContractError("description, source, and dtype must be non-empty")
        if self.feature_family not in FEATURE_FAMILIES:
            raise ContractError(f"unsupported feature_family: {self.feature_family!r}")
        if not isinstance(self.provenance_reference, str) or not self.provenance_reference:
            raise ContractError("provenance_reference must be a non-empty string")
        if self.provenance_sha256 is not None and (
            not isinstance(self.provenance_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.provenance_sha256) is None
        ):
            raise ContractError("provenance_sha256 must be null or lowercase SHA-256")
        audit_statuses = (
            self.prefix_invariance_status,
            self.future_intervention_status,
            self.same_row_target_leakage_status,
            self.pit_availability_status,
            self.publication_date_status,
            self.restatement_availability_status,
        )
        if any(status not in FEATURE_AUDIT_STATUSES for status in audit_statuses):
            raise ContractError("feature audit statuses must use the closed audit vocabulary")
        if not isinstance(self.audit_notes, str) or not self.audit_notes:
            raise ContractError("audit_notes must be a non-empty string")
        if self.availability not in {"point_in_time", "lagged", "evaluation_only"}:
            raise ContractError(f"unsupported availability: {self.availability!r}")
        boolean_fields = (
            self.evaluation_only,
            self.allowed_for_fit,
            self.allowed_for_predict,
            self.point_in_time_safe,
            self.uses_revised_data,
        )
        if not all(isinstance(value, bool) for value in boolean_fields):
            raise ContractError("feature safety flags must be booleans")
        if (
            not isinstance(self.availability_lag_sessions, int)
            or isinstance(self.availability_lag_sessions, bool)
            or self.availability_lag_sessions < 0
        ):
            raise ContractError("availability_lag_sessions must be a non-negative integer")
        if (
            not isinstance(self.lookahead_sessions, int)
            or isinstance(self.lookahead_sessions, bool)
            or self.lookahead_sessions < 0
        ):
            raise ContractError("lookahead_sessions must be a non-negative integer")

        is_truth = self.column_name == EVALUATION_ONLY_TRUTH_COLUMN
        must_be_diagnostic = (
            self.evaluation_only
            or self.availability == "evaluation_only"
            or self.lookahead_sessions > 0
            or self.uses_revised_data
            or not self.point_in_time_safe
            or is_truth
            or any(status in {"FAIL", "BROKEN"} for status in audit_statuses)
        )
        if must_be_diagnostic and (self.allowed_for_fit or self.allowed_for_predict):
            raise ContractError(
                f"unsafe feature {self.feature_id!r} cannot be allowed for fit or predict"
            )
        if is_truth and not (
            self.evaluation_only
            and self.availability == "evaluation_only"
            and not self.point_in_time_safe
        ):
            raise ContractError("true_fair_pe must be explicitly evaluation-only and non-PIT")
        if self.evaluation_only != (self.availability == "evaluation_only"):
            raise ContractError("evaluation_only and availability must agree")


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    model_version: str
    family: str
    feature_ids: tuple[str, ...]
    training_target: str
    prediction_name: str
    output_semantics: str
    deterministic: bool

    def __post_init__(self) -> None:
        _require_identifier(self.model_id, field="model_id")
        _require_identifier(self.family, field="family")
        if not self.model_version or not self.prediction_name or not self.output_semantics:
            raise ContractError(
                "model_version, prediction_name, and output_semantics must be non-empty"
            )
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ContractError("feature_ids must be unique")
        for feature_id in self.feature_ids:
            _require_identifier(feature_id, field="feature_id")
        if self.training_target == EVALUATION_ONLY_TRUTH_COLUMN:
            raise ContractError("true_fair_pe is evaluation-only and cannot be a training target")
        if not isinstance(self.deterministic, bool):
            raise ContractError("deterministic must be a boolean")

    @property
    def registration_id(self) -> str:
        return f"{self.model_id}@{self.model_version}"


@dataclass(frozen=True)
class FitContext:
    experiment_id: str
    fold_id: str
    seed: int
    train_end: pd.Timestamp
    target_name: str
    feature_metadata: tuple[FeatureMetadata, ...]

    def __post_init__(self) -> None:
        if self.target_name == EVALUATION_ONLY_TRUTH_COLUMN:
            raise ContractError("true_fair_pe is evaluation-only and cannot enter FitContext")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ContractError("seed must be an integer")
        if pd.isna(self.train_end):
            raise ContractError("train_end must be a valid timestamp")


@dataclass(frozen=True)
class PredictContext:
    experiment_id: str
    fold_id: str
    seed: int
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp
    feature_metadata: tuple[FeatureMetadata, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ContractError("seed must be an integer")
        if pd.isna(self.prediction_start) or pd.isna(self.prediction_end):
            raise ContractError("prediction bounds must be valid timestamps")
        if self.prediction_start > self.prediction_end:
            raise ContractError("prediction_start must be on or before prediction_end")


@runtime_checkable
class PEModel(Protocol):
    """Minimal interface shared by every Model Lab candidate."""

    def fit(self, x: pd.DataFrame, y: pd.Series, *, context: FitContext) -> PEModel:
        """Fit only on PIT-safe inputs and a non-evaluation target."""

    def predict(self, x: pd.DataFrame, *, context: PredictContext) -> pd.Series:
        """Return one positive-P/E prediction candidate per input row."""

    def metadata(self) -> ModelMetadata:
        """Return immutable registration metadata."""


def validate_feature_frame(
    frame: pd.DataFrame,
    features: Sequence[FeatureMetadata],
    *,
    phase: PredictionPhase,
    require_exact_columns: bool = True,
    identity_columns: Sequence[str] = DEFAULT_MODEL_IDENTITY_COLUMNS,
) -> None:
    """Reject undeclared, evaluation-only, revised, or future-looking columns."""

    if phase not in {"fit", "predict"}:
        raise ContractError(f"unsupported phase: {phase!r}")
    if frame.columns.has_duplicates:
        raise ContractError("model input columns must be unique")
    control_columns = sorted(MODEL_CONTROL_COLUMNS.intersection(frame.columns))
    if control_columns:
        raise ContractError(f"model input contains reserved control columns: {control_columns}")
    identity_overlap = sorted(set(identity_columns).intersection(frame.columns))
    if identity_overlap:
        raise ContractError(
            f"model input contains identity columns that require a separate API: {identity_overlap}"
        )

    ids = [feature.feature_id for feature in features]
    columns = [feature.column_name for feature in features]
    if len(ids) != len(set(ids)) or len(columns) != len(set(columns)):
        raise ContractError("selected feature ids and columns must be unique")
    for feature in features:
        allowed = feature.allowed_for_fit if phase == "fit" else feature.allowed_for_predict
        if not allowed:
            raise ContractError(f"feature {feature.feature_id!r} is not allowed for {phase}")
        if (
            feature.evaluation_only
            or feature.lookahead_sessions > 0
            or feature.uses_revised_data
            or not feature.point_in_time_safe
        ):
            raise ContractError(f"feature {feature.feature_id!r} is not point-in-time safe")

    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ContractError(f"model input is missing declared columns: {missing}")
    extra = sorted(set(frame.columns).difference(columns))
    if require_exact_columns and extra:
        raise ContractError(f"model input contains undeclared columns: {extra}")


def validate_model_call(
    metadata: ModelMetadata,
    frame: pd.DataFrame,
    features: Sequence[FeatureMetadata],
    *,
    phase: PredictionPhase,
    target_name: str | None = None,
) -> None:
    selected_ids = tuple(feature.feature_id for feature in features)
    if selected_ids != metadata.feature_ids:
        raise ContractError(
            f"model feature order {metadata.feature_ids!r} does not match {selected_ids!r}"
        )
    if phase == "fit":
        if target_name is None or target_name != metadata.training_target:
            raise ContractError("FitContext target does not match model metadata")
        if target_name == EVALUATION_ONLY_TRUTH_COLUMN:
            raise ContractError("true_fair_pe is evaluation-only")
    validate_feature_frame(frame, features, phase=phase)
