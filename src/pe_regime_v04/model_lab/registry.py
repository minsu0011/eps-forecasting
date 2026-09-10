"""Append-only dual CSV/JSON registries and dependency-free JSON schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    EVALUATION_ONLY_TRUTH_COLUMN,
    FEATURE_AUDIT_STATUSES,
    FEATURE_FAMILIES,
    AvailabilityKind,
    ContractError,
    FeatureAuditStatus,
    FeatureFamily,
    FeatureMetadata,
    ModelMetadata,
)


REGISTRY_FORMAT_VERSION = 1
SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MODEL_STATUSES = (
    "PROMOTED",
    "PROMISING",
    "COMPUTE_INEFFICIENT",
    "DATA_INSUFFICIENT",
    "INCUMBENT",
    "PASS",
    "RESEARCH_ONLY",
    "SATURATED",
    "REJECTED",
    "BROKEN",
    "UNTESTED",
)
MODEL_TRACKS = ("A", "B", "C")
MODEL_TRACK_SEMANTICS = {
    "A": "fundamental/PIT estimand excluding same-row price and observed P/E",
    "B": "market-conditioned latent valuation estimand; never label as intrinsic",
    "C": "hybrid estimand combining fundamental and current or lagged market information",
}
TUNING_STATUSES = ("UNTESTED", "RUNNING", "PASS", "FAIL", "NOT_APPLICABLE", "BROKEN")
LOCKED_STATUSES = ("UNTESTED", "LOCKED", "UNLOCKED", "NOT_APPLICABLE", "BROKEN")
HELDOUT_STATUSES = (
    "NOT_OPENED",
    "RESERVED_SPENT",
    "PASS",
    "FAIL",
    "NOT_APPLICABLE",
    "BROKEN",
)
_MODEL_FIELDS = (
    "model_id",
    "family",
    "variant",
    "version",
    "registry_revision",
    "track",
    "estimand",
    "external_reference",
    "paper",
    "repository",
    "package",
    "license",
    "target",
    "feature_set",
    "uses_same_row_price",
    "uses_same_row_observed_pe",
    "causal",
    "pit_safe",
    "train_window",
    "refit_frequency",
    "hyperparameters",
    "tuning_status",
    "locked_status",
    "heldout_status",
    "fair_log_mae",
    "fair_log_rmse",
    "worst_seed",
    "compute_time",
    "status",
    "notes",
    "entrypoint",
    "feature_ids",
    "prediction_name",
    "output_semantics",
    "deterministic",
    "description",
    "parameters_sha256",
)
_FEATURE_FIELDS = (
    "feature_id",
    "column_name",
    "description",
    "source",
    "dtype",
    "availability",
    "availability_lag_sessions",
    "lookahead_sessions",
    "evaluation_only",
    "allowed_for_fit",
    "allowed_for_predict",
    "point_in_time_safe",
    "uses_revised_data",
    "feature_family",
    "provenance_reference",
    "provenance_sha256",
    "prefix_invariance_status",
    "future_intervention_status",
    "same_row_target_leakage_status",
    "pit_availability_status",
    "publication_date_status",
    "restatement_availability_status",
    "audit_notes",
)
_MODEL_MUTABLE_FIELDS = (
    "tuning_status",
    "locked_status",
    "heldout_status",
    "fair_log_mae",
    "fair_log_rmse",
    "worst_seed",
    "compute_time",
    "status",
    "notes",
)
_MODEL_DEFINITION_FIELDS = tuple(
    field for field in _MODEL_FIELDS if field not in {"registry_revision", *_MODEL_MUTABLE_FIELDS}
)
_STATUS_TRANSITIONS = {
    "UNTESTED": frozenset(
        {
            "UNTESTED",
            "PASS",
            "PROMISING",
            "RESEARCH_ONLY",
            "SATURATED",
            "REJECTED",
            "COMPUTE_INEFFICIENT",
            "DATA_INSUFFICIENT",
            "BROKEN",
        }
    ),
    "PASS": frozenset(
        {
            "PASS",
            "PROMOTED",
            "PROMISING",
            "INCUMBENT",
            "RESEARCH_ONLY",
            "SATURATED",
            "REJECTED",
            "COMPUTE_INEFFICIENT",
            "DATA_INSUFFICIENT",
            "BROKEN",
        }
    ),
    "PROMISING": frozenset(
        {
            "PROMISING",
            "PASS",
            "PROMOTED",
            "RESEARCH_ONLY",
            "SATURATED",
            "REJECTED",
            "COMPUTE_INEFFICIENT",
            "DATA_INSUFFICIENT",
            "BROKEN",
        }
    ),
    "PROMOTED": frozenset({"PROMOTED", "INCUMBENT", "SATURATED", "REJECTED", "BROKEN"}),
    "INCUMBENT": frozenset({"INCUMBENT", "PROMOTED", "SATURATED", "REJECTED", "BROKEN"}),
    "RESEARCH_ONLY": frozenset(
        {
            "RESEARCH_ONLY",
            "PROMISING",
            "PASS",
            "SATURATED",
            "REJECTED",
            "COMPUTE_INEFFICIENT",
            "DATA_INSUFFICIENT",
            "BROKEN",
        }
    ),
    "SATURATED": frozenset({"SATURATED"}),
    "REJECTED": frozenset({"REJECTED"}),
    "COMPUTE_INEFFICIENT": frozenset({"COMPUTE_INEFFICIENT"}),
    "DATA_INSUFFICIENT": frozenset({"DATA_INSUFFICIENT"}),
    "BROKEN": frozenset({"BROKEN"}),
}
_TUNING_TRANSITIONS = {
    "UNTESTED": frozenset(TUNING_STATUSES),
    "RUNNING": frozenset({"RUNNING", "PASS", "FAIL", "BROKEN"}),
    "PASS": frozenset({"PASS"}),
    "FAIL": frozenset({"FAIL"}),
    "NOT_APPLICABLE": frozenset({"NOT_APPLICABLE"}),
    "BROKEN": frozenset({"BROKEN"}),
}
_LOCKED_TRANSITIONS = {
    "UNTESTED": frozenset(LOCKED_STATUSES),
    "UNLOCKED": frozenset({"UNLOCKED", "LOCKED", "BROKEN"}),
    "LOCKED": frozenset({"LOCKED"}),
    "NOT_APPLICABLE": frozenset({"NOT_APPLICABLE"}),
    "BROKEN": frozenset({"BROKEN"}),
}
_HELDOUT_TRANSITIONS = {
    "NOT_OPENED": frozenset({"NOT_OPENED", "RESERVED_SPENT", "NOT_APPLICABLE", "BROKEN"}),
    "RESERVED_SPENT": frozenset({"RESERVED_SPENT", "PASS", "FAIL", "BROKEN"}),
    "PASS": frozenset({"PASS"}),
    "FAIL": frozenset({"FAIL"}),
    "NOT_APPLICABLE": frozenset({"NOT_APPLICABLE"}),
    "BROKEN": frozenset({"BROKEN"}),
}


class RegistryError(ContractError):
    """Raised for malformed, conflicting, or non-append-only registries."""


@dataclass(frozen=True)
class RegistryPaths:
    csv_path: Path
    json_path: Path


@dataclass(frozen=True)
class RegistrySnapshot:
    paths: RegistryPaths
    registry_type: str
    record_count: int
    logical_sha256: str
    csv_sha256: str
    json_sha256: str


@dataclass(frozen=True)
class ModelRegistration:
    model_id: str
    family: str
    variant: str
    version: str
    registry_revision: int
    track: str
    estimand: str
    external_reference: bool
    paper: str | None
    repository: str | None
    package: str | None
    license: str | None
    target: str
    feature_set: str
    uses_same_row_price: bool
    uses_same_row_observed_pe: bool
    causal: bool
    pit_safe: bool
    train_window: str
    refit_frequency: str
    hyperparameters: Mapping[str, Any]
    tuning_status: str
    locked_status: str
    heldout_status: str
    fair_log_mae: float | None
    fair_log_rmse: float | None
    worst_seed: int | None
    compute_time: float | None
    status: str
    notes: str
    entrypoint: str
    feature_ids: tuple[str, ...]
    prediction_name: str
    output_semantics: str
    deterministic: bool
    description: str
    parameters_sha256: str

    def __post_init__(self) -> None:
        ModelMetadata(
            model_id=self.model_id,
            model_version=self.version,
            family=self.family,
            feature_ids=self.feature_ids,
            training_target=self.target,
            prediction_name=self.prediction_name,
            output_semantics=self.output_semantics,
            deterministic=self.deterministic,
        )
        if not self.variant or not isinstance(self.variant, str):
            raise RegistryError("variant must be a non-empty string")
        if (
            isinstance(self.registry_revision, bool)
            or not isinstance(self.registry_revision, int)
            or self.registry_revision < 0
        ):
            raise RegistryError("registry_revision must be a non-negative integer")
        if self.track not in MODEL_TRACKS:
            raise RegistryError(f"unsupported model track: {self.track!r}")
        for field, value in (
            ("estimand", self.estimand),
            ("feature_set", self.feature_set),
            ("train_window", self.train_window),
            ("refit_frequency", self.refit_frequency),
            ("notes", self.notes),
        ):
            if not isinstance(value, str) or not value:
                raise RegistryError(f"{field} must be a non-empty string")
        for field, value in (
            ("external_reference", self.external_reference),
            ("uses_same_row_price", self.uses_same_row_price),
            ("uses_same_row_observed_pe", self.uses_same_row_observed_pe),
            ("causal", self.causal),
            ("pit_safe", self.pit_safe),
        ):
            if not isinstance(value, bool):
                raise RegistryError(f"{field} must be a boolean")
        reference_fields = (self.paper, self.repository, self.package, self.license)
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in reference_fields
        ):
            raise RegistryError("reference fields must be null or non-empty strings")
        if self.external_reference and (
            not any(value is not None for value in reference_fields[:3]) or self.license is None
        ):
            raise RegistryError("external models require a source reference and license")
        if self.track == "A" and (
            self.uses_same_row_price or self.uses_same_row_observed_pe or not self.pit_safe
        ):
            raise RegistryError("Track A must be PIT-safe and exclude same-row market valuation")
        if not isinstance(self.hyperparameters, Mapping):
            raise RegistryError("hyperparameters must be a JSON object")
        canonical_parameters = _canonical_json(dict(self.hyperparameters))
        if _SHA256.fullmatch(self.parameters_sha256) is None:
            raise RegistryError("parameters_sha256 must be lowercase SHA-256")
        if hashlib.sha256(canonical_parameters).hexdigest() != self.parameters_sha256:
            raise RegistryError("parameters_sha256 must seal canonical hyperparameters")
        if ":" not in self.entrypoint or any(character.isspace() for character in self.entrypoint):
            raise RegistryError("entrypoint must use module.path:object syntax")
        if self.tuning_status not in TUNING_STATUSES:
            raise RegistryError(f"unsupported tuning_status: {self.tuning_status!r}")
        if self.locked_status not in LOCKED_STATUSES:
            raise RegistryError(f"unsupported locked_status: {self.locked_status!r}")
        if self.heldout_status not in HELDOUT_STATUSES:
            raise RegistryError(f"unsupported heldout_status: {self.heldout_status!r}")
        if self.status not in MODEL_STATUSES:
            raise RegistryError(f"unsupported model status: {self.status!r}")
        for field, value in (
            ("fair_log_mae", self.fair_log_mae),
            ("fair_log_rmse", self.fair_log_rmse),
            ("compute_time", self.compute_time),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not float(value) >= 0.0
                or not float(value) < float("inf")
            ):
                raise RegistryError(f"{field} must be null or a finite non-negative number")
        if self.worst_seed is not None and (
            isinstance(self.worst_seed, bool) or not isinstance(self.worst_seed, int)
        ):
            raise RegistryError("worst_seed must be null or an integer")
        result_values = (
            self.fair_log_mae,
            self.fair_log_rmse,
            self.worst_seed,
            self.compute_time,
        )
        if self.registry_revision == 0 and (
            self.tuning_status != "UNTESTED"
            or self.locked_status != "UNTESTED"
            or self.heldout_status != "NOT_OPENED"
            or self.status != "UNTESTED"
            or any(value is not None for value in result_values)
        ):
            raise RegistryError(
                "registry_revision 0 must be an untested, unopened definition event"
            )
        if self.locked_status == "LOCKED" and self.tuning_status != "PASS":
            raise RegistryError("LOCKED models require tuning_status PASS")
        if self.heldout_status in {"RESERVED_SPENT", "PASS", "FAIL"} and (
            self.locked_status != "LOCKED"
        ):
            raise RegistryError("opened heldout states require a locked model")
        if self.heldout_status in {"PASS", "FAIL"} and any(
            value is None for value in result_values
        ):
            raise RegistryError("completed heldout states require complete fair results")
        if self.status in {"PASS", "PROMISING", "PROMOTED", "INCUMBENT"} and (
            self.tuning_status != "PASS"
        ):
            raise RegistryError(f"{self.status} models require tuning_status PASS")
        if self.status in {"PROMOTED", "INCUMBENT"} and (
            self.locked_status != "LOCKED" or self.heldout_status != "PASS"
        ):
            raise RegistryError(
                f"{self.status} models require locked_status LOCKED and heldout_status PASS"
            )
        if self.status == "UNTESTED" and any(value is not None for value in result_values):
            raise RegistryError("UNTESTED models cannot claim evaluation results")
        if self.status in {"PASS", "PROMOTED", "PROMISING", "INCUMBENT"} and any(
            value is None for value in result_values
        ):
            raise RegistryError(f"{self.status} models require complete fair evaluation results")
        if not self.description:
            raise RegistryError("model description must be non-empty")

    @property
    def registration_id(self) -> str:
        return f"{self.definition_id}#{self.registry_revision}"

    @property
    def definition_id(self) -> str:
        return f"{self.model_id}@{self.variant}@{self.version}"

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["feature_ids"] = list(self.feature_ids)
        record["hyperparameters"] = dict(self.hyperparameters)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ModelRegistration:
        _require_exact_fields(record, _MODEL_FIELDS, kind="model")
        payload = dict(record)
        feature_ids = payload["feature_ids"]
        if not isinstance(feature_ids, list) or not all(
            isinstance(value, str) for value in feature_ids
        ):
            raise RegistryError("model feature_ids must be an array of strings")
        payload["feature_ids"] = tuple(feature_ids)
        hyperparameters = payload["hyperparameters"]
        if not isinstance(hyperparameters, dict):
            raise RegistryError("model hyperparameters must be an object")
        return cls(**payload)


@dataclass(frozen=True)
class FeatureRegistration:
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
        self.to_metadata()

    def to_metadata(self) -> FeatureMetadata:
        return FeatureMetadata(**asdict(self))

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FeatureRegistration:
        _require_exact_fields(record, _FEATURE_FIELDS, kind="feature")
        return cls(**dict(record))


def _require_exact_fields(record: Mapping[str, Any], expected: Sequence[str], *, kind: str) -> None:
    if not isinstance(record, Mapping):
        raise RegistryError(f"{kind} record must be an object")
    missing = sorted(set(expected).difference(record))
    extra = sorted(set(record).difference(expected))
    if missing or extra:
        raise RegistryError(f"{kind} record fields mismatch; missing={missing}, extra={extra}")


def _string_schema(
    *, pattern: str | None = None, enum: Sequence[str] | None = None
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "minLength": 1}
    if pattern is not None:
        schema["pattern"] = pattern
    if enum is not None:
        schema["enum"] = list(enum)
    return schema


def _nullable_string_schema() -> dict[str, Any]:
    return {"type": ["string", "null"], "minLength": 1}


def _nullable_number_schema() -> dict[str, Any]:
    return {"type": ["number", "null"], "minimum": 0}


def _registry_envelope_schema(
    registry_type: str, record_schema: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "$schema": SCHEMA_URI,
        "$id": f"https://pe-regime.local/model-lab/{registry_type}-registry.schema.json",
        "title": f"Model Lab {registry_type.title()} Registry",
        "type": "object",
        "additionalProperties": False,
        "required": ["format_version", "registry_type", "records", "registry_sha256"],
        "properties": {
            "format_version": {"const": REGISTRY_FORMAT_VERSION},
            "registry_type": {"const": registry_type},
            "records": {"type": "array", "items": dict(record_schema)},
            "registry_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }


def model_registry_json_schema() -> dict[str, Any]:
    identifier = "^[a-z][a-z0-9_.-]*$"
    record = {
        "type": "object",
        "additionalProperties": False,
        "required": list(_MODEL_FIELDS),
        "properties": {
            "model_id": _string_schema(pattern=identifier),
            "family": _string_schema(pattern=identifier),
            "variant": _string_schema(),
            "version": _string_schema(),
            "registry_revision": {"type": "integer", "minimum": 0},
            "track": _string_schema(enum=MODEL_TRACKS),
            "estimand": _string_schema(),
            "external_reference": {"type": "boolean"},
            "paper": _nullable_string_schema(),
            "repository": _nullable_string_schema(),
            "package": _nullable_string_schema(),
            "license": _nullable_string_schema(),
            "target": _string_schema(),
            "feature_set": _string_schema(),
            "uses_same_row_price": {"type": "boolean"},
            "uses_same_row_observed_pe": {"type": "boolean"},
            "causal": {"type": "boolean"},
            "pit_safe": {"type": "boolean"},
            "train_window": _string_schema(),
            "refit_frequency": _string_schema(),
            "hyperparameters": {"type": "object"},
            "tuning_status": _string_schema(enum=TUNING_STATUSES),
            "locked_status": _string_schema(enum=LOCKED_STATUSES),
            "heldout_status": _string_schema(enum=HELDOUT_STATUSES),
            "fair_log_mae": _nullable_number_schema(),
            "fair_log_rmse": _nullable_number_schema(),
            "worst_seed": {"type": ["integer", "null"]},
            "compute_time": _nullable_number_schema(),
            "status": _string_schema(enum=MODEL_STATUSES),
            "notes": _string_schema(),
            "entrypoint": _string_schema(pattern=r"^[^:\s]+:[^:\s]+$"),
            "feature_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": _string_schema(pattern=identifier),
            },
            "prediction_name": _string_schema(),
            "output_semantics": _string_schema(),
            "deterministic": {"type": "boolean"},
            "description": _string_schema(),
            "parameters_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }
    return _registry_envelope_schema("model", record)


def feature_registry_json_schema() -> dict[str, Any]:
    identifier = "^[a-z][a-z0-9_.-]*$"
    record = {
        "type": "object",
        "additionalProperties": False,
        "required": list(_FEATURE_FIELDS),
        "properties": {
            "feature_id": _string_schema(pattern=identifier),
            "column_name": _string_schema(),
            "description": _string_schema(),
            "source": _string_schema(),
            "dtype": _string_schema(),
            "availability": _string_schema(enum=("point_in_time", "lagged", "evaluation_only")),
            "availability_lag_sessions": {"type": "integer", "minimum": 0},
            "lookahead_sessions": {"type": "integer", "minimum": 0},
            "evaluation_only": {"type": "boolean"},
            "allowed_for_fit": {"type": "boolean"},
            "allowed_for_predict": {"type": "boolean"},
            "point_in_time_safe": {"type": "boolean"},
            "uses_revised_data": {"type": "boolean"},
            "feature_family": _string_schema(enum=FEATURE_FAMILIES),
            "provenance_reference": _string_schema(),
            "provenance_sha256": {
                "type": ["string", "null"],
                "pattern": "^[0-9a-f]{64}$",
            },
            "prefix_invariance_status": _string_schema(enum=FEATURE_AUDIT_STATUSES),
            "future_intervention_status": _string_schema(enum=FEATURE_AUDIT_STATUSES),
            "same_row_target_leakage_status": _string_schema(enum=FEATURE_AUDIT_STATUSES),
            "pit_availability_status": _string_schema(enum=FEATURE_AUDIT_STATUSES),
            "publication_date_status": _string_schema(enum=FEATURE_AUDIT_STATUSES),
            "restatement_availability_status": _string_schema(enum=FEATURE_AUDIT_STATUSES),
            "audit_notes": _string_schema(),
        },
    }
    return _registry_envelope_schema("feature", record)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RegistryError("registry payload must be canonical JSON") from exc


def _seal_registry(registry_type: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": REGISTRY_FORMAT_VERSION,
        "registry_type": registry_type,
        "records": [dict(record) for record in records],
    }
    payload["registry_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def _verify_envelope(payload: Mapping[str, Any], *, registry_type: str) -> None:
    expected_fields = {"format_version", "registry_type", "records", "registry_sha256"}
    if set(payload) != expected_fields:
        raise RegistryError("registry envelope fields are invalid")
    if payload["format_version"] != REGISTRY_FORMAT_VERSION:
        raise RegistryError("unsupported registry format version")
    if payload["registry_type"] != registry_type:
        raise RegistryError("registry type mismatch")
    records = payload["records"]
    if not isinstance(records, list):
        raise RegistryError("registry records must be an array")
    logical = dict(payload)
    recorded = logical.pop("registry_sha256")
    actual = hashlib.sha256(_canonical_json(logical)).hexdigest()
    if recorded != actual:
        raise RegistryError("registry logical SHA-256 mismatch")


def _csv_text(registry_type: str, records: Sequence[Mapping[str, Any]]) -> str:
    fields = _MODEL_FIELDS if registry_type == "model" else _FEATURE_FIELDS
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in records:
        row = dict(record)
        for field in fields:
            value = row[field]
            if isinstance(value, bool):
                row[field] = "true" if value else "false"
            elif isinstance(value, list):
                row[field] = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
            elif isinstance(value, Mapping):
                row[field] = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            elif value is None:
                row[field] = ""
        writer.writerow(row)
    return stream.getvalue()


def _parse_bool(value: str, *, field: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise RegistryError(f"CSV field {field!r} must be true or false")


def _parse_optional_float(value: str, *, field: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RegistryError(f"CSV field {field!r} must be empty or numeric") from exc


def _parse_optional_int(value: str, *, field: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise RegistryError(f"CSV field {field!r} must be empty or an integer") from exc


def _read_csv(path: Path, *, registry_type: str) -> list[dict[str, Any]]:
    fields = _MODEL_FIELDS if registry_type == "model" else _FEATURE_FIELDS
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise RegistryError("CSV registry header mismatch")
        records: list[dict[str, Any]] = []
        for row in reader:
            record: dict[str, Any] = dict(row)
            if registry_type == "model":
                try:
                    feature_ids = json.loads(record["feature_ids"])
                    hyperparameters = json.loads(record["hyperparameters"])
                except json.JSONDecodeError as exc:
                    raise RegistryError(
                        "CSV feature_ids and hyperparameters must contain JSON"
                    ) from exc
                record["feature_ids"] = feature_ids
                record["hyperparameters"] = hyperparameters
                for field in (
                    "external_reference",
                    "uses_same_row_price",
                    "uses_same_row_observed_pe",
                    "causal",
                    "pit_safe",
                    "deterministic",
                ):
                    record[field] = _parse_bool(record[field], field=field)
                for field in ("paper", "repository", "package", "license"):
                    record[field] = record[field] or None
                for field in ("fair_log_mae", "fair_log_rmse", "compute_time"):
                    record[field] = _parse_optional_float(record[field], field=field)
                record["worst_seed"] = _parse_optional_int(record["worst_seed"], field="worst_seed")
                try:
                    record["registry_revision"] = int(record["registry_revision"])
                except ValueError as exc:
                    raise RegistryError("CSV field 'registry_revision' must be an integer") from exc
            else:
                for field in (
                    "evaluation_only",
                    "allowed_for_fit",
                    "allowed_for_predict",
                    "point_in_time_safe",
                    "uses_revised_data",
                ):
                    record[field] = _parse_bool(record[field], field=field)
                for field in ("availability_lag_sessions", "lookahead_sessions"):
                    try:
                        record[field] = int(record[field])
                    except ValueError as exc:
                        raise RegistryError(f"CSV field {field!r} must be an integer") from exc
                record["provenance_sha256"] = record["provenance_sha256"] or None
            records.append(record)
    return records


def _records_from_json(path: Path, *, registry_type: str) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read registry JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RegistryError("registry JSON root must be an object")
    _verify_envelope(payload, registry_type=registry_type)
    return list(payload["records"]), str(payload["registry_sha256"])


def _validate_model_transition(previous: ModelRegistration, current: ModelRegistration) -> None:
    if current.registry_revision != previous.registry_revision + 1:
        raise RegistryError("model registry revisions must be contiguous and monotonic")
    previous_record = previous.to_record()
    current_record = current.to_record()
    changed_definition = [
        field
        for field in _MODEL_DEFINITION_FIELDS
        if current_record[field] != previous_record[field]
    ]
    if changed_definition:
        raise RegistryError(
            f"model definition fields are immutable across registry revisions: {changed_definition}"
        )
    for field, transitions in (
        ("status", _STATUS_TRANSITIONS),
        ("tuning_status", _TUNING_TRANSITIONS),
        ("locked_status", _LOCKED_TRANSITIONS),
        ("heldout_status", _HELDOUT_TRANSITIONS),
    ):
        old = getattr(previous, field)
        new = getattr(current, field)
        if new not in transitions[old]:
            raise RegistryError(f"illegal {field} transition: {old} -> {new}")
    for field in ("fair_log_mae", "fair_log_rmse", "worst_seed", "compute_time"):
        old = getattr(previous, field)
        new = getattr(current, field)
        if old is not None and new != old:
            raise RegistryError(f"published model result {field} is immutable")
    if all(getattr(previous, field) == getattr(current, field) for field in _MODEL_MUTABLE_FIELDS):
        raise RegistryError("model registry revision must change lifecycle or result state")


def _validate_model_history(records: Sequence[ModelRegistration]) -> None:
    histories: dict[str, list[ModelRegistration]] = {}
    for record in records:
        history = histories.setdefault(record.definition_id, [])
        if not history:
            if record.registry_revision != 0:
                raise RegistryError("a model definition history must begin at registry_revision 0")
        else:
            _validate_model_transition(history[-1], record)
        history.append(record)


def _validate_records(
    records: Sequence[Mapping[str, Any]], *, registry_type: str
) -> list[dict[str, Any]]:
    if registry_type == "model":
        parsed = [ModelRegistration.from_record(record) for record in records]
        keys = [record.registration_id for record in parsed]
        _validate_model_history(parsed)
        normalized = [record.to_record() for record in parsed]
    elif registry_type == "feature":
        parsed = [FeatureRegistration.from_record(record) for record in records]
        keys = [record.feature_id for record in parsed]
        normalized = [record.to_record() for record in parsed]
    else:
        raise RegistryError(f"unsupported registry type: {registry_type!r}")
    if len(keys) != len(set(keys)):
        raise RegistryError(f"duplicate {registry_type} registry keys")
    return normalized


def _read_registry(paths: RegistryPaths, *, registry_type: str) -> tuple[list[dict[str, Any]], str]:
    if not paths.csv_path.exists() or not paths.json_path.exists():
        raise RegistryError("both CSV and JSON registry files are required")
    json_records, logical_sha256 = _records_from_json(paths.json_path, registry_type=registry_type)
    json_records = _validate_records(json_records, registry_type=registry_type)
    csv_records = _validate_records(
        _read_csv(paths.csv_path, registry_type=registry_type),
        registry_type=registry_type,
    )
    if csv_records != json_records:
        raise RegistryError("CSV and JSON registry projections differ")
    return json_records, logical_sha256


def load_model_registry_history(paths: RegistryPaths) -> tuple[ModelRegistration, ...]:
    records, _ = _read_registry(paths, registry_type="model")
    return tuple(ModelRegistration.from_record(record) for record in records)


def load_model_registry(paths: RegistryPaths) -> tuple[ModelRegistration, ...]:
    """Return the latest validated event for every immutable model definition."""

    latest: dict[str, ModelRegistration] = {}
    order: list[str] = []
    for record in load_model_registry_history(paths):
        if record.definition_id not in latest:
            order.append(record.definition_id)
        latest[record.definition_id] = record
    return tuple(latest[definition_id] for definition_id in order)


def load_feature_registry(paths: RegistryPaths) -> tuple[FeatureRegistration, ...]:
    records, _ = _read_registry(paths, registry_type="feature")
    return tuple(FeatureRegistration.from_record(record) for record in records)


def validate_registry_cross_references(
    models: Sequence[ModelRegistration],
    features: Sequence[FeatureRegistration],
    *,
    formal_run: bool = False,
) -> None:
    if not isinstance(formal_run, bool):
        raise RegistryError("formal_run must be a boolean")
    feature_map = {feature.feature_id: feature for feature in features}
    if len(feature_map) != len(features):
        raise RegistryError("feature ids must be unique")
    evaluation_columns = {feature.column_name for feature in features if feature.evaluation_only}
    for model in models:
        missing = sorted(set(model.feature_ids).difference(feature_map))
        if missing:
            raise RegistryError(f"model {model.registration_id!r} has unknown features: {missing}")
        for feature_id in model.feature_ids:
            feature = feature_map[feature_id]
            if not feature.allowed_for_fit or not feature.allowed_for_predict:
                raise RegistryError(
                    f"model {model.registration_id!r} references unsafe feature {feature_id!r}"
                )
            if formal_run:
                audit_statuses = (
                    feature.prefix_invariance_status,
                    feature.future_intervention_status,
                    feature.same_row_target_leakage_status,
                    feature.pit_availability_status,
                    feature.publication_date_status,
                    feature.restatement_availability_status,
                )
                if any(status not in {"PASS", "NOT_APPLICABLE"} for status in audit_statuses):
                    raise RegistryError(
                        f"formal runs require completed feature audits for {feature_id!r}"
                    )
        if model.target == EVALUATION_ONLY_TRUTH_COLUMN or model.target in evaluation_columns:
            raise RegistryError(
                f"model {model.registration_id!r} uses an evaluation-only training target"
            )


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    if temporary.exists():
        raise FileExistsError(f"registry temporary path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_registry(
    paths: RegistryPaths,
    records: Sequence[Mapping[str, Any]],
    *,
    registry_type: str,
) -> RegistrySnapshot:
    normalized = _validate_records(records, registry_type=registry_type)
    existing: list[dict[str, Any]] = []
    if paths.csv_path.exists() or paths.json_path.exists():
        existing, _ = _read_registry(paths, registry_type=registry_type)
        if normalized[: len(existing)] != existing or len(normalized) < len(existing):
            raise RegistryError("registry updates must preserve the exact existing prefix")
        if normalized == existing:
            return _snapshot(paths, registry_type=registry_type)

    envelope = _seal_registry(registry_type, normalized)
    json_bytes = (
        json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    )
    csv_bytes = _csv_text(registry_type, normalized).encode("utf-8")
    _atomic_write(paths.json_path, json_bytes)
    _atomic_write(paths.csv_path, csv_bytes)
    return _snapshot(paths, registry_type=registry_type)


def write_model_registry(
    paths: RegistryPaths, records: Sequence[ModelRegistration]
) -> RegistrySnapshot:
    return _write_registry(paths, [record.to_record() for record in records], registry_type="model")


def write_feature_registry(
    paths: RegistryPaths, records: Sequence[FeatureRegistration]
) -> RegistrySnapshot:
    return _write_registry(
        paths, [record.to_record() for record in records], registry_type="feature"
    )


def append_model_registration(
    paths: RegistryPaths, registration: ModelRegistration
) -> RegistrySnapshot:
    existing = list(load_model_registry_history(paths)) if paths.csv_path.exists() else []
    matches = [
        record for record in existing if record.registration_id == registration.registration_id
    ]
    if matches:
        if matches[0] != registration:
            raise RegistryError(f"model registration conflict: {registration.registration_id}")
        return _snapshot(paths, registry_type="model")
    return write_model_registry(paths, [*existing, registration])


def append_feature_registration(
    paths: RegistryPaths, registration: FeatureRegistration
) -> RegistrySnapshot:
    existing = list(load_feature_registry(paths)) if paths.csv_path.exists() else []
    matches = [record for record in existing if record.feature_id == registration.feature_id]
    if matches:
        if matches[0] != registration:
            raise RegistryError(f"feature registration conflict: {registration.feature_id}")
        return _snapshot(paths, registry_type="feature")
    return write_feature_registry(paths, [*existing, registration])


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(paths: RegistryPaths, *, registry_type: str) -> RegistrySnapshot:
    records, logical_sha256 = _read_registry(paths, registry_type=registry_type)
    return RegistrySnapshot(
        paths=paths,
        registry_type=registry_type,
        record_count=len(records),
        logical_sha256=logical_sha256,
        csv_sha256=_file_sha256(paths.csv_path),
        json_sha256=_file_sha256(paths.json_path),
    )


def write_immutable_schema(path: Path, schema: Mapping[str, Any]) -> str:
    data = json.dumps(schema, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if path.exists():
        if path.read_bytes() != data:
            raise RegistryError("schema files are immutable")
    else:
        _atomic_write(path, data)
    return hashlib.sha256(data).hexdigest()
