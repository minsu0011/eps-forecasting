"""Safe adapter between the 150-column canonical frame and Model Lab models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import hashlib
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, Sequence

import numpy as np
import pandas as pd

from .contracts import (
    EVALUATION_ONLY_TRUTH_COLUMN,
    MODEL_CONTROL_COLUMNS,
    ContractError,
    FeatureMetadata,
    PredictionPhase,
    validate_feature_frame,
)


CANONICAL_INPUT_COLUMNS = 150
FORMAL_EVALUATION_START = pd.Timestamp("2015-01-02")
CANONICAL_IDENTITY_COLUMNS = ("date",)
DEFAULT_IDENTITY_COLUMNS = ("seed", "date")
PIT_SIDECAR_SERIALIZATION = "model_lab_pit_long_csv_utf8_lf_v1"
_PIT_CUTOFF_POLICIES_UTC = MappingProxyType({"DGP_FUNDAMENTAL_1230_UTC": time(12, 30, 0)})
PIT_CUTOFF_POLICY_IDS = tuple(_PIT_CUTOFF_POLICIES_UTC)


@dataclass(frozen=True)
class PreparedModelInput:
    identity: pd.DataFrame
    features: pd.DataFrame
    feature_metadata: tuple[FeatureMetadata, ...]
    source_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.identity.index.equals(self.features.index):
            raise ContractError("identity and feature indices must match exactly")
        if len(self.source_columns) != CANONICAL_INPUT_COLUMNS or len(
            set(self.source_columns)
        ) != len(self.source_columns):
            raise ContractError(
                "source_columns must preserve the unique canonical 150-column header"
            )


@dataclass(frozen=True)
class PITFeatureSidecarSelection:
    """Wide model input plus a replayable, hashed as-of selection trace."""

    prepared: PreparedModelInput
    selection_provenance: pd.DataFrame
    raw_sidecar_sha256: str
    schema_sha256: str
    sidecar_serialization: str
    cutoff_policy_id: str
    decision_cutoff_sha256: str
    normalized_source_sha256: str
    selection_sha256: str
    factor_source_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        hashes = (
            self.raw_sidecar_sha256,
            self.schema_sha256,
            self.decision_cutoff_sha256,
            self.normalized_source_sha256,
            self.selection_sha256,
            *(value for _, value in self.factor_source_sha256),
        )
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in hashes
        ):
            raise ContractError("sidecar provenance hashes must be lowercase SHA-256")
        if not self.cutoff_policy_id or not isinstance(self.cutoff_policy_id, str):
            raise ContractError("cutoff_policy_id must be a non-empty string")
        if self.sidecar_serialization != PIT_SIDECAR_SERIALIZATION:
            raise ContractError("sidecar_serialization is not the sealed PIT format")


def prepare_model_input(
    canonical_frame: pd.DataFrame,
    features: Sequence[FeatureMetadata],
    *,
    identity_columns: Sequence[str] = CANONICAL_IDENTITY_COLUMNS,
    phase: PredictionPhase = "predict",
) -> PreparedModelInput:
    """Select declared PIT features without ever carrying evaluation truth.

    The width check applies to the complete canonical input frame, including
    identity columns. The fixed 150-column contract cannot be bypassed by a
    caller override.
    """

    if canonical_frame.columns.has_duplicates:
        raise ContractError("canonical input columns must be unique")
    expected_index = pd.RangeIndex(len(canonical_frame))
    if canonical_frame.index.name is not None or not canonical_frame.index.equals(expected_index):
        raise ContractError(
            "canonical input must use an unnamed zero-based RangeIndex; "
            "experiment identity belongs in detached metadata"
        )
    if len(canonical_frame.columns) != CANONICAL_INPUT_COLUMNS:
        raise ContractError(
            f"canonical input width must be {CANONICAL_INPUT_COLUMNS}, got "
            f"{len(canonical_frame.columns)}"
        )
    control_columns = sorted(MODEL_CONTROL_COLUMNS.intersection(canonical_frame.columns))
    if control_columns:
        raise ContractError(
            f"canonical model input contains reserved control columns: {control_columns}"
        )
    identity_names = tuple(identity_columns)
    if not identity_names or len(set(identity_names)) != len(identity_names):
        raise ContractError("identity_columns must be non-empty and unique")
    missing_identity = sorted(set(identity_names).difference(canonical_frame.columns))
    if missing_identity:
        raise ContractError(f"canonical input is missing identity columns: {missing_identity}")
    if canonical_frame.loc[:, list(identity_names)].isna().any().any():
        raise ContractError("identity columns cannot contain missing values")

    selected = tuple(features)
    feature_columns = [feature.column_name for feature in selected]
    identity_features = sorted(set(identity_names).intersection(feature_columns))
    if identity_features:
        raise ContractError(
            f"identity columns cannot be selected as model features: {identity_features}"
        )
    missing_features = sorted(set(feature_columns).difference(canonical_frame.columns))
    if missing_features:
        raise ContractError(f"canonical input is missing feature columns: {missing_features}")
    model_frame = canonical_frame.loc[:, feature_columns].copy()
    validate_feature_frame(model_frame, selected, phase=phase)
    identity = canonical_frame.loc[:, list(identity_names)].copy()
    if "date" in identity:
        dates = pd.to_datetime(identity["date"], errors="coerce")
        if dates.isna().any():
            raise ContractError("identity date contains invalid timestamps")
        identity["date"] = dates
    return PreparedModelInput(
        identity=identity,
        features=model_frame,
        feature_metadata=selected,
        source_columns=tuple(canonical_frame.columns),
    )


def merge_pit_feature_sidecar(
    prepared: PreparedModelInput,
    sidecar: pd.DataFrame,
    features: Sequence[FeatureMetadata],
    *,
    identity_columns: Sequence[str] = CANONICAL_IDENTITY_COLUMNS,
    available_at_column: str | None = "available_at",
    phase: PredictionPhase = "predict",
) -> NoReturn:
    """Reject legacy wide sidecars before they can create model input.

    The arguments remain only to give existing callers a deterministic failure.
    No legacy-wide data or acknowledgement can return a model-consumable type.
    """

    raise ContractError(
        "legacy wide sidecars cannot produce fit/predict input; use a verified long-form "
        "PIT artifact"
    )


PIT_LONG_SIDECAR_COLUMNS = (
    "entity_id",
    "factor_id",
    "observed_at",
    "available_at",
    "effective_session",
    "source_id",
    "revision_id",
    "is_missing",
    "value",
)


def _timezone_aware_utc(values: pd.Series, *, field: str) -> pd.Series:
    converted: list[pd.Timestamp] = []
    for value in values.tolist():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{field} must contain valid timestamps") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ContractError(f"{field} must be explicitly timezone-aware")
        converted.append(timestamp.tz_convert("UTC"))
    return pd.Series(pd.DatetimeIndex(converted), index=values.index, name=field)


def _session_dates(values: pd.Series, *, field: str) -> pd.Series:
    converted: list[pd.Timestamp] = []
    for value in values.tolist():
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{field} must contain valid session dates") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is not None:
            raise ContractError(f"{field} must contain timezone-naive session dates")
        if timestamp != timestamp.normalize():
            raise ContractError(f"{field} must contain date-only session labels")
        converted.append(timestamp)
    return pd.Series(pd.DatetimeIndex(converted), index=values.index, name=field)


def _hash_scalar(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return {"nonfinite": "nan"}
        if np.isposinf(numeric):
            return {"nonfinite": "+inf"}
        if np.isneginf(numeric):
            return {"nonfinite": "-inf"}
        return numeric
    if value is None or value is pd.NA:
        return None
    return value


def _logical_frame_sha256(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    sort_columns: Sequence[str],
) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(sort_columns), kind="mergesort")
    records = [
        {column: _hash_scalar(value) for column, value in row.items()}
        for row in ordered.to_dict(orient="records")
    ]
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_PIT_LONG_LOGICAL_TYPES = MappingProxyType(
    {
        "entity_id": "non_empty_string",
        "factor_id": "non_empty_string",
        "observed_at": "timezone_aware_utc_timestamp",
        "available_at": "timezone_aware_utc_timestamp",
        "effective_session": "timezone_naive_date",
        "source_id": "non_empty_string",
        "revision_id": "non_negative_integer",
        "is_missing": "boolean",
        "value": "finite_number_or_explicitly_masked_missing",
    }
)


def _normalize_pit_long_sidecar(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.has_duplicates or tuple(frame.columns) != PIT_LONG_SIDECAR_COLUMNS:
        raise ContractError("long-form feature sidecar must use the exact sealed column order")
    if frame.empty:
        raise ContractError("long-form feature sidecar must be non-empty")
    source = frame.copy()
    for field in ("entity_id", "factor_id", "source_id"):
        if (
            source[field].isna().any()
            or not source[field].map(lambda value: isinstance(value, str) and bool(value)).all()
        ):
            raise ContractError(f"{field} must contain non-empty strings")
    source_controls = sorted(
        set(source.loc[source["factor_id"].isin(MODEL_CONTROL_COLUMNS), "factor_id"])
    )
    if source_controls:
        raise ContractError(
            f"reserved control factors cannot enter an as-of feature sidecar: {source_controls}"
        )
    source["observed_at"] = _timezone_aware_utc(source["observed_at"], field="observed_at")
    source["available_at"] = _timezone_aware_utc(source["available_at"], field="available_at")
    if source["available_at"].lt(source["observed_at"]).any():
        raise ContractError("available_at cannot precede observed_at")
    source["effective_session"] = _session_dates(
        source["effective_session"], field="effective_session"
    )

    revisions: list[int] = []
    for value in source["revision_id"].tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ContractError("revision_id must contain non-negative integers")
        revision = int(value)
        if revision < 0:
            raise ContractError("revision_id must contain non-negative integers")
        revisions.append(revision)
    source["revision_id"] = pd.Series(revisions, index=source.index, dtype=np.int64)
    if not source["is_missing"].map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ContractError("is_missing must contain booleans")
    source["is_missing"] = source["is_missing"].astype(bool)

    numeric_values: list[float] = []
    for value in source["value"].tolist():
        if value is None or value is pd.NA:
            numeric_values.append(np.nan)
            continue
        if isinstance(value, (bool, np.bool_)):
            raise ContractError("sidecar values must be numeric or explicitly missing")
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ContractError("sidecar values must be numeric or explicitly missing") from exc
    source["value"] = pd.Series(numeric_values, index=source.index, dtype=np.float64)
    unmasked = ~source["is_missing"]
    if not np.isfinite(source.loc[unmasked, "value"].to_numpy(dtype=np.float64)).all():
        raise ContractError("non-finite sidecar values must be explicitly masked")

    record_key = [
        "entity_id",
        "factor_id",
        "effective_session",
        "source_id",
        "revision_id",
    ]
    if source.duplicated(record_key).any():
        raise ContractError("long-form sidecar contains duplicate/conflicting revisions")
    if source.groupby(["entity_id", "factor_id"], sort=False)["source_id"].nunique().gt(1).any():
        raise ContractError("each entity/factor history must have one unambiguous source_id")
    revision_groups = source.groupby(
        ["entity_id", "factor_id", "effective_session", "source_id"], sort=False
    )
    for _, group in revision_groups:
        ordered_revisions = group.sort_values("revision_id", kind="mergesort")
        if (
            not ordered_revisions["observed_at"].is_monotonic_increasing
            or not ordered_revisions["available_at"].is_monotonic_increasing
        ):
            raise ContractError(
                "sidecar revision chronology cannot roll back observed_at or available_at"
            )
    return source


def _pit_schema_sha256(normalized: pd.DataFrame) -> str:
    payload = {
        "serialization": PIT_SIDECAR_SERIALIZATION,
        "columns": [
            {
                "name": column,
                "logical_type": _PIT_LONG_LOGICAL_TYPES[column],
                "normalized_dtype": str(normalized[column].dtype),
            }
            for column in PIT_LONG_SIDECAR_COLUMNS
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


class VerifiedPITSidecar:
    """Immutable raw-byte snapshot verified before long-form PIT selection."""

    __slots__ = (
        "_raw_bytes",
        "raw_sidecar_sha256",
        "schema_sha256",
        "serialization",
        "_sealed",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(self, "_sealed"):
            raise AttributeError("VerifiedPITSidecar is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        raw_bytes: bytes,
        *,
        serialization: str = PIT_SIDECAR_SERIALIZATION,
        expected_raw_sidecar_sha256: str | None = None,
        expected_schema_sha256: str | None = None,
    ) -> None:
        if serialization != PIT_SIDECAR_SERIALIZATION:
            raise ContractError(f"unsupported PIT sidecar serialization: {serialization!r}")
        if not isinstance(raw_bytes, bytes):
            raise ContractError("PIT sidecar raw artifact must be bytes")
        snapshot = bytes(raw_bytes)
        if not snapshot or not snapshot.endswith(b"\n") or b"\r" in snapshot:
            raise ContractError("PIT sidecar CSV must use canonical non-empty LF bytes")
        if snapshot.startswith(b"\xef\xbb\xbf"):
            raise ContractError("PIT sidecar CSV must be UTF-8 without BOM")
        raw_sha256 = hashlib.sha256(snapshot).hexdigest()
        if expected_raw_sidecar_sha256 is not None and expected_raw_sidecar_sha256 != raw_sha256:
            raise ContractError(
                "expected_raw_sidecar_sha256 does not match the recomputed artifact hash"
            )
        try:
            parsed = pd.read_csv(io.BytesIO(snapshot))
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            raise ContractError("PIT sidecar CSV bytes cannot be parsed") from exc
        canonical = parsed.to_csv(index=False, lineterminator="\n").encode("utf-8")
        if canonical != snapshot:
            raise ContractError("PIT sidecar CSV bytes are not canonical for the declared format")
        normalized = _normalize_pit_long_sidecar(parsed)
        schema_sha256 = _pit_schema_sha256(normalized)
        if expected_schema_sha256 is not None and expected_schema_sha256 != schema_sha256:
            raise ContractError(
                "expected_schema_sha256 does not match the recomputed artifact hash"
            )
        self._raw_bytes = snapshot
        self.raw_sidecar_sha256 = raw_sha256
        self.schema_sha256 = schema_sha256
        self.serialization = serialization
        self._sealed = True

    def to_frame(self) -> pd.DataFrame:
        if hashlib.sha256(self._raw_bytes).hexdigest() != self.raw_sidecar_sha256:
            raise ContractError("verified PIT sidecar raw snapshot changed")
        parsed = pd.read_csv(io.BytesIO(self._raw_bytes))
        normalized = _normalize_pit_long_sidecar(parsed)
        if _pit_schema_sha256(normalized) != self.schema_sha256:
            raise ContractError("verified PIT sidecar schema changed")
        return normalized


def load_verified_pit_feature_sidecar(
    path: Path,
    *,
    expected_raw_sidecar_sha256: str | None = None,
    expected_schema_sha256: str | None = None,
) -> VerifiedPITSidecar:
    """Read and verify a sealed PIT sidecar file into an immutable snapshot."""

    try:
        raw_bytes = Path(path).read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read PIT sidecar artifact: {path}") from exc
    return VerifiedPITSidecar(
        raw_bytes,
        expected_raw_sidecar_sha256=expected_raw_sidecar_sha256,
        expected_schema_sha256=expected_schema_sha256,
    )


def prepare_pit_feature_sidecar_asof(
    prepared: PreparedModelInput,
    sidecar: VerifiedPITSidecar,
    features: Sequence[FeatureMetadata],
    *,
    decision_cutoffs: pd.Series,
    cutoff_policy_id: str,
    entity_ids: str | pd.Series | None = None,
    phase: PredictionPhase = "predict",
) -> PITFeatureSidecarSelection:
    """Select long-form factor revisions strictly as of each decision instant.

    The source format is sealed by :data:`PIT_LONG_SIDECAR_COLUMNS`. Selection
    requires explicit timezone-aware observation, availability, and decision
    instants. Every decision/factor pair needs an eligible record; actual
    missingness must therefore be represented by ``is_missing=true`` rather
    than an absent row.
    """

    if type(sidecar) is not VerifiedPITSidecar:
        raise ContractError("as-of selection requires a VerifiedPITSidecar artifact")
    if cutoff_policy_id not in _PIT_CUTOFF_POLICIES_UTC:
        raise ContractError(f"unknown code-owned cutoff policy: {cutoff_policy_id!r}")
    expected_cutoff = _PIT_CUTOFF_POLICIES_UTC[cutoff_policy_id]

    selected = tuple(features)
    if not selected:
        raise ContractError("as-of sidecar features must be non-empty")
    if not decision_cutoffs.index.equals(prepared.features.index):
        raise ContractError("decision_cutoffs index must exactly match prepared input")
    if "date" not in prepared.identity:
        raise ContractError("prepared identity must include an effective session date")

    feature_ids = [feature.feature_id for feature in selected]
    feature_columns = [feature.column_name for feature in selected]
    if len(feature_ids) != len(set(feature_ids)) or len(feature_columns) != len(
        set(feature_columns)
    ):
        raise ContractError("as-of sidecar feature ids and columns must be unique")
    if EVALUATION_ONLY_TRUTH_COLUMN in {*feature_ids, *feature_columns}:
        raise ContractError("true_fair_pe must never enter an as-of feature sidecar")
    existing_ids = {feature.feature_id for feature in prepared.feature_metadata}
    if existing_ids.intersection(feature_ids) or set(prepared.features).intersection(
        feature_columns
    ):
        raise ContractError("as-of sidecar features cannot replace prepared features")
    canonical_collisions = sorted(set(prepared.source_columns).intersection(feature_columns))
    if canonical_collisions:
        raise ContractError(
            f"as-of sidecar features cannot overwrite the canonical header: {canonical_collisions}"
        )

    source = sidecar.to_frame()
    selected_source = source.loc[source["factor_id"].isin(feature_ids)].copy()
    if set(selected_source["factor_id"].unique()) != set(feature_ids):
        missing = sorted(set(feature_ids).difference(selected_source["factor_id"]))
        raise ContractError(f"long-form sidecar is missing registered factors: {missing}")
    source_hash_columns = PIT_LONG_SIDECAR_COLUMNS
    source_sort = (
        "entity_id",
        "factor_id",
        "effective_session",
        "observed_at",
        "available_at",
        "source_id",
        "revision_id",
    )
    normalized_source_sha256 = _logical_frame_sha256(
        selected_source,
        columns=source_hash_columns,
        sort_columns=source_sort,
    )
    factor_hashes = tuple(
        (
            feature.feature_id,
            _logical_frame_sha256(
                selected_source.loc[selected_source["factor_id"].eq(feature.feature_id)],
                columns=source_hash_columns,
                sort_columns=source_sort,
            ),
        )
        for feature in selected
    )
    for feature, (_, factor_hash) in zip(selected, factor_hashes, strict=True):
        if feature.provenance_sha256 is not None and feature.provenance_sha256 != factor_hash:
            raise ContractError(
                f"feature {feature.feature_id!r} provenance_sha256 does not bind its source"
            )

    decisions = _timezone_aware_utc(decision_cutoffs, field="decision_cutoffs")
    sessions = _session_dates(prepared.identity["date"], field="prepared identity date")
    decision_session = decisions.dt.tz_localize(None).dt.normalize()
    if not decision_session.equals(sessions):
        raise ContractError(
            "decision_cutoffs must fall on the exact prepared effective session in UTC"
        )
    decision_times = decisions.map(lambda value: value.time().replace(tzinfo=None))
    if not decision_times.eq(expected_cutoff).all():
        raise ContractError("decision_cutoffs do not match the sealed UTC cutoff time")
    cutoff_frame = pd.DataFrame(
        {
            "row_ordinal": np.arange(len(decisions), dtype=np.int64),
            "decision_session": sessions,
            "decision_at": decisions,
        }
    )
    decision_cutoff_sha256 = _logical_frame_sha256(
        cutoff_frame,
        columns=("row_ordinal", "decision_session", "decision_at"),
        sort_columns=("row_ordinal",),
    )
    if entity_ids is None:
        if "entity_id" not in prepared.identity:
            raise ContractError("entity_ids must be supplied when identity has no entity_id")
        entities = prepared.identity["entity_id"].copy()
    elif isinstance(entity_ids, str):
        if not entity_ids:
            raise ContractError("entity_ids scalar must be non-empty")
        entities = pd.Series(entity_ids, index=prepared.features.index)
    elif isinstance(entity_ids, pd.Series):
        if not entity_ids.index.equals(prepared.features.index):
            raise ContractError("entity_ids index must exactly match prepared input")
        entities = entity_ids.copy()
    else:
        raise ContractError("entity_ids must be a string, aligned Series, or identity column")
    if (
        entities.isna().any()
        or not entities.map(lambda value: isinstance(value, str) and bool(value)).all()
    ):
        raise ContractError("entity_ids must contain non-empty strings")

    histories = {
        key: group.sort_values(
            ["effective_session", "observed_at", "available_at", "revision_id"],
            kind="mergesort",
        )
        for key, group in selected_source.groupby(["entity_id", "factor_id"], sort=False)
    }
    selections: list[dict[str, object]] = []
    for row_ordinal, (entity, decision_at, session) in enumerate(
        zip(entities.tolist(), decisions.tolist(), sessions.tolist(), strict=True)
    ):
        for feature in selected:
            history = histories.get((entity, feature.feature_id))
            if history is None:
                raise ContractError(
                    f"no sidecar history for entity={entity!r}, factor={feature.feature_id!r}"
                )
            eligible = history.loc[
                history["observed_at"].le(decision_at)
                & history["available_at"].le(decision_at)
                & history["effective_session"].le(session)
            ]
            if eligible.empty:
                raise ContractError(
                    "no eligible sidecar revision at decision cutoff for "
                    f"entity={entity!r}, factor={feature.feature_id!r}, row={row_ordinal}"
                )
            latest_session = eligible["effective_session"].max()
            latest_session_records = eligible.loc[eligible["effective_session"].eq(latest_session)]
            chosen = latest_session_records.loc[latest_session_records["revision_id"].idxmax()]
            selections.append(
                {
                    "row_ordinal": row_ordinal,
                    "decision_at": decision_at,
                    "decision_session": session,
                    **{column: chosen[column] for column in PIT_LONG_SIDECAR_COLUMNS},
                }
            )

    provenance = pd.DataFrame(selections)
    selection_columns = (
        "row_ordinal",
        "decision_at",
        "decision_session",
        *PIT_LONG_SIDECAR_COLUMNS,
    )
    selection_sort = ("row_ordinal", "factor_id")
    selection_sha256 = _logical_frame_sha256(
        provenance,
        columns=selection_columns,
        sort_columns=selection_sort,
    )
    wide = pd.DataFrame(index=prepared.features.index)
    for feature in selected:
        factor_rows = provenance.loc[provenance["factor_id"].eq(feature.feature_id)].sort_values(
            "row_ordinal", kind="mergesort"
        )
        values = factor_rows["value"].to_numpy(dtype=np.float64, copy=True)
        values[factor_rows["is_missing"].to_numpy(dtype=bool)] = np.nan
        wide[feature.column_name] = values
    validate_feature_frame(wide, selected, phase=phase)
    combined = pd.concat([prepared.features, wide], axis=1)
    return PITFeatureSidecarSelection(
        prepared=PreparedModelInput(
            identity=prepared.identity.copy(),
            features=combined,
            feature_metadata=(*prepared.feature_metadata, *selected),
            source_columns=prepared.source_columns,
        ),
        selection_provenance=provenance.loc[:, list(selection_columns)].copy(),
        raw_sidecar_sha256=sidecar.raw_sidecar_sha256,
        schema_sha256=sidecar.schema_sha256,
        sidecar_serialization=sidecar.serialization,
        cutoff_policy_id=cutoff_policy_id,
        decision_cutoff_sha256=decision_cutoff_sha256,
        normalized_source_sha256=normalized_source_sha256,
        selection_sha256=selection_sha256,
        factor_source_sha256=factor_hashes,
    )


def attach_prediction_identity(
    prepared: PreparedModelInput,
    prediction: pd.Series,
    *,
    seed: int,
    model_id: str,
    fold_id: str,
) -> pd.DataFrame:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ContractError("seed must be supplied as external integer experiment metadata")
    if not prediction.index.equals(prepared.features.index):
        raise ContractError("prediction index must exactly match prepared model input")
    numeric = pd.to_numeric(prediction, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    output = prepared.identity.copy()
    output.insert(0, "seed", seed)
    output["fold_id"] = fold_id
    output["model_id"] = model_id
    output["prediction"] = numeric
    return output


def prepare_evaluation_truth(
    truth_frame: pd.DataFrame,
    *,
    seed: int,
    date_column: str = "date",
) -> pd.DataFrame:
    """Attach external seed metadata to a detached per-seed truth file."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ContractError("seed must be supplied as external integer experiment metadata")
    required = {date_column, EVALUATION_ONLY_TRUTH_COLUMN}
    missing = sorted(required.difference(truth_frame.columns))
    if missing:
        raise ContractError(f"detached truth is missing columns: {missing}")
    dates = pd.to_datetime(truth_frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ContractError("detached truth contains invalid dates")
    if dates.duplicated().any():
        raise ContractError("detached truth dates must be unique within one seed")
    output = pd.DataFrame(
        {
            "seed": seed,
            "date": dates,
            EVALUATION_ONLY_TRUTH_COLUMN: pd.to_numeric(
                truth_frame[EVALUATION_ONLY_TRUTH_COLUMN], errors="coerce"
            ).to_numpy(dtype=np.float64, na_value=np.nan),
        },
        index=truth_frame.index,
    )
    return output


def merge_evaluation_truth(
    predictions: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    evaluation_start: str | pd.Timestamp = FORMAL_EVALUATION_START,
) -> pd.DataFrame:
    """Merge separate truth only after predictions exist, using identity/date keys."""

    keys = tuple(identity_columns)
    required_prediction = {*keys, "model_id", "prediction"}
    missing_prediction = sorted(required_prediction.difference(predictions.columns))
    if missing_prediction:
        raise ContractError(f"prediction frame is missing columns: {missing_prediction}")
    if EVALUATION_ONLY_TRUTH_COLUMN in predictions.columns:
        raise ContractError("predictions must be produced before evaluation truth is attached")
    required_truth = {*keys, EVALUATION_ONLY_TRUTH_COLUMN}
    missing_truth = sorted(required_truth.difference(truth.columns))
    if missing_truth:
        raise ContractError(f"truth frame is missing columns: {missing_truth}")
    if truth.duplicated(list(keys)).any():
        raise ContractError("truth identity keys must be unique")

    prediction_frame = predictions.copy()
    truth_frame = truth.loc[:, [*keys, EVALUATION_ONLY_TRUTH_COLUMN]].copy()
    if "date" not in keys:
        raise ContractError("identity_columns must include date for an explicit chronological join")
    prediction_frame["date"] = pd.to_datetime(prediction_frame["date"], errors="coerce")
    truth_frame["date"] = pd.to_datetime(truth_frame["date"], errors="coerce")
    if prediction_frame["date"].isna().any() or truth_frame["date"].isna().any():
        raise ContractError("prediction and truth dates must be valid")
    cutoff = pd.Timestamp(evaluation_start)
    prediction_frame = prediction_frame.loc[prediction_frame["date"] >= cutoff].copy()
    truth_frame = truth_frame.loc[truth_frame["date"] >= cutoff].copy()

    merged = prediction_frame.merge(
        truth_frame,
        how="left",
        on=list(keys),
        validate="many_to_one",
        sort=False,
    )
    if merged[EVALUATION_ONLY_TRUTH_COLUMN].isna().any():
        missing = int(merged[EVALUATION_ONLY_TRUTH_COLUMN].isna().sum())
        raise ContractError(f"evaluation truth is missing for {missing} prediction rows")
    sort_columns = [column for column in (*keys, "model_id", "fold_id") if column in merged]
    return merged.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def validate_identity_truth_consistency(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
) -> None:
    """Require one exact numeric truth value for every shared identity."""

    keys = tuple(identity_columns)
    required = {*keys, EVALUATION_ONLY_TRUTH_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContractError(f"truth consistency frame is missing columns: {missing}")

    def truth_token(value: object) -> str:
        if isinstance(value, (bool, np.bool_)):
            raise ContractError("evaluation truth must be numeric, never boolean")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ContractError("evaluation truth must be numeric") from exc
        if np.isnan(numeric):
            return "nan"
        return numeric.hex()

    checked = frame.loc[:, [*keys, EVALUATION_ONLY_TRUTH_COLUMN]].copy()
    checked["_truth_token"] = checked[EVALUATION_ONLY_TRUTH_COLUMN].map(truth_token)
    conflicts = (
        checked.groupby(list(keys), sort=False, dropna=False)["_truth_token"]
        .nunique(dropna=False)
        .gt(1)
    )
    if conflicts.any():
        raise ContractError("evaluation truth must be identical for every shared identity")


def evaluation_identity_sha256(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
) -> str:
    """Hash the exact unique identity set used by a metric computation."""

    keys = tuple(identity_columns)
    missing = sorted(set(keys).difference(frame.columns))
    if missing:
        raise ContractError(f"identity hash frame is missing columns: {missing}")
    identities = frame.loc[:, list(keys)].drop_duplicates().copy()
    return _logical_frame_sha256(
        identities,
        columns=keys,
        sort_columns=keys,
    )


def apply_identical_common_mask(
    prediction_frame: pd.DataFrame,
    *,
    required_model_ids: Sequence[str],
    identity_columns: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
) -> pd.DataFrame:
    """Return the exact same positive-finite truth/prediction rows for every model."""

    models = tuple(required_model_ids)
    if not models or len(models) != len(set(models)):
        raise ContractError("required_model_ids must be non-empty and unique")
    keys = tuple(identity_columns)
    required = {*keys, "model_id", "prediction", EVALUATION_ONLY_TRUTH_COLUMN}
    missing = sorted(required.difference(prediction_frame.columns))
    if missing:
        raise ContractError(f"prediction frame is missing columns: {missing}")
    selected = prediction_frame.loc[prediction_frame["model_id"].isin(models)].copy()
    if set(selected["model_id"].unique()) != set(models):
        raise ContractError("every required model must be present")
    if selected.duplicated([*keys, "model_id"]).any():
        raise ContractError("model predictions must be unique per identity key")
    validate_identity_truth_consistency(selected, identity_columns=keys)
    identity_model_counts = selected.groupby(list(keys), sort=False, dropna=False)[
        "model_id"
    ].nunique()
    if not identity_model_counts.eq(len(models)).all():
        raise ContractError("common-mask models must have identical pre-mask identity sets")

    prediction = pd.to_numeric(selected["prediction"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    truth = pd.to_numeric(selected[EVALUATION_ONLY_TRUTH_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    selected["_valid"] = (
        np.isfinite(prediction) & (prediction > 0.0) & np.isfinite(truth) & (truth > 0.0)
    )
    grouped = selected.groupby(list(keys), sort=False, dropna=False)
    valid_identity = grouped.agg(
        model_count=("model_id", "nunique"),
        all_valid=("_valid", "all"),
        truth_count=(EVALUATION_ONLY_TRUTH_COLUMN, "nunique"),
    )
    valid_identity = valid_identity.loc[
        (valid_identity["model_count"] == len(models))
        & valid_identity["all_valid"]
        & (valid_identity["truth_count"] == 1)
    ].reset_index()[list(keys)]
    common = selected.merge(valid_identity, how="inner", on=list(keys), validate="many_to_one")
    common = common.drop(columns="_valid")
    counts = common.groupby("model_id", sort=False).size()
    if len(counts) != len(models) or counts.nunique() != 1:
        raise ContractError("failed to construct an identical common mask")
    return common.sort_values([*keys, "model_id"], kind="mergesort").reset_index(drop=True)
