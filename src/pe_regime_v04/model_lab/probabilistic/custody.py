"""Single-read, content-addressed custody for PIT features, labels, and predictions."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..folds import PITFold
from .authorization import ExecutionAuthorization, canonical_csv_bytes
from .binding import LOCKED_FEATURE_SIDECAR_SHA256
from .contracts import (
    IDENTITY_COLUMNS,
    PROBABILISTIC_DESIGN_SHA256,
    ProbabilisticContractError,
    VerifiedPredictionBatch,
    _make_verified_prediction_batch_from_receipt_bytes,
    canonical_json_bytes,
    identity_sha256,
    logical_frame_sha256,
    normalize_identity_frame,
    require_no_evaluation_truth,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .coverage import _validate_prediction_frame
from .nested import FORMAL_SESSIONS_PER_SEED
from .spec import FEATURE_COLUMNS


FEATURE_PROVENANCE_SCHEMA = "expected_pe_model_zoo.probabilistic_pit_features.v2"
PREDICTION_RECEIPT_SCHEMA = "expected_pe_model_zoo.probabilistic_prediction_receipt.v2"
REFERENCE_PREDICTION_RECEIPT_SCHEMA = (
    "expected_pe_model_zoo.probabilistic_reference_prediction_receipt.v2"
)
REFERENCE_IDS = ("rolling_log_quantiles_252", "point_residual_quantiles_252")
_FEATURE_TOKEN = object()
_LABEL_TOKEN = object()
_POINT_COMPARATOR_TOKEN = object()


def write_content_addressed_csv(directory: Path, stem: str, frame: pd.DataFrame) -> Path:
    """Create a new object by digest; never overwrite an existing path."""

    raw = canonical_csv_bytes(frame)
    digest = sha256_bytes(raw)
    path = Path(directory) / f"{stem}.{digest}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ProbabilisticContractError("content-addressed CSV path contains other bytes")
    return path


def write_content_addressed_json(directory: Path, stem: str, payload: Mapping[str, Any]) -> Path:
    raw = canonical_json_bytes(payload) + b"\n"
    digest = sha256_bytes(raw)
    path = Path(directory) / f"{stem}.{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ProbabilisticContractError("content-addressed JSON path contains other bytes")
    return path


def _require_content_address(path: Path, digest: str) -> None:
    if digest not in Path(path).name.split("."):
        raise ProbabilisticContractError("artifact filename does not contain its SHA-256 address")


def _read_csv_once(
    path: Path, *, expected_raw_sha256: str, context: str
) -> tuple[bytes, pd.DataFrame]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError(f"{context} bytes are unavailable") from exc
    digest = sha256_bytes(raw)
    if digest != expected_raw_sha256:
        raise ProbabilisticContractError(f"{context} differs from authorization")
    _require_content_address(Path(path), digest)
    try:
        frame = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ProbabilisticContractError(f"{context} cannot be decoded") from exc
    return raw, frame


def _normalize_pit_frame(
    frame: pd.DataFrame, *, payload_columns: tuple[str, ...], context: str
) -> pd.DataFrame:
    required = ("seed", "entity_id", "date", "ordered_position", *payload_columns)
    if tuple(frame.columns) != required:
        raise ProbabilisticContractError(f"{context} schema/order differs")
    require_no_evaluation_truth(frame.columns, context=context)
    output = frame.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce", utc=True).dt.tz_convert(None)
    positions = pd.to_numeric(output["ordered_position"], errors="coerce")
    seeds = pd.to_numeric(output["seed"], errors="coerce")
    if output["date"].isna().any() or positions.isna().any() or seeds.isna().any():
        raise ProbabilisticContractError(f"{context} identity contains invalid values")
    if (
        not np.equal(positions, np.floor(positions)).all()
        or not np.equal(seeds, np.floor(seeds)).all()
    ):
        raise ProbabilisticContractError(f"{context} seed/position must be integer")
    output["ordered_position"] = positions.astype(np.int64)
    output["seed"] = seeds.astype(np.int64)
    for _, group in output.groupby(["seed", "entity_id"], sort=False, dropna=False):
        if len(group) != FORMAL_SESSIONS_PER_SEED or not np.array_equal(
            group["ordered_position"].to_numpy(), np.arange(FORMAL_SESSIONS_PER_SEED)
        ):
            raise ProbabilisticContractError(
                f"{context} must contain exact ordered positions 0..1799 per seed/entity"
            )
        if not group["date"].is_monotonic_increasing or group["date"].duplicated().any():
            raise ProbabilisticContractError(f"{context} dates must be unique and chronological")
    return output


class VerifiedPITFeatureArtifact:
    __slots__ = (
        "_raw",
        "_raw_sha256",
        "_frame",
        "_frame_sha256",
        "_provenance_raw",
        "_provenance_raw_sha256",
        "_provenance",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _FEATURE_TOKEN:
            raise ProbabilisticContractError("PIT features must come from the custody factory")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        raw: bytes,
        frame: pd.DataFrame,
        provenance_raw: bytes,
        provenance: Mapping[str, Any],
    ) -> None:
        if token is not _FEATURE_TOKEN:
            raise ProbabilisticContractError("invalid PIT feature factory token")
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_frame_sha256", logical_frame_sha256(frame))
        object.__setattr__(self, "_provenance_raw", bytes(provenance_raw))
        object.__setattr__(self, "_provenance_raw_sha256", sha256_bytes(provenance_raw))
        object.__setattr__(self, "_provenance", dict(provenance))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedPITFeatureArtifact is immutable")

    @property
    def raw_sha256(self) -> str:
        return self._raw_sha256

    @property
    def provenance(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(json.dumps(self._provenance)))

    def _group(self, *, seed: int, entity_id: str) -> pd.DataFrame:
        self.verify_integrity()
        mask = (self._frame["seed"] == seed) & (self._frame["entity_id"] == entity_id)
        group = self._frame.loc[mask].copy()
        if len(group) != FORMAL_SESSIONS_PER_SEED:
            raise ProbabilisticContractError("authorized PIT feature group is absent")
        return group.reset_index(drop=True)

    def verify_integrity(self) -> None:
        if (
            sha256_bytes(self._raw) != self._raw_sha256
            or self._raw_sha256 != self._provenance["feature_artifact_raw_sha256"]
        ):
            raise ProbabilisticContractError("PIT feature raw bytes changed")
        if (
            sha256_bytes(self._provenance_raw) != self._provenance_raw_sha256
            or json.loads(self._provenance_raw) != self._provenance
        ):
            raise ProbabilisticContractError("PIT feature provenance bytes changed")
        if logical_frame_sha256(self._frame) != self._frame_sha256:
            raise ProbabilisticContractError("PIT feature frame changed after loading")


class VerifiedTrainingLabelArtifact:
    __slots__ = ("_raw", "_raw_sha256", "_frame", "_frame_sha256")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _LABEL_TOKEN:
            raise ProbabilisticContractError("training labels must come from the custody factory")
        return super().__new__(cls)

    def __init__(self, token: object, *, raw: bytes, frame: pd.DataFrame) -> None:
        if token is not _LABEL_TOKEN:
            raise ProbabilisticContractError("invalid label factory token")
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_frame_sha256", logical_frame_sha256(frame))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedTrainingLabelArtifact is immutable")

    @property
    def raw_sha256(self) -> str:
        return self._raw_sha256

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw) != self._raw_sha256:
            raise ProbabilisticContractError("training label raw bytes changed")
        if logical_frame_sha256(self._frame) != self._frame_sha256:
            raise ProbabilisticContractError("training label frame changed after loading")

    def _group(self, *, seed: int, entity_id: str) -> pd.DataFrame:
        self.verify_integrity()
        mask = (self._frame["seed"] == seed) & (self._frame["entity_id"] == entity_id)
        group = self._frame.loc[mask].copy()
        if len(group) != FORMAL_SESSIONS_PER_SEED:
            raise ProbabilisticContractError("authorized training-label group is absent")
        return group.reset_index(drop=True)


class VerifiedPointComparatorArtifact:
    """One precommitted common-mask point comparator; IDs cannot be relabeled."""

    __slots__ = (
        "_authorization_sha256",
        "_comparator_id",
        "_frame",
        "_frame_sha256",
        "_identity_sha256",
        "_raw",
        "_raw_sha256",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _POINT_COMPARATOR_TOKEN:
            raise ProbabilisticContractError(
                "point comparators must come from the content-addressed custody factory"
            )
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        comparator_id: str,
        raw: bytes,
        frame: pd.DataFrame,
        authorization_sha256: str,
    ) -> None:
        if token is not _POINT_COMPARATOR_TOKEN:
            raise ProbabilisticContractError("invalid point-comparator factory token")
        identity = normalize_identity_frame(frame, context="point comparator", sort=False)
        object.__setattr__(self, "_authorization_sha256", authorization_sha256)
        object.__setattr__(self, "_comparator_id", comparator_id)
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_frame_sha256", logical_frame_sha256(frame))
        object.__setattr__(self, "_identity_sha256", identity_sha256(identity, sort=False))
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedPointComparatorArtifact is immutable")

    @property
    def authorization_sha256(self) -> str:
        return self._authorization_sha256

    @property
    def comparator_id(self) -> str:
        return self._comparator_id

    @property
    def frame(self) -> pd.DataFrame:
        self.verify_integrity()
        return self._frame.copy(deep=True)

    @property
    def frame_sha256(self) -> str:
        return self._frame_sha256

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    @property
    def raw_sha256(self) -> str:
        return self._raw_sha256

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw) != self._raw_sha256:
            raise ProbabilisticContractError("point comparator raw bytes changed")
        if logical_frame_sha256(self._frame) != self._frame_sha256:
            raise ProbabilisticContractError("point comparator frame changed")


def load_verified_point_comparator(
    path: Path,
    *,
    comparator_id: str,
    authorization: ExecutionAuthorization,
) -> VerifiedPointComparatorArtifact:
    """Load one of the two distinct precommitted common-mask point comparators."""

    if comparator_id not in {"v04_expected_pe", "ml_expected_pe"}:
        raise ProbabilisticContractError("point comparator ID is not predeclared")
    authorization.verify_integrity()
    bindings = authorization.bindings["comparator_prediction_sha256_by_id"]
    if set(bindings) != {"point_history", "v04_expected_pe", "ml_expected_pe"}:
        raise ProbabilisticContractError("point comparator binding set differs")
    if bindings["v04_expected_pe"] == bindings["ml_expected_pe"]:
        raise ProbabilisticContractError("point comparator artifacts must be distinct")
    raw, frame = _read_csv_once(
        path,
        expected_raw_sha256=bindings[comparator_id],
        context=f"{comparator_id} point comparator",
    )
    expected_columns = (*IDENTITY_COLUMNS, "expected_pe")
    if tuple(frame.columns) != expected_columns:
        raise ProbabilisticContractError("point comparator schema/order differs")
    identity = normalize_identity_frame(frame, context="point comparator", sort=False)
    authorized_identity = normalize_identity_frame(
        authorization.identity_frame(), context="authorized point comparator", sort=False
    )
    if not identity.equals(authorized_identity):
        raise ProbabilisticContractError("point comparator common-mask identity differs")
    values = pd.to_numeric(frame["expected_pe"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ProbabilisticContractError("point comparator values must be positive and finite")
    normalized = identity.copy()
    normalized["expected_pe"] = values
    return VerifiedPointComparatorArtifact(
        _POINT_COMPARATOR_TOKEN,
        comparator_id=comparator_id,
        raw=raw,
        frame=normalized,
        authorization_sha256=authorization.raw_sha256,
    )


def seal_point_comparator_custody_receipt(
    artifact: VerifiedPointComparatorArtifact,
    directory: Path,
    *,
    authorization: ExecutionAuthorization,
) -> Path:
    """Persist a deterministic receipt for one precommitted comparator file."""

    if not isinstance(artifact, VerifiedPointComparatorArtifact):
        raise ProbabilisticContractError("point-comparator receipt requires verified custody")
    artifact.verify_integrity()
    authorization.verify_integrity()
    if artifact.authorization_sha256 != authorization.raw_sha256:
        raise ProbabilisticContractError("point-comparator authorization differs")
    payload = seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.point_comparator_custody.v4",
            "authorization_raw_sha256": authorization.raw_sha256,
            "source_closure_sha256": authorization.source_closure_sha256,
            "comparator_id": artifact.comparator_id,
            "comparator_raw_sha256": artifact.raw_sha256,
            "comparator_logical_sha256": artifact.frame_sha256,
            "common_identity_sha256": artifact.identity_sha256,
        }
    )
    return write_content_addressed_json(
        Path(directory), f"point_comparator_receipt_{artifact.comparator_id}", payload
    )


def load_verified_point_comparator_with_receipt(
    prediction_path: Path,
    receipt_path: Path,
    *,
    comparator_id: str,
    authorization: ExecutionAuthorization,
) -> VerifiedPointComparatorArtifact:
    """Load distinct comparator bytes plus their immutable custody receipt."""

    artifact = load_verified_point_comparator(
        prediction_path, comparator_id=comparator_id, authorization=authorization
    )
    try:
        raw = Path(receipt_path).read_bytes()
        receipt = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError(
            "point-comparator custody receipt is unavailable or invalid"
        ) from exc
    if sha256_bytes(raw) not in Path(receipt_path).name.split("."):
        raise ProbabilisticContractError(
            "point-comparator custody receipt is not content-addressed"
        )
    verify_payload_seal(receipt)
    expected = seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.point_comparator_custody.v4",
            "authorization_raw_sha256": authorization.raw_sha256,
            "source_closure_sha256": authorization.source_closure_sha256,
            "comparator_id": artifact.comparator_id,
            "comparator_raw_sha256": artifact.raw_sha256,
            "comparator_logical_sha256": artifact.frame_sha256,
            "common_identity_sha256": artifact.identity_sha256,
        }
    )
    if receipt != expected:
        raise ProbabilisticContractError(
            "point-comparator custody receipt differs from prediction/source bytes"
        )
    return artifact


def build_feature_provenance(*, feature_artifact_raw_sha256: str) -> dict[str, Any]:
    return seal_payload(
        {
            "schema_version": FEATURE_PROVENANCE_SCHEMA,
            "design_sha256": PROBABILISTIC_DESIGN_SHA256,
            "track": "C",
            "feature_sidecar_sha256": LOCKED_FEATURE_SIDECAR_SHA256,
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_artifact_raw_sha256": feature_artifact_raw_sha256,
            "same_row_price_permitted": True,
            "same_row_observed_pe_present": False,
            "true_fair_pe_present": False,
            "pit_policy": "features materialized at-or-before each row decision timestamp",
        }
    )


def load_verified_pit_features(
    path: Path,
    provenance_path: Path,
    *,
    authorization: ExecutionAuthorization,
) -> VerifiedPITFeatureArtifact:
    authorization.verify_integrity()
    bindings = authorization.bindings
    raw, frame = _read_csv_once(
        path,
        expected_raw_sha256=bindings["feature_artifact_raw_sha256"],
        context="PIT feature artifact",
    )
    try:
        provenance_raw = Path(provenance_path).read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError("PIT feature provenance is unavailable") from exc
    if sha256_bytes(provenance_raw) != bindings["feature_provenance_sha256"]:
        raise ProbabilisticContractError("PIT feature provenance differs from authorization")
    _require_content_address(Path(provenance_path), sha256_bytes(provenance_raw))
    try:
        provenance = json.loads(provenance_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("PIT feature provenance cannot be decoded") from exc
    verify_payload_seal(provenance)
    expected = build_feature_provenance(feature_artifact_raw_sha256=sha256_bytes(raw))
    if provenance != expected:
        raise ProbabilisticContractError("PIT feature provenance contract differs")
    normalized = _normalize_pit_frame(
        frame, payload_columns=FEATURE_COLUMNS, context="PIT feature artifact"
    )
    for column in FEATURE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    values = normalized.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    # Missing predictors are part of the locked feature contract: qlinear and
    # NGBoost use fold-training medians, while HistGradientBoosting handles
    # them natively.  Infinite values have no such declared treatment.
    if np.isinf(values).any():
        raise ProbabilisticContractError("PIT features must not contain infinity")
    return VerifiedPITFeatureArtifact(
        _FEATURE_TOKEN,
        raw=raw,
        frame=normalized,
        provenance_raw=provenance_raw,
        provenance=provenance,
    )


def load_verified_training_labels(
    path: Path,
    *,
    authorization: ExecutionAuthorization,
) -> VerifiedTrainingLabelArtifact:
    authorization.verify_integrity()
    raw, frame = _read_csv_once(
        path,
        expected_raw_sha256=authorization.bindings["training_label_artifact_raw_sha256"],
        context="training label artifact",
    )
    normalized = _normalize_pit_frame(
        frame,
        payload_columns=("label_available_at", "observed_pe"),
        context="training label artifact",
    )
    normalized["label_available_at"] = pd.to_datetime(
        normalized["label_available_at"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    observed = pd.to_numeric(normalized["observed_pe"], errors="coerce").to_numpy(dtype=np.float64)
    if normalized["label_available_at"].isna().any() or np.isinf(observed).any():
        raise ProbabilisticContractError("training labels/availability are invalid")
    normalized["observed_pe"] = observed
    return VerifiedTrainingLabelArtifact(_LABEL_TOKEN, raw=raw, frame=normalized)


def materialize_authorized_fold(
    *,
    features: VerifiedPITFeatureArtifact,
    labels: VerifiedTrainingLabelArtifact,
    authorization: ExecutionAuthorization,
    fold: PITFold,
    seed: int,
    entity_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test frames only from factory-held rows and exact fold positions."""

    if not isinstance(features, VerifiedPITFeatureArtifact) or not isinstance(
        labels, VerifiedTrainingLabelArtifact
    ):
        raise ProbabilisticContractError("fold materialization requires verified custody objects")
    feature_group = features._group(seed=seed, entity_id=entity_id)
    label_group = labels._group(seed=seed, entity_id=entity_id)
    fold = require_canonical_authorized_fold(
        labels=labels,
        authorization=authorization,
        fold=fold,
        seed=seed,
        entity_id=entity_id,
    )
    keys = ["seed", "entity_id", "date", "ordered_position"]
    if not feature_group[keys].equals(label_group[keys]):
        raise ProbabilisticContractError("PIT feature and training-label row identities differ")
    train_positions = np.asarray(fold.train_positions, dtype=np.int64)
    test_positions = np.asarray(fold.test_positions, dtype=np.int64)
    if not np.array_equal(feature_group.iloc[train_positions]["ordered_position"], train_positions):
        raise ProbabilisticContractError("fold train positions differ from PIT artifact positions")
    if not np.array_equal(feature_group.iloc[test_positions]["ordered_position"], test_positions):
        raise ProbabilisticContractError("fold test positions differ from PIT artifact positions")
    train = feature_group.iloc[train_positions].copy().reset_index(drop=True)
    train_labels = label_group.iloc[train_positions].copy().reset_index(drop=True)
    if (train_labels["label_available_at"] > pd.Timestamp(fold.label_information_cutoff)).any():
        raise ProbabilisticContractError("training labels were unavailable at fold cutoff")
    train["observed_pe"] = train_labels["observed_pe"].to_numpy(copy=True)
    eligible = np.isfinite(train["observed_pe"].to_numpy(dtype=np.float64)) & (
        train["observed_pe"].to_numpy(dtype=np.float64) > 0.0
    )
    train = train.loc[eligible].reset_index(drop=True)
    if len(train) < 252:
        raise ProbabilisticContractError(
            "authorized fold has fewer than 252 positive finite observed targets"
        )
    test = feature_group.iloc[test_positions].copy().reset_index(drop=True)
    authorized_rows = authorization.identity_frame()
    expected = authorized_rows.loc[
        (authorized_rows["seed"] == seed)
        & (authorized_rows["entity_id"] == entity_id)
        & (authorized_rows["fold_id"] == fold.fold_id)
    ].reset_index(drop=True)
    actual = test.loc[:, ["seed", "entity_id", "date", "ordered_position"]].copy()
    expected_no_fold = expected.loc[:, ["seed", "entity_id", "date", "ordered_position"]]
    if not actual.equals(expected_no_fold):
        raise ProbabilisticContractError(
            "test positions/identities differ from authorized fold rows"
        )
    return train, test


def require_canonical_authorized_fold(
    *,
    labels: VerifiedTrainingLabelArtifact,
    authorization: ExecutionAuthorization,
    fold: PITFold,
    seed: int,
    entity_id: str,
) -> PITFold:
    """Reconstruct and require the one exact scheduler-owned outer fold.

    A :class:`PITFold` is only a request key at this boundary.  Its complete
    membership and temporal metadata are compared with a deterministic rebuild
    from the content-addressed 1,800-row label artifact before any row can be
    materialized.  Consequently a caller cannot shorten, reorder, age, or move
    a training window while retaining an authorized test identity.
    """

    if not isinstance(labels, VerifiedTrainingLabelArtifact) or not isinstance(fold, PITFold):
        raise ProbabilisticContractError("fold verification requires verified labels and PITFold")
    authorization.verify_integrity()
    group = labels._group(seed=seed, entity_id=entity_id)
    from .nested import build_outer_folds

    canonical_by_id = {
        item.fold_id: item
        for item in build_outer_folds(
            group,
            label_available_at_column="label_available_at",
            require_formal_shape=True,
        )
    }
    canonical = canonical_by_id.get(fold.fold_id)
    if canonical is None:
        raise ProbabilisticContractError("fold ID is absent from the canonical outer plan")
    authorized = authorization.identity_frame()
    authorized = authorized.loc[
        (authorized["seed"] == seed)
        & (authorized["entity_id"] == entity_id)
        & (authorized["fold_id"] == fold.fold_id)
    ].reset_index(drop=True)
    if authorized.empty or not np.array_equal(
        authorized["ordered_position"].to_numpy(dtype=np.int64),
        np.asarray(canonical.test_positions, dtype=np.int64),
    ):
        raise ProbabilisticContractError("canonical fold test identity is not authorized")
    if fold != canonical:
        raise ProbabilisticContractError(
            "caller fold differs from exact canonical membership/positions/cutoff"
        )
    return canonical


def seal_prediction_artifact(
    predictions: pd.DataFrame,
    directory: Path,
    *,
    authorization: ExecutionAuthorization,
) -> tuple[Path, Path]:
    """Publish one immutable candidate object and a provenance-complete receipt."""

    authorization.verify_integrity()
    normalized, identity_hash, prediction_hash = _validate_prediction_frame(
        predictions, authorized_identity=authorization.identity_frame()
    )
    raw = canonical_csv_bytes(normalized)
    raw_hash = sha256_bytes(raw)
    prediction_path = write_content_addressed_csv(directory, "predictions", normalized)
    model_id = str(normalized["model_id"].iloc[0])
    bindings = authorization.bindings
    receipt = seal_payload(
        {
            "schema_version": PREDICTION_RECEIPT_SCHEMA,
            "design_sha256": PROBABILISTIC_DESIGN_SHA256,
            "authorization_raw_sha256": authorization.raw_sha256,
            "model_id": model_id,
            "prediction_raw_sha256": raw_hash,
            "prediction_logical_sha256": prediction_hash,
            "identity_logical_sha256": identity_hash,
            "adapter_binding_sha256": bindings["adapter_binding_sha256_by_id"][model_id],
            "candidate_source_snapshot_sha256": bindings["candidate_source_snapshot_sha256"],
            "environment_manifest_sha256": bindings["environment_manifest_sha256_by_id"][model_id],
            "feature_artifact_raw_sha256": bindings["feature_artifact_raw_sha256"],
            "feature_provenance_sha256": bindings["feature_provenance_sha256"],
            "fold_manifest_sha256": bindings["fold_manifest_sha256"],
            "common_mask_manifest_sha256": bindings["common_mask_manifest_sha256"],
        }
    )
    receipt_path = write_content_addressed_json(directory, "prediction_receipt", receipt)
    return prediction_path, receipt_path


def load_verified_prediction_artifact(
    prediction_path: Path,
    receipt_path: Path,
    *,
    authorization: ExecutionAuthorization,
) -> VerifiedPredictionBatch:
    """Read prediction and receipt exactly once into evaluator-safe private custody."""

    authorization.verify_integrity()
    try:
        raw = Path(prediction_path).read_bytes()
        receipt_raw = Path(receipt_path).read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError("prediction custody artifacts are unavailable") from exc
    raw_hash = sha256_bytes(raw)
    receipt_hash = sha256_bytes(receipt_raw)
    _require_content_address(Path(prediction_path), raw_hash)
    _require_content_address(Path(receipt_path), receipt_hash)
    try:
        receipt = json.loads(receipt_raw)
        frame = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise ProbabilisticContractError("prediction custody artifacts cannot be decoded") from exc
    verify_payload_seal(receipt)
    if receipt.get("schema_version") != PREDICTION_RECEIPT_SCHEMA:
        raise ProbabilisticContractError("prediction receipt schema differs")
    normalized, identity_hash, prediction_hash = _validate_prediction_frame(
        frame, authorized_identity=authorization.identity_frame()
    )
    model_id = str(normalized["model_id"].iloc[0])
    bindings = authorization.bindings
    expected_receipt = seal_payload(
        {
            "schema_version": PREDICTION_RECEIPT_SCHEMA,
            "design_sha256": PROBABILISTIC_DESIGN_SHA256,
            "authorization_raw_sha256": authorization.raw_sha256,
            "model_id": model_id,
            "prediction_raw_sha256": raw_hash,
            "prediction_logical_sha256": prediction_hash,
            "identity_logical_sha256": identity_hash,
            "adapter_binding_sha256": bindings["adapter_binding_sha256_by_id"][model_id],
            "candidate_source_snapshot_sha256": bindings["candidate_source_snapshot_sha256"],
            "environment_manifest_sha256": bindings["environment_manifest_sha256_by_id"][model_id],
            "feature_artifact_raw_sha256": bindings["feature_artifact_raw_sha256"],
            "feature_provenance_sha256": bindings["feature_provenance_sha256"],
            "fold_manifest_sha256": bindings["fold_manifest_sha256"],
            "common_mask_manifest_sha256": bindings["common_mask_manifest_sha256"],
        }
    )
    if receipt != expected_receipt:
        raise ProbabilisticContractError("prediction receipt differs from authorization/provenance")
    return _make_verified_prediction_batch_from_receipt_bytes(
        normalized,
        identity_hash=identity_hash,
        prediction_hash=prediction_hash,
        raw_bytes=raw,
        raw_hash=raw_hash,
        receipt_hash=receipt_hash,
        receipt_bytes=receipt_raw,
        authorization_hash=authorization.raw_sha256,
    )


def _reference_prediction_receipt(
    *,
    authorization: ExecutionAuthorization,
    reference_id: str,
    prediction_raw_sha256: str,
    prediction_logical_sha256: str,
    identity_logical_sha256: str,
) -> dict[str, Any]:
    bindings = authorization.bindings
    return seal_payload(
        {
            "schema_version": REFERENCE_PREDICTION_RECEIPT_SCHEMA,
            "design_sha256": PROBABILISTIC_DESIGN_SHA256,
            "authorization_raw_sha256": authorization.raw_sha256,
            "reference_id": reference_id,
            "prediction_raw_sha256": prediction_raw_sha256,
            "prediction_logical_sha256": prediction_logical_sha256,
            "identity_logical_sha256": identity_logical_sha256,
            "reference_binding_sha256": bindings["probabilistic_reference_sha256_by_id"][
                reference_id
            ],
            "training_label_artifact_raw_sha256": bindings["training_label_artifact_raw_sha256"],
            "point_history_raw_sha256": bindings["comparator_prediction_sha256_by_id"][
                "point_history"
            ],
            "fold_manifest_sha256": bindings["fold_manifest_sha256"],
            "common_mask_manifest_sha256": bindings["common_mask_manifest_sha256"],
        }
    )


def seal_reference_prediction_artifact(
    predictions: pd.DataFrame,
    directory: Path,
    *,
    authorization: ExecutionAuthorization,
) -> tuple[Path, Path]:
    """Seal one full-common-mask causal reference distribution."""

    authorization.verify_integrity()
    normalized, identity_hash, prediction_hash = _validate_prediction_frame(
        predictions,
        authorized_identity=authorization.identity_frame(),
        allowed_model_ids=REFERENCE_IDS,
        context="reference",
    )
    reference_id = str(normalized["model_id"].iloc[0])
    raw = canonical_csv_bytes(normalized)
    prediction_path = write_content_addressed_csv(
        directory, f"reference_predictions_{reference_id}", normalized
    )
    receipt = _reference_prediction_receipt(
        authorization=authorization,
        reference_id=reference_id,
        prediction_raw_sha256=sha256_bytes(raw),
        prediction_logical_sha256=prediction_hash,
        identity_logical_sha256=identity_hash,
    )
    receipt_path = write_content_addressed_json(
        directory, f"reference_prediction_receipt_{reference_id}", receipt
    )
    return prediction_path, receipt_path


def load_verified_reference_prediction_artifact(
    prediction_path: Path,
    receipt_path: Path,
    *,
    authorization: ExecutionAuthorization,
) -> VerifiedPredictionBatch:
    """Load a sealed full-common-mask reference under its exact source/config binding."""

    authorization.verify_integrity()
    try:
        raw = Path(prediction_path).read_bytes()
        receipt_raw = Path(receipt_path).read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError("reference prediction artifacts are unavailable") from exc
    raw_hash = sha256_bytes(raw)
    receipt_hash = sha256_bytes(receipt_raw)
    _require_content_address(Path(prediction_path), raw_hash)
    _require_content_address(Path(receipt_path), receipt_hash)
    try:
        receipt = json.loads(receipt_raw)
        frame = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise ProbabilisticContractError(
            "reference prediction artifacts cannot be decoded"
        ) from exc
    verify_payload_seal(receipt)
    normalized, identity_hash, prediction_hash = _validate_prediction_frame(
        frame,
        authorized_identity=authorization.identity_frame(),
        allowed_model_ids=REFERENCE_IDS,
        context="reference",
    )
    reference_id = str(normalized["model_id"].iloc[0])
    expected = _reference_prediction_receipt(
        authorization=authorization,
        reference_id=reference_id,
        prediction_raw_sha256=raw_hash,
        prediction_logical_sha256=prediction_hash,
        identity_logical_sha256=identity_hash,
    )
    if receipt != expected:
        raise ProbabilisticContractError("reference receipt differs from authorization/provenance")
    return _make_verified_prediction_batch_from_receipt_bytes(
        normalized,
        identity_hash=identity_hash,
        prediction_hash=prediction_hash,
        raw_bytes=raw,
        raw_hash=raw_hash,
        receipt_hash=receipt_hash,
        receipt_bytes=receipt_raw,
        authorization_hash=authorization.raw_sha256,
    )
