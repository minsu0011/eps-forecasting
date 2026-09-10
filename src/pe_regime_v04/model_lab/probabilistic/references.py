"""Factory-only causal reference distributions with exact grouped residual shifts."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..folds import PITFold
from .authorization import ExecutionAuthorization, canonical_csv_bytes
from .contracts import (
    IDENTITY_COLUMNS,
    PROBABILISTIC_DESIGN_SHA256,
    QUANTILE_LEVELS,
    ProbabilisticContractError,
    QuantilePredictionBatch,
    VerifiedPredictionBatch,
    canonical_json_bytes,
    logical_frame_sha256,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .crossing import make_prediction_batch
from .custody import VerifiedTrainingLabelArtifact, require_canonical_authorized_fold
from .nested import FORMAL_SESSIONS_PER_SEED


REFERENCE_WINDOW = 252
REFERENCE_IDS = ("rolling_log_quantiles_252", "point_residual_quantiles_252")
POINT_HISTORY_SCHEMA = "expected_pe_model_zoo.probabilistic_point_history.v2"
_POINT_HISTORY_TOKEN = object()
_REFERENCE_FOLD_TOKEN = object()
_REFERENCE_ACCUMULATOR_TOKEN = object()
_REFERENCE_EXECUTION_RECEIPT_TOKEN = object()


class VerifiedPointPredictionHistory:
    __slots__ = ("_raw", "_raw_sha256", "_frame", "_frame_sha256")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _POINT_HISTORY_TOKEN:
            raise ProbabilisticContractError(
                "point prediction history must come from its content-addressed factory"
            )
        return super().__new__(cls)

    def __init__(self, token: object, *, raw: bytes, frame: pd.DataFrame) -> None:
        if token is not _POINT_HISTORY_TOKEN:
            raise ProbabilisticContractError("invalid point-history factory token")
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_frame_sha256", logical_frame_sha256(frame))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedPointPredictionHistory is immutable")

    def _group(self, *, seed: int, entity_id: str) -> pd.DataFrame:
        if sha256_bytes(self._raw) != self._raw_sha256:
            raise ProbabilisticContractError("point history raw bytes changed")
        if logical_frame_sha256(self._frame) != self._frame_sha256:
            raise ProbabilisticContractError("point history frame changed after loading")
        mask = (self._frame["seed"] == seed) & (self._frame["entity_id"] == entity_id)
        group = self._frame.loc[mask].reset_index(drop=True)
        if len(group) != FORMAL_SESSIONS_PER_SEED:
            raise ProbabilisticContractError("point-history seed/entity group is absent")
        return group.copy()


class VerifiedReferenceFoldBatch:
    """Opaque output emitted only by one of the two causal reference factories."""

    __slots__ = (
        "_authorization_sha256",
        "_batch",
        "_entity_id",
        "_fold_id",
        "_identity",
        "_reference_id",
        "_seed",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _REFERENCE_FOLD_TOKEN:
            raise ProbabilisticContractError(
                "reference fold batches must come from a causal reference factory"
            )
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        authorization: ExecutionAuthorization,
        batch: QuantilePredictionBatch,
        fold: PITFold,
        seed: int,
        entity_id: str,
    ) -> None:
        if token is not _REFERENCE_FOLD_TOKEN:
            raise ProbabilisticContractError("invalid reference-fold factory token")
        identity = authorization.identity_frame()
        identity = identity.loc[
            (identity["seed"] == seed)
            & (identity["entity_id"] == entity_id)
            & (identity["fold_id"] == fold.fold_id)
        ].reset_index(drop=True)
        if not np.array_equal(
            identity["ordered_position"].to_numpy(dtype=np.int64),
            np.asarray(fold.test_positions, dtype=np.int64),
        ):
            raise ProbabilisticContractError("reference fold identity differs from authorization")
        if batch.rows != len(identity):
            raise ProbabilisticContractError("reference fold prediction row count differs")
        object.__setattr__(self, "_authorization_sha256", authorization.raw_sha256)
        object.__setattr__(self, "_batch", batch)
        object.__setattr__(self, "_entity_id", entity_id)
        object.__setattr__(self, "_fold_id", fold.fold_id)
        object.__setattr__(self, "_identity", identity.copy(deep=True))
        object.__setattr__(self, "_reference_id", batch.model_id)
        object.__setattr__(self, "_seed", seed)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedReferenceFoldBatch is immutable")

    @property
    def pe_quantiles(self) -> np.ndarray:
        return self._batch.pe_quantiles.copy()

    @property
    def rows(self) -> int:
        return self._batch.rows


class VerifiedReferenceExecutionReceipt:
    """Full-mask receipt proving reference rows came from causal fold factories."""

    __slots__ = ("_payload", "_raw", "_raw_sha256")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _REFERENCE_EXECUTION_RECEIPT_TOKEN:
            raise ProbabilisticContractError(
                "reference execution receipts must come from the accumulator"
            )
        return super().__new__(cls)

    def __init__(self, token: object, *, payload: dict[str, object]) -> None:
        if token is not _REFERENCE_EXECUTION_RECEIPT_TOKEN:
            raise ProbabilisticContractError("invalid reference execution receipt token")
        sealed = seal_payload(payload)
        raw = canonical_json_bytes(sealed)
        object.__setattr__(self, "_payload", sealed)
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedReferenceExecutionReceipt is immutable")

    @property
    def authorization_sha256(self) -> str:
        return str(self._payload["authorization_raw_sha256"])

    @property
    def participant_id(self) -> str:
        return str(self._payload["reference_id"])

    @property
    def provenance(self) -> dict[str, object]:
        self.verify_integrity()
        return json.loads(json.dumps(self._payload["provenance"]))

    @property
    def raw_sha256(self) -> str:
        return self._raw_sha256

    @property
    def raw_bytes(self) -> bytes:
        self.verify_integrity()
        return bytes(self._raw)

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw) != self._raw_sha256:
            raise ProbabilisticContractError("reference execution receipt bytes changed")
        decoded = json.loads(self._raw)
        verify_payload_seal(decoded)
        if decoded != self._payload:
            raise ProbabilisticContractError("reference execution receipt payload changed")


class ReferenceExecutionAccumulator:
    """Factory-only accumulator binding every reference fold to the sealed full output."""

    __slots__ = (
        "_authorization",
        "_finalized",
        "_fold_keys",
        "_outputs",
        "_reference_id",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _REFERENCE_ACCUMULATOR_TOKEN:
            raise ProbabilisticContractError("reference accumulators must come from the factory")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        authorization: ExecutionAuthorization,
        reference_id: str,
    ) -> None:
        if token is not _REFERENCE_ACCUMULATOR_TOKEN:
            raise ProbabilisticContractError("invalid reference accumulator token")
        object.__setattr__(self, "_authorization", authorization)
        object.__setattr__(self, "_finalized", False)
        object.__setattr__(self, "_fold_keys", set())
        object.__setattr__(self, "_outputs", [])
        object.__setattr__(self, "_reference_id", reference_id)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("ReferenceExecutionAccumulator is immutable")

    def record(self, result: VerifiedReferenceFoldBatch) -> None:
        if self._finalized:
            raise ProbabilisticContractError("reference accumulator is finalized")
        if not isinstance(result, VerifiedReferenceFoldBatch):
            raise ProbabilisticContractError("reference accumulator rejects arbitrary batches")
        if (
            result._authorization_sha256 != self._authorization.raw_sha256
            or result._reference_id != self._reference_id
        ):
            raise ProbabilisticContractError("reference fold execution binding differs")
        key = (result._seed, result._entity_id, result._fold_id)
        if key in self._fold_keys:
            raise ProbabilisticContractError("reference fold retry is forbidden")
        self._fold_keys.add(key)
        output = pd.concat(
            [
                result._identity,
                pd.Series(
                    self._reference_id,
                    index=result._identity.index,
                    name="model_id",
                ),
                result._batch.output_frame(),
            ],
            axis=1,
        )
        self._outputs.append(output.copy(deep=True))

    def finalize(self, predictions: VerifiedPredictionBatch) -> VerifiedReferenceExecutionReceipt:
        if self._finalized:
            raise ProbabilisticContractError("reference accumulator is single-use")
        object.__setattr__(self, "_finalized", True)
        predictions.verify_integrity()
        if predictions.authorization_sha256 != self._authorization.raw_sha256:
            raise ProbabilisticContractError("reference prediction authorization differs")
        if tuple(pd.unique(predictions.frame["model_id"])) != (self._reference_id,):
            raise ProbabilisticContractError("reference sealed participant differs")
        identity = self._authorization.identity_frame()
        expected_fold_keys = set(
            identity.loc[:, ["seed", "entity_id", "fold_id"]].itertuples(index=False, name=None)
        )
        if self._fold_keys != expected_fold_keys:
            raise ProbabilisticContractError("reference accumulator did not observe every fold")
        observed = pd.concat(self._outputs, ignore_index=True)
        identity_columns = list(IDENTITY_COLUMNS)
        if observed.duplicated(identity_columns).any():
            raise ProbabilisticContractError("reference accumulator observed duplicate rows")
        observed_indexed = observed.set_index(identity_columns, drop=False)
        expected_index = pd.MultiIndex.from_frame(identity.loc[:, identity_columns])
        try:
            observed = observed_indexed.loc[expected_index].reset_index(drop=True)
        except KeyError as exc:
            raise ProbabilisticContractError(
                "reference accumulator common identity differs"
            ) from exc
        if (
            logical_frame_sha256(observed) != predictions.prediction_sha256
            or sha256_bytes(canonical_csv_bytes(observed)) != predictions.raw_sha256
        ):
            raise ProbabilisticContractError(
                "sealed reference differs from causal fold factory outputs"
            )
        provenance = {
            "prediction_raw_sha256": predictions.raw_sha256,
            "prediction_logical_sha256": predictions.prediction_sha256,
            "prediction_receipt_sha256": predictions.receipt_sha256,
            "common_identity_sha256": predictions.identity_sha256,
            "reference_binding_sha256": self._authorization.bindings[
                "probabilistic_reference_sha256_by_id"
            ][self._reference_id],
            "authorization_raw_sha256": self._authorization.raw_sha256,
            "source_closure_sha256": self._authorization.source_closure_sha256,
            "observed_fold_count": len(self._fold_keys),
        }
        return VerifiedReferenceExecutionReceipt(
            _REFERENCE_EXECUTION_RECEIPT_TOKEN,
            payload={
                "schema_version": (
                    "expected_pe_model_zoo.probabilistic_reference_execution_receipt.v2"
                ),
                "authorization_raw_sha256": self._authorization.raw_sha256,
                "reference_id": self._reference_id,
                "provenance": provenance,
            },
        )

    def seal(
        self, directory: Path
    ) -> tuple[VerifiedPredictionBatch, VerifiedReferenceExecutionReceipt]:
        """Seal only the accumulated causal rows, then verify them before evaluation."""

        if self._finalized:
            raise ProbabilisticContractError("reference accumulator is single-use")
        identity = self._authorization.identity_frame()
        expected_fold_keys = set(
            identity.loc[:, ["seed", "entity_id", "fold_id"]].itertuples(index=False, name=None)
        )
        if self._fold_keys != expected_fold_keys:
            raise ProbabilisticContractError("reference accumulator did not observe every fold")
        observed = pd.concat(self._outputs, ignore_index=True)
        identity_columns = list(IDENTITY_COLUMNS)
        observed_indexed = observed.set_index(identity_columns, drop=False)
        expected_index = pd.MultiIndex.from_frame(identity.loc[:, identity_columns])
        try:
            observed = observed_indexed.loc[expected_index].reset_index(drop=True)
        except KeyError as exc:
            raise ProbabilisticContractError(
                "reference accumulator common identity differs"
            ) from exc
        from .custody import (
            load_verified_reference_prediction_artifact,
            seal_reference_prediction_artifact,
        )

        prediction_path, receipt_path = seal_reference_prediction_artifact(
            observed, directory, authorization=self._authorization
        )
        predictions = load_verified_reference_prediction_artifact(
            prediction_path, receipt_path, authorization=self._authorization
        )
        return predictions, self.finalize(predictions)


def create_reference_execution_accumulator(
    *, authorization: ExecutionAuthorization, reference_id: str
) -> ReferenceExecutionAccumulator:
    authorization.verify_integrity()
    if reference_id not in REFERENCE_IDS:
        raise ProbabilisticContractError("reference accumulator ID is not locked")
    _verify_reference_binding(authorization=authorization, reference_id=reference_id)
    return ReferenceExecutionAccumulator(
        _REFERENCE_ACCUMULATOR_TOKEN,
        authorization=authorization,
        reference_id=reference_id,
    )


def write_reference_execution_receipt(
    receipt: VerifiedReferenceExecutionReceipt, directory: Path
) -> Path:
    if not isinstance(receipt, VerifiedReferenceExecutionReceipt):
        raise ProbabilisticContractError("reference receipt writer requires verified custody")
    raw = receipt.raw_bytes
    path = Path(directory) / (
        f"reference_execution_receipt_{receipt.participant_id}.{sha256_bytes(raw)}.json"
    )
    from .artifacts import immutable_write_bytes

    immutable_write_bytes(path, raw)
    return path


def load_verified_reference_execution_receipt(
    path: Path,
    *,
    authorization: ExecutionAuthorization,
    predictions: VerifiedPredictionBatch,
    reference_id: str,
) -> VerifiedReferenceExecutionReceipt:
    authorization.verify_integrity()
    predictions.verify_integrity()
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError(
            "reference execution receipt is unavailable or invalid"
        ) from exc
    if sha256_bytes(raw) not in Path(path).name.split("."):
        raise ProbabilisticContractError("reference execution receipt is not content-addressed")
    verify_payload_seal(payload)
    receipt = VerifiedReferenceExecutionReceipt(_REFERENCE_EXECUTION_RECEIPT_TOKEN, payload=payload)
    provenance = receipt.provenance
    frame_ids = tuple(pd.unique(predictions.frame["model_id"]))
    if (
        receipt.raw_bytes != raw
        or reference_id not in REFERENCE_IDS
        or frame_ids != (reference_id,)
        or receipt.participant_id != reference_id
        or receipt.authorization_sha256 != authorization.raw_sha256
        or provenance.get("prediction_raw_sha256") != predictions.raw_sha256
        or provenance.get("prediction_logical_sha256") != predictions.prediction_sha256
        or provenance.get("prediction_receipt_sha256") != predictions.receipt_sha256
        or provenance.get("common_identity_sha256") != predictions.identity_sha256
        or provenance.get("source_closure_sha256") != authorization.source_closure_sha256
    ):
        raise ProbabilisticContractError(
            "reference execution receipt differs from prediction/source custody"
        )
    return receipt


def reference_binding_sha256_by_id() -> dict[str, str]:
    source_hash = sha256_bytes(Path(__file__).read_bytes())
    return {
        model_id: sha256_bytes(
            canonical_json_bytes(
                seal_payload(
                    {
                        "schema_version": "expected_pe_model_zoo.probabilistic_reference_binding.v2",
                        "design_sha256": PROBABILISTIC_DESIGN_SHA256,
                        "model_id": model_id,
                        "source_sha256": source_hash,
                        "window": REFERENCE_WINDOW,
                        "quantiles": list(QUANTILE_LEVELS),
                        "quantile_method": "inverted_cdf",
                        "group_keys": ["seed", "entity_id"],
                        "shift": "exact_fold_train_positions_tail_252",
                    }
                )
            )
        )
        for model_id in ("rolling_log_quantiles_252", "point_residual_quantiles_252")
    }


def _verify_reference_binding(*, authorization: ExecutionAuthorization, reference_id: str) -> None:
    expected = authorization.bindings["probabilistic_reference_sha256_by_id"].get(reference_id)
    if expected != reference_binding_sha256_by_id()[reference_id]:
        raise ProbabilisticContractError("reference source/config binding differs")


def load_verified_point_prediction_history(
    path: Path,
    *,
    authorization: ExecutionAuthorization,
    comparator_id: str,
) -> VerifiedPointPredictionHistory:
    """Load one precommitted point history; no caller assertion flags are accepted."""

    authorization.verify_integrity()
    expected_hash = authorization.bindings["comparator_prediction_sha256_by_id"].get(comparator_id)
    if expected_hash is None:
        raise ProbabilisticContractError("point comparator id is not precommitted")
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError("point prediction history is unavailable") from exc
    digest = sha256_bytes(raw)
    if digest != expected_hash or digest not in Path(path).name.split("."):
        raise ProbabilisticContractError("point prediction history differs from its address")
    try:
        frame = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ProbabilisticContractError("point prediction history cannot be decoded") from exc
    expected_columns = (
        "seed",
        "entity_id",
        "date",
        "ordered_position",
        "point_expected_pe",
    )
    if tuple(frame.columns) != expected_columns:
        raise ProbabilisticContractError("point prediction history schema/order differs")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
    positions = pd.to_numeric(frame["ordered_position"], errors="coerce")
    point = pd.to_numeric(frame["point_expected_pe"], errors="coerce").to_numpy(dtype=np.float64)
    if (
        frame["date"].isna().any()
        or positions.isna().any()
        or not np.equal(positions, np.floor(positions)).all()
        or np.isinf(point).any()
    ):
        raise ProbabilisticContractError("point prediction history values are invalid")
    frame["ordered_position"] = positions.astype(np.int64)
    frame["point_expected_pe"] = point
    for _, group in frame.groupby(["seed", "entity_id"], sort=False, dropna=False):
        if len(group) != FORMAL_SESSIONS_PER_SEED or not np.array_equal(
            group["ordered_position"].to_numpy(), np.arange(FORMAL_SESSIONS_PER_SEED)
        ):
            raise ProbabilisticContractError(
                "point history must contain exact positions 0..1799 per seed/entity"
            )
        if not group["date"].is_monotonic_increasing or group["date"].duplicated().any():
            raise ProbabilisticContractError("point history dates are not chronological")
    return VerifiedPointPredictionHistory(_POINT_HISTORY_TOKEN, raw=raw, frame=frame)


def rolling_log_quantiles_252(
    *,
    labels: VerifiedTrainingLabelArtifact,
    authorization: ExecutionAuthorization,
    fold: PITFold,
    seed: int,
    entity_id: str,
):
    """Use exactly the final 252 authorized, available labels in one group."""

    _verify_reference_binding(authorization=authorization, reference_id="rolling_log_quantiles_252")
    if not isinstance(labels, VerifiedTrainingLabelArtifact):
        raise ProbabilisticContractError("rolling reference requires verified labels")
    fold = require_canonical_authorized_fold(
        labels=labels,
        authorization=authorization,
        fold=fold,
        seed=seed,
        entity_id=entity_id,
    )
    group = labels._group(seed=seed, entity_id=entity_id)
    positions = np.asarray(fold.train_positions, dtype=np.int64)
    if len(positions) < REFERENCE_WINDOW:
        raise ProbabilisticContractError("rolling reference needs 252 causal labels")
    selected_positions = positions[-REFERENCE_WINDOW:]
    history = group.iloc[selected_positions]
    if not np.array_equal(history["ordered_position"].to_numpy(), selected_positions):
        raise ProbabilisticContractError("rolling reference shift differs from fold history")
    if (history["label_available_at"] > pd.Timestamp(fold.label_information_cutoff)).any():
        raise ProbabilisticContractError("rolling reference includes unavailable labels")
    values = history["observed_pe"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ProbabilisticContractError(
            "rolling reference latest 252 labels must be positive and finite"
        )
    quantiles = np.quantile(np.log(values), QUANTILE_LEVELS, method="inverted_cdf")
    raw = np.repeat(quantiles[None, :], len(fold.test_positions), axis=0)
    batch = make_prediction_batch(model_id="rolling_log_quantiles_252", raw_log_quantiles=raw)
    return VerifiedReferenceFoldBatch(
        _REFERENCE_FOLD_TOKEN,
        authorization=authorization,
        batch=batch,
        fold=fold,
        seed=seed,
        entity_id=entity_id,
    )


def point_residual_quantiles_252(
    *,
    point_history: VerifiedPointPredictionHistory,
    labels: VerifiedTrainingLabelArtifact,
    authorization: ExecutionAuthorization,
    fold: PITFold,
    seed: int,
    entity_id: str,
):
    """Apply exact prior grouped residuals; current rows can never enter history."""

    _verify_reference_binding(
        authorization=authorization, reference_id="point_residual_quantiles_252"
    )
    if not isinstance(point_history, VerifiedPointPredictionHistory) or not isinstance(
        labels, VerifiedTrainingLabelArtifact
    ):
        raise ProbabilisticContractError("residual reference requires verified artifacts")
    fold = require_canonical_authorized_fold(
        labels=labels,
        authorization=authorization,
        fold=fold,
        seed=seed,
        entity_id=entity_id,
    )
    points = point_history._group(seed=seed, entity_id=entity_id)
    label_group = labels._group(seed=seed, entity_id=entity_id)
    identity_columns = ["seed", "entity_id", "date", "ordered_position"]
    if not points[identity_columns].equals(label_group[identity_columns]):
        raise ProbabilisticContractError("point and label history identities differ")
    train_positions = np.asarray(fold.train_positions, dtype=np.int64)
    test_positions = np.asarray(fold.test_positions, dtype=np.int64)
    if len(train_positions) < REFERENCE_WINDOW:
        raise ProbabilisticContractError("residual reference needs 252 causal residuals")
    prior_positions = train_positions[-REFERENCE_WINDOW:]
    if np.max(prior_positions) >= np.min(test_positions):
        raise ProbabilisticContractError("residual history is not strictly prior")
    prior_points = points.iloc[prior_positions]
    prior_labels = label_group.iloc[prior_positions]
    if not np.array_equal(prior_points["ordered_position"], prior_positions) or not np.array_equal(
        prior_labels["ordered_position"], prior_positions
    ):
        raise ProbabilisticContractError("residual reference does not use exact shifted positions")
    if (prior_labels["label_available_at"] > pd.Timestamp(fold.label_information_cutoff)).any():
        raise ProbabilisticContractError("residual reference includes unavailable labels")
    current = points.iloc[test_positions]
    if not np.array_equal(current["ordered_position"], test_positions):
        raise ProbabilisticContractError("current point predictions differ from fold positions")
    prior_label_values = prior_labels["observed_pe"].to_numpy(dtype=np.float64)
    prior_point_values = prior_points["point_expected_pe"].to_numpy(dtype=np.float64)
    current_point_values = current["point_expected_pe"].to_numpy(dtype=np.float64)
    if any(
        not np.isfinite(values).all() or (values <= 0.0).any()
        for values in (prior_label_values, prior_point_values, current_point_values)
    ):
        raise ProbabilisticContractError(
            "residual reference selected label/point window must be positive and finite"
        )
    residual = np.log(prior_label_values) - np.log(prior_point_values)
    residual_quantiles = np.quantile(residual, QUANTILE_LEVELS, method="inverted_cdf")
    raw = np.log(current_point_values)[:, None] + (residual_quantiles[None, :])
    batch = make_prediction_batch(model_id="point_residual_quantiles_252", raw_log_quantiles=raw)
    return VerifiedReferenceFoldBatch(
        _REFERENCE_FOLD_TOKEN,
        authorization=authorization,
        batch=batch,
        fold=fold,
        seed=seed,
        entity_id=entity_id,
    )
