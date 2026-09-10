"""Detached evaluator: the only probabilistic module allowed to compute scores."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

from .authorization import ExecutionAuthorization, FORMAL_SCOPE, SCORE_FREE_SCOPE
from .contracts import (
    EVALUATION_ONLY_COLUMN,
    IDENTITY_COLUMNS,
    PE_QUANTILE_COLUMNS,
    NORMAL_QUANTILE_Z,
    QUANTILE_LEVELS,
    ProbabilisticContractError,
    VerifiedPredictionBatch,
    canonical_json_bytes,
    identity_sha256,
    logical_frame_sha256,
    normalize_identity_frame,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .custody import REFERENCE_IDS, VerifiedPointComparatorArtifact
from .references import VerifiedReferenceExecutionReceipt
from .spec import CANDIDATE_IDS


_DETACHED_TRUTH_TOKEN = object()
_EVALUATION_RECEIPT_TOKEN = object()


def create_evaluation_session(*_: object, **__: object) -> None:
    """Tombstoned V3 API; evaluation is available only through the physical CLI."""

    raise ProbabilisticContractError(
        "in-memory evaluation sessions are tombstoned; use the physical evaluator process"
    )


class DetachedTruth:
    """Evaluator-private truth bytes; callers receive no mutable truth frame."""

    __slots__ = (
        "_raw",
        "_frame",
        "_file_sha256",
        "_identity_sha256",
        "_logical_sha256",
        "_authorization_sha256",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _DETACHED_TRUTH_TOKEN:
            raise ProbabilisticContractError("DetachedTruth must be loaded by the evaluator")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        raw: bytes,
        frame: pd.DataFrame,
        file_sha256: str,
        identity_sha256_value: str,
        logical_sha256: str,
        authorization_sha256: str,
    ) -> None:
        if token is not _DETACHED_TRUTH_TOKEN:
            raise ProbabilisticContractError("invalid detached-truth factory token")
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_file_sha256", file_sha256)
        object.__setattr__(self, "_identity_sha256", identity_sha256_value)
        object.__setattr__(self, "_logical_sha256", logical_sha256)
        object.__setattr__(self, "_authorization_sha256", authorization_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("DetachedTruth is immutable")

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw) != self._file_sha256:
            raise ProbabilisticContractError("detached truth raw bytes changed")
        if logical_frame_sha256(self._frame) != self._logical_sha256:
            raise ProbabilisticContractError("detached truth frame changed")


class VerifiedEvaluationReceipt(Mapping[str, Any]):
    """Opaque evaluator-owned metric and provenance receipt."""

    __slots__ = ("_payload", "_raw", "_raw_sha256")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _EVALUATION_RECEIPT_TOKEN:
            raise ProbabilisticContractError(
                "evaluation receipts must come from verified evaluator custody"
            )
        return super().__new__(cls)

    def __init__(self, token: object, *, payload: Mapping[str, Any]) -> None:
        if token is not _EVALUATION_RECEIPT_TOKEN:
            raise ProbabilisticContractError("invalid evaluator receipt factory token")
        sealed = seal_payload(payload)
        raw = canonical_json_bytes(sealed)
        object.__setattr__(self, "_payload", sealed)
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedEvaluationReceipt is immutable")

    def __getitem__(self, key: str) -> Any:
        self.verify_integrity()
        if key not in self._payload["metrics"]:
            raise KeyError(key)
        return json.loads(json.dumps(self._payload["metrics"][key]))

    def __iter__(self) -> Iterator[str]:
        self.verify_integrity()
        return iter(tuple(self._payload["metrics"]))

    def __len__(self) -> int:
        return len(self._payload["metrics"])

    @property
    def authorization_sha256(self) -> str:
        return str(self._payload["authorization_raw_sha256"])

    @property
    def participant_id(self) -> str:
        return str(self._payload["participant_id"])

    @property
    def receipt_kind(self) -> str:
        return str(self._payload["receipt_kind"])

    @property
    def raw_sha256(self) -> str:
        return self._raw_sha256

    @property
    def raw_bytes(self) -> bytes:
        self.verify_integrity()
        return bytes(self._raw)

    @property
    def provenance(self) -> Mapping[str, Any]:
        self.verify_integrity()
        return json.loads(json.dumps(self._payload["provenance"]))

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw) != self._raw_sha256:
            raise ProbabilisticContractError("evaluation receipt raw bytes changed")
        decoded = json.loads(self._raw)
        verify_payload_seal(decoded)
        if decoded != self._payload:
            raise ProbabilisticContractError("evaluation receipt payload changed")

    def _metrics_copy(self) -> dict[str, Any]:
        self.verify_integrity()
        return json.loads(json.dumps(self._payload["metrics"]))


PREDECLARED_SLICE_COLUMNS = ("dgp_regime", "earnings_staleness_slice")


@dataclass(frozen=True)
class EvaluationSlices:
    frame: pd.DataFrame
    identity_sha256: str


def verify_evaluation_slices(frame: pd.DataFrame) -> EvaluationSlices:
    expected = (*IDENTITY_COLUMNS, *PREDECLARED_SLICE_COLUMNS)
    if tuple(frame.columns) != expected:
        raise ProbabilisticContractError("evaluation slice schema differs from predeclaration")
    identity = normalize_identity_frame(frame, context="evaluation slices", sort=False)
    if frame.loc[:, list(PREDECLARED_SLICE_COLUMNS)].isna().any().any():
        raise ProbabilisticContractError("evaluation slice labels must be complete")
    return EvaluationSlices(
        frame=frame.copy(deep=True), identity_sha256=identity_sha256(identity, sort=False)
    )


def load_detached_truth(*_: object, **__: object) -> None:
    """Tombstoned: truth cannot be opened through an in-process library call."""

    raise ProbabilisticContractError(
        "in-process truth loading is tombstoned; use the physical evaluator process"
    )


def _load_detached_truth_after_physical_preflight(
    path: Path,
    *,
    authorization: ExecutionAuthorization,
    preflight: object,
) -> DetachedTruth:
    """Child-only truth loader reached after complete disk-receipt verification."""

    from .physical_evaluator import VerifiedPhysicalPreflight

    if not isinstance(preflight, VerifiedPhysicalPreflight):
        raise ProbabilisticContractError("truth loader requires physical preflight custody")
    preflight.verify_for_truth_open(authorization=authorization)
    expected_sha256 = authorization.bindings["detached_truth_manifest_sha256"]
    if authorization.scope == FORMAL_SCOPE:
        from .spent import UPSTREAM_INPUTS
        from .spent_inputs import assemble_detached_truth

        repo_root = Path(__file__).resolve().parents[4]
        expected_path = (repo_root / UPSTREAM_INPUTS["evaluate"]["path"]).resolve()
        if Path(path).resolve() != expected_path:
            raise ProbabilisticContractError("formal detached-truth manifest path differs")
        if expected_sha256 != UPSTREAM_INPUTS["evaluate"]["raw_sha256"]:
            raise ProbabilisticContractError("formal detached-truth manifest pin differs")
        frame, _, manifest_raw = assemble_detached_truth(repo_root, authorization.identity_frame())
        identity = normalize_identity_frame(frame, context="detached truth", sort=False)
        return DetachedTruth(
            _DETACHED_TRUTH_TOKEN,
            raw=manifest_raw,
            frame=frame,
            file_sha256=expected_sha256,
            identity_sha256_value=identity_sha256(identity, sort=False),
            logical_sha256=logical_frame_sha256(frame),
            authorization_sha256=authorization.raw_sha256,
        )
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError("detached truth bytes are unavailable") from exc
    if sha256_bytes(raw) != expected_sha256:
        raise ProbabilisticContractError("detached truth bytes differ from authorization")
    if expected_sha256 not in Path(path).name.split("."):
        raise ProbabilisticContractError("detached truth filename is not content-addressed")
    if Path(path).suffix.casefold() != ".csv":
        raise ProbabilisticContractError("detached truth must be canonical CSV")
    try:
        frame = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ProbabilisticContractError("detached truth cannot be decoded") from exc
    expected = (*IDENTITY_COLUMNS, EVALUATION_ONLY_COLUMN)
    if tuple(frame.columns) != expected:
        raise ProbabilisticContractError("detached truth schema must be identity plus true_fair_pe")
    identity = normalize_identity_frame(frame, context="detached truth", sort=False)
    values = pd.to_numeric(frame[EVALUATION_ONLY_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ProbabilisticContractError("detached fair truth must be positive and finite")
    authorized_identity = normalize_identity_frame(
        authorization.identity_frame(), context="authorized truth identity", sort=False
    )
    if not identity.equals(authorized_identity):
        raise ProbabilisticContractError("detached truth identity/order differs from authorization")
    return DetachedTruth(
        _DETACHED_TRUTH_TOKEN,
        raw=raw,
        frame=frame.copy(),
        file_sha256=expected_sha256,
        identity_sha256_value=identity_sha256(identity, sort=False),
        logical_sha256=logical_frame_sha256(frame),
        authorization_sha256=authorization.raw_sha256,
    )


def evaluate_distribution(
    predictions: VerifiedPredictionBatch,
    truth: DetachedTruth,
    *,
    slices: EvaluationSlices | None = None,
) -> dict[str, Any]:
    """Evaluate ordered log-space quantiles on the exact verified common mask."""

    if not isinstance(predictions, VerifiedPredictionBatch):
        raise ProbabilisticContractError("evaluator requires an opaque verified prediction batch")
    predictions.verify_integrity()
    if not isinstance(truth, DetachedTruth):
        raise ProbabilisticContractError("evaluator requires an opaque detached truth batch")
    truth.verify_integrity()
    if predictions.authorization_sha256 != truth._authorization_sha256:
        raise ProbabilisticContractError("truth and prediction authorizations differ")
    if predictions.identity_sha256 != truth._identity_sha256:
        raise ProbabilisticContractError(
            "truth/prediction identities differ; mask shrink forbidden"
        )
    predicted_identity = normalize_identity_frame(
        predictions.frame, context="verified predictions", sort=False
    )
    truth_identity = normalize_identity_frame(truth._frame, context="detached truth", sort=False)
    if identity_sha256(predicted_identity, sort=False) != identity_sha256(
        truth_identity, sort=False
    ):
        raise ProbabilisticContractError("truth order differs from sealed prediction order")
    pe_quantiles = predictions.frame.loc[:, list(PE_QUANTILE_COLUMNS)].to_numpy(dtype=np.float64)
    if (pe_quantiles <= 0.0).any() or not np.isfinite(pe_quantiles).all():
        raise ProbabilisticContractError("verified quantiles became invalid")
    q = np.log(pe_quantiles)
    z = np.log(truth._frame[EVALUATION_ONLY_COLUMN].to_numpy(dtype=np.float64))
    pinball_by_level = {
        f"p{int(tau * 100):02d}": float(np.mean(_pinball(z, q[:, index], tau)))
        for index, tau in enumerate(QUANTILE_LEVELS)
    }
    calibration_by_level = {
        f"p{int(tau * 100):02d}": float(np.mean(z <= q[:, index]) - tau)
        for index, tau in enumerate(QUANTILE_LEVELS)
    }
    is80, d80, under80, over80 = _interval_score(z, q[:, 0], q[:, 4], alpha=0.20)
    is50, d50, under50, over50 = _interval_score(z, q[:, 1], q[:, 3], alpha=0.50)
    absolute = np.abs(z - q[:, 2])
    wis = (0.5 * absolute + 0.10 * is80 + 0.25 * is50) / 2.5
    result: dict[str, Any] = {
        "row_count": int(len(z)),
        "identity_sha256": predictions.identity_sha256,
        "prediction_sha256": predictions.prediction_sha256,
        "truth_manifest_sha256": truth._file_sha256,
        "mean_pinball_loss": float(np.mean(list(pinball_by_level.values()))),
        "pinball_by_level": pinball_by_level,
        "wis": float(np.mean(wis)),
        "wis_components": {
            "dispersion": float(np.mean((0.10 * d80 + 0.25 * d50) / 2.5)),
            "underprediction_penalty": float(np.mean((0.10 * under80 + 0.25 * under50) / 2.5)),
            "overprediction_penalty": float(np.mean((0.10 * over80 + 0.25 * over50) / 2.5)),
            "median_absolute_error_component": float(np.mean(0.5 * absolute / 2.5)),
        },
        "calibration_error_by_level": calibration_by_level,
        "mean_absolute_quantile_calibration_error": float(
            np.mean(np.abs(list(calibration_by_level.values())))
        ),
        "maximum_absolute_quantile_calibration_error": float(
            np.max(np.abs(list(calibration_by_level.values())))
        ),
        "p10_p90_coverage": float(np.mean((z >= q[:, 0]) & (z <= q[:, 4]))),
        "p25_p75_coverage": float(np.mean((z >= q[:, 1]) & (z <= q[:, 3]))),
        "sharpness": _sharpness(q),
        "fair_median": _fair_median(z, q[:, 2]),
        "per_seed": _per_seed(predictions.frame["seed"], z, q, wis),
    }
    result["p10_p90_coverage_error"] = result["p10_p90_coverage"] - 0.80
    result["p25_p75_coverage_error"] = result["p25_p75_coverage"] - 0.50
    density_columns = {"density_loc", "density_scale"}
    present = density_columns.intersection(predictions.frame.columns)
    if present and present != density_columns:
        raise ProbabilisticContractError("NGBoost density parameters are incomplete")
    if present:
        result["ngboost_density"] = _normal_density_metrics(predictions.frame, z)
    else:
        result["quantile_pit_status"] = "NOT_COMPUTED_NO_TAIL_COMPLETE_CDF"
    if slices is not None:
        if slices.identity_sha256 != predictions.identity_sha256:
            raise ProbabilisticContractError("evaluation slice identity differs from common mask")
        result["predeclared_slices"] = _slice_metrics(slices.frame, z, q, wis)
    return result


def _verified_distribution_provenance(
    *,
    predictions: VerifiedPredictionBatch,
    truth: DetachedTruth,
    authorization: ExecutionAuthorization,
    participant_id: str,
) -> dict[str, Any]:
    if not isinstance(predictions, VerifiedPredictionBatch):
        raise ProbabilisticContractError("distribution receipt requires verified predictions")
    if not isinstance(truth, DetachedTruth):
        raise ProbabilisticContractError("distribution receipt requires detached truth custody")
    if not isinstance(authorization, ExecutionAuthorization):
        raise ProbabilisticContractError("distribution receipt requires factory authorization")
    authorization.verify_integrity()
    predictions.verify_integrity()
    truth.verify_integrity()
    frame = predictions.frame
    model_ids = tuple(pd.unique(frame["model_id"]))
    if model_ids != (participant_id,):
        raise ProbabilisticContractError("distribution participant ID differs from prediction")
    if (
        predictions.authorization_sha256 != authorization.raw_sha256
        or truth._authorization_sha256 != authorization.raw_sha256
    ):
        raise ProbabilisticContractError("distribution receipt authorization differs")
    if predictions.identity_sha256 != truth._identity_sha256:
        raise ProbabilisticContractError("distribution receipt common identity differs")
    return {
        "evidence_mode": "VERIFIED_CONTENT_ADDRESSED_EVALUATION",
        "prediction_raw_sha256": predictions.raw_sha256,
        "prediction_logical_sha256": predictions.prediction_sha256,
        "prediction_receipt_sha256": predictions.receipt_sha256,
        "common_identity_sha256": predictions.identity_sha256,
        "detached_truth_raw_sha256": truth._file_sha256,
        "detached_truth_logical_sha256": truth._logical_sha256,
        "authorization_raw_sha256": authorization.raw_sha256,
    }


def evaluate_candidate_receipt(
    predictions: VerifiedPredictionBatch,
    truth: DetachedTruth,
    *,
    authorization: ExecutionAuthorization,
    candidate_id: str,
    slices: EvaluationSlices | None = None,
) -> VerifiedEvaluationReceipt:
    """Score one locked candidate only from verified prediction and truth custody."""

    if candidate_id not in CANDIDATE_IDS:
        raise ProbabilisticContractError("candidate evaluation ID is not locked")
    provenance = _verified_distribution_provenance(
        predictions=predictions,
        truth=truth,
        authorization=authorization,
        participant_id=candidate_id,
    )
    metrics = evaluate_distribution(predictions, truth, slices=slices)
    return VerifiedEvaluationReceipt(
        _EVALUATION_RECEIPT_TOKEN,
        payload={
            "schema_version": "expected_pe_model_zoo.probabilistic_evaluation_receipt.v2",
            "authorization_raw_sha256": authorization.raw_sha256,
            "participant_id": candidate_id,
            "receipt_kind": "candidate_distribution",
            "provenance": provenance,
            "metrics": metrics,
        },
    )


def evaluate_reference_receipt(
    predictions: VerifiedPredictionBatch,
    truth: DetachedTruth,
    *,
    execution_receipt: VerifiedReferenceExecutionReceipt,
    authorization: ExecutionAuthorization,
    reference_id: str,
    slices: EvaluationSlices | None = None,
) -> VerifiedEvaluationReceipt:
    """Score one sealed full-mask causal reference distribution."""

    if reference_id not in REFERENCE_IDS:
        raise ProbabilisticContractError("reference evaluation ID is not locked")
    if not isinstance(execution_receipt, VerifiedReferenceExecutionReceipt):
        raise ProbabilisticContractError("reference evaluation requires a causal execution receipt")
    execution_receipt.verify_integrity()
    if (
        execution_receipt.participant_id != reference_id
        or execution_receipt.authorization_sha256 != authorization.raw_sha256
    ):
        raise ProbabilisticContractError("reference execution receipt binding differs")
    provenance = _verified_distribution_provenance(
        predictions=predictions,
        truth=truth,
        authorization=authorization,
        participant_id=reference_id,
    )
    execution_provenance = execution_receipt.provenance
    for field in (
        "prediction_raw_sha256",
        "prediction_logical_sha256",
        "prediction_receipt_sha256",
        "common_identity_sha256",
        "authorization_raw_sha256",
    ):
        if provenance[field] != execution_provenance[field]:
            raise ProbabilisticContractError(
                f"reference execution/evaluator provenance differs: {field}"
            )
    if (
        execution_provenance["reference_binding_sha256"]
        != authorization.bindings["probabilistic_reference_sha256_by_id"][reference_id]
    ):
        raise ProbabilisticContractError("reference source/config receipt differs")
    provenance["reference_execution_receipt_sha256"] = execution_receipt.raw_sha256
    provenance["reference_binding_sha256"] = execution_provenance["reference_binding_sha256"]
    metrics = evaluate_distribution(predictions, truth, slices=slices)
    return VerifiedEvaluationReceipt(
        _EVALUATION_RECEIPT_TOKEN,
        payload={
            "schema_version": "expected_pe_model_zoo.probabilistic_evaluation_receipt.v2",
            "authorization_raw_sha256": authorization.raw_sha256,
            "participant_id": reference_id,
            "receipt_kind": "reference_distribution",
            "provenance": provenance,
            "metrics": metrics,
        },
    )


def evaluate_point_comparator_receipt(
    comparator: VerifiedPointComparatorArtifact,
    truth: DetachedTruth,
    *,
    authorization: ExecutionAuthorization,
    comparator_id: str,
) -> VerifiedEvaluationReceipt:
    """Score one exact precommitted point artifact; relabeling is rejected."""

    if comparator_id not in {"v04_expected_pe", "ml_expected_pe"}:
        raise ProbabilisticContractError("point comparator evaluation ID is not locked")
    if not isinstance(comparator, VerifiedPointComparatorArtifact):
        raise ProbabilisticContractError("point evaluation requires comparator custody")
    if not isinstance(truth, DetachedTruth):
        raise ProbabilisticContractError("point evaluation requires detached truth custody")
    authorization.verify_integrity()
    comparator.verify_integrity()
    truth.verify_integrity()
    if comparator.comparator_id != comparator_id:
        raise ProbabilisticContractError("point comparator cannot be relabeled")
    if (
        comparator.authorization_sha256 != authorization.raw_sha256
        or truth._authorization_sha256 != authorization.raw_sha256
    ):
        raise ProbabilisticContractError("point comparator authorization differs")
    if comparator.identity_sha256 != truth._identity_sha256:
        raise ProbabilisticContractError("point comparator common identity differs")
    frame = comparator.frame
    z = np.log(truth._frame[EVALUATION_ONLY_COLUMN].to_numpy(dtype=np.float64))
    point = np.log(frame["expected_pe"].to_numpy(dtype=np.float64))
    error = point - z
    seeds = frame["seed"].to_numpy()
    per_seed = {
        str(seed): {
            "fair_log_mae": float(np.mean(np.abs(error[seeds == seed]))),
        }
        for seed in sorted(pd.unique(seeds), key=str)
    }
    metrics = {
        "fair_log_mae": float(np.mean(np.abs(error))),
        "fair_log_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "per_seed": per_seed,
    }
    return VerifiedEvaluationReceipt(
        _EVALUATION_RECEIPT_TOKEN,
        payload={
            "schema_version": "expected_pe_model_zoo.probabilistic_evaluation_receipt.v2",
            "authorization_raw_sha256": authorization.raw_sha256,
            "participant_id": comparator_id,
            "receipt_kind": "point_comparator",
            "provenance": {
                "evidence_mode": "VERIFIED_CONTENT_ADDRESSED_EVALUATION",
                "comparator_raw_sha256": comparator.raw_sha256,
                "comparator_logical_sha256": comparator.frame_sha256,
                "common_identity_sha256": comparator.identity_sha256,
                "detached_truth_raw_sha256": truth._file_sha256,
                "detached_truth_logical_sha256": truth._logical_sha256,
                "authorization_raw_sha256": authorization.raw_sha256,
            },
            "metrics": metrics,
        },
    )


def _score_free_candidate_metrics(wis: float, mae: float) -> dict[str, Any]:
    return {
        "wis": wis,
        "mean_pinball_loss": wis / 2,
        "fair_median": {"fair_log_mae": mae, "fair_log_rmse": mae * 1.1},
        "calibration_error_by_level": {f"p{x}": 0.0 for x in (10, 25, 50, 75, 90)},
        "mean_absolute_quantile_calibration_error": 0.0,
        "maximum_absolute_quantile_calibration_error": 0.0,
        "p10_p90_coverage_error": 0.0,
        "p25_p75_coverage_error": 0.0,
        "per_seed": {
            "SYNTHETIC": {
                "wis": wis,
                "fair_log_mae": mae,
                "p10_p90_coverage": 0.8,
                "p25_p75_coverage": 0.5,
            }
        },
    }


def score_free_evaluation_receipts(
    authorization: ExecutionAuthorization,
) -> tuple[
    dict[str, VerifiedEvaluationReceipt],
    dict[str, VerifiedEvaluationReceipt],
    dict[str, VerifiedEvaluationReceipt],
]:
    """Build a fixed, zero-input synthetic receipt set for governance wiring tests."""

    authorization.verify_integrity()
    if authorization.scope != SCORE_FREE_SCOPE:
        raise ProbabilisticContractError("fixed synthetic receipts require score-free scope")
    candidate_metrics = {
        "qlinear_l1_with_regime_v1": _score_free_candidate_metrics(0.80, 0.80),
        "qhistgb_with_regime_v1": _score_free_candidate_metrics(0.75, 0.79),
        "ngboost_normal_crps_with_regime_v1": _score_free_candidate_metrics(0.78, 0.795),
    }
    reference_metrics = {
        "rolling_log_quantiles_252": _score_free_candidate_metrics(1.0, 1.0),
        "point_residual_quantiles_252": _score_free_candidate_metrics(0.9, 0.95),
    }
    point_metrics = {
        "v04_expected_pe": {
            "fair_log_mae": 1.0,
            "fair_log_rmse": 1.1,
            "per_seed": {"SYNTHETIC": {"fair_log_mae": 1.0}},
        },
        "ml_expected_pe": {
            "fair_log_mae": 0.9,
            "fair_log_rmse": 1.0,
            "per_seed": {"SYNTHETIC": {"fair_log_mae": 0.9}},
        },
    }
    common_identity = identity_sha256(authorization.identity_frame(), sort=False)
    truth_raw = sha256_bytes(b"FIXED_SCORE_FREE_EVALUATOR_TRUTH_V2")

    def fixed_receipt(
        participant_id: str, receipt_kind: str, metrics: Mapping[str, Any]
    ) -> VerifiedEvaluationReceipt:
        if receipt_kind == "candidate_distribution":
            fixture_raw = sha256_bytes(f"SCORE_FREE_FIXED_PREDICTION:{participant_id}".encode())
        else:
            fixture_raw = sha256_bytes(
                canonical_json_bytes(
                    {
                        "participant_id": participant_id,
                        "receipt_kind": receipt_kind,
                        "metrics": metrics,
                    }
                )
            )
        if receipt_kind in {"candidate_distribution", "reference_distribution"}:
            artifact_provenance = {
                "prediction_raw_sha256": fixture_raw,
                "prediction_logical_sha256": fixture_raw,
                "prediction_receipt_sha256": fixture_raw,
            }
        else:
            artifact_provenance = {
                "comparator_raw_sha256": fixture_raw,
                "comparator_logical_sha256": fixture_raw,
            }
        return VerifiedEvaluationReceipt(
            _EVALUATION_RECEIPT_TOKEN,
            payload={
                "schema_version": "expected_pe_model_zoo.probabilistic_evaluation_receipt.v2",
                "authorization_raw_sha256": authorization.raw_sha256,
                "participant_id": participant_id,
                "receipt_kind": receipt_kind,
                "provenance": {
                    "evidence_mode": "SCORE_FREE_FIXED_INTERNAL_FIXTURE",
                    **artifact_provenance,
                    "common_identity_sha256": common_identity,
                    "detached_truth_raw_sha256": truth_raw,
                    "detached_truth_logical_sha256": truth_raw,
                    "authorization_raw_sha256": authorization.raw_sha256,
                },
                "metrics": metrics,
            },
        )

    return (
        {
            participant_id: fixed_receipt(participant_id, "candidate_distribution", metrics)
            for participant_id, metrics in candidate_metrics.items()
        },
        {
            participant_id: fixed_receipt(participant_id, "reference_distribution", metrics)
            for participant_id, metrics in reference_metrics.items()
        },
        {
            participant_id: fixed_receipt(participant_id, "point_comparator", metrics)
            for participant_id, metrics in point_metrics.items()
        },
    )


def _pinball(z: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    return tau * np.maximum(z - q, 0.0) + (1.0 - tau) * np.maximum(q - z, 0.0)


def _interval_score(
    z: np.ndarray, lower: np.ndarray, upper: np.ndarray, *, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dispersion = upper - lower
    under = (2.0 / alpha) * (lower - z) * (z < lower)
    over = (2.0 / alpha) * (z - upper) * (z > upper)
    return dispersion + under + over, dispersion, under, over


def _sharpness(q: np.ndarray) -> dict[str, float]:
    width80 = q[:, 4] - q[:, 0]
    width50 = q[:, 3] - q[:, 1]
    return {
        "p10_p90_mean": float(np.mean(width80)),
        "p10_p90_median": float(np.median(width80)),
        "p10_p90_p90": float(np.quantile(width80, 0.90)),
        "p25_p75_mean": float(np.mean(width50)),
        "p25_p75_median": float(np.median(width50)),
        "p25_p75_p90": float(np.quantile(width50, 0.90)),
        "interval_collapse_frequency": float(np.mean((width80 == 0.0) | (width50 == 0.0))),
    }


def _fair_median(z: np.ndarray, median: np.ndarray) -> dict[str, float]:
    error = median - z
    absolute = np.abs(error)
    return {
        "fair_log_mae": float(np.mean(absolute)),
        "fair_log_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_abs_log_error": float(np.median(absolute)),
        "p90_abs_log_error": float(np.quantile(absolute, 0.90)),
        "p95_abs_log_error": float(np.quantile(absolute, 0.95)),
        "fair_log_bias": float(np.mean(error)),
    }


def _per_seed(
    seeds: pd.Series, z: np.ndarray, q: np.ndarray, wis: np.ndarray
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    values = seeds.to_numpy()
    for seed in sorted(pd.unique(values), key=str):
        mask = values == seed
        median = _fair_median(z[mask], q[mask, 2])
        output[str(seed)] = {
            "rows": int(mask.sum()),
            "wis": float(np.mean(wis[mask])),
            "fair_log_mae": median["fair_log_mae"],
            "p10_p90_coverage": float(np.mean((z[mask] >= q[mask, 0]) & (z[mask] <= q[mask, 4]))),
            "p25_p75_coverage": float(np.mean((z[mask] >= q[mask, 1]) & (z[mask] <= q[mask, 3]))),
        }
    return output


def _normal_density_metrics(frame: pd.DataFrame, z: np.ndarray) -> dict[str, Any]:
    loc = frame["density_loc"].to_numpy(dtype=np.float64)
    scale = frame["density_scale"].to_numpy(dtype=np.float64)
    if not np.isfinite(loc).all() or not np.isfinite(scale).all() or (scale <= 0.0).any():
        raise ProbabilisticContractError("NGBoost scale validity failed")
    expected_pe = np.exp(loc[:, None] + scale[:, None] * NORMAL_QUANTILE_Z[None, :])
    published_pe = frame.loc[:, list(PE_QUANTILE_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.array_equal(expected_pe, published_pe):
        raise ProbabilisticContractError(
            "NGBoost density parameters and published quantiles are inconsistent"
        )
    standardized = (z - loc) / scale
    crps = scale * (
        standardized * (2.0 * norm.cdf(standardized) - 1.0)
        + 2.0 * norm.pdf(standardized)
        - 1.0 / math.sqrt(math.pi)
    )
    nll = np.log(scale) + 0.5 * math.log(2.0 * math.pi) + 0.5 * standardized**2
    pit = norm.cdf(standardized)
    histogram, _ = np.histogram(pit, bins=np.linspace(0.0, 1.0, 11))
    return {
        "analytic_crps": float(np.mean(crps)),
        "negative_log_likelihood": float(np.mean(nll)),
        "scale_validity": "PASS",
        "scale_minimum": float(np.min(scale)),
        "pit_mean": float(np.mean(pit)),
        "pit_variance": float(np.var(pit)),
        "pit_histogram_10_equal_bins": histogram.astype(int).tolist(),
    }


def _slice_metrics(
    frame: pd.DataFrame,
    z: np.ndarray,
    q: np.ndarray,
    wis: np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for column in PREDECLARED_SLICE_COLUMNS:
        output[column] = {}
        for value in sorted(pd.unique(frame[column]), key=str):
            mask = frame[column].to_numpy() == value
            output[column][str(value)] = {
                "rows": int(mask.sum()),
                "wis": float(np.mean(wis[mask])),
                "fair_log_mae": float(np.mean(np.abs(q[mask, 2] - z[mask]))),
            }
    return output


def _cheap_screen_decision(
    candidate: Mapping[str, Any],
    *,
    stronger_probabilistic_reference: Mapping[str, Any],
    stronger_point_comparator: Mapping[str, Any],
    crossing_row_rate: float,
    crossing_p95_magnitude: float,
    post_repair_crossing_rate: float,
    common_mask_coverage: float,
    runtime_minutes: float,
    aggregate_rss_gib: float,
) -> dict[str, Any]:
    """Apply the design thresholds once; inputs must be predeclared spent-role metrics."""

    calibration = candidate["calibration_error_by_level"]
    per_seed = candidate["per_seed"]
    guards = {
        "common_mask_coverage": common_mask_coverage == 1.0,
        "post_repair_crossing_rate": post_repair_crossing_rate == 0.0,
        "raw_crossing_row_rate": crossing_row_rate <= 0.05,
        "raw_crossing_p95_magnitude": crossing_p95_magnitude <= 0.02,
        "mean_abs_calibration": candidate["mean_absolute_quantile_calibration_error"] <= 0.05,
        "max_abs_calibration": candidate["maximum_absolute_quantile_calibration_error"] <= 0.10,
        "p10_p90_coverage": abs(candidate["p10_p90_coverage_error"]) <= 0.10,
        "p25_p75_coverage": abs(candidate["p25_p75_coverage_error"]) <= 0.10,
        "worst_seed_coverage": all(
            abs(value["p10_p90_coverage"] - 0.80) <= 0.15
            and abs(value["p25_p75_coverage"] - 0.50) <= 0.15
            for value in per_seed.values()
        ),
        "runtime": runtime_minutes <= 90.0,
        "rss": aggregate_rss_gib <= 64.0,
        "finite_calibration": all(math.isfinite(float(value)) for value in calibration.values()),
    }
    candidate_mae = candidate["fair_median"]["fair_log_mae"]
    point_mae = stronger_point_comparator["fair_log_mae"]
    candidate_rmse = candidate["fair_median"]["fair_log_rmse"]
    point_rmse = stronger_point_comparator["fair_log_rmse"]
    candidate_seed_ids = set(per_seed)
    if candidate_seed_ids != set(stronger_probabilistic_reference["per_seed"]):
        raise ProbabilisticContractError("reference per-seed support differs from candidate")
    if candidate_seed_ids != set(stronger_point_comparator["per_seed"]):
        raise ProbabilisticContractError("point comparator per-seed support differs from candidate")
    worst_seed_wis_degradation = max(
        _relative_degradation(
            per_seed[seed]["wis"], stronger_probabilistic_reference["per_seed"][seed]["wis"]
        )
        for seed in candidate_seed_ids
    )
    worst_seed_mae_degradation = max(
        _relative_degradation(
            per_seed[seed]["fair_log_mae"],
            stronger_point_comparator["per_seed"][seed]["fair_log_mae"],
        )
        for seed in candidate_seed_ids
    )
    guards["worst_seed_wis_degradation"] = worst_seed_wis_degradation <= 0.05
    guards["worst_seed_fair_log_mae_degradation"] = worst_seed_mae_degradation <= 0.02
    wis_improvement = (
        stronger_probabilistic_reference["wis"] - candidate["wis"]
    ) / stronger_probabilistic_reference["wis"]
    pinball_improvement = (
        stronger_probabilistic_reference["mean_pinball_loss"] - candidate["mean_pinball_loss"]
    ) / stronger_probabilistic_reference["mean_pinball_loss"]
    mae_improvement = (point_mae - candidate_mae) / point_mae
    rmse_improvement = (point_rmse - candidate_rmse) / point_rmse
    joint = (
        wis_improvement >= 0.01
        and pinball_improvement >= 0.01
        and max(mae_improvement, rmse_improvement) >= 0.005
        and min(mae_improvement, rmse_improvement) >= -0.005
    )
    distribution_only = (
        wis_improvement >= 0.02
        and pinball_improvement >= 0.02
        and mae_improvement >= -0.005
        and rmse_improvement >= -0.005
    )
    path = (
        "JOINT_ACCURACY_DISTRIBUTION"
        if joint
        else ("DISTRIBUTION_ONLY_VALUE" if distribution_only else "NO_ADVANCEMENT")
    )
    return {
        "guards": guards,
        "all_guards_pass": all(guards.values()),
        "advancement_path": path if all(guards.values()) else "NO_ADVANCEMENT",
        "relative_improvements": {
            "wis": wis_improvement,
            "mean_pinball_loss": pinball_improvement,
            "fair_log_mae": mae_improvement,
            "fair_log_rmse": rmse_improvement,
            "worst_seed_wis_degradation": worst_seed_wis_degradation,
            "worst_seed_fair_log_mae_degradation": worst_seed_mae_degradation,
        },
    }


def _relative_degradation(candidate_error: float, reference_error: float) -> float:
    candidate_value = float(candidate_error)
    reference_value = float(reference_error)
    if not math.isfinite(candidate_value) or not math.isfinite(reference_value):
        raise ProbabilisticContractError("screen comparison metrics must be finite")
    if reference_value <= 0.0:
        if candidate_value <= reference_value:
            return 0.0
        raise ProbabilisticContractError("positive reference error is required for degradation")
    return (candidate_value - reference_value) / reference_value
