"""Score-free contracts for the isolated probabilistic Expected-P/E wave."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..contracts import ContractError


PROBABILISTIC_DESIGN_SHA256 = "ae2442c8dfde4c70f45e3efe5a20f80a13ed7c8545a055dd05d57e931e433652"
PROBABILISTIC_DESIGN_SCHEMA = "expected_pe_model_zoo.probabilistic_wave_design.v1"
PROBABILISTIC_DESIGN_STATUS = (
    "DESIGN_ONLY_SCORE_FREE_NO_MODEL_RUN_NO_SEED_RESERVATION_NO_HELDOUT_ACCESS"
)
EVALUATION_ONLY_COLUMN = "true_fair_pe"
QUANTILE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90)
QUANTILE_LABELS = ("p10", "p25", "p50", "p75", "p90")
LOG_QUANTILE_COLUMNS = tuple(f"predicted_log_pe_{label}" for label in QUANTILE_LABELS)
PE_QUANTILE_COLUMNS = tuple(f"predicted_pe_{label}" for label in QUANTILE_LABELS)
UNCERTAINTY_COLUMNS = (
    "uncertainty_log_iqr",
    "uncertainty_log_idr",
    "uncertainty_robust_sigma",
    "uncertainty_p90_p10_ratio",
)
IDENTITY_COLUMNS = ("seed", "entity_id", "date", "ordered_position", "fold_id")
MODEL_OUTPUT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "model_id",
    *PE_QUANTILE_COLUMNS,
    "expected_pe",
    *UNCERTAINTY_COLUMNS,
)
DENSITY_OUTPUT_COLUMNS = ("density_loc", "density_scale")
NORMAL_QUANTILE_Z = np.asarray(
    (
        -1.2815515655446004,
        -0.6744897501960817,
        0.0,
        0.6744897501960817,
        1.2815515655446004,
    ),
    dtype=np.float64,
)
NORMAL_QUANTILE_Z.setflags(write=False)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERIFIED_PREDICTION_TOKEN = object()


class ProbabilisticContractError(ContractError):
    """Raised when a distributional artifact is ambiguous, unsafe, or mutable."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProbabilisticContractError("value is not finite canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ProbabilisticContractError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProbabilisticContractError(f"{field} must be a lowercase SHA-256")
    return value


def seal_payload(payload: Mapping[str, Any], *, field: str = "manifest_sha256") -> dict[str, Any]:
    output = dict(payload)
    output.pop(field, None)
    output[field] = sha256_bytes(canonical_json_bytes(output))
    return output


def verify_payload_seal(payload: Mapping[str, Any], *, field: str = "manifest_sha256") -> None:
    recorded = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    expected = sha256_bytes(canonical_json_bytes(unsigned))
    if recorded != expected:
        raise ProbabilisticContractError(f"{field} is missing or invalid")


def verify_probabilistic_design(path: Path) -> dict[str, Any]:
    """Bind implementation bytes to the exact score-free design."""

    path = Path(path)
    if sha256_file(path) != PROBABILISTIC_DESIGN_SHA256:
        raise ProbabilisticContractError("probabilistic DESIGN.json SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("probabilistic DESIGN.json is unreadable") from exc
    if not isinstance(value, dict):
        raise ProbabilisticContractError("probabilistic design root must be an object")
    if value.get("schema_version") != PROBABILISTIC_DESIGN_SCHEMA:
        raise ProbabilisticContractError("probabilistic design schema differs")
    if value.get("status") != PROBABILISTIC_DESIGN_STATUS:
        raise ProbabilisticContractError("probabilistic design is not score-free")
    scope = value.get("scope_guards")
    if not isinstance(scope, dict):
        raise ProbabilisticContractError("probabilistic design scope guards are missing")
    required_false = (
        "source_edits",
        "test_edits",
        "registry_edits",
        "model_fit",
        "predictions_generated",
        "scores_read_or_generated",
        "structural_candidate_scores_read",
        "fresh_seed_named_selected_or_reserved",
        "heldout_opened",
        "post_score_candidate_substitution",
    )
    if any(scope.get(field) is not False for field in required_false):
        raise ProbabilisticContractError("probabilistic design score-free guards changed")
    if scope.get("candidate_ceiling") != 3:
        raise ProbabilisticContractError("probabilistic candidate ceiling differs")
    return value


def require_no_evaluation_truth(columns: Iterable[object], *, context: str) -> None:
    if EVALUATION_ONLY_COLUMN.casefold() in {str(column).casefold() for column in columns}:
        raise ProbabilisticContractError(f"{context} must not contain {EVALUATION_ONLY_COLUMN}")


def require_unique_columns(frame: pd.DataFrame, *, context: str) -> None:
    if frame.columns.has_duplicates:
        raise ProbabilisticContractError(f"{context} columns must be unique")


def normalize_dates(values: Iterable[object], *, context: str) -> pd.Series:
    dates = pd.to_datetime(pd.Series(values), errors="coerce", utc=True)
    if dates.isna().any():
        raise ProbabilisticContractError(f"{context} contains invalid dates")
    return dates.dt.tz_convert(None)


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    require_unique_columns(frame, context="logical frame")
    normalized = frame.copy()
    for column in normalized:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
        elif pd.api.types.is_float_dtype(normalized[column]):
            values = normalized[column].to_numpy(dtype=np.float64, na_value=np.nan)
            if np.isinf(values).any():
                raise ProbabilisticContractError("logical frame contains infinite values")
    rendered = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")
    return sha256_bytes(rendered)


def normalize_identity_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = IDENTITY_COLUMNS,
    context: str,
    sort: bool = False,
) -> pd.DataFrame:
    require_unique_columns(frame, context=context)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ProbabilisticContractError(f"{context} is missing identity columns: {missing}")
    output = frame.loc[:, list(columns)].copy()
    if "date" in output:
        output["date"] = normalize_dates(output["date"], context=f"{context}.date")
    if output.isna().any().any():
        raise ProbabilisticContractError(f"{context} identity contains missing values")
    if output.duplicated(list(columns)).any():
        raise ProbabilisticContractError(f"{context} identity is not one-to-one")
    if sort:
        output = output.sort_values(list(columns), kind="mergesort")
    return output.reset_index(drop=True)


def identity_sha256(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] = IDENTITY_COLUMNS,
    sort: bool = False,
) -> str:
    return logical_frame_sha256(
        normalize_identity_frame(frame, columns=columns, context="identity hash", sort=sort)
    )


@dataclass(frozen=True)
class DistributionModelMetadata:
    model_id: str
    family: str
    variant: str
    package: str
    license: str
    feature_ids: tuple[str, ...]
    quantile_levels: tuple[float, ...]
    full_density: bool
    training_target: str = "log(observed_pe)"
    track: str = "C"
    uses_same_row_price: bool = True
    uses_same_row_observed_pe: bool = False
    crossing_policy: str = "stable_monotone_rearrangement_in_log_space"

    def __post_init__(self) -> None:
        if not self.model_id or not self.family or not self.variant:
            raise ProbabilisticContractError("model identity fields must be non-empty")
        if self.quantile_levels != QUANTILE_LEVELS:
            raise ProbabilisticContractError("model quantile levels differ from the design")
        if self.training_target != "log(observed_pe)":
            raise ProbabilisticContractError("probabilistic target must be log(observed_pe)")
        if self.track != "C" or not self.uses_same_row_price:
            raise ProbabilisticContractError("probabilistic candidates must remain Track C")
        if self.uses_same_row_observed_pe:
            raise ProbabilisticContractError("same-row observed P/E is forbidden")
        if len(self.feature_ids) != len(set(self.feature_ids)):
            raise ProbabilisticContractError("model feature ids must be unique")


@dataclass(frozen=True)
class CrossingDiagnostics:
    rows: int
    crossing_rows: int
    crossing_row_rate: float
    maximum_crossing_pair_count: int
    p95_total_negative_adjacent_gap: float
    post_repair_crossing_rows: int
    interval_collapse_rows: int

    def __post_init__(self) -> None:
        numeric = (
            self.crossing_row_rate,
            self.p95_total_negative_adjacent_gap,
        )
        if self.rows < 1 or any(not math.isfinite(value) or value < 0.0 for value in numeric):
            raise ProbabilisticContractError("crossing diagnostics are invalid")
        counts = (
            self.crossing_rows,
            self.maximum_crossing_pair_count,
            self.post_repair_crossing_rows,
            self.interval_collapse_rows,
        )
        if any(count < 0 for count in counts):
            raise ProbabilisticContractError("crossing diagnostic counts are invalid")


def _immutable_float_matrix(
    value: object,
    *,
    rows: int | None,
    columns: int,
    context: str,
) -> np.ndarray:
    output = np.array(value, dtype=np.float64, copy=True)
    if output.ndim != 2 or output.shape[1] != columns:
        raise ProbabilisticContractError(f"{context} must have shape (n,{columns})")
    if rows is not None and output.shape[0] != rows:
        raise ProbabilisticContractError(f"{context} row count differs")
    if not np.isfinite(output).all():
        raise ProbabilisticContractError(f"{context} must be finite")
    output.setflags(write=False)
    return output


@dataclass(frozen=True)
class QuantilePredictionBatch:
    model_id: str
    raw_log_quantiles: np.ndarray
    repaired_log_quantiles: np.ndarray
    pe_quantiles: np.ndarray
    diagnostics: CrossingDiagnostics
    density_parameters: Mapping[str, np.ndarray] | None = None

    def __post_init__(self) -> None:
        raw = _immutable_float_matrix(
            self.raw_log_quantiles,
            rows=None,
            columns=len(QUANTILE_LEVELS),
            context="raw log quantiles",
        )
        repaired = _immutable_float_matrix(
            self.repaired_log_quantiles,
            rows=raw.shape[0],
            columns=len(QUANTILE_LEVELS),
            context="repaired log quantiles",
        )
        pe = _immutable_float_matrix(
            self.pe_quantiles,
            rows=raw.shape[0],
            columns=len(QUANTILE_LEVELS),
            context="P/E quantiles",
        )
        if raw.shape[0] < 1:
            raise ProbabilisticContractError("prediction batch must be non-empty")
        if (np.diff(repaired, axis=1) < 0.0).any():
            raise ProbabilisticContractError("repaired log quantiles cross")
        if (pe <= 0.0).any() or (np.diff(pe, axis=1) < 0.0).any():
            raise ProbabilisticContractError("P/E quantiles must be positive and ordered")
        if not np.array_equal(np.exp(repaired), pe):
            raise ProbabilisticContractError("P/E quantiles are not exact exp(log quantiles)")
        parameters: dict[str, np.ndarray] | None = None
        if self.density_parameters is not None:
            parameters = {}
            for name, values in sorted(self.density_parameters.items()):
                array = np.array(values, dtype=np.float64, copy=True)
                if array.shape != (raw.shape[0],) or not np.isfinite(array).all():
                    raise ProbabilisticContractError(
                        f"density parameter {name!r} must be finite and row-aligned"
                    )
                array.setflags(write=False)
                parameters[str(name)] = array
            if set(parameters) != {"loc", "scale"}:
                raise ProbabilisticContractError(
                    "Normal density parameters must be exactly loc and scale"
                )
            loc = parameters["loc"]
            scale = parameters["scale"]
            if (scale <= 0.0).any():
                raise ProbabilisticContractError("Normal density scale must be positive")
            expected = loc[:, None] + scale[:, None] * NORMAL_QUANTILE_Z[None, :]
            if not np.array_equal(raw, expected) or not np.array_equal(repaired, expected):
                raise ProbabilisticContractError(
                    "Normal density loc/scale and published quantiles are inconsistent"
                )
        object.__setattr__(self, "raw_log_quantiles", raw)
        object.__setattr__(self, "repaired_log_quantiles", repaired)
        object.__setattr__(self, "pe_quantiles", pe)
        object.__setattr__(
            self,
            "density_parameters",
            None if parameters is None else MappingProxyType(parameters),
        )

    @property
    def rows(self) -> int:
        return int(self.pe_quantiles.shape[0])

    def output_frame(self, *, index: pd.Index | None = None) -> pd.DataFrame:
        output = pd.DataFrame(self.pe_quantiles, columns=PE_QUANTILE_COLUMNS, index=index)
        output["expected_pe"] = output["predicted_pe_p50"].to_numpy(copy=True)
        log_iqr = self.repaired_log_quantiles[:, 3] - self.repaired_log_quantiles[:, 1]
        log_idr = self.repaired_log_quantiles[:, 4] - self.repaired_log_quantiles[:, 0]
        output["uncertainty_log_iqr"] = log_iqr
        output["uncertainty_log_idr"] = log_idr
        output["uncertainty_robust_sigma"] = log_iqr / 1.3489795003921634
        output["uncertainty_p90_p10_ratio"] = self.pe_quantiles[:, 4] / self.pe_quantiles[:, 0]
        if self.density_parameters is not None:
            for name, values in self.density_parameters.items():
                output[f"density_{name}"] = values
        return output


class VerifiedPredictionBatch:
    """Opaque single-read prediction bytes accepted by the evaluator boundary."""

    __slots__ = (
        "_frame",
        "_identity_sha256",
        "_prediction_sha256",
        "_raw_bytes",
        "_raw_sha256",
        "_receipt_sha256",
        "_receipt_bytes",
        "_authorization_sha256",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _VERIFIED_PREDICTION_TOKEN:
            raise ProbabilisticContractError(
                "VerifiedPredictionBatch must come from the content-addressed loader"
            )
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        frame: pd.DataFrame,
        identity_sha256: str,
        prediction_sha256: str,
        raw_bytes: bytes,
        raw_sha256: str,
        receipt_sha256: str,
        receipt_bytes: bytes,
        authorization_sha256: str,
    ) -> None:
        if token is not _VERIFIED_PREDICTION_TOKEN:
            raise ProbabilisticContractError("invalid verified-prediction factory token")
        require_sha256(identity_sha256, field="identity_sha256")
        require_sha256(prediction_sha256, field="prediction_sha256")
        require_sha256(raw_sha256, field="raw_sha256")
        require_sha256(receipt_sha256, field="receipt_sha256")
        require_sha256(authorization_sha256, field="authorization_sha256")
        require_no_evaluation_truth(frame.columns, context="verified prediction batch")
        if sha256_bytes(raw_bytes) != raw_sha256:
            raise ProbabilisticContractError("prediction raw bytes differ from their address")
        if sha256_bytes(receipt_bytes) != receipt_sha256:
            raise ProbabilisticContractError("prediction receipt bytes differ from their address")
        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_identity_sha256", identity_sha256)
        object.__setattr__(self, "_prediction_sha256", prediction_sha256)
        object.__setattr__(self, "_raw_bytes", bytes(raw_bytes))
        object.__setattr__(self, "_raw_sha256", raw_sha256)
        object.__setattr__(self, "_receipt_sha256", receipt_sha256)
        object.__setattr__(self, "_receipt_bytes", bytes(receipt_bytes))
        object.__setattr__(self, "_authorization_sha256", authorization_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedPredictionBatch is immutable")

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy(deep=True)

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    @property
    def prediction_sha256(self) -> str:
        return self._prediction_sha256

    @property
    def raw_sha256(self) -> str:
        return self._raw_sha256

    @property
    def receipt_sha256(self) -> str:
        return self._receipt_sha256

    @property
    def authorization_sha256(self) -> str:
        return self._authorization_sha256

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw_bytes) != self._raw_sha256:
            raise ProbabilisticContractError("verified prediction raw bytes changed")
        if logical_frame_sha256(self._frame) != self._prediction_sha256:
            raise ProbabilisticContractError("verified prediction frame changed")
        identity = normalize_identity_frame(
            self._frame, context="verified prediction identity", sort=False
        )
        if identity_sha256(identity, sort=False) != self._identity_sha256:
            raise ProbabilisticContractError("verified prediction identity hash changed")
        if sha256_bytes(self._receipt_bytes) != self._receipt_sha256:
            raise ProbabilisticContractError("verified prediction receipt bytes changed")
        try:
            receipt = json.loads(self._receipt_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProbabilisticContractError(
                "verified prediction receipt cannot be decoded"
            ) from exc
        verify_payload_seal(receipt)
        participant = receipt.get("model_id", receipt.get("reference_id"))
        frame_ids = tuple(pd.unique(self._frame["model_id"]))
        if (
            len(frame_ids) != 1
            or participant != frame_ids[0]
            or receipt.get("authorization_raw_sha256") != self._authorization_sha256
            or receipt.get("prediction_raw_sha256") != self._raw_sha256
            or receipt.get("prediction_logical_sha256") != self._prediction_sha256
            or receipt.get("identity_logical_sha256") != self._identity_sha256
        ):
            raise ProbabilisticContractError(
                "verified prediction receipt provenance differs from custody bytes"
            )


def _make_verified_prediction_batch(*_: object, **__: object) -> VerifiedPredictionBatch:
    """Tombstoned: in-memory caller-minted custody is forbidden."""

    raise ProbabilisticContractError(
        "in-memory prediction-batch construction is tombstoned; load receipt files"
    )


def _make_verified_prediction_batch_from_receipt_bytes(
    frame: pd.DataFrame,
    *,
    identity_hash: str,
    prediction_hash: str,
    raw_bytes: bytes,
    raw_hash: str,
    receipt_hash: str,
    receipt_bytes: bytes,
    authorization_hash: str,
) -> VerifiedPredictionBatch:
    return VerifiedPredictionBatch(
        _VERIFIED_PREDICTION_TOKEN,
        frame=frame,
        identity_sha256=identity_hash,
        prediction_sha256=prediction_hash,
        raw_bytes=raw_bytes,
        raw_sha256=raw_hash,
        receipt_sha256=receipt_hash,
        receipt_bytes=receipt_bytes,
        authorization_sha256=authorization_hash,
    )
