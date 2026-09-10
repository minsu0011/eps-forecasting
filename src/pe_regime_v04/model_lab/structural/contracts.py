"""Fail-closed contracts shared by the score-free Structural Wave kernels.

This namespace is deliberately isolated from the frozen Phase-1 and Wave-1
registries.  The implementation is executable on synthetic inputs, but it has
no code path that opens evaluation truth or emits a formal candidate score.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..contracts import ContractError


STRUCTURAL_DESIGN_ID = "model_zoo_structural_wave_design_20260819"
STRUCTURAL_DESIGN_SHA256 = "6a1516dc0b75bfa93ffadbdff253cb634dccf3ce7f579bc713033b70db33c4cf"
EVALUATION_ONLY_COLUMN = "true_fair_pe"
IDENTITY_COLUMNS = ("seed", "date", "outer_fold_id")
FULL_META_IDENTITY_COLUMNS = (
    "seed",
    "date",
    "outer_fold_id",
    "inner_fold_id",
    "base_model_id",
    "base_source_sha256",
    "base_config_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class StructuralContractError(ContractError):
    """Raised when Structural Wave evidence is ambiguous, mutable, or unsafe."""


_KERNEL_BINDING_TOKEN = object()
_DECOMPOSITION_TRACK_BY_CANDIDATE = {
    "decomp_block_ridge_ar1_lag1": "A",
    "decomp_block_ridge_ar1_current": "C",
}


@dataclass(frozen=True, init=False)
class KernelBinding:
    """Factory-only content binding for one authorized decomposition candidate."""

    candidate_id: str
    track: str
    source_sha256: str
    config_sha256: str
    environment_sha256: str
    fold_sha256: str
    authorization_sha256: str
    design_sha256: str = STRUCTURAL_DESIGN_SHA256

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError(
            "KernelBinding is factory-only; derive it from sealed authorization"
        )

    @classmethod
    def _mint(cls, *, token: object, **values: Any) -> "KernelBinding":
        if token is not _KERNEL_BINDING_TOKEN:
            raise StructuralContractError("invalid kernel-binding mint token")
        output = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(output, name, value)
        output._validate()
        return output

    @classmethod
    def from_authorization(
        cls,
        authorization: Any,
        *,
        candidate_id: str,
        fold_sha256: str,
    ) -> "KernelBinding":
        # Local import avoids an authorization -> contracts import cycle.
        from .authorization import StructuralExecutionAuthorization

        if not isinstance(authorization, StructuralExecutionAuthorization):
            raise StructuralContractError("kernel binding requires sealed authorization")
        authorization.require_candidate(candidate_id)
        try:
            track = _DECOMPOSITION_TRACK_BY_CANDIDATE[candidate_id]
        except KeyError as exc:
            raise StructuralContractError(
                "kernel binding is restricted to the two decomposition candidates"
            ) from exc
        require_sha256(fold_sha256, field="fold_sha256")
        snapshot = json.loads(authorization.execution_snapshot_path.read_text(encoding="utf-8"))
        verify_payload_seal(snapshot)
        inventory = snapshot.get("source_inventory")
        if not isinstance(inventory, list):
            raise StructuralContractError("execution snapshot inventory is missing")
        decomposition_path = "src/pe_regime_v04/model_lab/structural/decomposition.py"
        matches = [row for row in inventory if row.get("path") == decomposition_path]
        if len(matches) != 1:
            raise StructuralContractError("decomposition source is not uniquely snapshotted")
        return cls._mint(
            token=_KERNEL_BINDING_TOKEN,
            candidate_id=candidate_id,
            track=track,
            source_sha256=str(matches[0]["sha256"]),
            config_sha256=STRUCTURAL_DESIGN_SHA256,
            environment_sha256=authorization.residual_base.environment_sha256,
            fold_sha256=fold_sha256,
            authorization_sha256=authorization.authorization_sha256,
            design_sha256=STRUCTURAL_DESIGN_SHA256,
        )

    def _validate(self) -> None:
        if _DECOMPOSITION_TRACK_BY_CANDIDATE.get(self.candidate_id) != self.track:
            raise StructuralContractError("kernel candidate/track binding is invalid")
        for field_name in (
            "source_sha256",
            "config_sha256",
            "environment_sha256",
            "fold_sha256",
            "authorization_sha256",
            "design_sha256",
        ):
            require_sha256(getattr(self, field_name), field=field_name)
        if self.design_sha256 != STRUCTURAL_DESIGN_SHA256:
            raise StructuralContractError("kernel binding references a different design")

    def verify(self, authorization: Any, *, candidate_id: str, track: str) -> None:
        expected = KernelBinding.from_authorization(
            authorization,
            candidate_id=candidate_id,
            fold_sha256=self.fold_sha256,
        )
        if self != expected or self.track != track:
            raise StructuralContractError("decomposition kernel binding changed")

    def as_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "track": self.track,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "environment_sha256": self.environment_sha256,
            "fold_sha256": self.fold_sha256,
            "authorization_sha256": self.authorization_sha256,
            "design_sha256": self.design_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        authorization: Any,
    ) -> "KernelBinding":
        if set(value) != {
            "candidate_id",
            "track",
            "source_sha256",
            "config_sha256",
            "environment_sha256",
            "fold_sha256",
            "authorization_sha256",
            "design_sha256",
        }:
            raise StructuralContractError("kernel-binding payload schema changed")
        output = cls._mint(token=_KERNEL_BINDING_TOKEN, **dict(value))
        output.verify(
            authorization,
            candidate_id=output.candidate_id,
            track=output.track,
        )
        return output


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one JSON serialization used for every structural seal."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StructuralContractError("value is not finite canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise StructuralContractError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StructuralContractError(f"{field} must be a lowercase SHA-256")
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
    if recorded != sha256_bytes(canonical_json_bytes(unsigned)):
        raise StructuralContractError(f"{field} is missing or invalid")


def verify_structural_design(path: Path) -> dict[str, Any]:
    """Bind source and tests to the exact frozen score-free design bytes."""

    path = Path(path)
    if sha256_file(path) != STRUCTURAL_DESIGN_SHA256:
        raise StructuralContractError("Structural Wave DESIGN.json SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralContractError("Structural Wave DESIGN.json is unreadable") from exc
    if not isinstance(value, dict) or value.get("design_id") != STRUCTURAL_DESIGN_ID:
        raise StructuralContractError("Structural Wave design identity mismatch")
    if value.get("state") != "DESIGN_ONLY_SCORE_FREE_NO_SEED_RESERVATION":
        raise StructuralContractError("Structural Wave design is not the score-free lock")
    attestations = value.get("attestations")
    if not isinstance(attestations, dict):
        raise StructuralContractError("Structural Wave design attestations are missing")
    required_false = (
        "wave1_candidate_predictions_read",
        "wave1_candidate_scores_read",
        "structural_candidate_predictions_generated",
        "structural_candidate_scores_generated",
        "true_fair_pe_consumed",
        "fresh_seed_selected_or_reserved",
        "heldout_opened",
    )
    if any(attestations.get(key) is not False for key in required_false):
        raise StructuralContractError("Structural Wave design score-free attestations changed")
    return value


def require_no_evaluation_truth(columns: Iterable[object], *, context: str) -> None:
    lowered = {str(column).casefold() for column in columns}
    if EVALUATION_ONLY_COLUMN.casefold() in lowered:
        raise StructuralContractError(f"{context} must not contain {EVALUATION_ONLY_COLUMN}")


def require_unique_columns(frame: pd.DataFrame, *, context: str) -> None:
    if frame.columns.has_duplicates:
        raise StructuralContractError(f"{context} columns must be unique")


def numeric_vector(values: Iterable[object], *, context: str) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    output = series.to_numpy(dtype=np.float64, na_value=np.nan)
    if output.ndim != 1:
        raise StructuralContractError(f"{context} must be one-dimensional")
    return output


def require_positive_finite(values: Iterable[object], *, context: str) -> np.ndarray:
    output = numeric_vector(values, context=context)
    if output.size == 0 or not np.isfinite(output).all() or (output <= 0.0).any():
        raise StructuralContractError(f"{context} must be non-empty, positive, and finite")
    return output


def normalize_dates(values: Iterable[object], *, context: str) -> pd.Series:
    dates = pd.to_datetime(pd.Series(values), errors="coerce", utc=True)
    if dates.isna().any():
        raise StructuralContractError(f"{context} contains invalid dates")
    return dates.dt.tz_convert(None)


def normalized_identity_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    context: str,
    sort: bool = True,
) -> pd.DataFrame:
    require_unique_columns(frame, context=context)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise StructuralContractError(f"{context} is missing identity columns: {missing}")
    output = frame.loc[:, list(columns)].copy()
    if "date" in output:
        output["date"] = normalize_dates(output["date"], context=f"{context}.date").to_numpy(
            copy=True
        )
    if output.isna().any().any():
        raise StructuralContractError(f"{context} identity contains missing values")
    if output.duplicated(list(columns)).any():
        raise StructuralContractError(f"{context} identity must be one-to-one")
    if sort:
        output = output.sort_values(list(columns), kind="mergesort").reset_index(drop=True)
    else:
        output = output.reset_index(drop=True)
    return output


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    """Hash logical values with an explicit schema and stable scalar encoding."""

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
                raise StructuralContractError("logical frame contains infinite values")
    rendered = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")
    return sha256_bytes(rendered)


def identity_sha256(frame: pd.DataFrame, *, columns: Sequence[str]) -> str:
    identity = normalized_identity_frame(
        frame,
        columns=columns,
        context="identity hash input",
        sort=True,
    )
    return logical_frame_sha256(identity)


def require_exact_identity(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    columns: Sequence[str],
    context: str,
) -> pd.DataFrame:
    """Reject mask shrink, duplicates, additions, row substitution, and reordering."""

    expected_identity = normalized_identity_frame(
        expected, columns=columns, context=f"{context}.expected", sort=False
    )
    actual_identity = normalized_identity_frame(
        actual, columns=columns, context=f"{context}.actual", sort=False
    )
    if len(expected_identity) != len(actual_identity):
        raise StructuralContractError(f"{context} row count differs")
    if not expected_identity.equals(actual_identity):
        raise StructuralContractError(f"{context} identity or row order differs")
    return actual_identity


def finite_float(value: object, *, field: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as exc:
        raise StructuralContractError(f"{field} must be numeric") from exc
    if not math.isfinite(output):
        raise StructuralContractError(f"{field} must be finite")
    return output
