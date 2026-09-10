"""Content-addressed, factory-only execution authorization and common-mask custody."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import (
    IDENTITY_COLUMNS,
    PROBABILISTIC_DESIGN_SHA256,
    ProbabilisticContractError,
    canonical_json_bytes,
    identity_sha256,
    logical_frame_sha256,
    normalize_identity_frame,
    require_sha256,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .nested import FORMAL_EVALUATION_ROWS_PER_SEED, FORMAL_SCORE_START_POSITION
from .spec import CANDIDATE_IDS
from .source_closure import VerifiedSourceClosure, load_verified_source_closure


AUTHORIZATION_SCHEMA = "expected_pe_model_zoo.probabilistic_execution_authorization.v2"
SCORE_FREE_SCOPE = "SCORE_FREE_SYNTHETIC_VALIDATION"
FORMAL_SCOPE = "SPENT_SCREEN_EXECUTION"
_AUTHORIZATION_TOKEN = object()
_REQUIRED_BINDINGS = (
    "adapter_binding_sha256_by_id",
    "candidate_source_snapshot_sha256",
    "common_mask_manifest_sha256",
    "comparator_prediction_sha256_by_id",
    "detached_truth_manifest_sha256",
    "environment_manifest_sha256_by_id",
    "feature_artifact_raw_sha256",
    "feature_provenance_sha256",
    "fold_manifest_sha256",
    "formal_identity_logical_sha256",
    "formal_identity_manifest_sha256",
    "probabilistic_reference_sha256_by_id",
    "spent_role_authorization_sha256",
    "training_label_artifact_raw_sha256",
)


def canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Stable round-trippable CSV bytes for content-addressed tabular contracts."""

    normalized = frame.copy()
    for column in normalized:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )
    return normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")


def _parse_identity_bytes(raw: bytes) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(raw), float_precision="round_trip")
    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ProbabilisticContractError("formal identity manifest is unreadable") from exc
    if tuple(frame.columns) != IDENTITY_COLUMNS:
        raise ProbabilisticContractError("formal identity manifest schema differs")
    identity = normalize_identity_frame(frame, context="formal identity manifest", sort=False)
    if not pd.api.types.is_integer_dtype(identity["seed"].dtype):
        raise ProbabilisticContractError("formal identity seed must be an integer")
    positions = pd.to_numeric(identity["ordered_position"], errors="coerce")
    if positions.isna().any() or not np.equal(positions, np.floor(positions)).all():
        raise ProbabilisticContractError("formal ordered positions must be integers")
    identity["ordered_position"] = positions.astype(np.int64)
    if not identity["fold_id"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ProbabilisticContractError("formal fold ids must be non-empty strings")
    expected_positions = np.arange(
        FORMAL_SCORE_START_POSITION,
        FORMAL_SCORE_START_POSITION + FORMAL_EVALUATION_ROWS_PER_SEED,
        dtype=np.int64,
    )
    for _, group in identity.groupby(["seed", "entity_id"], sort=False, dropna=False):
        if len(group) != FORMAL_EVALUATION_ROWS_PER_SEED or not np.array_equal(
            group["ordered_position"].to_numpy(dtype=np.int64), expected_positions
        ):
            raise ProbabilisticContractError(
                "each formal seed/entity must contain exact positions 504..1799"
            )
        if not group["date"].is_monotonic_increasing:
            raise ProbabilisticContractError("formal identity dates must be chronological")
        block_ids = group["fold_id"].to_numpy(dtype=object)
        starts = np.r_[True, block_ids[1:] != block_ids[:-1]]
        seen = block_ids[starts]
        if len(seen) != len(set(seen)):
            raise ProbabilisticContractError("formal fold ids must occupy one contiguous block")
        sizes = np.diff(np.r_[np.flatnonzero(starts), len(group)])
        if len(sizes) != 62 or not np.array_equal(
            sizes, np.asarray([21] * 61 + [15], dtype=np.int64)
        ):
            raise ProbabilisticContractError("formal fold blocks must be 61x21 plus terminal 15")
    return identity


def common_mask_manifest(identity: pd.DataFrame) -> dict[str, Any]:
    normalized = _parse_identity_bytes(canonical_csv_bytes(identity))
    groups = normalized.groupby(["seed", "entity_id"], sort=False, dropna=False).ngroups
    payload = {
        "schema_version": "expected_pe_model_zoo.probabilistic_common_mask.v2",
        "design_sha256": PROBABILISTIC_DESIGN_SHA256,
        "identity_columns": list(IDENTITY_COLUMNS),
        "identity_logical_sha256": identity_sha256(normalized, sort=False),
        "fold_manifest_sha256": logical_frame_sha256(normalized),
        "groups": int(groups),
        "rows_per_seed_entity": FORMAL_EVALUATION_ROWS_PER_SEED,
        "score_start_position": FORMAL_SCORE_START_POSITION,
        "natural_coverage_required": 1.0,
        "mask_shrink_allowed": False,
    }
    return seal_payload(payload)


class ExecutionAuthorization:
    """Opaque authorization whose identity rows and bindings cannot be caller-replaced."""

    __slots__ = (
        "_payload_bytes",
        "_payload",
        "_identity_raw",
        "_identity",
        "_source_closure",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _AUTHORIZATION_TOKEN:
            raise ProbabilisticContractError(
                "ExecutionAuthorization must come from the content-addressed loader"
            )
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        payload_bytes: bytes,
        payload: Mapping[str, Any],
        identity_raw: bytes,
        identity: pd.DataFrame,
        source_closure: VerifiedSourceClosure,
    ) -> None:
        if token is not _AUTHORIZATION_TOKEN:
            raise ProbabilisticContractError("invalid execution-authorization factory token")
        object.__setattr__(self, "_payload_bytes", bytes(payload_bytes))
        object.__setattr__(self, "_payload", dict(payload))
        object.__setattr__(self, "_identity_raw", bytes(identity_raw))
        object.__setattr__(self, "_identity", identity.copy(deep=True))
        object.__setattr__(self, "_source_closure", source_closure)
        self.verify_integrity()

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("ExecutionAuthorization is immutable")

    @property
    def raw_sha256(self) -> str:
        return sha256_bytes(self._payload_bytes)

    @property
    def scope(self) -> str:
        return str(self._payload["authorization_scope"])

    @property
    def evaluation_authorized(self) -> bool:
        return bool(self._payload["synthetic_evaluation_authorized"])

    @property
    def payload(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(json.dumps(self._payload)))

    @property
    def bindings(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(json.dumps(self._payload["required_bindings"])))

    def identity_frame(self) -> pd.DataFrame:
        self.verify_integrity()
        return self._identity.copy(deep=True)

    @property
    def source_closure_sha256(self) -> str:
        self.verify_integrity()
        return self._source_closure.raw_sha256

    def verify_integrity(self) -> None:
        self._source_closure.verify_integrity()
        payload = json.loads(self._payload_bytes)
        if payload != self._payload:
            raise ProbabilisticContractError("authorization payload bytes changed")
        verify_payload_seal(payload)
        if (
            sha256_bytes(self._identity_raw)
            != payload["required_bindings"]["formal_identity_manifest_sha256"]
        ):
            raise ProbabilisticContractError("authorization identity raw bytes changed")
        identity = _parse_identity_bytes(self._identity_raw)
        if (
            identity_sha256(identity, sort=False)
            != payload["required_bindings"]["formal_identity_logical_sha256"]
        ):
            raise ProbabilisticContractError("authorization identity logical hash changed")
        if not identity.equals(self._identity):
            raise ProbabilisticContractError("authorization identity frame changed")


def build_execution_authorization_payload(
    *,
    identity_raw: bytes,
    bindings: Mapping[str, Any],
    authorization_scope: str = SCORE_FREE_SCOPE,
    formal_execution_authorized: bool = False,
    synthetic_evaluation_authorized: bool = True,
) -> dict[str, Any]:
    """Build a precommit body; an external audit must still pin its raw SHA-256."""

    identity = _parse_identity_bytes(identity_raw)
    mask = common_mask_manifest(identity)
    required = dict(bindings)
    required["formal_identity_manifest_sha256"] = sha256_bytes(identity_raw)
    required["formal_identity_logical_sha256"] = identity_sha256(identity, sort=False)
    required["fold_manifest_sha256"] = logical_frame_sha256(identity)
    required["common_mask_manifest_sha256"] = sha256_bytes(canonical_json_bytes(mask))
    if set(required) != set(_REQUIRED_BINDINGS):
        missing = sorted(set(_REQUIRED_BINDINGS).difference(required))
        extra = sorted(set(required).difference(_REQUIRED_BINDINGS))
        raise ProbabilisticContractError(
            f"execution authorization bindings differ; missing={missing}, extra={extra}"
        )
    return seal_payload(
        {
            "schema_version": AUTHORIZATION_SCHEMA,
            "design_sha256": PROBABILISTIC_DESIGN_SHA256,
            "authorization_scope": authorization_scope,
            "candidate_ids": list(CANDIDATE_IDS),
            "prediction_execution_authorized": True,
            "synthetic_evaluation_authorized": synthetic_evaluation_authorized,
            "formal_execution_authorized": formal_execution_authorized,
            "fresh_seed_reserved_or_opened": False,
            "heldout_opened": False,
            "score_computation_authorized": False,
            "required_bindings": required,
        }
    )


def _verify_hash_mapping(value: object, *, field: str, expected_keys: set[str] | None) -> None:
    if not isinstance(value, dict) or not value:
        raise ProbabilisticContractError(f"{field} must be a non-empty SHA-256 mapping")
    if expected_keys is not None and set(value) != expected_keys:
        raise ProbabilisticContractError(f"{field} key set differs from the precommit")
    for key, digest in value.items():
        if not isinstance(key, str) or not key:
            raise ProbabilisticContractError(f"{field} contains an invalid key")
        require_sha256(digest, field=f"{field}.{key}")


def load_execution_authorization(
    precommit_path: Path,
    identity_manifest_path: Path,
    source_closure_path: Path,
    *,
    expected_precommit_sha256: str,
    require_formal: bool = False,
    formal_activation_path: Path | None = None,
    expected_formal_activation_sha256: str | None = None,
) -> ExecutionAuthorization:
    """Read precommit and identity bytes once and return only an externally addressed grant."""

    require_sha256(expected_precommit_sha256, field="expected_precommit_sha256")
    try:
        payload_raw = Path(precommit_path).read_bytes()
        identity_raw = Path(identity_manifest_path).read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError(
            "execution authorization artifacts are unavailable"
        ) from exc
    if sha256_bytes(payload_raw) != expected_precommit_sha256:
        raise ProbabilisticContractError("execution precommit differs from external address")
    if expected_precommit_sha256 not in Path(precommit_path).name.split("."):
        raise ProbabilisticContractError("execution precommit filename is not content-addressed")
    try:
        payload = json.loads(payload_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("execution precommit cannot be decoded") from exc
    if not isinstance(payload, dict):
        raise ProbabilisticContractError("execution precommit must be an object")
    verify_payload_seal(payload)
    if payload.get("schema_version") != AUTHORIZATION_SCHEMA:
        raise ProbabilisticContractError("execution authorization schema differs")
    if payload.get("design_sha256") != PROBABILISTIC_DESIGN_SHA256:
        raise ProbabilisticContractError("execution authorization design differs")
    if tuple(payload.get("candidate_ids", ())) != CANDIDATE_IDS:
        raise ProbabilisticContractError("execution authorization candidates differ")
    scope = payload.get("authorization_scope")
    if scope not in {SCORE_FREE_SCOPE, FORMAL_SCOPE}:
        raise ProbabilisticContractError("execution authorization scope is invalid")
    if payload.get("prediction_execution_authorized") is not True:
        raise ProbabilisticContractError("prediction execution is not authorized")
    if payload.get("score_computation_authorized") is not False:
        raise ProbabilisticContractError("score-free authorization cannot authorize scoring")
    if (
        payload.get("fresh_seed_reserved_or_opened") is not False
        or payload.get("heldout_opened") is not False
    ):
        raise ProbabilisticContractError("execution authorization custody flags changed")
    if require_formal and (
        scope != FORMAL_SCOPE or payload.get("formal_execution_authorized") is not True
    ):
        raise ProbabilisticContractError("formal spent-screen execution is not authorized")
    if require_formal and (
        formal_activation_path is None or expected_formal_activation_sha256 is None
    ):
        raise ProbabilisticContractError(
            "formal authorization requires the content-addressed activation/policy pin"
        )
    if not require_formal and payload.get("formal_execution_authorized") is not False:
        raise ProbabilisticContractError(
            "score-free validation cannot impersonate formal execution"
        )
    bindings = payload.get("required_bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(_REQUIRED_BINDINGS):
        raise ProbabilisticContractError("execution authorization binding schema differs")
    scalar_hashes = set(_REQUIRED_BINDINGS).difference(
        {
            "adapter_binding_sha256_by_id",
            "comparator_prediction_sha256_by_id",
            "probabilistic_reference_sha256_by_id",
            "environment_manifest_sha256_by_id",
        }
    )
    for field in scalar_hashes:
        require_sha256(bindings[field], field=field)
    _verify_hash_mapping(
        bindings["adapter_binding_sha256_by_id"],
        field="adapter_binding_sha256_by_id",
        expected_keys=set(CANDIDATE_IDS),
    )
    _verify_hash_mapping(
        bindings["comparator_prediction_sha256_by_id"],
        field="comparator_prediction_sha256_by_id",
        expected_keys={"point_history", "v04_expected_pe", "ml_expected_pe"},
    )
    if (
        bindings["comparator_prediction_sha256_by_id"]["v04_expected_pe"]
        == bindings["comparator_prediction_sha256_by_id"]["ml_expected_pe"]
    ):
        raise ProbabilisticContractError("point comparator artifact hashes must be distinct")
    _verify_hash_mapping(
        bindings["probabilistic_reference_sha256_by_id"],
        field="probabilistic_reference_sha256_by_id",
        expected_keys={"rolling_log_quantiles_252", "point_residual_quantiles_252"},
    )
    _verify_hash_mapping(
        bindings["environment_manifest_sha256_by_id"],
        field="environment_manifest_sha256_by_id",
        expected_keys=set(CANDIDATE_IDS),
    )
    identity = _parse_identity_bytes(identity_raw)
    if sha256_bytes(identity_raw) != bindings["formal_identity_manifest_sha256"]:
        raise ProbabilisticContractError("formal identity raw hash differs from precommit")
    if bindings["formal_identity_manifest_sha256"] not in Path(identity_manifest_path).name.split(
        "."
    ):
        raise ProbabilisticContractError("formal identity filename is not content-addressed")
    if identity_sha256(identity, sort=False) != bindings["formal_identity_logical_sha256"]:
        raise ProbabilisticContractError("formal identity logical hash differs from precommit")
    if logical_frame_sha256(identity) != bindings["fold_manifest_sha256"]:
        raise ProbabilisticContractError("formal fold rows differ from precommit")
    mask = common_mask_manifest(identity)
    if sha256_bytes(canonical_json_bytes(mask)) != bindings["common_mask_manifest_sha256"]:
        raise ProbabilisticContractError("executable common-mask manifest differs")
    source_closure = load_verified_source_closure(
        source_closure_path,
        expected_raw_sha256=bindings["candidate_source_snapshot_sha256"],
        repo_root=Path(__file__).resolve().parents[4],
    )
    if require_formal:
        from .spent import verify_formal_activation

        verify_formal_activation(
            formal_activation_path,
            expected_raw_sha256=expected_formal_activation_sha256,
            repo_root=Path(__file__).resolve().parents[4],
            authorization_raw_sha256=expected_precommit_sha256,
            source_snapshot_raw_sha256=bindings["candidate_source_snapshot_sha256"],
            identity_raw_sha256=bindings["formal_identity_manifest_sha256"],
            authorization_bindings=bindings,
        )
    return ExecutionAuthorization(
        _AUTHORIZATION_TOKEN,
        payload_bytes=payload_raw,
        payload=payload,
        identity_raw=identity_raw,
        identity=identity,
        source_closure=source_closure,
    )


@dataclass(frozen=True)
class AuthorizationFiles:
    precommit: Path
    identity_manifest: Path
    precommit_sha256: str
