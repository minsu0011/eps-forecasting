"""Physical truth-boundary gates for the isolated two-survivor continuation.

The parent may verify and hand off addresses, but only a distinct, single-use child
that reloaded the same V3 request and independent GO may obtain the truth address.
Metric computation is intentionally not performed by this score-free activation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .artifacts import immutable_write_bytes
from .contracts import (
    ProbabilisticContractError,
    canonical_json_bytes,
    require_sha256,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .two_survivor_continuation import (
    TwoSurvivorLaunchAuthority,
    VerifiedTwoSurvivorBundle,
    authority_binding,
    load_two_survivor_bundle,
    require_two_survivor_authority,
)


HANDOFF_SCHEMA = "expected_pe_model_zoo.probabilistic_two_survivor_handoff.v1"
CLAIM_SCHEMA = "expected_pe_model_zoo.probabilistic_two_survivor_handoff_claim.v1"
DETACHED_TRUTH_ADDRESS = {
    "path": "outputs/model_zoo_wave1_screen_20260819/EVALUATE_INPUTS.json",
    "raw_sha256": "6cc6c750905784d615dc5ba200843b296b6f29fe1a1209d859da5fd266eda98d",
    "logical_sha256": "5f39e7495f81485695cb38443e5a8771bf76406e10d735e2d221745861f2f70a",
}

_PREFLIGHT_TOKEN = object()


def _relative_path(path: Path, root: Path, *, context: str) -> str:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ProbabilisticContractError(f"{context} escaped repository root") from exc
    if path.is_symlink():
        raise ProbabilisticContractError(f"{context} may not be a symlink alias")
    return relative.as_posix()


def build_two_survivor_handoff_payload(
    *,
    bundle: VerifiedTwoSurvivorBundle,
    authority: TwoSurvivorLaunchAuthority,
    parent_pid: int | None = None,
) -> dict[str, Any]:
    authority = require_two_survivor_authority(authority)
    if not isinstance(bundle, VerifiedTwoSurvivorBundle):
        raise ProbabilisticContractError("handoff requires verified two-survivor bundle")
    bundle.verify_integrity()
    if bundle.authority is not authority:
        # Identity is intentional: callers cannot mix two independently reloaded
        # capabilities even when they merely claim the same hash strings.
        raise ProbabilisticContractError("bundle/handoff authority capability differs")
    pid = os.getpid() if parent_pid is None else parent_pid
    if not isinstance(pid, int) or pid <= 0:
        raise ProbabilisticContractError("handoff parent PID is invalid")
    return seal_payload(
        {
            "schema_version": HANDOFF_SCHEMA,
            "authority": authority_binding(authority),
            "bundle": {
                "path": _relative_path(bundle.path, authority.repo_root, context="bundle"),
                "raw_sha256": bundle.raw_sha256,
            },
            # This address is copied from a frozen constant.  The parent does not
            # stat, open, hash, parse, or otherwise observe the addressed bytes.
            "detached_truth_address": dict(DETACHED_TRUTH_ADDRESS),
            "parent_pid": pid,
            "truth_address_source": "FROZEN_CONSTANT_NO_PARENT_FILE_ACCESS",
            "parent_truth_read": False,
            "child_process_required": True,
            "single_use_claim_required": True,
            "metric_computation_started": False,
            "fresh_seed_reserved_or_opened": False,
            "heldout_opened": False,
        }
    )


def write_two_survivor_handoff(
    payload: Mapping[str, Any], directory: Path
) -> tuple[Path, Path]:
    if payload.get("schema_version") != HANDOFF_SCHEMA:
        raise ProbabilisticContractError("two-survivor handoff schema differs")
    verify_payload_seal(payload)
    raw = canonical_json_bytes(payload)
    digest = sha256_bytes(raw)
    handoff_path = Path(directory) / f"two_survivor_handoff.{digest}.json"
    claim_path = Path(directory) / f"two_survivor_handoff.{digest}.claim.json"
    immutable_write_bytes(handoff_path, raw)
    if claim_path.exists():
        raise ProbabilisticContractError("two-survivor handoff was already claimed")
    return handoff_path, claim_path


class VerifiedTwoSurvivorChildPreflight:
    __slots__ = (
        "_authority",
        "_bundle",
        "_claim_path",
        "_handoff_path",
        "_handoff_raw_sha256",
        "_payload",
        "_used",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _PREFLIGHT_TOKEN:
            raise ProbabilisticContractError("child preflight requires the physical loader")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        authority: TwoSurvivorLaunchAuthority,
        bundle: VerifiedTwoSurvivorBundle,
        claim_path: Path,
        handoff_path: Path,
        handoff_raw_sha256: str,
        payload: Mapping[str, Any],
    ) -> None:
        if token is not _PREFLIGHT_TOKEN:
            raise ProbabilisticContractError("invalid child-preflight factory token")
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_bundle", bundle)
        object.__setattr__(self, "_claim_path", Path(claim_path).resolve())
        object.__setattr__(self, "_handoff_path", Path(handoff_path).resolve())
        object.__setattr__(self, "_handoff_raw_sha256", handoff_raw_sha256)
        object.__setattr__(self, "_payload", json.loads(json.dumps(payload)))
        object.__setattr__(self, "_used", False)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedTwoSurvivorChildPreflight is immutable")

    @property
    def bundle(self) -> VerifiedTwoSurvivorBundle:
        self.verify_integrity()
        return self._bundle

    @property
    def authority(self) -> TwoSurvivorLaunchAuthority:
        self.verify_integrity()
        return self._authority

    def verify_integrity(self) -> None:
        self._authority.verify_integrity()
        self._bundle.verify_integrity()
        try:
            raw = self._handoff_path.read_bytes()
            claim_raw = self._claim_path.read_bytes()
            claim = json.loads(claim_raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProbabilisticContractError("child preflight custody is unavailable") from exc
        if sha256_bytes(raw) != self._handoff_raw_sha256:
            raise ProbabilisticContractError("handoff bytes changed after child preflight")
        verify_payload_seal(claim)
        if (
            claim.get("schema_version") != CLAIM_SCHEMA
            or claim.get("handoff_raw_sha256") != self._handoff_raw_sha256
            or claim.get("authority") != authority_binding(self._authority)
            or claim.get("child_pid") != os.getpid()
            or claim.get("single_use") is not True
        ):
            raise ProbabilisticContractError("child claim receipt differs")

    def consume_truth_address(self) -> Mapping[str, str]:
        """Return the frozen address once; the caller still performs child-only loading."""

        self.verify_integrity()
        if self._used:
            raise ProbabilisticContractError("truth preflight is single-use")
        object.__setattr__(self, "_used", True)
        return MappingProxyType(dict(DETACHED_TRUTH_ADDRESS))


def _exclusive_claim(
    path: Path,
    *,
    handoff_raw_sha256: str,
    authority: TwoSurvivorLaunchAuthority,
) -> None:
    payload = seal_payload(
        {
            "schema_version": CLAIM_SCHEMA,
            "handoff_raw_sha256": handoff_raw_sha256,
            "authority": authority_binding(authority),
            "child_pid": os.getpid(),
            "single_use": True,
            "retry_allowed": False,
        }
    )
    raw = canonical_json_bytes(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ProbabilisticContractError("two-survivor handoff replay is forbidden") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Preserve a claim even if receipt persistence fails.  A partial claim
        # blocks retry and therefore remains fail-closed.
        raise


def load_two_survivor_child_preflight(
    handoff_path: Path,
    *,
    expected_handoff_raw_sha256: str,
    claim_path: Path,
    authority: TwoSurvivorLaunchAuthority,
) -> VerifiedTwoSurvivorChildPreflight:
    authority = require_two_survivor_authority(authority)
    expected = require_sha256(expected_handoff_raw_sha256, field="handoff raw")
    handoff_path = Path(handoff_path).resolve()
    try:
        raw = handoff_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("two-survivor handoff is unavailable") from exc
    if sha256_bytes(raw) != expected or expected not in handoff_path.name.split("."):
        raise ProbabilisticContractError("two-survivor handoff differs from external raw pin")
    verify_payload_seal(payload)
    if (
        payload.get("schema_version") != HANDOFF_SCHEMA
        or payload.get("authority") != authority_binding(authority)
        or payload.get("detached_truth_address") != DETACHED_TRUTH_ADDRESS
        or payload.get("truth_address_source")
        != "FROZEN_CONSTANT_NO_PARENT_FILE_ACCESS"
        or payload.get("parent_truth_read") is not False
        or payload.get("child_process_required") is not True
        or payload.get("single_use_claim_required") is not True
        or payload.get("metric_computation_started") is not False
    ):
        raise ProbabilisticContractError("two-survivor handoff policy differs")
    if payload.get("parent_pid") == os.getpid():
        raise ProbabilisticContractError("truth preflight requires a distinct child process")
    bundle_record = payload.get("bundle")
    if not isinstance(bundle_record, dict) or set(bundle_record) != {"path", "raw_sha256"}:
        raise ProbabilisticContractError("handoff bundle record differs")
    bundle_path = (authority.repo_root / bundle_record["path"]).resolve()
    _relative_path(bundle_path, authority.repo_root, context="child bundle")
    bundle = load_two_survivor_bundle(
        bundle_path,
        expected_raw_sha256=bundle_record["raw_sha256"],
        authority=authority,
    )
    claim_path = Path(claim_path).resolve()
    _relative_path(claim_path, authority.repo_root, context="child claim")
    _exclusive_claim(
        claim_path,
        handoff_raw_sha256=expected,
        authority=authority,
    )
    return VerifiedTwoSurvivorChildPreflight(
        _PREFLIGHT_TOKEN,
        authority=authority,
        bundle=bundle,
        claim_path=claim_path,
        handoff_path=handoff_path,
        handoff_raw_sha256=expected,
        payload=payload,
    )


def require_two_survivor_child_preflight(
    value: object,
) -> VerifiedTwoSurvivorChildPreflight:
    if not isinstance(value, VerifiedTwoSurvivorChildPreflight):
        raise ProbabilisticContractError(
            "truth address requires exact two-survivor child preflight"
        )
    value.verify_integrity()
    return value

