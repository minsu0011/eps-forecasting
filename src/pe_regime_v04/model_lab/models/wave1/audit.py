"""Machine-verifiable independent-audit trust record for Wave-1."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from ...contracts import ContractError
from .artifacts import canonical_json_bytes, load_sealed_json, verify_file_record
from .spec import DESIGN_LOCK_SHA256, EVIDENCE_SEEDS


class AuditTrustError(ContractError):
    """Raised when an audit candidate or independent GO record is not trustworthy."""


AUDIT_CHECK_NAMES = (
    "wave1_targeted_pytest",
    "full_pytest",
    "ruff",
    "py_compile",
)
AUDIT_APPROVAL_SCOPE = (
    "registry_definition_append",
    "execution_binding",
    "execution_precommit",
    "predict",
    "evaluate",
    "registry_stage1_result_append",
)
MUTABLE_REGISTRY_RELATIVE_PATHS = (
    "research/model_zoo/feature_registry.csv",
    "research/model_zoo/feature_registry.json",
    "research/model_zoo/model_registry.csv",
    "research/model_zoo/model_registry.json",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_AUDIT_CANDIDATE_FIELDS = {
    "format_version",
    "mode",
    "design_lock_sha256",
    "audit_scope",
    "runtime_inventory",
    "runtime_inventory_sha256",
    "checks",
    "mg1_schema_check",
    "candidate_scores_seen",
    "candidate_predictions_run",
    "fresh_or_heldout_opened",
    "execution_binding_written",
    "execution_precommit_written",
    "registries_mutated",
    "manifest_sha256",
}
_AUDIT_GO_FIELDS = {
    "format_version",
    "mode",
    "decision",
    "auditor_role",
    "approval_scope",
    "audit_candidate",
    "audit_candidate_manifest_sha256",
    "runtime_inventory_sha256",
    "candidate_scores_seen",
    "candidate_predictions_run",
    "fresh_or_heldout_opened",
    "manifest_sha256",
}


@dataclass(frozen=True)
class VerifiedIndependentAudit:
    go: dict[str, Any]
    candidate: dict[str, Any]
    candidate_path: Path


def audit_inventory_sha256(inventory: Iterable[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        canonical_json_bytes([dict(item) for item in inventory])
    ).hexdigest()


def _path_under_root(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuditTrustError(f"audit inventory path is not relative: {relative_path!r}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AuditTrustError(
            f"audit inventory path escapes the project root: {relative_path!r}"
        ) from exc
    return candidate


def verify_audit_candidate(
    path: Path,
    *,
    project_root: Path,
    permitted_current_drift: Iterable[str] = (),
) -> dict[str, Any]:
    """Verify the sealed no-score audit candidate and every non-exempt live byte."""

    try:
        candidate = load_sealed_json(path, expected_mode="wave1_audit_candidate")
    except ContractError as exc:
        raise AuditTrustError(f"audit candidate seal is invalid: {exc}") from exc
    if set(candidate) != _AUDIT_CANDIDATE_FIELDS or candidate["format_version"] != 1:
        raise AuditTrustError("audit candidate fields/format are invalid")
    expected_state = {
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "audit_scope": "WAVE1_NO_SCORE_PRE_EXECUTION",
        "candidate_scores_seen": False,
        "candidate_predictions_run": False,
        "fresh_or_heldout_opened": False,
        "execution_binding_written": False,
        "execution_precommit_written": False,
        "registries_mutated": False,
    }
    if any(candidate.get(key) != value for key, value in expected_state.items()):
        raise AuditTrustError("audit candidate state declarations are invalid")
    inventory = candidate["runtime_inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise AuditTrustError("audit candidate runtime inventory is empty")
    if candidate["runtime_inventory_sha256"] != audit_inventory_sha256(inventory):
        raise AuditTrustError("audit candidate runtime inventory SHA-256 differs")
    permitted = set(permitted_current_drift)
    if not permitted.issubset(MUTABLE_REGISTRY_RELATIVE_PATHS):
        raise AuditTrustError("audit verifier was given an unsupported drift exemption")
    labels: list[str] = []
    root = Path(project_root).resolve(strict=True)
    for item in inventory:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path",
            "bytes",
            "sha256",
        }:
            raise AuditTrustError("audit inventory entry fields are invalid")
        relative_path = item["relative_path"]
        if not isinstance(relative_path, str) or not relative_path:
            raise AuditTrustError("audit inventory relative_path is invalid")
        labels.append(relative_path)
        if (
            isinstance(item["bytes"], bool)
            or not isinstance(item["bytes"], int)
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or _SHA256.fullmatch(item["sha256"]) is None
        ):
            raise AuditTrustError(f"audit inventory digest is invalid: {relative_path}")
        if relative_path in permitted:
            continue
        live = _path_under_root(root, relative_path)
        if not live.is_file():
            raise AuditTrustError(f"audited live file is missing: {relative_path}")
        if (
            live.stat().st_size != item["bytes"]
            or hashlib.sha256(live.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise AuditTrustError(f"audited live file drifted: {relative_path}")
    if labels != sorted(labels) or len(labels) != len(set(labels)):
        raise AuditTrustError("audit inventory must have unique sorted paths")
    checks = candidate["checks"]
    if (
        not isinstance(checks, list)
        or not all(isinstance(item, Mapping) for item in checks)
        or tuple(item.get("name") for item in checks) != AUDIT_CHECK_NAMES
    ):
        raise AuditTrustError("audit candidate check universe/order differs")
    for item in checks:
        if not isinstance(item, Mapping) or set(item) != {
            "name",
            "command",
            "exit_code",
            "stdout_sha256",
            "stderr_sha256",
            "stdout_base64",
            "stderr_base64",
            "stdout_tail",
            "stderr_tail",
            "summary",
        }:
            raise AuditTrustError("audit check fields are invalid")
        if item["exit_code"] != 0 or item["summary"] != "PASS":
            raise AuditTrustError(f"audit check did not pass: {item.get('name')}")
        if any(
            not isinstance(item[key], str) or not item[key]
            for key in ("command", "stdout_sha256", "stderr_sha256")
        ) or any(
            not isinstance(item[key], str)
            for key in (
                "stdout_base64",
                "stderr_base64",
                "stdout_tail",
                "stderr_tail",
            )
        ):
            raise AuditTrustError("audit check command/digests are invalid")
        if any(
            _SHA256.fullmatch(item[key]) is None
            for key in ("stdout_sha256", "stderr_sha256")
        ):
            raise AuditTrustError("audit check output digest is not SHA-256")
        try:
            stdout = base64.b64decode(item["stdout_base64"], validate=True)
            stderr = base64.b64decode(item["stderr_base64"], validate=True)
        except (ValueError, binascii.Error) as exc:
            raise AuditTrustError("audit check output is not canonical base64") from exc
        if (
            base64.b64encode(stdout).decode("ascii") != item["stdout_base64"]
            or base64.b64encode(stderr).decode("ascii") != item["stderr_base64"]
        ):
            raise AuditTrustError("audit check output base64 encoding is not canonical")
        if (
            hashlib.sha256(stdout).hexdigest() != item["stdout_sha256"]
            or hashlib.sha256(stderr).hexdigest() != item["stderr_sha256"]
        ):
            raise AuditTrustError("audit check embedded output differs from its digest")
    mg1 = candidate["mg1_schema_check"]
    if not isinstance(mg1, Mapping) or mg1.get("status") != "PASS":
        raise AuditTrustError("MG1 no-score schema check did not pass")
    if tuple(mg1.get("seeds", ())) != EVIDENCE_SEEDS:
        raise AuditTrustError("MG1 no-score schema check seed universe differs")
    if (
        mg1.get("candidate_predictions_run") is not False
        or mg1.get("candidate_scores_seen") is not False
        or mg1.get("fresh_or_heldout_opened") is not False
    ):
        raise AuditTrustError("MG1 schema evidence claims prohibited execution/access")
    return candidate


def verify_independent_audit_go(
    path: Path,
    *,
    project_root: Path,
    permitted_current_drift: Iterable[str] = (),
) -> VerifiedIndependentAudit:
    """Verify a canonical GO bound to the exact audited Wave1 snapshot."""

    try:
        go = load_sealed_json(path, expected_mode="wave1_independent_audit_go")
    except ContractError as exc:
        raise AuditTrustError(f"independent audit GO seal is invalid: {exc}") from exc
    if set(go) != _AUDIT_GO_FIELDS or go["format_version"] != 1:
        raise AuditTrustError("independent audit GO fields/format are invalid")
    expected = {
        "decision": "GO",
        "auditor_role": "independent_wave1_auditor",
        "approval_scope": list(AUDIT_APPROVAL_SCOPE),
        "candidate_scores_seen": False,
        "candidate_predictions_run": False,
        "fresh_or_heldout_opened": False,
    }
    if any(go.get(key) != value for key, value in expected.items()):
        raise AuditTrustError("independent audit GO scope/state is invalid")
    try:
        candidate_path = verify_file_record(
            go["audit_candidate"], context="independent-audit candidate"
        )
    except ContractError as exc:
        raise AuditTrustError(
            f"independent audit candidate file record is invalid: {exc}"
        ) from exc
    candidate = verify_audit_candidate(
        candidate_path,
        project_root=project_root,
        permitted_current_drift=permitted_current_drift,
    )
    if (
        go["audit_candidate_manifest_sha256"] != candidate["manifest_sha256"]
        or go["runtime_inventory_sha256"] != candidate["runtime_inventory_sha256"]
    ):
        raise AuditTrustError("independent audit GO does not bind the exact audit candidate")
    return VerifiedIndependentAudit(
        go=go,
        candidate=candidate,
        candidate_path=candidate_path,
    )
