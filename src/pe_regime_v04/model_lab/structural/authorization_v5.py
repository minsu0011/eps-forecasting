"""Isolated V5 authorization loader; V4 authority bytes remain untouched."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import authorization as core
from .contracts import STRUCTURAL_DESIGN_SHA256, canonical_json_bytes, sha256_bytes, sha256_file


AUTHORITY_POLICY_V5_NAME = "authority_policy_v5.py"


def load_structural_execution_authorization_v5(
    project_root: Path,
    *,
    external_policy_sha256: str,
    scope: core.AuthorizationScope = "SYNTHETIC_NO_SCORE",
) -> core.StructuralExecutionAuthorization:
    """Mint V5 capability from the new policy while preserving all V4 base locks."""

    if scope not in ("SYNTHETIC_NO_SCORE", "FORMAL_SPENT"):
        raise core.StructuralContractError("unknown structural authorization scope")
    root = Path(project_root).resolve()
    decision_path = (
        root / "outputs/model_zoo_structural_wave_screen_20260819/TRIGGER_DECISION.json"
    ).resolve()
    base_path = (
        root / f"outputs/model_zoo_structural_wave_screen_20260819/{core.BASE_BINDING_LOCK_NAME}"
    ).resolve()
    snapshot_path = (
        root / f"outputs/model_zoo_structural_wave_screen_20260819/{core.EXECUTION_SNAPSHOT_NAME}"
    ).resolve()
    policy_path = (
        root / f"src/pe_regime_v04/model_lab/structural/{AUTHORITY_POLICY_V5_NAME}"
    ).resolve()
    policy = core._load_authority_policy(
        policy_path,
        expected_raw_sha256=external_policy_sha256,
    )
    activation_path = (root / str(policy.get("activation_relative_path", ""))).resolve()
    for path in (decision_path, base_path, snapshot_path, policy_path, activation_path):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise core.StructuralContractError(
                "V5 authorization path escapes project root"
            ) from exc
    if sha256_file(decision_path) != core.TRIGGER_DECISION_FILE_SHA256:
        raise core.StructuralContractError("TRIGGER_DECISION.json file SHA-256 changed")
    decision = core._read_json(decision_path, context="trigger decision")
    base_lock = core._read_json(base_path, context="base-binding lock")
    snapshot = core._read_json(snapshot_path, context="execution snapshot")
    activation = core._read_json(activation_path, context="V5 audit activation")
    core._verify_decision(decision)
    core._verify_base_lock(base_lock)
    core._verify_snapshot(snapshot, root)
    core._verify_authority_policy(
        policy,
        trigger=decision,
        base_lock=base_lock,
        snapshot=snapshot,
        activation=activation,
        project_root=root,
    )
    residual = core.BoundBaseIdentity._from_mapping(base_lock["residual_base"])
    pair = tuple(core.BoundBaseIdentity._from_mapping(item) for item in base_lock["ensemble_pair"])
    spent = policy.get("formal_spent_activated") is True
    if scope == "FORMAL_SPENT" and not spent:
        raise core.StructuralContractError(
            "V5 formal spent execution is blocked pending independent audit GO"
        )
    values: dict[str, Any] = {
        "scope": scope,
        "project_root": root,
        "trigger_decision_path": decision_path,
        "trigger_decision_sha256": core.TRIGGER_DECISION_FILE_SHA256,
        "trigger_decision_logical_sha256": str(decision["manifest_sha256"]),
        "base_binding_lock_path": base_path,
        "base_binding_lock_sha256": sha256_file(base_path),
        "base_binding_lock_logical_sha256": str(base_lock["manifest_sha256"]),
        "execution_snapshot_path": snapshot_path,
        "execution_snapshot_sha256": sha256_file(snapshot_path),
        "execution_snapshot_logical_sha256": str(snapshot["manifest_sha256"]),
        "authority_policy_path": policy_path,
        "authority_policy_sha256": external_policy_sha256,
        "authority_policy_logical_sha256": str(policy["manifest_sha256"]),
        "audit_activation_path": activation_path,
        "audit_activation_sha256": sha256_file(activation_path),
        "audit_activation_logical_sha256": str(activation["manifest_sha256"]),
        "enabled_candidates": core.EXPECTED_ENABLED,
        "disabled_candidates": core.EXPECTED_DISABLED,
        "residual_base": residual,
        "pair": (pair[0], pair[1]),
        "spent_execution_authorized": spent,
    }
    unsigned = {
        "format_version": 1,
        "scope": scope,
        "design_sha256": STRUCTURAL_DESIGN_SHA256,
        "trigger_decision_sha256": values["trigger_decision_sha256"],
        "trigger_decision_logical_sha256": values["trigger_decision_logical_sha256"],
        "base_binding_lock_sha256": values["base_binding_lock_sha256"],
        "base_binding_lock_logical_sha256": values["base_binding_lock_logical_sha256"],
        "execution_snapshot_sha256": values["execution_snapshot_sha256"],
        "execution_snapshot_logical_sha256": values["execution_snapshot_logical_sha256"],
        "authority_policy_sha256": values["authority_policy_sha256"],
        "authority_policy_logical_sha256": values["authority_policy_logical_sha256"],
        "audit_activation_sha256": values["audit_activation_sha256"],
        "audit_activation_logical_sha256": values["audit_activation_logical_sha256"],
        "enabled_candidates": list(core.EXPECTED_ENABLED),
        "disabled_candidates": list(core.EXPECTED_DISABLED),
        "residual_base": residual.as_dict(),
        "pair": [item.as_dict() for item in pair],
        "spent_execution_authorized": spent,
    }
    values["authorization_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    output = core.StructuralExecutionAuthorization._mint(token=core._AUTHORIZATION_TOKEN, **values)
    output.verify()
    return output
