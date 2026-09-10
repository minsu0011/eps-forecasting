"""Opaque, content-addressed authorization for Structural Wave entry points.

The trigger decision is necessary but deliberately not sufficient.  A kernel
may run only when the decision, exact base bindings, and the execution source
snapshot all still match their sealed bytes.  The currently permitted scope is
``SYNTHETIC_NO_SCORE``; spent-data execution remains fail-closed until a later
independent audit explicitly authorizes the exact snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping

from .contracts import (
    STRUCTURAL_DESIGN_SHA256,
    StructuralContractError,
    canonical_json_bytes,
    require_sha256,
    sha256_bytes,
    sha256_file,
    verify_payload_seal,
)


TRIGGER_DECISION_FILE_SHA256 = "45b6dee894ed72809c49902f4f78ff058a465db6ebfe8903a21c659883504765"
TRIGGER_DECISION_LOGICAL_SHA256 = "a3440ae5a3aa81767738d59e4f60ae391e0034d6e223c6f8513089042c1b7023"
EXPECTED_ENABLED = (
    "decomp_block_ridge_ar1_lag1",
    "decomp_block_ridge_ar1_current",
    "residual_ar1_nested_oof",
    "stack_geometric_equal_pair",
    "stack_simplex_pair_frozen",
)
EXPECTED_DISABLED = ("residual_huber_nested_oof",)
EXPECTED_RESIDUAL_BASE = "v04_expected_pe"
EXPECTED_PAIR = ("xgboost_cpu_common", "spline_ridge_common")
EXPECTED_TRIGGER_TUPLE = {
    "T_A_TRACK_GAP": True,
    "T_DIRECT_PLATEAU_OR_TAIL": True,
    "T_COMPLEMENTARITY": True,
    "T_STABLE_BIAS": False,
    "T_SERIAL_RESIDUAL": True,
}

AuthorizationScope = Literal["SYNTHETIC_NO_SCORE", "FORMAL_SPENT"]
_AUTHORIZATION_TOKEN = object()
BASE_BINDING_LOCK_NAME = "BASE_BINDING_LOCK_V4.json"
EXECUTION_SNAPSHOT_NAME = "EXECUTION_SNAPSHOT_V4.json"
AUTHORITY_POLICY_NAME = "authority_policy_v4.py"


@dataclass(frozen=True)
class BoundBaseIdentity:
    """Exact replay identity loaded from the sealed base-binding lock."""

    model_id: str
    family: str
    track: str
    decision_cutoff: str
    source_sha256: str
    config_sha256: str
    environment_sha256: str
    feature_registry_sha256: str
    feature_ids: tuple[str, ...]
    replay_contract_sha256: str

    def __post_init__(self) -> None:
        if not self.model_id or not self.family or not self.track or not self.decision_cutoff:
            raise StructuralContractError("base execution identity has an empty semantic field")
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise StructuralContractError("base execution feature ids must be unique")
        for field in (
            "source_sha256",
            "config_sha256",
            "environment_sha256",
            "feature_registry_sha256",
            "replay_contract_sha256",
        ):
            require_sha256(getattr(self, field), field=field)

    @classmethod
    def _from_mapping(cls, value: Mapping[str, Any]) -> "BoundBaseIdentity":
        required = {
            "model_id",
            "family",
            "track",
            "decision_cutoff",
            "source_sha256",
            "config_sha256",
            "environment_sha256",
            "feature_registry_sha256",
            "feature_ids",
            "replay_contract_sha256",
        }
        if set(value) != required or not isinstance(value.get("feature_ids"), list):
            raise StructuralContractError("base execution binding schema changed")
        return cls(
            model_id=str(value["model_id"]),
            family=str(value["family"]),
            track=str(value["track"]),
            decision_cutoff=str(value["decision_cutoff"]),
            source_sha256=str(value["source_sha256"]),
            config_sha256=str(value["config_sha256"]),
            environment_sha256=str(value["environment_sha256"]),
            feature_registry_sha256=str(value["feature_registry_sha256"]),
            feature_ids=tuple(str(item) for item in value["feature_ids"]),
            replay_contract_sha256=str(value["replay_contract_sha256"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "track": self.track,
            "decision_cutoff": self.decision_cutoff,
            "source_sha256": self.source_sha256,
            "config_sha256": self.config_sha256,
            "environment_sha256": self.environment_sha256,
            "feature_registry_sha256": self.feature_registry_sha256,
            "feature_ids": list(self.feature_ids),
            "replay_contract_sha256": self.replay_contract_sha256,
        }


@dataclass(frozen=True, init=False)
class StructuralExecutionAuthorization:
    """Factory-only capability checked again at every formal kernel call."""

    scope: AuthorizationScope
    project_root: Path
    trigger_decision_path: Path
    trigger_decision_sha256: str
    trigger_decision_logical_sha256: str
    base_binding_lock_path: Path
    base_binding_lock_sha256: str
    base_binding_lock_logical_sha256: str
    execution_snapshot_path: Path
    execution_snapshot_sha256: str
    execution_snapshot_logical_sha256: str
    authority_policy_path: Path
    authority_policy_sha256: str
    authority_policy_logical_sha256: str
    audit_activation_path: Path
    audit_activation_sha256: str
    audit_activation_logical_sha256: str
    enabled_candidates: tuple[str, ...]
    disabled_candidates: tuple[str, ...]
    residual_base: BoundBaseIdentity
    pair: tuple[BoundBaseIdentity, BoundBaseIdentity]
    spent_execution_authorized: bool
    authorization_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError(
            "StructuralExecutionAuthorization is factory-only; load sealed evidence"
        )

    @classmethod
    def _mint(cls, *, token: object, **values: Any) -> "StructuralExecutionAuthorization":
        if token is not _AUTHORIZATION_TOKEN:
            raise StructuralContractError("invalid structural authorization mint token")
        output = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(output, name, value)
        return output

    def _unsigned_binding(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "scope": self.scope,
            "design_sha256": STRUCTURAL_DESIGN_SHA256,
            "trigger_decision_sha256": self.trigger_decision_sha256,
            "trigger_decision_logical_sha256": self.trigger_decision_logical_sha256,
            "base_binding_lock_sha256": self.base_binding_lock_sha256,
            "base_binding_lock_logical_sha256": self.base_binding_lock_logical_sha256,
            "execution_snapshot_sha256": self.execution_snapshot_sha256,
            "execution_snapshot_logical_sha256": self.execution_snapshot_logical_sha256,
            "authority_policy_sha256": self.authority_policy_sha256,
            "authority_policy_logical_sha256": self.authority_policy_logical_sha256,
            "audit_activation_sha256": self.audit_activation_sha256,
            "audit_activation_logical_sha256": self.audit_activation_logical_sha256,
            "enabled_candidates": list(self.enabled_candidates),
            "disabled_candidates": list(self.disabled_candidates),
            "residual_base": self.residual_base.as_dict(),
            "pair": [item.as_dict() for item in self.pair],
            "spent_execution_authorized": self.spent_execution_authorized,
        }

    def verify(self) -> None:
        """Re-open all seals and source inventory; stale capabilities are invalid."""

        if sha256_file(self.trigger_decision_path) != self.trigger_decision_sha256:
            raise StructuralContractError("sealed trigger decision bytes changed")
        if sha256_file(self.base_binding_lock_path) != self.base_binding_lock_sha256:
            raise StructuralContractError("sealed base-binding lock bytes changed")
        if sha256_file(self.execution_snapshot_path) != self.execution_snapshot_sha256:
            raise StructuralContractError("sealed execution snapshot bytes changed")
        if sha256_file(self.authority_policy_path) != self.authority_policy_sha256:
            raise StructuralContractError("externally pinned authority policy bytes changed")
        if sha256_file(self.audit_activation_path) != self.audit_activation_sha256:
            raise StructuralContractError("pinned independent audit activation bytes changed")
        decision = _read_json(self.trigger_decision_path, context="trigger decision")
        base_lock = _read_json(self.base_binding_lock_path, context="base-binding lock")
        snapshot = _read_json(self.execution_snapshot_path, context="execution snapshot")
        policy = _load_authority_policy(
            self.authority_policy_path,
            expected_raw_sha256=self.authority_policy_sha256,
        )
        activation = _read_json(self.audit_activation_path, context="audit activation")
        _verify_decision(decision)
        _verify_base_lock(base_lock)
        _verify_snapshot(snapshot, self.project_root)
        _verify_authority_policy(
            policy,
            trigger=decision,
            base_lock=base_lock,
            snapshot=snapshot,
            activation=activation,
            project_root=self.project_root,
        )
        if decision["manifest_sha256"] != self.trigger_decision_logical_sha256:
            raise StructuralContractError("trigger decision logical seal changed")
        if base_lock["manifest_sha256"] != self.base_binding_lock_logical_sha256:
            raise StructuralContractError("base-binding lock logical seal changed")
        if snapshot["manifest_sha256"] != self.execution_snapshot_logical_sha256:
            raise StructuralContractError("execution snapshot logical seal changed")
        if policy["manifest_sha256"] != self.authority_policy_logical_sha256:
            raise StructuralContractError("authority policy logical seal changed")
        if activation["manifest_sha256"] != self.audit_activation_logical_sha256:
            raise StructuralContractError("audit activation logical seal changed")
        if (
            sha256_bytes(canonical_json_bytes(self._unsigned_binding()))
            != self.authorization_sha256
        ):
            raise StructuralContractError("structural authorization content changed")
        if self.scope == "FORMAL_SPENT" and not self.spent_execution_authorized:
            raise StructuralContractError(
                "formal spent execution is blocked pending repeat independent audit GO"
            )

    def require_candidate(self, candidate_id: str) -> None:
        self.verify()
        if candidate_id not in self.enabled_candidates:
            if candidate_id == "residual_huber_nested_oof":
                raise StructuralContractError("disabled Huber candidate is uncallable")
            raise StructuralContractError(f"candidate is not enabled: {candidate_id}")

    def require_residual_base(self, candidate_id: str, base_model_id: str) -> BoundBaseIdentity:
        self.require_candidate(candidate_id)
        if candidate_id != "residual_ar1_nested_oof":
            raise StructuralContractError("only the enabled AR1 residual candidate is callable")
        if base_model_id != EXPECTED_RESIDUAL_BASE or base_model_id != self.residual_base.model_id:
            raise StructuralContractError("residual base must be exactly v04_expected_pe")
        return self.residual_base

    def require_pair(
        self, candidate_id: str, base0_model_id: str, base1_model_id: str
    ) -> tuple[BoundBaseIdentity, BoundBaseIdentity]:
        self.require_candidate(candidate_id)
        if candidate_id not in {
            "stack_geometric_equal_pair",
            "stack_simplex_pair_frozen",
        }:
            raise StructuralContractError("candidate is not an authorized ensemble")
        actual = (base0_model_id, base1_model_id)
        if actual != EXPECTED_PAIR or actual != tuple(item.model_id for item in self.pair):
            raise StructuralContractError(
                "ensemble pair/order must be exactly xgboost_cpu_common+spline_ridge_common"
            )
        return self.pair

    def require_formal_spent_go(self) -> None:
        self.verify()
        if self.scope != "FORMAL_SPENT" or not self.spent_execution_authorized:
            raise StructuralContractError(
                "spent-data execution is blocked pending repeat independent audit GO"
            )


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralContractError(f"cannot read {context}: {path}") from exc
    if not isinstance(value, dict):
        raise StructuralContractError(f"{context} must be a JSON object")
    return value


def _verify_decision(value: Mapping[str, Any]) -> None:
    verify_payload_seal(value)
    if value.get("manifest_sha256") != TRIGGER_DECISION_LOGICAL_SHA256:
        raise StructuralContractError("trigger decision is not the frozen final resolution")
    design = value.get("structural_design")
    if not isinstance(design, Mapping) or design.get("sha256") != STRUCTURAL_DESIGN_SHA256:
        raise StructuralContractError("trigger decision design changed")
    if value.get("triggers") != EXPECTED_TRIGGER_TUPLE:
        raise StructuralContractError("trigger decision tuple changed")
    if tuple(value.get("enabled_candidates", ())) != EXPECTED_ENABLED:
        raise StructuralContractError("trigger decision enabled set/order changed")
    if tuple(value.get("disabled_candidates", ())) != EXPECTED_DISABLED:
        raise StructuralContractError("trigger decision disabled set changed")
    if value.get("selected_residual_base_model_id") != EXPECTED_RESIDUAL_BASE:
        raise StructuralContractError("trigger decision residual base changed")
    selected_pair = value.get("selected_complementary_pair")
    if (
        not isinstance(selected_pair, Mapping)
        or (selected_pair.get("primary_model_id"), selected_pair.get("secondary_model_id"))
        != EXPECTED_PAIR
    ):
        raise StructuralContractError("trigger decision selected pair/order changed")


def _verify_base_lock(value: Mapping[str, Any]) -> None:
    verify_payload_seal(value)
    if value.get("mode") != "structural_exact_base_execution_binding_lock":
        raise StructuralContractError("base-binding lock mode changed")
    if value.get("design_sha256") != STRUCTURAL_DESIGN_SHA256:
        raise StructuralContractError("base-binding lock design changed")
    residual = value.get("residual_base")
    pair = value.get("ensemble_pair")
    if not isinstance(residual, Mapping) or not isinstance(pair, list) or len(pair) != 2:
        raise StructuralContractError("base-binding lock payload is incomplete")
    residual_binding = BoundBaseIdentity._from_mapping(residual)
    pair_bindings = tuple(BoundBaseIdentity._from_mapping(item) for item in pair)
    if residual_binding.model_id != EXPECTED_RESIDUAL_BASE:
        raise StructuralContractError("base-binding lock residual base changed")
    if tuple(item.model_id for item in pair_bindings) != EXPECTED_PAIR:
        raise StructuralContractError("base-binding lock pair/order changed")
    evidence = value.get("binding_evidence")
    if not isinstance(evidence, Mapping):
        raise StructuralContractError("base-binding lock execution evidence is missing")
    if "dgp_execution_contract" in evidence:
        raise StructuralContractError("unrelated mutable DGP contract is forbidden")
    required_evidence = {
        "predict_inputs",
        "wave1_execution_binding",
        "model_registry_snapshot",
        "feature_registry_snapshot",
        "environment_snapshot",
        "residual_v04_replay",
        "wave1_pair_replay",
        "spent_surfaces",
    }
    if set(evidence) != required_evidence:
        raise StructuralContractError("base-binding lock evidence schema changed")


def _verify_snapshot(value: Mapping[str, Any], project_root: Path) -> None:
    verify_payload_seal(value)
    if value.get("mode") != "structural_content_addressed_execution_snapshot":
        raise StructuralContractError("execution snapshot mode changed")
    if value.get("design_sha256") != STRUCTURAL_DESIGN_SHA256:
        raise StructuralContractError("execution snapshot design changed")
    if value.get("trigger_decision_sha256") != TRIGGER_DECISION_FILE_SHA256:
        raise StructuralContractError("execution snapshot trigger binding changed")
    inventory = value.get("source_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise StructuralContractError("execution snapshot source inventory is missing")
    seen: set[str] = set()
    for row in inventory:
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise StructuralContractError("execution snapshot inventory schema changed")
        relative = str(row["path"]).replace("\\", "/")
        if relative in seen or relative.startswith("/") or ".." in Path(relative).parts:
            raise StructuralContractError("execution snapshot inventory path is unsafe")
        seen.add(relative)
        path = project_root / Path(relative)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise StructuralContractError(f"snapshotted source is missing: {relative}") from exc
        if size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise StructuralContractError(f"snapshotted source changed: {relative}")
    mandatory = {
        "src/pe_regime_v04/model_lab/structural/authorization.py",
        "src/pe_regime_v04/model_lab/structural/contracts.py",
        "src/pe_regime_v04/model_lab/structural/decomposition.py",
        "src/pe_regime_v04/model_lab/structural/dispatcher.py",
        "src/pe_regime_v04/model_lab/structural/ensemble.py",
        "src/pe_regime_v04/model_lab/structural/meta.py",
        "src/pe_regime_v04/model_lab/structural/nested.py",
        "src/pe_regime_v04/model_lab/structural/replay.py",
        "src/pe_regime_v04/model_lab/structural/residuals.py",
        "src/pe_regime_v04/model_lab/structural/resources.py",
        "src/pe_regime_v04/model_lab/structural/runner.py",
        "src/pe_regime_v04/model_lab/structural/track_a_audit.py",
        "src/pe_regime_v04/model_lab/contracts.py",
        "src/pe_regime_v04/model_lab/folds.py",
        "src/pe_regime_v04/model_lab/models/wave1/adapters.py",
        "src/pe_regime_v04/model_lab/models/wave1/artifacts.py",
        "src/pe_regime_v04/model_lab/models/wave1/spec.py",
        "src/pe_regime_v04/pipeline.py",
        "config/v04_bottleneck.yaml",
        "outputs/model_zoo_wave1_screen_20260819/PREDICT_INPUTS.json",
        "outputs/model_zoo_structural_wave_screen_20260819/BASE_BINDING_LOCK_V4.json",
        "outputs/model_zoo_structural_wave_screen_20260819/TRACK_A_DATA_BOUND_AUDIT.json",
        "outputs/model_zoo_structural_wave_screen_20260819/NO_SCORE_BACKEND_BENCHMARK.json",
    }
    if not mandatory.issubset(seen):
        raise StructuralContractError("execution snapshot omits a mandatory execution dependency")
    forbidden = {
        "research/model_zoo/dgp_suite/EXECUTION_CONTRACT.json",
        "research/model_zoo/dgp_suite/RUNTIME_POLICY.json",
        "research/model_zoo/dgp_suite/RUNTIME_MANIFEST.json",
    }
    if forbidden.intersection(seen):
        raise StructuralContractError("execution snapshot includes unrelated mutable DGP state")
    if value.get("dependency_scope") != ("TRANSITIVE_STRUCTURAL_V04_PHASE1_WAVE1_RUNTIME_ONLY"):
        raise StructuralContractError("execution snapshot dependency scope changed")
    if value.get("authority_policy_layer") != ("EXTERNAL_RAW_SHA_PIN_REQUIRED_NOT_IN_SNAPSHOT"):
        raise StructuralContractError("execution snapshot authority-policy layer changed")
    if "src/pe_regime_v04/model_lab/structural/authority_policy_v4.py" in seen:
        raise StructuralContractError("acyclic external policy must not self-enter snapshot")
    if value.get("spent_execution_authorized") is not False:
        raise StructuralContractError("pre-audit snapshot must not authorize spent-data execution")


def _load_authority_policy(path: Path, *, expected_raw_sha256: str) -> dict[str, Any]:
    """Load one externally pinned generated module from the exact verified bytes."""

    require_sha256(expected_raw_sha256, field="external_policy_sha256")
    try:
        encoded = Path(path).read_bytes()
    except OSError as exc:
        raise StructuralContractError("authority policy module is unreadable") from exc
    if sha256_bytes(encoded) != expected_raw_sha256:
        raise StructuralContractError("authority policy differs from the external audit pin")
    namespace: dict[str, Any] = {"__builtins__": {}}
    try:
        exec(compile(encoded, str(path), "exec"), namespace, namespace)  # noqa: S102
    except Exception as exc:
        raise StructuralContractError("authority policy module cannot be loaded") from exc
    value = namespace.get("AUTHORITY_POLICY")
    if not isinstance(value, dict):
        raise StructuralContractError("authority policy module has no policy object")
    return value


def _verify_authority_policy(
    policy: Mapping[str, Any],
    *,
    trigger: Mapping[str, Any],
    base_lock: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    activation: Mapping[str, Any],
    project_root: Path,
) -> None:
    verify_payload_seal(policy)
    verify_payload_seal(activation)
    if policy.get("format_version") != 4 or policy.get("mode") != (
        "structural_v4_two_layer_external_audit_pinned_authority"
    ):
        raise StructuralContractError("authority policy schema/mode changed")
    expected = {
        "trigger_decision": (
            project_root
            / "outputs/model_zoo_structural_wave_screen_20260819/TRIGGER_DECISION.json",
            trigger,
        ),
        "base_binding_lock_v4": (
            project_root
            / "outputs/model_zoo_structural_wave_screen_20260819/BASE_BINDING_LOCK_V4.json",
            base_lock,
        ),
        "execution_snapshot_v4": (
            project_root
            / "outputs/model_zoo_structural_wave_screen_20260819/EXECUTION_SNAPSHOT_V4.json",
            snapshot,
        ),
        "independent_audit_activation": (
            project_root / str(policy.get("activation_relative_path", "")),
            activation,
        ),
    }
    bindings = policy.get("exact_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected):
        raise StructuralContractError("authority policy exact bindings changed")
    for name, (path, payload) in expected.items():
        row = bindings[name]
        if not isinstance(row, Mapping) or set(row) != {"raw_sha256", "logical_sha256"}:
            raise StructuralContractError("authority policy binding schema changed")
        if row["raw_sha256"] != sha256_file(path):
            raise StructuralContractError(f"authority policy raw binding changed: {name}")
        if row["logical_sha256"] != payload.get("manifest_sha256"):
            raise StructuralContractError(f"authority policy logical binding changed: {name}")
    if policy.get("source_inventory_sha256") != snapshot.get("source_inventory_sha256"):
        raise StructuralContractError("authority policy source-closure binding changed")
    if policy.get("formal_spent_activated") is True:
        decision = activation.get("decision")
        if (
            not isinstance(decision, Mapping)
            or decision.get("spent_seed_screen") != "GO"
            or decision.get("open_p0") != 0
            or decision.get("open_p1") != 0
        ):
            raise StructuralContractError("formal activation lacks independent zero-P0/P1 GO")
    elif policy.get("formal_spent_activated") is not False:
        raise StructuralContractError("authority policy activation flag is invalid")


def load_structural_execution_authorization(
    project_root: Path,
    *,
    external_policy_sha256: str,
    scope: AuthorizationScope = "SYNTHETIC_NO_SCORE",
) -> StructuralExecutionAuthorization:
    """Mint an authorization only after revalidating every immutable dependency."""

    if scope not in ("SYNTHETIC_NO_SCORE", "FORMAL_SPENT"):
        raise StructuralContractError("unknown structural authorization scope")
    root = Path(project_root).resolve()
    decision_path = (
        root / "outputs/model_zoo_structural_wave_screen_20260819/TRIGGER_DECISION.json"
    ).resolve()
    base_path = (
        root / f"outputs/model_zoo_structural_wave_screen_20260819/{BASE_BINDING_LOCK_NAME}"
    ).resolve()
    snapshot_path = (
        root / f"outputs/model_zoo_structural_wave_screen_20260819/{EXECUTION_SNAPSHOT_NAME}"
    ).resolve()
    policy_path = (
        root / f"src/pe_regime_v04/model_lab/structural/{AUTHORITY_POLICY_NAME}"
    ).resolve()
    policy = _load_authority_policy(
        policy_path,
        expected_raw_sha256=external_policy_sha256,
    )
    activation_path = (root / str(policy.get("activation_relative_path", ""))).resolve()
    for path in (decision_path, base_path, snapshot_path, policy_path, activation_path):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise StructuralContractError("authorization path escapes project root") from exc
    if sha256_file(decision_path) != TRIGGER_DECISION_FILE_SHA256:
        raise StructuralContractError("TRIGGER_DECISION.json file SHA-256 changed")
    decision = _read_json(decision_path, context="trigger decision")
    base_lock = _read_json(base_path, context="base-binding lock")
    snapshot = _read_json(snapshot_path, context="execution snapshot")
    activation = _read_json(activation_path, context="audit activation")
    _verify_decision(decision)
    _verify_base_lock(base_lock)
    _verify_snapshot(snapshot, root)
    _verify_authority_policy(
        policy,
        trigger=decision,
        base_lock=base_lock,
        snapshot=snapshot,
        activation=activation,
        project_root=root,
    )
    residual = BoundBaseIdentity._from_mapping(base_lock["residual_base"])
    pair = tuple(BoundBaseIdentity._from_mapping(item) for item in base_lock["ensemble_pair"])
    spent = policy.get("formal_spent_activated") is True
    if scope == "FORMAL_SPENT":
        # This pre-audit snapshot is intentionally unable to satisfy this gate.
        if not spent:
            raise StructuralContractError(
                "formal spent execution is blocked pending repeat independent audit GO"
            )
    values: dict[str, Any] = {
        "scope": scope,
        "project_root": root,
        "trigger_decision_path": decision_path,
        "trigger_decision_sha256": TRIGGER_DECISION_FILE_SHA256,
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
        "enabled_candidates": EXPECTED_ENABLED,
        "disabled_candidates": EXPECTED_DISABLED,
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
        "enabled_candidates": list(EXPECTED_ENABLED),
        "disabled_candidates": list(EXPECTED_DISABLED),
        "residual_base": residual.as_dict(),
        "pair": [item.as_dict() for item in pair],
        "spent_execution_authorized": spent,
    }
    values["authorization_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    output = StructuralExecutionAuthorization._mint(token=_AUTHORIZATION_TOKEN, **values)
    output.verify()
    return output


def disabled_huber_entrypoint(*_args: object, **_kwargs: object) -> None:
    """Stable public tombstone: the final trigger makes Huber uncallable."""

    raise StructuralContractError("disabled Huber candidate is uncallable")
