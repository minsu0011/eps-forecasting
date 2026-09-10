"""Fail-closed authorization for the two physical Wave-1 execution phases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import sys
from typing import Any, Literal, Mapping

from ...contracts import ContractError
from .artifacts import (
    canonical_json_bytes,
    file_record,
    load_sealed_json,
    verify_file_record,
)
from .audit import MUTABLE_REGISTRY_RELATIVE_PATHS, verify_independent_audit_go
from .spec import DESIGN_LOCK_SHA256


ExecutionMode = Literal["predict", "evaluate"]


class ExecutionAuthorizationError(ContractError):
    """Raised before numerical work when a bound execution input has drifted."""


@dataclass(frozen=True)
class ExecutionAuthorization:
    mode: ExecutionMode
    precommit: dict[str, Any]
    binding: dict[str, Any]
    precommit_record: dict[str, Any]
    binding_record: dict[str, Any]


_PRECOMMIT_FIELDS = {
    "format_version",
    "mode",
    "design_lock_sha256",
    "execution_binding",
    "independent_audit_go",
    "predict_inputs_content_sha256",
    "evaluate_inputs_content_sha256",
    "candidate_scores_at_commit",
    "candidate_predictions_at_commit",
    "formal_evidence",
    "fresh_seeds_consumed",
    "heldout_opened",
    "manifest_sha256",
}

_BINDING_FIELDS = {
    "format_version",
    "mode",
    "design_lock_sha256",
    "original_precommit_sha256",
    "independent_audit_go",
    "candidate_scores_at_binding",
    "candidate_predictions_at_binding",
    "fresh_seeds_consumed",
    "heldout_opened",
    "runtime_verified_files",
    "opaque_external_artifact_digests",
    "model_universe",
    "execution_policy",
    "fold_policy",
    "environment_policy",
    "registry_state",
    "manifest_sha256",
}

_PINNED_RUNTIME_PACKAGES = {
    "catboost": "1.2.10",
    "lightgbm": "4.6.0",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "pyarrow": "21.0.0",
    "PyYAML": "6.0.2",
    "scikit-learn": "1.7.2",
    "scipy": "1.15.3",
    "statsmodels": "0.14.6",
    "threadpoolctl": "3.6.0",
    "xgboost": "3.2.0",
}

RESULT_APPEND_DRIFT_LABELS = (
    "research/model_zoo/model_registry.csv",
    "research/model_zoo/model_registry.json",
)


def input_content_sha256(payload: Mapping[str, Any]) -> str:
    """Hash input content while excluding its final precommit backlink and own seal."""

    content = dict(payload)
    content.pop("execution_precommit", None)
    content.pop("manifest_sha256", None)
    return hashlib.sha256(canonical_json_bytes(content)).hexdigest()


def _verify_runtime_environment() -> None:
    if platform.python_implementation() != "CPython" or sys.version_info[:2] != (3, 10):
        raise ExecutionAuthorizationError("Wave1 requires CPython 3.10")
    actual = {
        name: importlib_metadata.version(name) for name in _PINNED_RUNTIME_PACKAGES
    }
    wrong = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in _PINNED_RUNTIME_PACKAGES.items()
        if actual[name] != expected
    }
    if wrong:
        raise ExecutionAuthorizationError(f"Wave1 runtime package pins differ: {wrong}")


def _verify_runtime_inventory(
    binding: Mapping[str, Any],
    *,
    permitted_live_drift_labels: tuple[str, ...] = (),
) -> None:
    inventory = binding.get("runtime_verified_files")
    if not isinstance(inventory, list) or not inventory:
        raise ExecutionAuthorizationError("execution binding runtime inventory is empty")
    labels: list[str] = []
    permitted = set(permitted_live_drift_labels)
    if not permitted.issubset(RESULT_APPEND_DRIFT_LABELS):
        raise ExecutionAuthorizationError("unsupported runtime-drift exemption")
    for item in inventory:
        if not isinstance(item, Mapping) or set(item) != {
            "label",
            "live_path",
            "snapshot",
        }:
            raise ExecutionAuthorizationError("runtime inventory entry fields are invalid")
        label = item["label"]
        if not isinstance(label, str) or not label:
            raise ExecutionAuthorizationError("runtime inventory labels must be non-empty")
        labels.append(label)
        try:
            verify_file_record(
                item["snapshot"],
                context=f"execution-bound snapshot {label}",
            )
        except ContractError as exc:
            raise ExecutionAuthorizationError(
                f"execution-bound snapshot {label} failed verification: {exc}"
            ) from exc
        live_path = Path(str(item["live_path"]))
        if not live_path.is_absolute() or not live_path.is_file():
            raise ExecutionAuthorizationError(f"live runtime file is missing for {label}")
        expected = item["snapshot"]
        if label not in permitted and (
            live_path.stat().st_size != int(expected["bytes"])
            or hashlib.sha256(live_path.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise ExecutionAuthorizationError(
                f"live runtime file differs from its execution snapshot: {label}"
            )
    if len(labels) != len(set(labels)):
        raise ExecutionAuthorizationError("runtime inventory labels must be unique")


def verify_historical_execution_snapshots(binding_path: Path) -> dict[str, Any]:
    """Audit immutable snapshots without requiring mutable live files to remain unchanged."""

    binding = load_sealed_json(binding_path, expected_mode="execution_binding")
    if set(binding) != _BINDING_FIELDS:
        raise ExecutionAuthorizationError("execution binding fields are invalid")
    for item in binding["runtime_verified_files"]:
        if set(item) != {"label", "live_path", "snapshot"}:
            raise ExecutionAuthorizationError("runtime inventory entry fields are invalid")
        verify_file_record(
            item["snapshot"],
            context=f"historical execution snapshot {item['label']}",
        )
    return binding


def _load_binding(
    record: Mapping[str, Any],
    *,
    permitted_live_drift_labels: tuple[str, ...] = (),
) -> tuple[Path, dict[str, Any]]:
    path = verify_file_record(record, context="execution binding")
    binding = load_sealed_json(path, expected_mode="execution_binding")
    if set(binding) != _BINDING_FIELDS:
        raise ExecutionAuthorizationError("execution binding fields are invalid")
    if binding["format_version"] != 1:
        raise ExecutionAuthorizationError("unsupported execution binding format")
    if binding["design_lock_sha256"] != DESIGN_LOCK_SHA256:
        raise ExecutionAuthorizationError("execution binding uses a different design lock")
    if binding["candidate_scores_at_binding"] != "unseen":
        raise ExecutionAuthorizationError("execution binding was not created score-blind")
    if binding["candidate_predictions_at_binding"] != "not_run":
        raise ExecutionAuthorizationError("execution binding followed candidate prediction")
    if binding["fresh_seeds_consumed"] is not False or binding["heldout_opened"] is not False:
        raise ExecutionAuthorizationError("execution binding claims prohibited evidence access")
    audit_go_path = verify_file_record(
        binding["independent_audit_go"], context="execution-bound independent audit GO"
    )
    try:
        verify_independent_audit_go(
            audit_go_path,
            project_root=Path(__file__).resolve().parents[5],
            permitted_current_drift=MUTABLE_REGISTRY_RELATIVE_PATHS,
        )
    except ContractError as exc:
        raise ExecutionAuthorizationError(
            f"execution-bound independent audit GO failed verification: {exc}"
        ) from exc
    _verify_runtime_inventory(
        binding,
        permitted_live_drift_labels=permitted_live_drift_labels,
    )
    return path, binding


def authorize_execution(
    execution_precommit_path: Path,
    input_manifest_path: Path,
    *,
    mode: ExecutionMode,
    permitted_live_drift_labels: tuple[str, ...] = (),
) -> ExecutionAuthorization:
    """Verify precommit, binding, source inventory, and the exact phase input."""

    if mode not in {"predict", "evaluate"}:
        raise ExecutionAuthorizationError(f"unsupported execution mode: {mode!r}")
    precommit = load_sealed_json(
        Path(execution_precommit_path), expected_mode="execution_precommit"
    )
    if set(precommit) != _PRECOMMIT_FIELDS or precommit["format_version"] != 1:
        raise ExecutionAuthorizationError("execution precommit fields/format are invalid")
    if precommit["design_lock_sha256"] != DESIGN_LOCK_SHA256:
        raise ExecutionAuthorizationError("execution precommit uses a different design lock")
    expected_states = {
        "candidate_scores_at_commit": "unseen",
        "candidate_predictions_at_commit": "not_run",
        "formal_evidence": False,
        "fresh_seeds_consumed": False,
        "heldout_opened": False,
    }
    if any(precommit.get(key) != value for key, value in expected_states.items()):
        raise ExecutionAuthorizationError("execution precommit state declarations are invalid")

    precommit_record = file_record(Path(execution_precommit_path))
    binding_record = dict(precommit["execution_binding"])
    _, binding = _load_binding(
        binding_record,
        permitted_live_drift_labels=permitted_live_drift_labels,
    )
    if precommit["independent_audit_go"] != binding["independent_audit_go"]:
        raise ExecutionAuthorizationError(
            "execution precommit and binding use different independent audit records"
        )

    inputs = load_sealed_json(
        Path(input_manifest_path), expected_mode=f"{mode}_inputs"
    )
    if inputs.get("execution_precommit") != precommit_record:
        raise ExecutionAuthorizationError("phase input does not bind this exact precommit file")
    if inputs.get("execution_binding") != binding_record:
        raise ExecutionAuthorizationError("phase input does not bind this exact execution binding")
    expected_content = precommit[f"{mode}_inputs_content_sha256"]
    if input_content_sha256(inputs) != expected_content:
        raise ExecutionAuthorizationError("phase input content differs from the precommit")
    _verify_runtime_environment()
    return ExecutionAuthorization(
        mode=mode,
        precommit=precommit,
        binding=binding,
        precommit_record=precommit_record,
        binding_record=binding_record,
    )
