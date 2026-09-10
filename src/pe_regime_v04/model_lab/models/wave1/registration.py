"""Planned append-only registry definitions; never invoked automatically."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from ...registry import (
    FeatureRegistration,
    ModelRegistration,
    RegistryPaths,
    RegistryError,
    RegistrySnapshot,
    append_feature_registration,
    append_model_registration,
    load_feature_registry,
    load_model_registry_history,
    write_feature_registry,
    write_model_registry,
)
from .spec import (
    CANDIDATE_MODEL_IDS,
    COMMON_FEATURES,
    MODEL_BY_ID,
    REGIME_EXTENSION,
    WAVE1_MODEL_DEFINITIONS,
    feature_metadata,
)


_REPOSITORIES = {
    "scikit-learn": "https://github.com/scikit-learn/scikit-learn",
    "catboost": "https://github.com/catboost/catboost",
    "xgboost": "https://github.com/dmlc/xgboost",
    "statsmodels": "https://github.com/statsmodels/statsmodels",
}

_PAPERS = {
    "scikit-learn": "https://jmlr.org/papers/v12/pedregosa11a.html",
    "catboost": "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html",
    "xgboost": "https://doi.org/10.1145/2939672.2939785",
    "statsmodels": "https://conference.scipy.org/proceedings/scipy2010/seabold.html",
}


@dataclass(frozen=True)
class DefinitionReconciliation:
    feature_snapshot: RegistrySnapshot
    model_snapshot: RegistrySnapshot
    feature_appended: tuple[str, ...]
    feature_skipped_exact: tuple[str, ...]
    model_appended: tuple[str, ...]
    model_skipped_exact: tuple[str, ...]


@dataclass(frozen=True)
class ResultReconciliation:
    model_snapshot: RegistrySnapshot
    result_appended: tuple[str, ...]
    result_skipped_exact: tuple[str, ...]


AppendHook = Callable[[str, str, int], None]

_AUDIT_REGISTRY_LABELS = {
    "feature_csv": "research/model_zoo/feature_registry.csv",
    "feature_json": "research/model_zoo/feature_registry.json",
    "model_csv": "research/model_zoo/model_registry.csv",
    "model_json": "research/model_zoo/model_registry.json",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def planned_feature_registrations() -> tuple[FeatureRegistration, ...]:
    metadata = feature_metadata(tuple((*COMMON_FEATURES, *REGIME_EXTENSION)))
    return tuple(FeatureRegistration(**item.__dict__) for item in metadata)


def _registration_hyperparameters(model_id: str) -> dict[str, Any]:
    definition = MODEL_BY_ID[model_id]
    payload = {
        "design_parameters": dict(definition.parameters),
        "target_transform": "natural_log_positive_finite_observed_pe",
        "prediction_transform": "exp_require_positive_finite",
        "fold_random_seed": "evidence_seed_plus_test_start_position",
        "train_weight": (
            "none"
            if definition.family == "state_space_local_linear_trend"
            else "where_nonfinite(eps_confidence,50).clip(5,100)/100"
        ),
        "all_missing_train_feature": "deterministic_drop_and_log",
        "constant_spline_feature": "deterministic_drop_and_log",
        "warning_policy": (
            "allow_exact_statsmodels_irregular_string_warning_only"
            if definition.family == "state_space_local_linear_trend"
            else "any_warning_fails_fold_no_retry"
        ),
        "threads": 1,
        "gpu": "off",
    }
    return payload


def planned_model_registrations() -> tuple[ModelRegistration, ...]:
    rows: list[ModelRegistration] = []
    for definition in WAVE1_MODEL_DEFINITIONS:
        package_name = definition.package.split("==", 1)[0]
        hyperparameters = _registration_hyperparameters(definition.model_id)
        target_history_only = definition.family == "state_space_local_linear_trend"
        rows.append(
            ModelRegistration(
                model_id=definition.model_id,
                family=definition.family,
                variant=definition.variant,
                version="wave1-b3a190-v1",
                registry_revision=0,
                track="B" if target_history_only else "C",
                estimand=(
                    "target-history market-conditioned expected P/E, not intrinsic fair value"
                    if target_history_only
                    else "hybrid fundamental-and-market expected P/E, not intrinsic fair value"
                ),
                external_reference=True,
                paper=_PAPERS[package_name],
                repository=_REPOSITORIES[package_name],
                package=definition.package,
                license=definition.license,
                target="log(observed_pe)",
                feature_set=definition.variant,
                uses_same_row_price=not target_history_only,
                uses_same_row_observed_pe=False,
                causal=True,
                pit_safe=True,
                train_window="rolling up to 1008 sessions; minimum 252 eligible rows",
                refit_frequency="every 21 sessions with terminal partial test allowed",
                hyperparameters=hyperparameters,
                tuning_status="UNTESTED",
                locked_status="UNTESTED",
                heldout_status="NOT_OPENED",
                fair_log_mae=None,
                fair_log_rmse=None,
                worst_seed=None,
                compute_time=None,
                status="UNTESTED",
                notes="Definition only; no score, fresh seed, or heldout was opened at registration planning.",
                entrypoint="pe_regime_v04.model_lab.models.wave1.adapters:build_wave1_model",
                feature_ids=tuple(definition.feature_columns),
                prediction_name="expected_pe",
                output_semantics="positive market-conditioned expected P/E research prediction",
                deterministic=True,
                description=(
                    f"Frozen Wave1 {definition.family} / {definition.variant} cheap-screen candidate"
                ),
                parameters_sha256=hashlib.sha256(_canonical_json(hyperparameters)).hexdigest(),
            )
        )
    return tuple(rows)


def append_planned_wave1_definitions(
    *,
    feature_paths: RegistryPaths,
    model_paths: RegistryPaths,
) -> tuple[RegistrySnapshot, RegistrySnapshot]:
    """Append via Phase1 APIs only after an external audit explicitly authorizes it."""

    result = reconcile_planned_wave1_definitions(
        feature_paths=feature_paths,
        model_paths=model_paths,
    )
    return result.feature_snapshot, result.model_snapshot


def reconcile_planned_wave1_definitions(
    *,
    feature_paths: RegistryPaths,
    model_paths: RegistryPaths,
    after_append: AppendHook | None = None,
) -> DefinitionReconciliation:
    """Append missing exact definitions and safely resume an interrupted transaction."""

    planned_features = planned_feature_registrations()
    planned_models = planned_model_registrations()
    existing_features = {item.feature_id: item for item in load_feature_registry(feature_paths)}
    existing_history = load_model_registry_history(model_paths)
    histories: dict[str, list[ModelRegistration]] = {}
    for item in existing_history:
        histories.setdefault(item.definition_id, []).append(item)

    for item in planned_features:
        existing = existing_features.get(item.feature_id)
        if existing is not None and existing != item:
            raise RegistryError(f"Wave1 feature definition conflict: {item.feature_id}")
    for item in planned_models:
        same_model_id = [row for row in existing_history if row.model_id == item.model_id]
        if any(row.definition_id != item.definition_id for row in same_model_id):
            raise RegistryError(f"Wave1 model definition identity conflict: {item.model_id}")
        history = histories.get(item.definition_id, [])
        if any(row.registry_revision != 0 for row in history):
            raise RegistryError(
                f"Wave1 definition transaction found extra revisions: {item.definition_id}"
            )
        if history and history[0] != item:
            raise RegistryError(f"Wave1 model definition conflict: {item.definition_id}")

    feature_appended: list[str] = []
    feature_skipped: list[str] = []
    feature_snapshot: RegistrySnapshot | None = None
    for item in planned_features:
        was_present = item.feature_id in existing_features
        feature_snapshot = append_feature_registration(feature_paths, item)
        if was_present:
            feature_skipped.append(item.feature_id)
        else:
            feature_appended.append(item.feature_id)
            existing_features[item.feature_id] = item
            if after_append is not None:
                after_append("feature", item.feature_id, len(feature_appended))

    model_appended: list[str] = []
    model_skipped: list[str] = []
    model_snapshot: RegistrySnapshot | None = None
    for item in planned_models:
        was_present = bool(histories.get(item.definition_id))
        model_snapshot = append_model_registration(model_paths, item)
        if was_present:
            model_skipped.append(item.registration_id)
        else:
            model_appended.append(item.registration_id)
            histories[item.definition_id] = [item]
            if after_append is not None:
                after_append("model", item.registration_id, len(model_appended))
    if feature_snapshot is None or model_snapshot is None:
        raise RuntimeError("planned Wave1 registry definitions are unexpectedly empty")
    return DefinitionReconciliation(
        feature_snapshot=feature_snapshot,
        model_snapshot=model_snapshot,
        feature_appended=tuple(feature_appended),
        feature_skipped_exact=tuple(feature_skipped),
        model_appended=tuple(model_appended),
        model_skipped_exact=tuple(model_skipped),
    )


def verify_definition_resume_state_against_audit(
    *,
    feature_paths: RegistryPaths,
    model_paths: RegistryPaths,
    audit_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove live registries equal the audited base plus an exact planned prefix."""

    audited = {str(item.get("relative_path")): item for item in audit_inventory}
    if any(label not in audited for label in _AUDIT_REGISTRY_LABELS.values()):
        raise RegistryError("audit inventory is missing registry baseline records")
    planned_features = planned_feature_registrations()
    planned_models = planned_model_registrations()
    current_features = load_feature_registry(feature_paths)
    current_models = load_model_registry_history(model_paths)
    planned_feature_ids = {item.feature_id for item in planned_features}
    planned_definition_ids = {item.definition_id for item in planned_models}

    feature_start = next(
        (
            index
            for index, item in enumerate(current_features)
            if item.feature_id in planned_feature_ids
        ),
        len(current_features),
    )
    model_start = next(
        (
            index
            for index, item in enumerate(current_models)
            if item.definition_id in planned_definition_ids
        ),
        len(current_models),
    )
    base_features = current_features[:feature_start]
    base_models = current_models[:model_start]
    feature_suffix = current_features[feature_start:]
    model_suffix = current_models[model_start:]
    if feature_suffix != planned_features[: len(feature_suffix)]:
        raise RegistryError("feature registry is not an exact planned-definition prefix")
    if model_suffix != planned_models[: len(model_suffix)]:
        raise RegistryError("model registry is not an exact planned-definition prefix")

    with tempfile.TemporaryDirectory(prefix="wave1-audited-registry-base-") as temporary:
        root = Path(temporary)
        base_feature_paths = RegistryPaths(root / "feature.csv", root / "feature.json")
        base_model_paths = RegistryPaths(root / "model.csv", root / "model.json")
        write_feature_registry(base_feature_paths, base_features)
        write_model_registry(base_model_paths, base_models)
        generated = {
            "feature_csv": base_feature_paths.csv_path,
            "feature_json": base_feature_paths.json_path,
            "model_csv": base_model_paths.csv_path,
            "model_json": base_model_paths.json_path,
        }
        for role, path in generated.items():
            expected = audited[_AUDIT_REGISTRY_LABELS[role]]
            if (
                path.stat().st_size != expected.get("bytes")
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected.get("sha256")
            ):
                raise RegistryError(
                    f"registry base differs from the independently audited bytes: {role}"
                )
    return {
        "audited_registry_before": {
            role: {
                "bytes": int(audited[label]["bytes"]),
                "sha256": str(audited[label]["sha256"]),
            }
            for role, label in _AUDIT_REGISTRY_LABELS.items()
        },
        "exact_partial_feature_count": len(feature_suffix),
        "exact_partial_model_count": len(model_suffix),
    }


def reconcile_wave1_result_records(
    model_paths: RegistryPaths,
    records: Sequence[ModelRegistration],
    *,
    after_append: AppendHook | None = None,
) -> ResultReconciliation:
    """Append exact Stage1 revision-one events and safely resume interruption."""

    desired = tuple(records)
    if tuple(item.model_id for item in desired) != CANDIDATE_MODEL_IDS:
        raise RegistryError("Wave1 result records differ from the candidate order/universe")
    if any(item.registry_revision != 1 for item in desired):
        raise RegistryError("Wave1 Stage1 result records must be registry revision one")
    planned = {item.definition_id: item for item in planned_model_registrations()}
    history = load_model_registry_history(model_paths)
    histories: dict[str, list[ModelRegistration]] = {}
    for item in history:
        histories.setdefault(item.definition_id, []).append(item)
    for item in desired:
        rows = histories.get(item.definition_id, [])
        expected_definition = planned.get(item.definition_id)
        if expected_definition is None or not rows or rows[0] != expected_definition:
            raise RegistryError(
                f"Wave1 Stage1 result is missing its exact definition: {item.definition_id}"
            )
        if any(row.registry_revision not in {0, 1} for row in rows):
            raise RegistryError(
                f"Wave1 Stage1 result transaction found extra revisions: {item.definition_id}"
            )
        revision_one = [row for row in rows if row.registry_revision == 1]
        if revision_one and revision_one[0] != item:
            raise RegistryError(f"Wave1 Stage1 result conflict: {item.registration_id}")

    appended: list[str] = []
    skipped: list[str] = []
    snapshot: RegistrySnapshot | None = None
    for item in desired:
        was_present = any(
            row.registry_revision == 1 for row in histories[item.definition_id]
        )
        snapshot = append_model_registration(model_paths, item)
        if was_present:
            skipped.append(item.registration_id)
        else:
            appended.append(item.registration_id)
            histories[item.definition_id].append(item)
            if after_append is not None:
                after_append("result", item.registration_id, len(appended))
    if snapshot is None:
        raise RuntimeError("Wave1 Stage1 result records are unexpectedly empty")
    return ResultReconciliation(
        model_snapshot=snapshot,
        result_appended=tuple(appended),
        result_skipped_exact=tuple(skipped),
    )


def verify_result_resume_state_against_binding(
    *,
    model_paths: RegistryPaths,
    runtime_verified_files: Sequence[Mapping[str, Any]],
    records: Sequence[ModelRegistration],
) -> int:
    """Prove live model history equals its binding snapshot plus an exact result prefix."""

    snapshots = {
        str(item.get("label")): item.get("snapshot") for item in runtime_verified_files
    }
    csv_record = snapshots.get("research/model_zoo/model_registry.csv")
    json_record = snapshots.get("research/model_zoo/model_registry.json")
    if not isinstance(csv_record, Mapping) or not isinstance(json_record, Mapping):
        raise RegistryError("execution binding is missing model-registry snapshots")
    baseline_paths = RegistryPaths(
        Path(str(csv_record["path"])), Path(str(json_record["path"]))
    )
    baseline = load_model_registry_history(baseline_paths)
    current = load_model_registry_history(model_paths)
    desired = tuple(records)
    if len(current) < len(baseline) or current[: len(baseline)] != baseline:
        raise RegistryError("live model registry differs from the execution-bound prefix")
    suffix = current[len(baseline) :]
    if suffix != desired[: len(suffix)]:
        raise RegistryError("live model registry is not an exact Stage1 result prefix")
    return len(suffix)


def planned_registration_payload() -> dict[str, Any]:
    return {
        "features": [item.to_record() for item in planned_feature_registrations()],
        "models": [item.to_record() for item in planned_model_registrations()],
    }
