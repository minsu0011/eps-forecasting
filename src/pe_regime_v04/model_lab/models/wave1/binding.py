"""Build score-blind Wave-1 input separation and immutable execution seals."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ...contracts import EVALUATION_ONLY_TRUTH_COLUMN, ContractError
from ...folds import generate_pit_folds
from .artifacts import (
    canonical_json_bytes,
    file_record,
    inspect_predict_seed_entry,
    load_wave1_seed_frames,
    logical_frame_sha256,
    seal_payload,
    sha256_file,
    verify_file_record,
    write_immutable_json,
)
from .authorization import input_content_sha256
from .prediction_mode import fold_execution_sha256, fold_schedule_sha256
from .runner import FOLD_SPEC
from .spec import (
    BASELINE_MODEL_IDS,
    CANDIDATE_MODEL_IDS,
    DESIGN_LOCK_SHA256,
    EVIDENCE_SEEDS,
    ORIGINAL_PRECOMMIT_SHA256,
    RESOURCE_POLICY,
)


class ExecutionBindingError(ContractError):
    """Raised while building an execution package from untrusted evidence records."""


RUNTIME_DEPENDENCY_RELATIVE_PATHS = (
    "pyproject.toml",
    "src/pe_regime_v04/__init__.py",
    "src/pe_regime_v04/model_lab/__init__.py",
    "src/pe_regime_v04/model_lab/analysis.py",
    "src/pe_regime_v04/model_lab/contracts.py",
    "src/pe_regime_v04/model_lab/dataset.py",
    "src/pe_regime_v04/model_lab/evaluator.py",
    "src/pe_regime_v04/model_lab/experiment.py",
    "src/pe_regime_v04/model_lab/folds.py",
    "src/pe_regime_v04/model_lab/matrices.py",
    "src/pe_regime_v04/model_lab/registry.py",
    "src/pe_regime_v04/model_lab/models/wave1/__init__.py",
    "src/pe_regime_v04/model_lab/models/wave1/adapters.py",
    "src/pe_regime_v04/model_lab/models/wave1/artifacts.py",
    "src/pe_regime_v04/model_lab/models/wave1/audit.py",
    "src/pe_regime_v04/model_lab/models/wave1/authorization.py",
    "src/pe_regime_v04/model_lab/models/wave1/binding.py",
    "src/pe_regime_v04/model_lab/models/wave1/evaluation.py",
    "src/pe_regime_v04/model_lab/models/wave1/evaluation_mode.py",
    "src/pe_regime_v04/model_lab/models/wave1/prediction_mode.py",
    "src/pe_regime_v04/model_lab/models/wave1/registration.py",
    "src/pe_regime_v04/model_lab/models/wave1/reporting.py",
    "src/pe_regime_v04/model_lab/models/wave1/resources.py",
    "src/pe_regime_v04/model_lab/models/wave1/runner.py",
    "src/pe_regime_v04/model_lab/models/wave1/spec.py",
    "src/pe_regime_v04/model_lab/models/__init__.py",
    "scripts/model_lab/wave1/evaluate.py",
    "scripts/model_lab/wave1/freeze_audit_candidate.py",
    "scripts/model_lab/wave1/no_score_dry_run.py",
    "scripts/model_lab/wave1/predict.py",
    "scripts/model_lab/wave1/prepare_execution.py",
    "scripts/model_lab/wave1/resource_probe.py",
    "scripts/model_lab/wave1/register_definitions.py",
    "scripts/model_lab/wave1/register_results.py",
    "tests/model_lab/conftest.py",
    "tests/model_lab/test_contracts_and_baseline.py",
    "tests/model_lab/test_dataset.py",
    "tests/model_lab/test_evaluator_and_analysis.py",
    "tests/model_lab/test_experiment.py",
    "tests/model_lab/test_folds.py",
    "tests/model_lab/test_matrices.py",
    "tests/model_lab/test_registry.py",
    "tests/model_lab/test_scripts.py",
    "tests/model_lab/test_wave1_adapters.py",
    "tests/model_lab/test_wave1_artifacts.py",
    "tests/model_lab/test_wave1_authorization.py",
    "tests/model_lab/test_wave1_evaluation.py",
    "tests/model_lab/test_wave1_registration.py",
    "tests/model_lab/test_wave1_reporting.py",
    "tests/model_lab/test_wave1_runner.py",
    "tests/model_lab/test_wave1_support.py",
    "research/model_zoo/API_CONTRACT.md",
    "research/model_zoo/feature_registry.csv",
    "research/model_zoo/feature_registry.json",
    "research/model_zoo/model_registry.csv",
    "research/model_zoo/model_registry.json",
    "research/model_zoo/schemas/feature_registry.schema.json",
    "research/model_zoo/schemas/model_registry.schema.json",
    "outputs/model_zoo_wave1_screen_20260819/DESIGN_LOCK.json",
    "outputs/model_zoo_wave1_screen_20260819/PRECOMMIT.json",
    "outputs/model_zoo_wave1_screen_20260819/MODEL_HYPOTHESES.md",
    "outputs/model_zoo_wave1_screen_20260819/EXECUTION_AMBIGUITY_RESOLUTIONS.json",
    "outputs/model_lab_environment_20260819/CHECKSUMS.sha256",
    "outputs/model_lab_environment_20260819/top_level_requirements.lock.txt",
    "outputs/model_lab_environment_20260819/fully_resolved_environment.lock.txt",
    "outputs/model_lab_environment_20260819/package_manifest.json",
    "outputs/model_lab_environment_20260819/validation_summary.json",
)


def _digest_record_without_path(path: Path, *, role: str, seed: int | None = None) -> dict:
    resolved = Path(path).resolve(strict=True)
    record: dict[str, Any] = {
        "role": role,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if seed is not None:
        record["seed"] = int(seed)
    return record


def _verify_v3_file_record(record: Mapping[str, Any], *, context: str) -> Path:
    if set(record) != {"format_version", "path", "bytes", "sha256"}:
        raise ExecutionBindingError(f"{context} record fields are invalid")
    if record["format_version"] != 3:
        raise ExecutionBindingError(f"{context} record format is not v3")
    return verify_file_record(
        {key: record[key] for key in ("path", "bytes", "sha256")},
        context=context,
    )


def _load_and_verify_mg1_result(path: Path, *, seed: int) -> tuple[dict, dict[str, Path]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionBindingError(f"cannot read MG1 result for seed {seed}") from exc
    if not isinstance(payload, dict) or payload.get("seed") != seed:
        raise ExecutionBindingError(f"MG1 result seed mismatch for {seed}")
    unsigned = dict(payload)
    recorded = unsigned.pop("result_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if recorded != expected:
        raise ExecutionBindingError(f"MG1 result logical seal differs for seed {seed}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ExecutionBindingError(f"MG1 result artifacts are missing for seed {seed}")
    paths = {
        role: _verify_v3_file_record(artifacts[role], context=f"MG1 {role} seed {seed}")
        for role in ("canonical_csv", "output_csv", "truth_csv")
    }
    return payload, paths


def _read_round_trip(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, float_precision="round_trip")


def collect_mg1_execution_inputs(
    result_paths: Mapping[int, Path],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Audit five spent-seed results and split predict-safe from evaluation-only inputs."""

    if tuple(sorted(result_paths)) != tuple(sorted(EVIDENCE_SEEDS)):
        raise ExecutionBindingError("MG1 result paths must cover the five exact spent seeds")
    predict_entries: list[dict] = []
    evaluate_entries: list[dict] = []
    opaque_digests: list[dict] = []
    fold_bindings: list[dict] = []
    for seed in EVIDENCE_SEEDS:
        result_path = Path(result_paths[seed]).resolve(strict=True)
        _, paths = _load_and_verify_mg1_result(result_path, seed=seed)
        canonical = _read_round_trip(paths["canonical_csv"])
        output = _read_round_trip(paths["output_csv"])
        truth = _read_round_trip(paths["truth_csv"])
        if canonical.empty or "symbol" not in canonical or "date" not in canonical:
            raise ExecutionBindingError(f"canonical identity is unavailable for seed {seed}")
        symbols = canonical["symbol"].astype(str)
        if symbols.nunique(dropna=False) != 1:
            raise ExecutionBindingError(f"canonical seed {seed} is not a single-symbol timeline")
        expected_symbol = str(symbols.iloc[0])
        identity = pd.DataFrame(
            {
                "date": pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d"),
                "symbol": symbols,
            }
        )
        entry = {
            "seed": seed,
            "source_record_bytes": result_path.stat().st_size,
            "source_record_sha256": sha256_file(result_path),
            "canonical_csv": file_record(paths["canonical_csv"]),
            "output_csv": file_record(paths["output_csv"]),
            "expected_rows": len(canonical),
            "expected_symbol": expected_symbol,
            "canonical_columns_sha256": hashlib.sha256(
                canonical_json_bytes(list(canonical.columns))
            ).hexdigest(),
            "output_columns_sha256": hashlib.sha256(
                canonical_json_bytes(list(output.columns))
            ).hexdigest(),
            "prefix150_logical_sha256": logical_frame_sha256(canonical),
            "identity_logical_sha256": logical_frame_sha256(identity),
        }
        inspect_predict_seed_entry(entry)
        model_frame, baselines = load_wave1_seed_frames(entry)
        if len(model_frame) != len(canonical) or len(baselines) != len(canonical):
            raise ExecutionBindingError(f"Wave1 real-manifest schema differs for seed {seed}")
        if tuple(truth.columns).count(EVALUATION_ONLY_TRUTH_COLUMN) != 1:
            raise ExecutionBindingError(f"truth column contract differs for seed {seed}")
        truth_dates = pd.to_datetime(truth["date"], errors="coerce")
        canonical_dates = pd.to_datetime(canonical["date"], errors="coerce")
        if truth_dates.isna().any() or not truth_dates.equals(canonical_dates):
            raise ExecutionBindingError(f"truth/canonical date identity differs for seed {seed}")
        predict_entries.append(entry)
        evaluate_entries.append(
            {
                "seed": seed,
                "truth_csv": file_record(paths["truth_csv"]),
                "expected_rows": len(truth),
            }
        )
        for role, artifact_path in (
            ("mg1_result_record", result_path),
            ("canonical150", paths["canonical_csv"]),
            ("output254", paths["output_csv"]),
            ("evaluation_only", paths["truth_csv"]),
        ):
            opaque_digests.append(
                _digest_record_without_path(artifact_path, role=role, seed=seed)
            )
        folds = generate_pit_folds(canonical.loc[:, ["date"]], FOLD_SPEC)
        fold_bindings.append(
            {
                "seed": seed,
                "rows": len(canonical),
                "schedule_sha256": fold_schedule_sha256(folds),
                "execution_sha256": fold_execution_sha256(folds),
            }
        )
    return predict_entries, evaluate_entries, opaque_digests, fold_bindings


def _snapshot_file(path: Path, snapshot_directory: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    snapshot = snapshot_directory / f"{digest}.snapshot"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if snapshot.stat().st_size != path.stat().st_size or sha256_file(snapshot) != digest:
            raise ExecutionBindingError("content-addressed snapshot collision or corruption")
    else:
        temporary = snapshot.with_name(f".{snapshot.name}.tmp.{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"snapshot temporary path already exists: {temporary}")
        try:
            with path.open("rb") as source, temporary.open("xb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, snapshot)
        finally:
            if temporary.exists():
                temporary.unlink()
    return file_record(snapshot)


def runtime_inventory(
    root: Path,
    additional_files: Sequence[Path],
    *,
    snapshot_directory: Path,
) -> list[dict]:
    """Bind all Wave-1/Phase-1 code, tests, environment locks, and registry state."""

    root = Path(root).resolve(strict=True)
    expected_labels = set(RUNTIME_DEPENDENCY_RELATIVE_PATHS)
    if len(expected_labels) != len(RUNTIME_DEPENDENCY_RELATIVE_PATHS):
        raise ExecutionBindingError("runtime dependency allowlist contains duplicate labels")
    paths = {label: root / label for label in RUNTIME_DEPENDENCY_RELATIVE_PATHS}
    missing = sorted(str(path) for path in paths.values() if not Path(path).is_file())
    if missing:
        raise ExecutionBindingError(f"runtime inventory contains missing files: {missing}")
    records: list[dict] = []
    for label in sorted(paths):
        path = paths[label].resolve(strict=True)
        records.append(
            {
                "label": label,
                "live_path": str(path),
                "snapshot": _snapshot_file(path, Path(snapshot_directory)),
            }
        )
    for additional in sorted(
        (Path(path).resolve(strict=True) for path in additional_files), key=lambda item: str(item)
    ):
        digest = sha256_file(additional)
        label = f"execution_evidence/{digest[:16]}_{additional.name}"
        records.append(
            {
                "label": label,
                "live_path": str(additional),
                "snapshot": _snapshot_file(additional, Path(snapshot_directory)),
            }
        )
    labels = [record["label"] for record in records]
    if len(labels) != len(set(labels)):
        raise ExecutionBindingError("runtime inventory labels collide")
    if set(labels[: len(expected_labels)]) != expected_labels:
        raise ExecutionBindingError("runtime dependency labels differ from the explicit allowlist")
    if len(records) != len(expected_labels) + len(additional_files):
        raise ExecutionBindingError("runtime dependency count differs from the explicit allowlist")
    return records


def build_execution_binding_payload(
    *,
    runtime_verified_files: Sequence[Mapping[str, Any]],
    opaque_external_artifact_digests: Sequence[Mapping[str, Any]],
    fold_bindings: Sequence[Mapping[str, Any]],
    registry_state: Mapping[str, Any],
    independent_audit_go: Mapping[str, Any],
) -> dict[str, Any]:
    design_path = next(
        (
            item["snapshot"]
            for item in runtime_verified_files
            if item["label"].endswith("/DESIGN_LOCK.json")
        ),
        None,
    )
    original_path = next(
        (
            item["snapshot"]
            for item in runtime_verified_files
            if item["label"].endswith("/PRECOMMIT.json")
        ),
        None,
    )
    if design_path is None or design_path["sha256"] != DESIGN_LOCK_SHA256:
        raise ExecutionBindingError("runtime inventory does not contain the exact DESIGN_LOCK")
    if original_path is None or original_path["sha256"] != ORIGINAL_PRECOMMIT_SHA256:
        raise ExecutionBindingError("runtime inventory does not contain the original PRECOMMIT")
    payload = {
        "format_version": 1,
        "mode": "execution_binding",
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "original_precommit_sha256": ORIGINAL_PRECOMMIT_SHA256,
        "independent_audit_go": dict(independent_audit_go),
        "candidate_scores_at_binding": "unseen",
        "candidate_predictions_at_binding": "not_run",
        "fresh_seeds_consumed": False,
        "heldout_opened": False,
        "runtime_verified_files": [dict(item) for item in runtime_verified_files],
        "opaque_external_artifact_digests": [
            dict(item) for item in opaque_external_artifact_digests
        ],
        "model_universe": {
            "candidate_model_ids": list(CANDIDATE_MODEL_IDS),
            "baseline_model_ids": list(BASELINE_MODEL_IDS),
            "evidence_seeds": list(EVIDENCE_SEEDS),
        },
        "execution_policy": {
            "physical_modes": ["predict", "evaluate"],
            "predict_truth_path_visibility": "prohibited",
            "candidate_fallback_or_retry": "prohibited",
            "target": "natural_log(positive_finite_observed_pe)",
            "supervised_training_weight": (
                "where_nonfinite(eps_confidence,50).clip(5,100)/100"
            ),
            "metrics": "unweighted",
            "outer_backend": "ThreadPoolExecutor",
            "outer_workers": 32,
            "inner_native_threads": 1,
            "warning_policy": (
                "one_global_pre_thread_error_policy_with_exact_statsmodels_"
                "SpecificationWarning_suppression"
            ),
            "round_trip_csv_authority": True,
        },
        "fold_policy": {
            "spec": {
                "min_train_sessions": FOLD_SPEC.min_train_sessions,
                "max_train_sessions": FOLD_SPEC.max_train_sessions,
                "test_sessions": FOLD_SPEC.test_sessions,
                "step_sessions": FOLD_SPEC.step_sessions,
                "embargo_sessions": FOLD_SPEC.embargo_sessions,
                "allow_partial_final_test": FOLD_SPEC.allow_partial_final_test,
            },
            "per_seed": [dict(item) for item in fold_bindings],
        },
        "environment_policy": {
            **dict(RESOURCE_POLICY),
            "python": "CPython==3.10.*",
            "runtime_packages": {
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
            },
        },
        "registry_state": dict(registry_state),
    }
    return seal_payload(payload)


def _base_predict_inputs(binding_path: Path, entries: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "format_version": 1,
        "mode": "predict_inputs",
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "execution_binding": file_record(binding_path),
        "evaluation_data_excluded": True,
        "seeds": [dict(entry) for entry in entries],
    }


def _base_evaluate_inputs(binding_path: Path, entries: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "format_version": 1,
        "mode": "evaluate_inputs",
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "execution_binding": file_record(binding_path),
        "predict_process_must_not_receive_this_manifest": True,
        "seeds": [dict(entry) for entry in entries],
    }


def write_execution_package(
    output_directory: Path,
    *,
    binding_payload: Mapping[str, Any],
    predict_entries: Sequence[Mapping[str, Any]],
    evaluate_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Write binding, precommit, and separately authorized inputs without a hash cycle."""

    output_directory = Path(output_directory)
    paths = {
        "binding": output_directory / "EXECUTION_BINDING.json",
        "precommit": output_directory / "EXECUTION_PRECOMMIT.json",
        "predict_inputs": output_directory / "PREDICT_INPUTS.json",
        "evaluate_inputs": output_directory / "EVALUATE_INPUTS.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("execution package files are immutable")
    write_immutable_json(paths["binding"], binding_payload)
    predict_base = _base_predict_inputs(paths["binding"], predict_entries)
    evaluate_base = _base_evaluate_inputs(paths["binding"], evaluate_entries)
    precommit = seal_payload(
        {
            "format_version": 1,
            "mode": "execution_precommit",
            "design_lock_sha256": DESIGN_LOCK_SHA256,
            "execution_binding": file_record(paths["binding"]),
            "independent_audit_go": dict(binding_payload["independent_audit_go"]),
            "predict_inputs_content_sha256": input_content_sha256(predict_base),
            "evaluate_inputs_content_sha256": input_content_sha256(evaluate_base),
            "candidate_scores_at_commit": "unseen",
            "candidate_predictions_at_commit": "not_run",
            "formal_evidence": False,
            "fresh_seeds_consumed": False,
            "heldout_opened": False,
        }
    )
    write_immutable_json(paths["precommit"], precommit)
    precommit_record = file_record(paths["precommit"])
    write_immutable_json(
        paths["predict_inputs"],
        seal_payload({**predict_base, "execution_precommit": precommit_record}),
    )
    write_immutable_json(
        paths["evaluate_inputs"],
        seal_payload({**evaluate_base, "execution_precommit": precommit_record}),
    )
    return paths


def opaque_root_evidence_digests(root: Path) -> list[dict]:
    """Hash MG1 anchors without exposing their paths to the prediction process."""

    root = Path(root).resolve(strict=True)
    relpaths = (
        "outputs/mg1/run_manifest.json",
        "outputs/mg1/promotion_policy.lock.json",
        "outputs/mg1/checkpoint.json",
        "outputs/v04_spent_seed_registry.json",
    )
    return [
        _digest_record_without_path(root / relpath, role=relpath.replace("/", "_"))
        for relpath in relpaths
    ]
