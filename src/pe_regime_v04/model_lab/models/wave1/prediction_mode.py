"""Physical prediction mode: accepts only sanitized inputs and emits sealed CSVs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import pandas as pd

from ... import folds as folds_module
from ...contracts import ContractError
from .artifacts import (
    DIAGNOSTIC_COLUMNS,
    PREDICTION_COLUMNS,
    file_record,
    load_predict_inputs,
    load_wave1_seed_frames,
    seal_payload,
    write_immutable_json,
    write_round_trip_csv,
)
from .authorization import authorize_execution
from .runner import FOLD_SPEC, run_seed_predictions
from .spec import BASELINE_MODEL_IDS, CANDIDATE_MODEL_IDS, DESIGN_LOCK_SHA256, EVIDENCE_SEEDS


SCHEDULE_SHA256 = "9b8bce1a68df3dd4655eab96450d6592aed2002d5bc71ec0d68a1d4c5b92d2ac"


def fold_schedule_sha256(folds) -> str:
    payload = {
        "starts": [int(fold.test_positions[0]) for fold in folds],
        "test_sizes": [len(fold.test_positions) for fold in folds],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _timestamp_text(value) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def fold_execution_payload(folds) -> dict[str, object]:
    """Bind every fold decision, not merely its observable test schedule."""

    generator_path = Path(folds_module.__file__).resolve(strict=True)
    generator_sha256 = hashlib.sha256(generator_path.read_bytes()).hexdigest()
    spec = {
        "min_train_sessions": FOLD_SPEC.min_train_sessions,
        "max_train_sessions": FOLD_SPEC.max_train_sessions,
        "test_sessions": FOLD_SPEC.test_sessions,
        "step_sessions": FOLD_SPEC.step_sessions,
        "embargo_sessions": FOLD_SPEC.embargo_sessions,
        "allow_partial_final_test": FOLD_SPEC.allow_partial_final_test,
    }
    rows = [
        {
            "fold_id": fold.fold_id,
            "train_positions": list(fold.train_positions),
            "test_positions": list(fold.test_positions),
            "train_start": _timestamp_text(fold.train_start),
            "train_end": _timestamp_text(fold.train_end),
            "test_start": _timestamp_text(fold.test_start),
            "test_end": _timestamp_text(fold.test_end),
            "label_information_cutoff": _timestamp_text(
                fold.label_information_cutoff
            ),
        }
        for fold in folds
    ]
    return {
        "generator_path": str(generator_path),
        "generator_sha256": generator_sha256,
        "spec": spec,
        "folds": rows,
    }


def fold_execution_sha256(folds) -> str:
    encoded = json.dumps(
        fold_execution_payload(folds),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def run_prediction_mode(
    predict_inputs_path: Path,
    execution_precommit_path: Path,
    output_directory: Path,
    *,
    outer_workers: int = 32,
    model_ids: Sequence[str] = CANDIDATE_MODEL_IDS,
    enforce_affinity: bool = True,
) -> Path:
    """Run no-truth prediction in the current, already CPU-sealed process."""

    authorization = authorize_execution(
        execution_precommit_path,
        predict_inputs_path,
        mode="predict",
    )
    inputs = load_predict_inputs(predict_inputs_path)
    seed_entries = inputs["seeds"]
    if not isinstance(seed_entries, list):
        raise ContractError("predict inputs seeds must be an array")
    seeds = tuple(int(entry["seed"]) for entry in seed_entries)
    if seeds != EVIDENCE_SEEDS:
        raise ContractError(f"predict inputs must use exact spent seeds {EVIDENCE_SEEDS}")
    output_directory = Path(output_directory)
    predictions_path = output_directory / "wave1_predictions.csv"
    diagnostics_path = output_directory / "wave1_fold_diagnostics.csv"
    manifest_path = output_directory / "PREDICTION_MANIFEST.json"
    if any(path.exists() for path in (predictions_path, diagnostics_path, manifest_path)):
        raise FileExistsError("Wave1 prediction outputs are immutable; choose a fresh directory")

    all_predictions: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    runtime_seconds: dict[str, float] = {
        model_id: 0.0 for model_id in (*model_ids, *BASELINE_MODEL_IDS)
    }
    resource_by_seed: list[dict[str, object]] = []
    schedule_hashes: list[str] = []
    execution_hashes: list[str] = []
    bound_folds = {
        int(item["seed"]): item
        for item in authorization.binding["fold_policy"]["per_seed"]
    }
    if set(bound_folds) != set(seeds):
        raise ContractError("execution binding fold-seed universe differs")
    started = time.perf_counter()
    for entry in seed_entries:
        seed = int(entry["seed"])
        model_frame, baselines = load_wave1_seed_frames(entry)
        result = run_seed_predictions(
            model_frame,
            baselines,
            seed=seed,
            model_ids=model_ids,
            outer_workers=outer_workers,
            enforce_affinity=enforce_affinity,
        )
        schedule_hash = fold_schedule_sha256(result.folds)
        if len(model_frame) == 1800 and schedule_hash != SCHEDULE_SHA256:
            raise ContractError("fold schedule differs from the sealed v0.4 partial-terminal schedule")
        execution_hash = fold_execution_sha256(result.folds)
        expected_fold = bound_folds[seed]
        if (
            int(expected_fold["rows"]) != len(model_frame)
            or expected_fold["schedule_sha256"] != schedule_hash
            or expected_fold["execution_sha256"] != execution_hash
        ):
            raise ContractError("runtime fold execution differs from the execution binding")
        schedule_hashes.append(schedule_hash)
        execution_hashes.append(execution_hash)
        all_predictions.append(result.predictions)
        all_diagnostics.append(result.diagnostics)
        for model_id, seconds in result.runtime_seconds_by_model.items():
            runtime_seconds[model_id] = runtime_seconds.get(model_id, 0.0) + float(seconds)
        resource_by_seed.append(
            {
                "seed": seed,
                **result.resource_summary,
                "affinity": list(result.affinity) if result.affinity is not None else None,
                "threadpool_inventory": result.threadpool_inventory,
            }
        )

    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        .sort_values(["seed", "date", "model_id"], kind="mergesort")
        .reset_index(drop=True)
        .loc[:, list(PREDICTION_COLUMNS)]
    )
    diagnostics = (
        pd.concat(all_diagnostics, ignore_index=True)
        .sort_values(["seed", "test_start_position", "model_id"], kind="mergesort")
        .reset_index(drop=True)
        .loc[:, list(DIAGNOSTIC_COLUMNS)]
    )
    prediction_record = write_round_trip_csv(
        predictions,
        predictions_path,
        columns=PREDICTION_COLUMNS,
    )
    diagnostic_record = write_round_trip_csv(
        diagnostics,
        diagnostics_path,
        columns=DIAGNOSTIC_COLUMNS,
    )
    expected_models = tuple((*model_ids, *BASELINE_MODEL_IDS))
    manifest = seal_payload(
        {
            "format_version": 1,
            "mode": "prediction_artifact",
            "design_lock_sha256": DESIGN_LOCK_SHA256,
            "execution_binding": authorization.binding_record,
            "execution_precommit": authorization.precommit_record,
            "predict_inputs": file_record(Path(predict_inputs_path)),
            "evaluation_data_used": False,
            "candidate_scores_computed": False,
            "seeds": list(seeds),
            "candidate_model_ids": list(model_ids),
            "baseline_model_ids": list(BASELINE_MODEL_IDS),
            "all_model_ids": list(expected_models),
            "outer_backend": "thread",
            "outer_workers": int(outer_workers),
            "inner_threads": 1,
            "gpu": "sealed_off",
            "fold_schedule_sha256_by_seed": schedule_hashes,
            "fold_execution_sha256_by_seed": execution_hashes,
            "prediction_rows": len(predictions),
            "diagnostic_rows": len(diagnostics),
            "predictions_csv": prediction_record,
            "fold_diagnostics_csv": diagnostic_record,
            "runtime_seconds_by_model": dict(sorted(runtime_seconds.items())),
            "runtime_boundary": (
                "sum of per-fold adapter preprocessing+fit+predict+validation wall seconds; "
                "input I/O, queue wait, and artifact serialization excluded"
            ),
            "end_to_end_wall_seconds_diagnostic_only": time.perf_counter() - started,
            "resource_by_seed": resource_by_seed,
        }
    )
    write_immutable_json(manifest_path, manifest)
    return manifest_path
