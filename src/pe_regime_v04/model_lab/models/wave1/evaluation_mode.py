"""Physical evaluation mode: consumes sealed predictions, then and only then truth."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...contracts import EVALUATION_ONLY_TRUTH_COLUMN, ContractError
from .artifacts import (
    PREDICTION_COLUMNS,
    file_record,
    load_sealed_json,
    seal_payload,
    verify_file_record,
    write_immutable_json,
    write_round_trip_csv,
)
from .authorization import ExecutionAuthorization, authorize_execution
from .evaluation import GATE_COLUMNS, evaluate_wave1_predictions
from .reporting import write_wave1_research_outputs
from .resources import assert_cpu_environment
from .spec import BASELINE_MODEL_IDS, CANDIDATE_MODEL_IDS, DESIGN_LOCK_SHA256, EVIDENCE_SEEDS


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, float_precision="round_trip")


def _load_prediction_manifest(path: Path, authorization: ExecutionAuthorization) -> dict:
    payload = load_sealed_json(path, expected_mode="prediction_artifact")
    if payload.get("design_lock_sha256") != DESIGN_LOCK_SHA256:
        raise ContractError("prediction artifact uses a different design lock")
    if payload.get("evaluation_data_used") is not False:
        raise ContractError("prediction artifact does not prove evaluation-data isolation")
    if payload.get("candidate_scores_computed") is not False:
        raise ContractError("prediction process must not compute candidate scores")
    if tuple(payload.get("candidate_model_ids", [])) != CANDIDATE_MODEL_IDS:
        raise ContractError("prediction artifact candidate model universe differs")
    if tuple(payload.get("baseline_model_ids", [])) != BASELINE_MODEL_IDS:
        raise ContractError("prediction artifact baseline model universe differs")
    if tuple(payload.get("seeds", [])) != EVIDENCE_SEEDS:
        raise ContractError("prediction artifact seed universe differs")
    if payload.get("execution_precommit") != authorization.precommit_record:
        raise ContractError("prediction artifact uses a different execution precommit")
    if payload.get("execution_binding") != authorization.binding_record:
        raise ContractError("prediction artifact uses a different execution binding")
    return payload


def _load_evaluation_inputs(path: Path) -> tuple[dict, dict[int, pd.DataFrame]]:
    payload = load_sealed_json(path, expected_mode="evaluate_inputs")
    required = {
        "format_version",
        "mode",
        "design_lock_sha256",
        "execution_binding",
        "execution_precommit",
        "predict_process_must_not_receive_this_manifest",
        "seeds",
        "manifest_sha256",
    }
    if set(payload) != required or payload["design_lock_sha256"] != DESIGN_LOCK_SHA256:
        raise ContractError("evaluation input manifest fields/design are invalid")
    if payload["predict_process_must_not_receive_this_manifest"] is not True:
        raise ContractError("evaluation manifest lacks the process-isolation assertion")
    seed_entries = payload["seeds"]
    if not isinstance(seed_entries, list):
        raise ContractError("evaluation seed entries must be an array")
    seeds = tuple(int(entry.get("seed")) for entry in seed_entries)
    if seeds != EVIDENCE_SEEDS:
        raise ContractError("evaluation input seed universe differs")
    truth_by_seed: dict[int, pd.DataFrame] = {}
    for entry in seed_entries:
        if set(entry) != {"seed", "truth_csv", "expected_rows"}:
            raise ContractError("evaluation seed entry fields are invalid")
        truth_path = verify_file_record(entry["truth_csv"], context="evaluation truth CSV")
        truth = _read_csv(truth_path)
        if len(truth) != int(entry["expected_rows"]):
            raise ContractError("truth CSV row count differs from the binding")
        if tuple(truth.columns).count(EVALUATION_ONLY_TRUTH_COLUMN) != 1:
            raise ContractError("truth CSV must contain true_fair_pe exactly once")
        truth_by_seed[int(entry["seed"])] = truth
    return payload, truth_by_seed


def run_evaluation_mode(
    prediction_manifest_path: Path,
    evaluation_inputs_path: Path,
    execution_precommit_path: Path,
    output_directory: Path,
) -> Path:
    assert_cpu_environment()
    authorization = authorize_execution(
        execution_precommit_path,
        evaluation_inputs_path,
        mode="evaluate",
    )
    prediction_manifest = _load_prediction_manifest(
        prediction_manifest_path,
        authorization,
    )
    _, truth_by_seed = _load_evaluation_inputs(evaluation_inputs_path)
    prediction_path = verify_file_record(
        prediction_manifest["predictions_csv"], context="sealed predictions CSV"
    )
    predictions = _read_csv(prediction_path)
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ContractError("sealed prediction CSV schema/order differs")
    runtime = {
        str(key): float(value)
        for key, value in prediction_manifest["runtime_seconds_by_model"].items()
    }
    result = evaluate_wave1_predictions(
        predictions,
        truth_by_seed,
        runtime_seconds_by_model=runtime,
        strict_expected_rows_per_seed=1296,
    )

    output_directory = Path(output_directory)
    paths = {
        "summary": output_directory / "wave1_summary.csv",
        "per_seed": output_directory / "wave1_per_seed.csv",
        "row_errors": output_directory / "wave1_row_errors.csv",
        "gates": output_directory / "wave1_gates.csv",
        "joined": output_directory / "wave1_joined_predictions.csv",
    }
    manifest_path = output_directory / "EVALUATION_MANIFEST.json"
    if manifest_path.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("Wave1 evaluation outputs are immutable; choose a fresh directory")
    artifacts = {
        "summary_csv": write_round_trip_csv(
            result.metrics.summary,
            paths["summary"],
            columns=tuple(result.metrics.summary.columns),
        ),
        "per_seed_csv": write_round_trip_csv(
            result.metrics.per_seed,
            paths["per_seed"],
            columns=tuple(result.metrics.per_seed.columns),
        ),
        "row_errors_csv": write_round_trip_csv(
            result.metrics.row_errors,
            paths["row_errors"],
            columns=tuple(result.metrics.row_errors.columns),
        ),
        "gates_csv": write_round_trip_csv(
            result.gates,
            paths["gates"],
            columns=GATE_COLUMNS,
        ),
        "joined_predictions_csv": write_round_trip_csv(
            result.joined_predictions,
            paths["joined"],
            columns=tuple(result.joined_predictions.columns),
        ),
    }
    research_artifacts = write_wave1_research_outputs(
        result,
        output_directory,
        execution_binding=authorization.binding_record,
        execution_precommit=authorization.precommit_record,
        candidate_model_ids=CANDIDATE_MODEL_IDS,
        baseline_model_ids=BASELINE_MODEL_IDS,
    )
    manifest = seal_payload(
        {
            "format_version": 1,
            "mode": "evaluation_artifact",
            "design_lock_sha256": DESIGN_LOCK_SHA256,
            "execution_binding": authorization.binding_record,
            "execution_precommit": authorization.precommit_record,
            "prediction_manifest": file_record(Path(prediction_manifest_path)),
            "evaluation_inputs": file_record(Path(evaluation_inputs_path)),
            "candidate_scores_computed": True,
            "fresh_seeds_consumed": False,
            "heldout_opened": False,
            "expected_identity_rows": result.expected_identity_rows,
            "common_identity_rows": result.common_identity_rows,
            "complementarity_baseline": result.complementarity_baseline,
            "advanced_model_ids": sorted(
                result.gates.loc[result.gates["advanced"], "model_id"].astype(str).tolist()
            ),
            **artifacts,
            **research_artifacts,
        }
    )
    write_immutable_json(manifest_path, manifest)
    return manifest_path
