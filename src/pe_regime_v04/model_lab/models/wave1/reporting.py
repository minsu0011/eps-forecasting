"""Immutable common-matrix, per-family, and registry-ready Wave-1 outputs."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ...contracts import EVALUATION_ONLY_TRUTH_COLUMN, ContractError
from ...matrices import canonicalize_oos_matrices, write_oos_matrices
from .artifacts import (
    canonical_json_bytes,
    file_record,
    seal_payload,
    write_immutable_json,
    write_immutable_text,
    write_round_trip_csv,
)
from .evaluation import Wave1Evaluation
from .registration import planned_model_registrations
from .spec import DESIGN_LOCK_SHA256, MODEL_BY_ID


def frame_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [
        {"name": str(column), "dtype": str(frame[column].dtype)} for column in frame.columns
    ]
    return hashlib.sha256(canonical_json_bytes(schema)).hexdigest()


def _common_mask_joined(
    result: Wave1Evaluation,
    *,
    model_ids: Sequence[str],
) -> pd.DataFrame:
    models = tuple(model_ids)
    joined = result.joined_predictions.copy()
    prediction = pd.to_numeric(joined["prediction"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    joined["_valid"] = np.isfinite(prediction) & (prediction > 0.0)
    valid = joined.pivot(
        index=["seed", "date"], columns="model_id", values="_valid"
    ).reindex(columns=list(models))
    common_keys = valid.index[valid.notna().all(axis=1) & valid.all(axis=1)].to_frame(
        index=False
    )
    common = joined.merge(
        common_keys,
        on=["seed", "date"],
        how="inner",
        validate="many_to_one",
    ).drop(columns="_valid")
    if len(common) != result.common_identity_rows * len(models):
        raise ContractError("common-mask OOS row count differs from evaluation evidence")
    return common


def _write_parquet_immutable(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable Parquet already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Parquet temporary path already exists: {temporary}")
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        reread = pd.read_parquet(temporary, engine="pyarrow")
        assert_frame_equal(frame, reread, check_exact=True, check_dtype=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        **file_record(path),
        "schema_sha256": frame_schema_sha256(frame),
        "rows": len(frame),
        "columns": list(frame.columns),
    }


def registry_ready_records(result: Wave1Evaluation) -> dict[str, Any]:
    summary = result.metrics.summary.set_index("model_id")
    gates = result.gates.set_index("model_id")
    records: list[dict[str, Any]] = []
    definitions = {item.model_id: item for item in planned_model_registrations()}
    for model_id, definition in definitions.items():
        metric = summary.loc[model_id]
        gate = gates.loc[model_id]
        passed = bool(gate["stage1_gate_pass"])
        advanced = bool(gate["advanced"])
        if advanced:
            screen_status = "SCREEN_ADVANCE_STAGE2_PROPOSED"
        elif passed:
            screen_status = "SCREEN_PASS_NOT_ADVANCED"
        else:
            screen_status = "SCREEN_REJECTED"
        event = replace(
            definition,
            registry_revision=1,
            # Wave1 uses already-spent seeds for a cheap screen.  It is not the
            # fresh Stage2 tuning event that may transition tuning_status.
            tuning_status="UNTESTED",
            locked_status="UNTESTED",
            heldout_status="NOT_OPENED",
            fair_log_mae=float(metric["fair_log_mae"]),
            fair_log_rmse=float(metric["fair_log_rmse"]),
            worst_seed=int(metric["worst_seed"]),
            compute_time=float(metric["runtime_seconds"]),
            status="RESEARCH_ONLY",
            notes=(
                "Spent-seed Wave1 cheap screen only; Stage2 tuning remains UNTESTED; "
                "no lock, fresh seed, or heldout access is authorized. "
                f"Screen={screen_status}; gate={gate['decision']}; "
                "target=log(observed_pe)."
            ),
        )
        records.append(event.to_record())
    screen_decisions = []
    for model_id in definitions:
        gate = gates.loc[model_id]
        if bool(gate["advanced"]):
            screen_status = "SCREEN_ADVANCE_STAGE2_PROPOSED"
        elif bool(gate["stage1_gate_pass"]):
            screen_status = "SCREEN_PASS_NOT_ADVANCED"
        else:
            screen_status = "SCREEN_REJECTED"
        screen_decisions.append(
            {
                "model_id": model_id,
                "screen_status": screen_status,
                "gate_decision": str(gate["decision"]),
                "stage2_tuning_authorized": False,
                "lock_authorized": False,
            }
        )
    return {
        "format_version": 1,
        "design_lock_sha256": DESIGN_LOCK_SHA256,
        "mutation_authorized": False,
        "evidence_stage": "SPENT_SEED_STAGE1_CHEAP_SCREEN",
        "stage2_tuning_authorized": False,
        "lock_authorized": False,
        "definition_records": [item.to_record() for item in definitions.values()],
        "result_records": records,
        "screen_decisions": screen_decisions,
    }


def _family_report_text(
    family: str,
    model_ids: Sequence[str],
    results: pd.DataFrame,
) -> str:
    lines = [
        f"# Wave-1 family report: {family}",
        "",
        "## Hypothesis and architecture",
        "",
        "This family contributes a distinct fixed inductive bias to expected P/E.",
        "",
        "Target: `log(observed_pe)` (natural log); predictions are exponentiated and positive.",
        "",
        "Evidence: spent-seed cheap screen only; no fresh or heldout evidence.",
        "",
        "Architecture/variants: "
        + ", ".join(
            f"`{model_id}` ({MODEL_BY_ID[model_id].variant})" for model_id in model_ids
        ),
        "",
        "Reference packages: "
        + ", ".join(
            sorted({MODEL_BY_ID[model_id].package for model_id in model_ids})
        ),
        "",
        "Features: "
        + "; ".join(
            f"{model_id}={len(MODEL_BY_ID[model_id].feature_columns)} locked columns"
            for model_id in model_ids
        ),
        "",
        "## Results and comparison",
        "",
        "| Model | MAE | RMSE | Worst seed | Runtime s | MAE gain | RMSE gain | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.fair_log_mae:.12g} | "
            f"{row.fair_log_rmse:.12g} | {row.worst_seed} | "
            f"{row.runtime_seconds:.12g} | {row.mae_relative_gain:.12g} | "
            f"{row.rmse_relative_gain:.12g} | {row.decision} |"
        )
    lines.extend(
        [
            "",
            "## Failure analysis and decision",
            "",
            "Failures are determined by full common coverage, both-primary-baseline gains, "
            "worst-seed cap, and the locked complementarity alternative. Fold failures remain "
            "NaN with no retry or fallback.",
            "",
            "The decision above is mechanical from the frozen Stage-1 gate; there is no "
            "report-time override. `advanced` and every component diagnostic remain in RESULTS.csv.",
            "",
        ]
    )
    return "\n".join(lines)


def write_wave1_research_outputs(
    result: Wave1Evaluation,
    output_directory: Path,
    *,
    execution_binding: Mapping[str, Any],
    execution_precommit: Mapping[str, Any],
    candidate_model_ids: Sequence[str],
    baseline_model_ids: Sequence[str],
) -> dict[str, Any]:
    """Emit common-mask and per-family artifacts without mutating registries."""

    all_models = tuple((*candidate_model_ids, *baseline_model_ids))
    common = _common_mask_joined(result, model_ids=all_models)
    matrix_input = common.loc[
        :, ["seed", "date", "model_id", "prediction", EVALUATION_ONLY_TRUTH_COLUMN]
    ]
    matrix_directory = Path(output_directory) / "common_oos_matrix"
    matrix = write_oos_matrices(
        matrix_input,
        matrix_directory,
        stem="COMMON_MASK_OOS_PREDICTIONS",
        identity_columns=("seed", "date"),
        parquet="required",
        parquet_engine="pyarrow",
    )
    canonical = canonicalize_oos_matrices(matrix_input, identity_columns=("seed", "date"))
    parquet_long = pd.read_parquet(matrix.long_parquet, engine="pyarrow")
    parquet_wide = pd.read_parquet(matrix.wide_parquet, engine="pyarrow")
    assert_frame_equal(canonical.long, parquet_long, check_exact=True, check_dtype=True)
    assert_frame_equal(canonical.wide, parquet_wide, check_exact=True, check_dtype=True)
    expected_wide = [
        "seed",
        "date",
        EVALUATION_ONLY_TRUTH_COLUMN,
        *[f"prediction__{model_id}" for model_id in sorted(all_models)],
    ]
    if list(canonical.wide.columns) != expected_wide:
        raise ContractError("common-mask wide matrix schema differs from the model universe")
    common_artifacts = {
        "long_csv": file_record(matrix.long_csv),
        "wide_csv": file_record(matrix.wide_csv),
        "long_parquet": file_record(matrix.long_parquet),
        "wide_parquet": file_record(matrix.wide_parquet),
        "long_schema_sha256": frame_schema_sha256(canonical.long),
        "wide_schema_sha256": frame_schema_sha256(canonical.wide),
        "identity_rows": len(canonical.wide),
        "model_columns": len(all_models),
    }

    family_artifacts: list[dict[str, Any]] = []
    summary = result.metrics.summary
    for family in sorted({MODEL_BY_ID[model_id].family for model_id in candidate_model_ids}):
        family_models = tuple(
            model_id
            for model_id in candidate_model_ids
            if MODEL_BY_ID[model_id].family == family
        )
        family_directory = Path(output_directory) / "families" / family
        paths = {
            "report": family_directory / "MODEL_REPORT.md",
            "results": family_directory / "RESULTS.csv",
            "config": family_directory / "CONFIG.json",
            "oos": family_directory / "OOS_PREDICTIONS.parquet",
        }
        if any(path.exists() for path in paths.values()):
            raise FileExistsError(f"family artifacts already exist for {family}")
        family_results = (
            summary.loc[summary["model_id"].isin(family_models)]
            .merge(
                result.gates,
                on="model_id",
                how="left",
                validate="one_to_one",
                suffixes=("", "_gate"),
            )
            .sort_values("model_id", kind="mergesort")
            .reset_index(drop=True)
        )
        results_record = write_round_trip_csv(
            family_results,
            paths["results"],
            columns=tuple(family_results.columns),
        )
        family_input = matrix_input.loc[matrix_input["model_id"].isin(family_models)]
        family_wide = canonicalize_oos_matrices(
            family_input, identity_columns=("seed", "date")
        ).wide
        oos_record = _write_parquet_immutable(family_wide, paths["oos"])
        config = seal_payload(
            {
                "format_version": 1,
                "mode": "wave1_family_config",
                "design_lock_sha256": DESIGN_LOCK_SHA256,
                "execution_binding": dict(execution_binding),
                "execution_precommit": dict(execution_precommit),
                "family": family,
                "model_ids": list(family_models),
                "target": {
                    "raw_column": "observed_pe",
                    "fit_transform": "natural_log",
                    "prediction_transform": "exponential",
                },
                "models": [
                    {
                        "model_id": model_id,
                        "variant": MODEL_BY_ID[model_id].variant,
                        "features": list(MODEL_BY_ID[model_id].feature_columns),
                        "parameters": dict(MODEL_BY_ID[model_id].parameters),
                        "package": MODEL_BY_ID[model_id].package,
                        "license": MODEL_BY_ID[model_id].license,
                    }
                    for model_id in family_models
                ],
            }
        )
        write_immutable_json(paths["config"], config)
        report_text = _family_report_text(family, family_models, family_results)
        write_immutable_text(paths["report"], report_text)
        family_artifacts.append(
            {
                "family": family,
                "model_ids": list(family_models),
                "model_report": file_record(paths["report"]),
                "results_csv": {
                    **results_record,
                    "schema_sha256": frame_schema_sha256(family_results),
                },
                "config_json": file_record(paths["config"]),
                "oos_predictions_parquet": oos_record,
            }
        )

    registry_payload = seal_payload(
        {
            **registry_ready_records(result),
            "mode": "registry_ready_wave1_decisions",
            "execution_binding": dict(execution_binding),
            "execution_precommit": dict(execution_precommit),
        }
    )
    registry_path = Path(output_directory) / "REGISTRY_READY_DECISIONS.json"
    write_immutable_json(registry_path, registry_payload)
    return {
        "common_oos_matrix": common_artifacts,
        "family_artifacts": family_artifacts,
        "registry_ready_decisions": file_record(registry_path),
    }
