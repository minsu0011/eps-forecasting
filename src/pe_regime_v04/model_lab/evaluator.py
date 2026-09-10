"""Evaluation-only fair-P/E metrics for Model Lab OOS predictions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import EVALUATION_ONLY_TRUTH_COLUMN, ContractError
from .dataset import (
    DEFAULT_IDENTITY_COLUMNS,
    FORMAL_EVALUATION_START,
    apply_identical_common_mask,
    evaluation_identity_sha256,
    validate_identity_truth_consistency,
)


SUMMARY_COLUMNS = (
    "model_id",
    "eligible_truth_rows",
    "evaluated_rows",
    "coverage",
    "natural_coverage",
    "fair_log_mae",
    "fair_log_rmse",
    "fair_abs_log_error_median",
    "fair_abs_log_error_p90",
    "fair_abs_log_error_p95",
    "fair_log_bias",
    "runtime_seconds",
    "worst_seed",
    "worst_seed_fair_log_mae",
    "identical_common_mask",
    "identity_sha256",
    "identity_row_count",
    "comparator_model_set_sha256",
    "full_coverage",
)


@dataclass(frozen=True)
class EvaluationResult:
    summary: pd.DataFrame
    per_seed: pd.DataFrame
    row_errors: pd.DataFrame


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)


def _metric_payload(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    eligible = np.isfinite(truth) & (truth > 0.0)
    valid = eligible & np.isfinite(prediction) & (prediction > 0.0)
    payload: dict[str, float | int] = {
        "eligible_truth_rows": int(eligible.sum()),
        "evaluated_rows": int(valid.sum()),
        "coverage": float(valid.sum() / eligible.sum()) if eligible.any() else 0.0,
        "fair_log_mae": math.nan,
        "fair_log_rmse": math.nan,
        "fair_abs_log_error_median": math.nan,
        "fair_abs_log_error_p90": math.nan,
        "fair_abs_log_error_p95": math.nan,
        "fair_log_bias": math.nan,
    }
    if valid.any():
        signed = np.log(prediction[valid] / truth[valid])
        absolute = np.abs(signed)
        payload.update(
            {
                "fair_log_mae": float(np.mean(absolute)),
                "fair_log_rmse": float(np.sqrt(np.mean(signed * signed))),
                "fair_abs_log_error_median": float(np.quantile(absolute, 0.50)),
                "fair_abs_log_error_p90": float(np.quantile(absolute, 0.90)),
                "fair_abs_log_error_p95": float(np.quantile(absolute, 0.95)),
                "fair_log_bias": float(np.mean(signed)),
            }
        )
    return payload


def evaluate_fair_predictions(
    prediction_frame: pd.DataFrame,
    *,
    runtime_seconds_by_model: Mapping[str, float] | None = None,
    common_mask_model_ids: Sequence[str] | None = None,
    identity_columns: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    evaluation_start: str | pd.Timestamp | None = FORMAL_EVALUATION_START,
) -> EvaluationResult:
    """Evaluate fair metrics; this is the only Model Lab API that consumes truth."""

    keys = tuple(identity_columns)
    required = {*keys, "model_id", "prediction", EVALUATION_ONLY_TRUTH_COLUMN}
    missing = sorted(required.difference(prediction_frame.columns))
    if missing:
        raise ContractError(f"evaluation frame is missing columns: {missing}")
    if "date" not in keys:
        raise ContractError("identity_columns must include date")
    frame = prediction_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ContractError("evaluation dates must be valid")
    if evaluation_start is not None:
        frame = frame.loc[frame["date"] >= pd.Timestamp(evaluation_start)].copy()
    if frame.empty:
        raise ContractError("no predictions remain in the evaluation interval")
    if frame["model_id"].isna().any() or (frame["model_id"].astype(str).str.len() == 0).any():
        raise ContractError("model_id must be non-empty")
    if frame.duplicated([*keys, "model_id"]).any():
        raise ContractError("predictions must be unique per model and identity key")
    validate_identity_truth_consistency(frame, identity_columns=keys)

    natural_coverage: dict[str, float] = {}
    for model_id, group in frame.groupby("model_id", sort=True):
        natural = _metric_payload(
            _numeric(group["prediction"]), _numeric(group[EVALUATION_ONLY_TRUTH_COLUMN])
        )
        natural_coverage[str(model_id)] = float(natural["coverage"])

    identical_common_mask = common_mask_model_ids is not None
    comparator_model_set_sha256: str | None = None
    if common_mask_model_ids is not None:
        comparator_models = tuple(sorted(common_mask_model_ids))
        comparator_model_set_sha256 = hashlib.sha256(
            json.dumps(
                comparator_models,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        frame = apply_identical_common_mask(
            frame,
            required_model_ids=common_mask_model_ids,
            identity_columns=keys,
        )

    runtime = {} if runtime_seconds_by_model is None else dict(runtime_seconds_by_model)
    evaluated_models = {str(value) for value in frame["model_id"].unique()}
    unknown_runtime = sorted(set(runtime).difference(evaluated_models))
    if unknown_runtime:
        raise ContractError(f"runtime mapping contains unknown models: {unknown_runtime}")
    for model_id, seconds in runtime.items():
        if not math.isfinite(float(seconds)) or float(seconds) < 0.0:
            raise ContractError(f"runtime for {model_id!r} must be finite and non-negative")

    error_frames: list[pd.DataFrame] = []
    per_seed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model_id, group in frame.groupby("model_id", sort=True):
        model_name = str(model_id)
        prediction = _numeric(group["prediction"])
        truth = _numeric(group[EVALUATION_ONLY_TRUTH_COLUMN])
        eligible = np.isfinite(truth) & (truth > 0.0)
        valid = eligible & np.isfinite(prediction) & (prediction > 0.0)
        signed = np.full(len(group), np.nan, dtype=np.float64)
        signed[valid] = np.log(prediction[valid] / truth[valid])
        errors = group.loc[:, [*keys, "model_id"]].copy()
        if "fold_id" in group:
            errors["fold_id"] = group["fold_id"].to_numpy()
        errors["eligible_truth"] = eligible
        errors["valid_prediction"] = valid
        errors["fair_log_error"] = signed
        errors["fair_abs_log_error"] = np.abs(signed)
        error_frames.append(errors)

        seed_metrics: list[dict[str, object]] = []
        if "seed" not in group:
            raise ContractError("evaluation requires a seed column for worst-seed reporting")
        for seed, seed_group in group.groupby("seed", sort=True):
            metrics = _metric_payload(
                _numeric(seed_group["prediction"]),
                _numeric(seed_group[EVALUATION_ONLY_TRUTH_COLUMN]),
            )
            row = {"model_id": model_name, "seed": seed, **metrics}
            seed_metrics.append(row)
            per_seed_rows.append(row)

        finite_seed_metrics = [
            row for row in seed_metrics if math.isfinite(float(row["fair_log_mae"]))
        ]
        if finite_seed_metrics:
            worst = max(
                finite_seed_metrics,
                key=lambda row: (float(row["fair_log_mae"]), str(row["seed"])),
            )
            worst_seed: object = worst["seed"]
            worst_seed_mae = float(worst["fair_log_mae"])
        else:
            worst_seed = None
            worst_seed_mae = math.nan

        aggregate = _metric_payload(prediction, truth)
        evaluated_identity = group.loc[valid, list(keys)]
        summary_rows.append(
            {
                "model_id": model_name,
                **aggregate,
                "natural_coverage": natural_coverage[model_name],
                "runtime_seconds": float(runtime.get(model_name, math.nan)),
                "worst_seed": worst_seed,
                "worst_seed_fair_log_mae": worst_seed_mae,
                "identical_common_mask": identical_common_mask,
                "identity_sha256": evaluation_identity_sha256(
                    evaluated_identity,
                    identity_columns=keys,
                ),
                "identity_row_count": int(len(evaluated_identity)),
                "comparator_model_set_sha256": comparator_model_set_sha256,
                "full_coverage": bool(
                    aggregate["eligible_truth_rows"] > 0
                    and aggregate["evaluated_rows"] == aggregate["eligible_truth_rows"]
                    and natural_coverage[model_name] == 1.0
                ),
            }
        )

    summary = pd.DataFrame(summary_rows).loc[:, list(SUMMARY_COLUMNS)]
    per_seed = (
        pd.DataFrame(per_seed_rows)
        .sort_values(["model_id", "seed"], kind="mergesort")
        .reset_index(drop=True)
    )
    row_errors = (
        pd.concat(error_frames, ignore_index=True)
        .sort_values([*keys, "model_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return EvaluationResult(summary=summary, per_seed=per_seed, row_errors=row_errors)
