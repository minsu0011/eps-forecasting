"""Deterministic leaderboard, Pareto, and oracle-complementarity analysis."""

from __future__ import annotations

import hashlib
from itertools import combinations
import json
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .contracts import EVALUATION_ONLY_TRUTH_COLUMN, ContractError
from .dataset import DEFAULT_IDENTITY_COLUMNS, validate_identity_truth_consistency


ORACLE_DISAGREEMENT_LOG_TOLERANCE = 1.0e-12


def build_leaderboard(
    summary: pd.DataFrame,
    *,
    comparator_model_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    required = {
        "model_id",
        "fair_log_mae",
        "fair_log_rmse",
        "fair_abs_log_error_p95",
        "coverage",
        "runtime_seconds",
        "worst_seed_fair_log_mae",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ContractError(f"leaderboard summary is missing columns: {missing}")
    if summary["model_id"].duplicated().any():
        raise ContractError("leaderboard requires one row per model")
    if len(summary) > 1:
        comparator_models = tuple(comparator_model_ids or ())
        if (
            not comparator_models
            or len(comparator_models) != len(set(comparator_models))
            or not all(isinstance(model, str) and model for model in comparator_models)
        ):
            raise ContractError(
                "comparative leaderboard requires explicit unique comparator_model_ids"
            )
        if set(summary["model_id"]) != set(comparator_models):
            raise ContractError("leaderboard model set does not match comparator_model_ids")
        binding_columns = {
            "identical_common_mask",
            "identity_sha256",
            "identity_row_count",
            "comparator_model_set_sha256",
            "full_coverage",
            "natural_coverage",
            "evaluated_rows",
        }
        missing_binding = sorted(binding_columns.difference(summary.columns))
        if missing_binding:
            raise ContractError(
                f"comparative leaderboard is missing mask bindings: {missing_binding}"
            )
        expected_model_set_sha256 = hashlib.sha256(
            json.dumps(
                tuple(sorted(comparator_models)),
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        identity_hashes = summary["identity_sha256"]
        if (
            not summary["identical_common_mask"]
            .map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value))
            .all()
            or identity_hashes.nunique(dropna=False) != 1
            or not identity_hashes.map(
                lambda value: isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
            ).all()
            or not summary["comparator_model_set_sha256"].eq(expected_model_set_sha256).all()
        ):
            raise ContractError("comparative leaderboard identity/model-set binding mismatch")
        row_counts = pd.to_numeric(summary["identity_row_count"], errors="coerce")
        evaluated_rows = pd.to_numeric(summary["evaluated_rows"], errors="coerce")
        if (
            row_counts.isna().any()
            or row_counts.nunique() != 1
            or not row_counts.eq(evaluated_rows).all()
            or not summary["full_coverage"]
            .map(lambda value: isinstance(value, (bool, np.bool_)) and bool(value))
            .all()
            or not pd.to_numeric(summary["coverage"], errors="coerce").eq(1.0).all()
            or not pd.to_numeric(summary["natural_coverage"], errors="coerce").eq(1.0).all()
        ):
            raise ContractError(
                "comparative leaderboard requires identical row counts and full coverage"
            )
    board = summary.copy()
    finite_metrics = [
        "fair_log_mae",
        "fair_log_rmse",
        "fair_abs_log_error_p95",
        "coverage",
        "worst_seed_fair_log_mae",
    ]
    for column in finite_metrics:
        numeric = pd.to_numeric(board[column], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        if not np.isfinite(numeric).all():
            raise ContractError(f"leaderboard metric {column!r} must be finite")
    runtime = pd.to_numeric(board["runtime_seconds"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    board["_runtime_sort"] = np.where(np.isfinite(runtime), runtime, np.inf)
    board = board.sort_values(
        [
            "fair_log_mae",
            "fair_log_rmse",
            "fair_abs_log_error_p95",
            "worst_seed_fair_log_mae",
            "_runtime_sort",
            "model_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1, dtype=np.int64))
    return board.drop(columns="_runtime_sort")


def pareto_frontier(
    summary: pd.DataFrame,
    *,
    minimize: Sequence[str] = (
        "fair_log_mae",
        "fair_log_rmse",
        "fair_abs_log_error_p95",
        "runtime_seconds",
        "worst_seed_fair_log_mae",
    ),
    maximize: Sequence[str] = ("coverage",),
) -> pd.DataFrame:
    objectives = [*minimize, *maximize]
    if not objectives or len(objectives) != len(set(objectives)):
        raise ContractError("Pareto objectives must be non-empty and unique")
    missing = sorted({"model_id", *objectives}.difference(summary.columns))
    if missing:
        raise ContractError(f"Pareto summary is missing columns: {missing}")
    values = summary.loc[:, objectives].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ContractError("Pareto objectives must all be finite")
    transformed = values.copy()
    for index, column in enumerate(objectives):
        if column in maximize:
            transformed[:, index] *= -1.0
    is_frontier = np.ones(len(summary), dtype=bool)
    for candidate in range(len(summary)):
        for other in range(len(summary)):
            if candidate == other:
                continue
            no_worse = np.all(transformed[other] <= transformed[candidate])
            strictly_better = np.any(transformed[other] < transformed[candidate])
            if no_worse and strictly_better:
                is_frontier[candidate] = False
                break
    frontier = summary.loc[is_frontier].copy()
    return frontier.sort_values(["fair_log_mae", "model_id"], kind="mergesort").reset_index(
        drop=True
    )


def oracle_complementarity(
    prediction_frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
) -> pd.DataFrame:
    """Measure pairwise error complementarity on a strict common mask.

    This is an evaluation-only oracle diagnostic.  It does not generate a
    trainable signal or authorize row-wise model selection.
    """

    keys = tuple(identity_columns)
    required = {*keys, "model_id", "prediction", EVALUATION_ONLY_TRUTH_COLUMN}
    missing = sorted(required.difference(prediction_frame.columns))
    if missing:
        raise ContractError(f"oracle input is missing columns: {missing}")
    if prediction_frame.duplicated([*keys, "model_id"]).any():
        raise ContractError("oracle input must be unique per model and identity")
    validate_identity_truth_consistency(prediction_frame, identity_columns=keys)
    raw_models = prediction_frame["model_id"].unique().tolist()
    if not all(isinstance(value, str) and value for value in raw_models):
        raise ContractError("oracle model_id values must be non-empty strings")
    models = sorted(raw_models)
    if len(models) < 2:
        raise ContractError("oracle complementarity requires at least two models")

    rows: list[dict[str, object]] = []
    for model_a, model_b in combinations(models, 2):
        a = prediction_frame.loc[
            prediction_frame["model_id"] == model_a,
            [*keys, "prediction", EVALUATION_ONLY_TRUTH_COLUMN],
        ].rename(
            columns={
                "prediction": "prediction_a",
                EVALUATION_ONLY_TRUTH_COLUMN: "truth_a",
            }
        )
        b = prediction_frame.loc[
            prediction_frame["model_id"] == model_b,
            [*keys, "prediction", EVALUATION_ONLY_TRUTH_COLUMN],
        ].rename(
            columns={
                "prediction": "prediction_b",
                EVALUATION_ONLY_TRUTH_COLUMN: "truth_b",
            }
        )
        paired = a.merge(b, how="inner", on=list(keys), validate="one_to_one")
        if len(paired) != len(a) or len(paired) != len(b):
            raise ContractError("oracle models must have identical identity sets")
        prediction_a = pd.to_numeric(paired["prediction_a"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        prediction_b = pd.to_numeric(paired["prediction_b"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        truth_a = pd.to_numeric(paired["truth_a"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        truth_b = pd.to_numeric(paired["truth_b"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        common = (
            np.isfinite(prediction_a)
            & (prediction_a > 0.0)
            & np.isfinite(prediction_b)
            & (prediction_b > 0.0)
            & np.isfinite(truth_a)
            & (truth_a > 0.0)
            & np.isfinite(truth_b)
            & (truth_b > 0.0)
            & (truth_a == truth_b)
        )
        if not common.any():
            raise ContractError(f"models {model_a!r} and {model_b!r} have no common fair rows")
        residual_a = np.log(prediction_a[common] / truth_a[common])
        residual_b = np.log(prediction_b[common] / truth_a[common])
        error_a = np.abs(residual_a)
        error_b = np.abs(residual_b)
        mae_a = float(np.mean(error_a))
        mae_b = float(np.mean(error_b))
        oracle_mae = float(np.mean(np.minimum(error_a, error_b)))
        best_single = min(mae_a, mae_b)
        if len(error_a) > 1 and np.std(error_a) > 0.0 and np.std(error_b) > 0.0:
            absolute_error_correlation = float(np.corrcoef(error_a, error_b)[0, 1])
        else:
            absolute_error_correlation = math.nan
        if len(residual_a) > 1 and np.std(residual_a) > 0.0 and np.std(residual_b) > 0.0:
            signed_residual_correlation = float(np.corrcoef(residual_a, residual_b)[0, 1])
        else:
            signed_residual_correlation = math.nan
        prediction_disagreement = np.abs(np.log(prediction_a[common] / prediction_b[common]))
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "common_rows": int(common.sum()),
                "model_a_fair_log_mae": mae_a,
                "model_b_fair_log_mae": mae_b,
                "best_single_fair_log_mae": best_single,
                "oracle_fair_log_mae": oracle_mae,
                "oracle_gain_vs_best_single": best_single - oracle_mae,
                "oracle_relative_gain_vs_best_single": (
                    (best_single - oracle_mae) / best_single if best_single > 0.0 else 0.0
                ),
                "model_a_win_fraction": float(np.mean(error_a < error_b)),
                "model_b_win_fraction": float(np.mean(error_b < error_a)),
                "tie_fraction": float(np.mean(error_a == error_b)),
                "signed_log_residual_correlation": signed_residual_correlation,
                "absolute_error_correlation": absolute_error_correlation,
                "prediction_disagreement_log_tolerance": (ORACLE_DISAGREEMENT_LOG_TOLERANCE),
                "prediction_disagreement_frequency": float(
                    np.mean(prediction_disagreement > ORACLE_DISAGREEMENT_LOG_TOLERANCE)
                ),
                "mean_absolute_log_prediction_disagreement": float(
                    np.mean(prediction_disagreement)
                ),
                "evaluation_only": True,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["oracle_gain_vs_best_single", "model_a", "model_b"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
