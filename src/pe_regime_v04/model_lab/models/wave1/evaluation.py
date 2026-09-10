"""Truth-isolated Wave-1 evaluation, gates, and deterministic advancement."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ...contracts import EVALUATION_ONLY_TRUTH_COLUMN, ContractError
from ...evaluator import EvaluationResult, evaluate_fair_predictions
from .artifacts import PREDICTION_COLUMNS
from .spec import (
    BASELINE_MODEL_IDS,
    CANDIDATE_MODEL_IDS,
    DIRECT_REGIME_REFERENCE_ID,
    EVALUATION_START,
    GATE_POLICY,
    MODEL_BY_ID,
    PRIMARY_BASELINE_IDS,
)


GATE_COLUMNS = (
    "model_id",
    "family",
    "coverage_pass",
    "fair_log_mae",
    "fair_log_rmse",
    "mae_primary_comparator",
    "rmse_primary_comparator",
    "mae_relative_gain",
    "rmse_relative_gain",
    "worst_seed_relative_degradation",
    "worst_seed_pass",
    "primary_path_pass",
    "complementarity_baseline",
    "absolute_error_pearson_correlation",
    "oracle_fair_log_mae",
    "oracle_relative_gain",
    "complementarity_path_pass",
    "direct_regime_reference_mae_relative_gain",
    "direct_regime_reference_rmse_relative_gain",
    "stage1_gate_pass",
    "advanced",
    "decision",
)


@dataclass(frozen=True)
class Wave1Evaluation:
    joined_predictions: pd.DataFrame
    metrics: EvaluationResult
    gates: pd.DataFrame
    common_identity_rows: int
    expected_identity_rows: int
    complementarity_baseline: str


def _metric_row(summary: pd.DataFrame, model_id: str) -> pd.Series:
    rows = summary.loc[summary["model_id"] == model_id]
    if len(rows) != 1:
        raise ContractError(f"evaluation summary must contain exactly one row for {model_id}")
    return rows.iloc[0]


def _relative_gain(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0.0 or not math.isfinite(candidate):
        return math.nan
    return (baseline - candidate) / baseline


def _strict_truth_frame(truth_by_seed: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed, truth in sorted(truth_by_seed.items()):
        required = {"date", EVALUATION_ONLY_TRUTH_COLUMN}
        missing = sorted(required.difference(truth.columns))
        if missing:
            raise ContractError(f"truth frame for seed {seed} is missing columns: {missing}")
        selected = truth.loc[:, ["date", EVALUATION_ONLY_TRUTH_COLUMN]].copy()
        selected.insert(0, "seed", int(seed))
        selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
        if selected["date"].isna().any() or selected.duplicated(["seed", "date"]).any():
            raise ContractError(f"truth frame for seed {seed} has invalid/duplicate dates")
        rows.append(selected)
    if not rows:
        raise ContractError("evaluation requires at least one truth frame")
    return pd.concat(rows, ignore_index=True)


def _error_vectors(
    joined: pd.DataFrame,
    *,
    candidate_id: str,
    baseline_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = joined.loc[
        joined["model_id"] == candidate_id,
        ["seed", "date", "prediction", EVALUATION_ONLY_TRUTH_COLUMN],
    ].rename(columns={"prediction": "candidate"})
    baseline = joined.loc[
        joined["model_id"] == baseline_id,
        ["seed", "date", "prediction", EVALUATION_ONLY_TRUTH_COLUMN],
    ].rename(
        columns={
            "prediction": "baseline",
            EVALUATION_ONLY_TRUTH_COLUMN: "baseline_truth",
        }
    )
    paired = candidate.merge(
        baseline,
        on=["seed", "date"],
        how="inner",
        validate="one_to_one",
    )
    truth = pd.to_numeric(paired[EVALUATION_ONLY_TRUTH_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    baseline_truth = pd.to_numeric(paired["baseline_truth"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    candidate_prediction = pd.to_numeric(paired["candidate"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    baseline_prediction = pd.to_numeric(paired["baseline"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    valid = (
        np.isfinite(truth)
        & (truth > 0.0)
        & (truth == baseline_truth)
        & np.isfinite(candidate_prediction)
        & (candidate_prediction > 0.0)
        & np.isfinite(baseline_prediction)
        & (baseline_prediction > 0.0)
    )
    return (
        np.abs(np.log(candidate_prediction[valid] / truth[valid])),
        np.abs(np.log(baseline_prediction[valid] / truth[valid])),
    )


def _worst_seed_degradation(
    per_seed: pd.DataFrame,
    *,
    candidate_id: str,
    seeds: Sequence[int],
) -> float:
    degradations: list[float] = []
    for seed in seeds:
        candidate_rows = per_seed.loc[
            (per_seed["model_id"] == candidate_id) & (per_seed["seed"] == seed)
        ]
        if len(candidate_rows) != 1:
            return math.inf
        for metric in ("fair_log_mae", "fair_log_rmse"):
            baseline_losses: list[float] = []
            for baseline_id in PRIMARY_BASELINE_IDS:
                baseline_rows = per_seed.loc[
                    (per_seed["model_id"] == baseline_id) & (per_seed["seed"] == seed)
                ]
                if len(baseline_rows) != 1:
                    return math.inf
                baseline_losses.append(float(baseline_rows.iloc[0][metric]))
            baseline = min(baseline_losses)
            candidate = float(candidate_rows.iloc[0][metric])
            if not math.isfinite(baseline) or baseline <= 0.0 or not math.isfinite(candidate):
                return math.inf
            degradations.append((candidate - baseline) / baseline)
    return max(degradations) if degradations else math.inf


def evaluate_wave1_predictions(
    predictions: pd.DataFrame,
    truth_by_seed: Mapping[int, pd.DataFrame],
    *,
    runtime_seconds_by_model: Mapping[str, float],
    candidate_model_ids: Sequence[str] = CANDIDATE_MODEL_IDS,
    baseline_model_ids: Sequence[str] = BASELINE_MODEL_IDS,
    strict_expected_rows_per_seed: int | None = 1296,
) -> Wave1Evaluation:
    """Evaluate only after a sealed prediction artifact has been produced."""

    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise ContractError("prediction artifact schema/order differs from the execution binding")
    candidates = tuple(candidate_model_ids)
    baselines = tuple(baseline_model_ids)
    models = (*candidates, *baselines)
    if len(models) != len(set(models)):
        raise ContractError("candidate and baseline model ids must be disjoint and unique")
    unknown_candidates = sorted(set(candidates).difference(MODEL_BY_ID))
    if unknown_candidates:
        raise ContractError(f"unknown candidate models: {unknown_candidates}")
    actual_models = set(predictions["model_id"].astype(str).unique())
    if actual_models != set(models):
        raise ContractError(
            f"prediction model universe mismatch: expected={sorted(models)}, actual={sorted(actual_models)}"
        )
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any() or frame.duplicated(["seed", "date", "model_id"]).any():
        raise ContractError("prediction identities must be valid and unique")
    frame = frame.loc[frame["date"] >= pd.Timestamp(EVALUATION_START)].copy()
    truth = _strict_truth_frame(truth_by_seed)
    truth = truth.loc[truth["date"] >= pd.Timestamp(EVALUATION_START)].copy()
    seeds = tuple(sorted(int(value) for value in truth_by_seed))
    if set(frame["seed"].astype(int).unique()) != set(seeds):
        raise ContractError("prediction/truth seed universe differs")

    counts = truth.groupby("seed", sort=True).size()
    if strict_expected_rows_per_seed is not None and not (
        counts == int(strict_expected_rows_per_seed)
    ).all():
        raise ContractError(
            f"truth evaluation rows per seed must equal {strict_expected_rows_per_seed}: {counts.to_dict()}"
        )
    expected_identity_rows = len(truth)
    expected_keys = truth.loc[:, ["seed", "date"]]
    for model_id in models:
        keys = frame.loc[frame["model_id"] == model_id, ["seed", "date"]]
        if len(keys) != expected_identity_rows:
            raise ContractError(f"model {model_id} row count differs from the truth key universe")
        merged_keys = expected_keys.merge(
            keys,
            how="outer",
            on=["seed", "date"],
            validate="one_to_one",
            indicator=True,
        )
        if not merged_keys["_merge"].eq("both").all():
            raise ContractError(f"model {model_id} identity keys differ from truth")

    joined = frame.merge(
        truth,
        on=["seed", "date"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    fair = pd.to_numeric(joined[EVALUATION_ONLY_TRUTH_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if not (np.isfinite(fair) & (fair > 0.0)).all():
        raise ContractError("evaluation truth must be entirely positive and finite")

    metrics = evaluate_fair_predictions(
        joined,
        runtime_seconds_by_model=runtime_seconds_by_model,
        common_mask_model_ids=models,
        identity_columns=("seed", "date"),
        evaluation_start=None,
    )
    prediction_values = pd.to_numeric(joined["prediction"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    valid_rows = np.isfinite(prediction_values) & (prediction_values > 0.0)
    valid_matrix = (
        joined.assign(_valid=valid_rows)
        .pivot(index=["seed", "date"], columns="model_id", values="_valid")
        .reindex(columns=list(models))
    )
    common_identity_rows = int(valid_matrix.all(axis=1).sum())
    coverage_complete = common_identity_rows == expected_identity_rows

    summary = metrics.summary
    primary_rows = {model_id: _metric_row(summary, model_id) for model_id in PRIMARY_BASELINE_IDS}
    primary_loss: dict[str, float] = {}
    for metric in ("fair_log_mae", "fair_log_rmse"):
        primary_loss[metric] = min(float(primary_rows[model_id][metric]) for model_id in PRIMARY_BASELINE_IDS)
    fixed_complementarity_baseline = min(
        PRIMARY_BASELINE_IDS,
        key=lambda model_id: (float(primary_rows[model_id]["fair_log_mae"]), model_id),
    )
    direct = _metric_row(summary, DIRECT_REGIME_REFERENCE_ID)

    gate_rows: list[dict[str, object]] = []
    for model_id in candidates:
        candidate = _metric_row(summary, model_id)
        mae = float(candidate["fair_log_mae"])
        rmse = float(candidate["fair_log_rmse"])
        mae_gain = _relative_gain(primary_loss["fair_log_mae"], mae)
        rmse_gain = _relative_gain(primary_loss["fair_log_rmse"], rmse)
        worst_degradation = _worst_seed_degradation(
            metrics.per_seed,
            candidate_id=model_id,
            seeds=seeds,
        )
        natural_coverage = float(candidate["natural_coverage"])
        coverage_pass = coverage_complete and natural_coverage == 1.0
        worst_pass = worst_degradation <= float(
            GATE_POLICY["max_worst_seed_relative_degradation_over_both_primary_metrics"]
        )
        primary_gain = float(GATE_POLICY["primary_metric_min_relative_gain"])
        other_floor = -float(GATE_POLICY["primary_other_metric_max_relative_degradation"])
        primary_path = (
            (mae_gain >= primary_gain and rmse_gain >= other_floor)
            or (rmse_gain >= primary_gain and mae_gain >= other_floor)
        )

        candidate_error, baseline_error = _error_vectors(
            joined,
            candidate_id=model_id,
            baseline_id=fixed_complementarity_baseline,
        )
        if (
            len(candidate_error) == expected_identity_rows
            and np.std(candidate_error) > 0.0
            and np.std(baseline_error) > 0.0
        ):
            correlation = float(np.corrcoef(candidate_error, baseline_error)[0, 1])
        else:
            correlation = math.nan
        oracle_mae = (
            float(np.mean(np.minimum(candidate_error, baseline_error)))
            if len(candidate_error) == expected_identity_rows
            else math.nan
        )
        baseline_mae = float(np.mean(baseline_error)) if len(baseline_error) else math.nan
        oracle_gain = _relative_gain(baseline_mae, oracle_mae)
        complementarity_path = (
            mae_gain
            >= -float(
                GATE_POLICY["complementarity_max_relative_degradation_each_primary_metric"]
            )
            and rmse_gain
            >= -float(
                GATE_POLICY["complementarity_max_relative_degradation_each_primary_metric"]
            )
            and math.isfinite(correlation)
            and correlation
            <= float(GATE_POLICY["complementarity_max_pearson_absolute_error_correlation"])
            and oracle_gain
            >= float(GATE_POLICY["complementarity_min_oracle_relative_gain_mae"])
        )
        stage1_pass = bool(coverage_pass and worst_pass and (primary_path or complementarity_path))
        gate_rows.append(
            {
                "model_id": model_id,
                "family": MODEL_BY_ID[model_id].family,
                "coverage_pass": bool(coverage_pass),
                "fair_log_mae": mae,
                "fair_log_rmse": rmse,
                "mae_primary_comparator": primary_loss["fair_log_mae"],
                "rmse_primary_comparator": primary_loss["fair_log_rmse"],
                "mae_relative_gain": mae_gain,
                "rmse_relative_gain": rmse_gain,
                "worst_seed_relative_degradation": worst_degradation,
                "worst_seed_pass": bool(worst_pass),
                "primary_path_pass": bool(primary_path),
                "complementarity_baseline": fixed_complementarity_baseline,
                "absolute_error_pearson_correlation": correlation,
                "oracle_fair_log_mae": oracle_mae,
                "oracle_relative_gain": oracle_gain,
                "complementarity_path_pass": bool(complementarity_path),
                "direct_regime_reference_mae_relative_gain": _relative_gain(
                    float(direct["fair_log_mae"]), mae
                ),
                "direct_regime_reference_rmse_relative_gain": _relative_gain(
                    float(direct["fair_log_rmse"]), rmse
                ),
                "stage1_gate_pass": stage1_pass,
                "advanced": False,
                "decision": "PASS_NOT_ADVANCED_CAP" if stage1_pass else "REJECT_GATE",
            }
        )

    gates = pd.DataFrame(gate_rows).loc[:, list(GATE_COLUMNS)]
    passing = gates.loc[gates["stage1_gate_pass"]].copy()
    if not passing.empty:
        family_winners = (
            passing.sort_values(
                ["family", "fair_log_mae", "fair_log_rmse", "model_id"],
                kind="mergesort",
            )
            .groupby("family", sort=True, as_index=False)
            .head(1)
            .sort_values(
                ["fair_log_mae", "fair_log_rmse", "family", "model_id"],
                kind="mergesort",
            )
            .head(int(GATE_POLICY["max_families_advanced"]))
        )
        advanced_ids = set(family_winners["model_id"].astype(str))
        gates.loc[gates["model_id"].isin(advanced_ids), "advanced"] = True
        gates.loc[gates["model_id"].isin(advanced_ids), "decision"] = "ADVANCE_STAGE2"
    gates = gates.sort_values(
        ["advanced", "fair_log_mae", "fair_log_rmse", "family", "model_id"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return Wave1Evaluation(
        joined_predictions=joined,
        metrics=metrics,
        gates=gates,
        common_identity_rows=common_identity_rows,
        expected_identity_rows=expected_identity_rows,
        complementarity_baseline=fixed_complementarity_baseline,
    )
