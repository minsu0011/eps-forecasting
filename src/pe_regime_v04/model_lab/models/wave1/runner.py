"""CPU-only deterministic fold runner for Wave-1 prediction mode."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import re
import time
from typing import Any, Sequence
import warnings

import numpy as np
import pandas as pd

from ...contracts import ContractError, FitContext, PredictContext
from ...folds import PITFold, PITFoldSpec, generate_pit_folds
from .adapters import Wave1StateSpaceAdapter, build_wave1_model
from .artifacts import DIAGNOSTIC_COLUMNS, PREDICTION_COLUMNS
from .resources import (
    ResourceGuard,
    ResourcePolicyError,
    assert_cpu_environment,
    assert_threadpools_one,
    set_full_process_affinity_0_31,
)
from .spec import (
    BASELINE_MODEL_IDS,
    CANDIDATE_MODEL_IDS,
    COMMON_FEATURES,
    MODEL_BY_ID,
    RESOURCE_POLICY,
    feature_metadata,
)


FOLD_SPEC = PITFoldSpec(
    min_train_sessions=252,
    max_train_sessions=1008,
    test_sessions=21,
    step_sessions=21,
    embargo_sessions=0,
    allow_partial_final_test=True,
)
STATE_SPACE_ALLOWED_WARNING = (
    "Value of `irregular` may be overridden when the trend component is specified using a model string."
)


@dataclass(frozen=True)
class FoldTask:
    seed: int
    fold: PITFold
    model_id: str
    model_frame: pd.DataFrame
    eligible_train_positions: tuple[int, ...]


@dataclass(frozen=True)
class FoldResult:
    predictions: pd.DataFrame
    diagnostic: dict[str, Any]


@dataclass(frozen=True)
class SeedPredictionResult:
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame
    runtime_seconds_by_model: dict[str, float]
    folds: tuple[PITFold, ...]
    resource_summary: dict[str, int]
    threadpool_inventory: list[dict[str, Any]]
    affinity: tuple[int, ...] | None


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _empty_predictions(task: FoldTask) -> pd.DataFrame:
    test = task.model_frame.iloc[list(task.fold.test_positions)]
    return pd.DataFrame(
        {
            "seed": task.seed,
            "date": pd.to_datetime(test["date"]).to_numpy(),
            "symbol": test["symbol"].astype(str).to_numpy(),
            "fold_id": task.fold.fold_id,
            "test_start_position": task.fold.test_positions[0],
            "model_id": task.model_id,
            "prediction": np.full(len(test), np.nan, dtype=np.float64),
        },
        columns=PREDICTION_COLUMNS,
    )


def _diagnostic_base(task: FoldTask) -> dict[str, Any]:
    return {
        "seed": task.seed,
        "fold_id": task.fold.fold_id,
        "test_start_position": task.fold.test_positions[0],
        "test_rows": len(task.fold.test_positions),
        "model_id": task.model_id,
        "fold_seed": task.seed + task.fold.test_positions[0],
        "status": "",
        "eligible_train_rows": len(task.eligible_train_positions),
        "active_features_json": "[]",
        "dropped_all_missing_json": "[]",
        "dropped_constant_spline_json": "[]",
        "warnings_json": "[]",
        "exception_type": "",
        "exception_message": "",
        "runtime_seconds": 0.0,
    }


def _run_fold_task(task: FoldTask, guard: ResourceGuard) -> FoldResult:
    guard.check()
    started = time.perf_counter()
    diagnostic = _diagnostic_base(task)
    predictions = _empty_predictions(task)
    if len(task.eligible_train_positions) < FOLD_SPEC.min_train_sessions:
        diagnostic["status"] = "SKIPPED_INSUFFICIENT_ELIGIBLE_TRAIN"
        diagnostic["runtime_seconds"] = time.perf_counter() - started
        return FoldResult(predictions=predictions, diagnostic=diagnostic)

    definition = MODEL_BY_ID[task.model_id]
    fold_seed = int(diagnostic["fold_seed"])
    train = task.model_frame.iloc[list(task.eligible_train_positions)]
    test = task.model_frame.iloc[list(task.fold.test_positions)]
    target = train["observed_pe"]
    model = build_wave1_model(task.model_id, fold_seed=fold_seed)
    if definition.feature_columns:
        x_train = train.loc[:, list(definition.feature_columns)].copy()
        x_test = test.loc[:, list(definition.feature_columns)].copy()
    else:
        x_train = pd.DataFrame(index=train.index)
        x_test = pd.DataFrame(index=test.index)
    fit_context = FitContext(
        experiment_id="wave1-spent-seed-cheap-screen",
        fold_id=task.fold.fold_id,
        seed=fold_seed,
        train_end=task.fold.train_end,
        target_name="observed_pe",
        feature_metadata=feature_metadata(tuple(definition.feature_columns)),
    )
    predict_context = PredictContext(
        experiment_id="wave1-spent-seed-cheap-screen",
        fold_id=task.fold.fold_id,
        seed=fold_seed,
        prediction_start=task.fold.test_start,
        prediction_end=task.fold.test_end,
        feature_metadata=feature_metadata(tuple(definition.feature_columns)),
    )
    try:
        model.fit(x_train, target, context=fit_context)
        prediction = model.predict(x_test, context=predict_context)
        warning_text = (
            [STATE_SPACE_ALLOWED_WARNING] if isinstance(model, Wave1StateSpaceAdapter) else []
        )
        diagnostics = model.diagnostics()
        predictions["prediction"] = prediction.to_numpy(dtype=np.float64)
        diagnostic.update(
            {
                "status": "OK",
                "active_features_json": _json_text(list(diagnostics.active_features)),
                "dropped_all_missing_json": _json_text(
                    list(diagnostics.dropped_all_missing_features)
                ),
                "dropped_constant_spline_json": _json_text(
                    list(diagnostics.dropped_constant_spline_features)
                ),
                "warnings_json": _json_text(warning_text),
            }
        )
    except ResourcePolicyError:
        raise
    except Exception as exc:  # a failed fold stays NaN; there is no retry or fallback
        diagnostic.update(
            {
                "status": "FAILED_NO_RETRY",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
        )
    diagnostic["runtime_seconds"] = time.perf_counter() - started
    guard.check()
    return FoldResult(predictions=predictions, diagnostic=diagnostic)


@contextmanager
def execution_warning_policy():
    """Install one global policy before threads; workers never mutate warning filters."""

    from statsmodels.tools.sm_exceptions import SpecificationWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore",
            message=f"^{re.escape(STATE_SPACE_ALLOWED_WARNING)}$",
            category=SpecificationWarning,
        )
        yield


def _training_eligible_positions(model_frame: pd.DataFrame, fold: PITFold) -> tuple[int, ...]:
    train_positions = np.asarray(fold.train_positions, dtype=np.int64)
    train = model_frame.iloc[train_positions]
    target = pd.to_numeric(train["observed_pe"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    # Row-wise feature missingness is handled by the locked fold-median imputer.
    # Tying target-history eligibility to feature availability would alter the
    # state-space estimand and silently discard otherwise usable supervised rows.
    eligible = np.isfinite(target) & (target > 0.0)
    return tuple(int(position) for position in train_positions[eligible])


def _baseline_predictions(
    baselines: pd.DataFrame,
    *,
    seed: int,
    folds: tuple[PITFold, ...],
) -> pd.DataFrame:
    if tuple(baselines.columns) != ("date", "symbol", *BASELINE_MODEL_IDS):
        raise ContractError("baseline frame must use the exact locked column order")
    rows: list[pd.DataFrame] = []
    for fold in folds:
        test = baselines.iloc[list(fold.test_positions)]
        for model_id in BASELINE_MODEL_IDS:
            prediction = pd.to_numeric(test[model_id], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
            rows.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "date": pd.to_datetime(test["date"]).to_numpy(),
                        "symbol": test["symbol"].astype(str).to_numpy(),
                        "fold_id": fold.fold_id,
                        "test_start_position": fold.test_positions[0],
                        "model_id": model_id,
                        "prediction": prediction,
                    },
                    columns=PREDICTION_COLUMNS,
                )
            )
    return pd.concat(rows, ignore_index=True)


def run_seed_predictions(
    model_frame: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    seed: int,
    model_ids: Sequence[str] = CANDIDATE_MODEL_IDS,
    outer_workers: int = 32,
    enforce_affinity: bool = True,
) -> SeedPredictionResult:
    """Generate candidate and baseline predictions without any evaluation input."""

    assert_cpu_environment()
    if "true_fair_pe" in model_frame or "true_fair_pe" in baselines:
        raise ContractError("predict runner rejects evaluation truth")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ContractError("evidence seed must be an integer")
    if not 1 <= int(outer_workers) <= int(RESOURCE_POLICY["outer_workers_max"]):
        raise ContractError("outer_workers must be in [1, 32]")
    selected = tuple(model_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ContractError("candidate model_ids must be non-empty and unique")
    unknown = sorted(set(selected).difference(CANDIDATE_MODEL_IDS))
    if unknown:
        raise ContractError(f"unknown Wave1 candidate model_ids: {unknown}")
    required = {"date", "symbol", "observed_pe", *COMMON_FEATURES}
    for model_id in selected:
        required.update(MODEL_BY_ID[model_id].feature_columns)
    missing = sorted(required.difference(model_frame.columns))
    if missing:
        raise ContractError(f"model frame is missing Wave1 columns: {missing}")
    if len(model_frame) != len(baselines):
        raise ContractError("model and baseline frames must have identical row counts")
    dates = pd.to_datetime(model_frame["date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ContractError("predict model dates must be valid, unique, and increasing")
    if not dates.equals(pd.to_datetime(baselines["date"], errors="coerce")):
        raise ContractError("model and baseline dates differ")
    if not model_frame["symbol"].astype(str).equals(baselines["symbol"].astype(str)):
        raise ContractError("model and baseline symbols differ")

    fold_frame = pd.DataFrame({"date": dates})
    folds = generate_pit_folds(fold_frame, FOLD_SPEC)
    eligibility = {fold.fold_id: _training_eligible_positions(model_frame, fold) for fold in folds}
    tasks = tuple(
        FoldTask(
            seed=seed,
            fold=fold,
            model_id=model_id,
            model_frame=model_frame,
            eligible_train_positions=eligibility[fold.fold_id],
        )
        for fold in folds
        for model_id in selected
    )
    affinity = set_full_process_affinity_0_31() if enforce_affinity else None

    from threadpoolctl import threadpool_limits

    with execution_warning_policy(), threadpool_limits(limits=1), ResourceGuard() as guard:
        inventory = assert_threadpools_one()
        if outer_workers == 1:
            results = [_run_fold_task(task, guard) for task in tasks]
        else:
            with ThreadPoolExecutor(
                max_workers=outer_workers,
                thread_name_prefix="wave1-fold",
            ) as executor:
                results = list(executor.map(lambda task: _run_fold_task(task, guard), tasks))
        guard.check()
        inventory = assert_threadpools_one()
        resource_summary = {
            "peak_process_rss_bytes": guard.peak_rss_bytes,
            "minimum_available_physical_bytes": int(guard.minimum_available_bytes or 0),
        }

    candidate_predictions = pd.concat(
        [result.predictions for result in results], ignore_index=True
    )
    baseline_predictions = _baseline_predictions(baselines, seed=seed, folds=folds)
    predictions = (
        pd.concat([candidate_predictions, baseline_predictions], ignore_index=True)
        .sort_values(["seed", "date", "model_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if predictions.duplicated(["seed", "date", "model_id"]).any():
        raise ContractError("predict output contains duplicate model/identity rows")
    diagnostics = (
        pd.DataFrame([result.diagnostic for result in results])
        .loc[:, list(DIAGNOSTIC_COLUMNS)]
        .sort_values(["seed", "test_start_position", "model_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    runtime_seconds_by_model = {
        str(model_id): float(group["runtime_seconds"].sum())
        for model_id, group in diagnostics.groupby("model_id", sort=True)
    }
    for baseline_id in BASELINE_MODEL_IDS:
        runtime_seconds_by_model[baseline_id] = 0.0
    expected_test_rows = sum(len(fold.test_positions) for fold in folds)
    expected_rows = expected_test_rows * (len(selected) + len(BASELINE_MODEL_IDS))
    if len(predictions) != expected_rows:
        raise ContractError(
            f"prediction row count mismatch: expected={expected_rows}, actual={len(predictions)}"
        )
    if not all(math.isfinite(value) and value >= 0.0 for value in runtime_seconds_by_model.values()):
        raise ContractError("model runtime totals must be finite and non-negative")
    return SeedPredictionResult(
        predictions=predictions.loc[:, list(PREDICTION_COLUMNS)],
        diagnostics=diagnostics,
        runtime_seconds_by_model=runtime_seconds_by_model,
        folds=folds,
        resource_summary=resource_summary,
        threadpool_inventory=inventory,
        affinity=affinity,
    )
