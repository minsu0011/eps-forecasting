from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .labels import future_return_proxy_labels, probability_log_loss_rows
from .utils import REGIME_ORDER, normalized_entropy, normalize_probabilities, sigmoid

_BASE_PROBABILITY_COLUMNS = ["p_bear", "p_sideways", "p_bull"]
_COMPONENT_PROBABILITY_COLUMNS = [
    "rule_p_bear",
    "rule_p_sideways",
    "rule_p_bull",
    "sjm3_p_bear",
    "sjm3_p_sideways",
    "sjm3_p_bull",
    "sjm2_gate_p_bear",
    "sjm2_gate_p_sideways",
    "sjm2_gate_p_bull",
    "hmm3_p_bear",
    "hmm3_p_sideways",
    "hmm3_p_bull",
]
_MARKET_COLUMNS = [
    "benchmark_return_1",
    "benchmark_return_5",
    "benchmark_return_21",
    "benchmark_return_63",
    "benchmark_return_126",
    "benchmark_return_252",
    "benchmark_realized_vol_10",
    "benchmark_realized_vol_20",
    "benchmark_realized_vol_63",
    "benchmark_price_vs_sma_50",
    "benchmark_price_vs_sma_200",
    "benchmark_sma_50_vs_200",
    "benchmark_sma_200_slope_20",
    "benchmark_drawdown_252",
    "benchmark_trend_efficiency_63",
    "benchmark_volume_log_z_63",
    "regime_entropy",
    "regime_agreement",
]


@dataclass(frozen=True)
class _RegimeFoldResult:
    test_start: int
    test_end: int
    probability: np.ndarray | None
    diagnostic: dict[str, Any]


def _positive_worker_count(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _parallel_settings(
    config: Mapping[str, Any],
    *,
    fold_count: int,
) -> tuple[str, int, int, int, int]:
    requested_outer = _positive_worker_count(
        config.get("outer_n_jobs", 1), name="regime_stacker.outer_n_jobs"
    )
    requested_inner = _positive_worker_count(
        config.get("n_jobs", 1), name="regime_stacker.n_jobs"
    )
    backend = str(config.get("parallel_backend", "thread")).strip().lower()
    if backend not in {"serial", "thread"}:
        raise ValueError("regime_stacker.parallel_backend must be 'serial' or 'thread'")
    cpu_limit = max(1, int(os.cpu_count() or 1))
    effective_outer = (
        1
        if backend == "serial"
        else min(requested_outer, max(1, fold_count), cpu_limit)
    )
    # Independent folds are the only parallel dimension.  Nested OpenMP workers
    # waste CPU on these small models and can make scheduling nondeterministic.
    effective_inner = 1
    execution_backend = "thread" if effective_outer > 1 else "serial"
    return (
        execution_backend,
        requested_outer,
        effective_outer,
        requested_inner,
        effective_inner,
    )


def _configured_class_weight(config: Mapping[str, Any]) -> None:
    value = config.get("class_weight")
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "natural"}:
        return None
    raise ValueError(
        "regime_stacker.class_weight must be null/natural for the forecast probability head"
    )


def _tree_sampling_settings(config: Mapping[str, Any]) -> tuple[float, int, bool, bool]:
    subsample = float(config.get("subsample", 0.90))
    if not 0.0 < subsample <= 1.0:
        raise ValueError("regime_stacker.subsample must be in (0, 1]")
    default_frequency = 1 if subsample < 1.0 else 0
    subsample_freq = int(config.get("subsample_freq", default_frequency))
    if subsample_freq < 0:
        raise ValueError("regime_stacker.subsample_freq must be non-negative")
    deterministic = bool(config.get("deterministic", True))
    force_col_wise = bool(config.get("force_col_wise", True))
    return subsample, subsample_freq, deterministic, force_col_wise


def _resolve_classifier_backend() -> tuple[str, str | None]:
    try:
        from lightgbm import LGBMClassifier  # noqa: F401

        return "lightgbm", None
    except ImportError as exc:
        return "sklearn_logistic_regression", f"{type(exc).__name__}: {exc}"


def _strict_probability_array(
    values: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("regime probabilities must have exactly three columns")
    with np.errstate(invalid="ignore"):
        row_sum = raw.sum(axis=1)
    valid = (
        np.isfinite(raw).all(axis=1)
        & (raw >= 0.0).all(axis=1)
        & (raw <= 1.0).all(axis=1)
        & np.isclose(row_sum, 1.0, atol=1e-6, rtol=0.0)
    )
    output = np.full_like(raw, np.nan, dtype=float)
    if valid.any():
        output[valid] = raw[valid] / row_sum[valid, None]
    return output, valid


def _paired_rolling_losses(
    baseline_rows: pd.Series,
    challenger_rows: pd.Series,
    *,
    horizon: int,
    window: int,
    min_history: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    window = _positive_worker_count(window, name="loss_window")
    min_history = _positive_worker_count(min_history, name="min_history")
    common = (
        baseline_rows.notna()
        & challenger_rows.notna()
        & np.isfinite(baseline_rows.to_numpy(dtype=float))
        & np.isfinite(challenger_rows.to_numpy(dtype=float))
    )
    matured_common = common.shift(horizon, fill_value=False)
    paired_count = (
        matured_common.astype(np.int64)
        .rolling(window, min_periods=1)
        .sum()
        .fillna(0)
        .astype(np.int64)
    )
    baseline_loss = (
        baseline_rows.where(common)
        .shift(horizon)
        .rolling(window, min_periods=min_history)
        .mean()
    )
    challenger_loss = (
        challenger_rows.where(common)
        .shift(horizon)
        .rolling(window, min_periods=min_history)
        .mean()
    )
    insufficient = paired_count < min_history
    baseline_loss.loc[insufficient] = np.nan
    challenger_loss.loc[insufficient] = np.nan
    return baseline_loss, challenger_loss, paired_count


def build_regime_stacker_features(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        column
        for column in (
            _BASE_PROBABILITY_COLUMNS
            + _COMPONENT_PROBABILITY_COLUMNS
            + _MARKET_COLUMNS
        )
        if column in frame.columns
    ]
    features = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    for column in _COMPONENT_PROBABILITY_COLUMNS:
        if column in features.columns:
            features[f"{column}_missing"] = features[column].isna().astype(float)
    # Log probabilities give the external classifiers a better-behaved linear scale.
    for column in _BASE_PROBABILITY_COLUMNS + _COMPONENT_PROBABILITY_COLUMNS:
        if column in features.columns:
            value = pd.to_numeric(features[column], errors="coerce")
            features[f"log_{column}"] = np.log(value.clip(lower=1e-6))
    return features


def _make_binary_classifier(config: Mapping[str, Any], seed: int):
    requested_outer = _positive_worker_count(
        config.get("outer_n_jobs", 1), name="regime_stacker.outer_n_jobs"
    )
    _, _, _, _, effective_inner = _parallel_settings(
        config, fold_count=requested_outer
    )
    class_weight = _configured_class_weight(config)
    subsample, subsample_freq, deterministic, force_col_wise = _tree_sampling_settings(
        config
    )
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=int(config.get("n_estimators", 100)),
            learning_rate=float(config.get("learning_rate", 0.03)),
            num_leaves=int(config.get("num_leaves", 7)),
            min_child_samples=int(config.get("min_child_samples", 25)),
            subsample=subsample,
            subsample_freq=subsample_freq,
            colsample_bytree=float(config.get("colsample_bytree", 0.85)),
            reg_lambda=float(config.get("reg_lambda", 1.0)),
            class_weight=class_weight,
            random_state=seed,
            n_jobs=effective_inner,
            deterministic=deterministic,
            force_col_wise=force_col_wise,
            verbosity=int(config.get("verbosity", -1)),
        )
    except ImportError:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                max_iter=1000,
                class_weight=class_weight,
                random_state=seed,
            ),
        )


def _fit_binary_probability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    active = X_train.columns[X_train.notna().any(axis=0)].tolist()
    if not active:
        raise ValueError("No active regime stacker features")
    X_train = X_train[active]
    X_test = X_test[active]
    imputer = SimpleImputer(strategy="median")
    train_array = imputer.fit_transform(X_train)
    test_array = imputer.transform(X_test)
    train_frame = pd.DataFrame(train_array, index=X_train.index, columns=active)
    test_frame = pd.DataFrame(test_array, index=X_test.index, columns=active)
    model = _make_binary_classifier(config, seed)
    model.fit(train_frame, y_train.astype(int))
    probability = np.asarray(model.predict_proba(test_frame), dtype=float)
    if hasattr(model, "classes_"):
        classes = np.asarray(model.classes_, dtype=int)
    else:
        classes = np.asarray(model[-1].classes_, dtype=int)
    positive_location = np.flatnonzero(classes == 1)
    if len(positive_location) != 1:
        raise RuntimeError(f"Binary classifier classes are invalid: {classes.tolist()}")
    positive_probability = probability[:, int(positive_location[0])]
    if (
        not np.isfinite(positive_probability).all()
        or (positive_probability < 0.0).any()
        or (positive_probability > 1.0).any()
    ):
        raise RuntimeError("Binary classifier returned invalid probabilities")
    module = model.__class__.__module__
    if module.startswith("lightgbm"):
        backend = "lightgbm"
        fallback_used = False
    elif module.startswith("sklearn"):
        backend = "sklearn_logistic_regression"
        fallback_used = True
    else:
        backend = f"custom:{module}.{model.__class__.__name__}"
        fallback_used = True
    subsample, subsample_freq, deterministic, force_col_wise = _tree_sampling_settings(
        config
    )
    return positive_probability, {
        "model_backend": backend,
        "model_fallback_used": fallback_used,
        "class_weight": _configured_class_weight(config),
        "calibration_status": "natural_prior_raw",
        "subsample": subsample,
        "subsample_freq": subsample_freq,
        "deterministic": deterministic,
        "force_col_wise": force_col_wise,
    }


def walk_forward_hierarchical_stacker(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.Series]:
    output = pd.DataFrame(
        np.nan,
        index=frame.index,
        columns=["stacker_p_bear", "stacker_p_sideways", "stacker_p_bull"],
    )
    diagnostics: list[dict[str, Any]] = []
    horizon = int(config.get("horizon", 21))
    labels = future_return_proxy_labels(
        frame["benchmark_close"],
        horizon=horizon,
        bull_threshold=float(config.get("bull_return_threshold", 0.03)),
        bear_threshold=float(config.get("bear_return_threshold", -0.03)),
    )
    if not bool(config.get("enabled", True)):
        return output, diagnostics, labels

    X = build_regime_stacker_features(frame)
    min_train = int(config.get("min_train", 252))
    train_window = int(config.get("train_window", 1008))
    refit_every = int(config.get("refit_every", 21))
    floor = float(config.get("probability_floor", 0.01))
    test_starts = list(range(min_train + horizon, len(frame), refit_every))
    (
        execution_backend,
        requested_outer,
        effective_outer,
        requested_inner,
        effective_inner,
    ) = _parallel_settings(config, fold_count=len(test_starts))
    class_weight = _configured_class_weight(config)
    subsample, subsample_freq, deterministic, force_col_wise = _tree_sampling_settings(
        config
    )
    model_backend, fallback_reason = _resolve_classifier_backend()
    execution_diagnostic = {
        "outer_n_jobs_requested": requested_outer,
        "outer_n_jobs_effective": effective_outer,
        "outer_backend": execution_backend,
        "inner_n_jobs_requested": requested_inner,
        "inner_n_jobs_effective": effective_inner,
        "backend": model_backend,
        "fallback_reason": fallback_reason,
        "class_weight": class_weight,
        "calibration_status": "natural_prior_raw",
        "subsample": subsample,
        "subsample_freq": subsample_freq,
        "deterministic": deterministic,
        "force_col_wise": force_col_wise,
    }

    def fit_fold(test_start: int) -> _RegimeFoldResult:
        test_end = min(len(frame), test_start + refit_every)
        # y[j] depends on close[j+h]. At test_start, only j < test_start-h is mature.
        latest_mature_target = test_start - horizon
        train_start = max(0, latest_mature_target - train_window)
        X_train = X.iloc[train_start:latest_mature_target].copy()
        y_train = labels.iloc[train_start:latest_mature_target].copy()
        paired = y_train.notna() & X_train.notna().any(axis=1)
        X_train = X_train.loc[paired]
        y_train = y_train.loc[paired].astype(int)
        class_counts = {
            REGIME_ORDER[class_index].lower(): int((y_train == class_index).sum())
            for class_index in range(3)
        }
        common_diagnostic = {
            "test_start": int(test_start),
            "test_end": int(test_end),
            "train_start": int(train_start),
            "train_end_exclusive": int(latest_mature_target),
            "train_rows": int(latest_mature_target - train_start),
            "paired_train_rows": int(len(X_train)),
            "usable_train_rows": int(len(X_train)),
            "required_usable_train_rows": int(min_train),
            "class_counts": class_counts,
            **execution_diagnostic,
        }
        if len(X_train) < min_train:
            return _RegimeFoldResult(
                test_start,
                test_end,
                None,
                {
                    **common_diagnostic,
                    "model_backend": "not_run",
                    "model_fallback_used": False,
                    "status": "skipped_insufficient_usable_train",
                },
            )
        if y_train.nunique() < 3:
            return _RegimeFoldResult(
                test_start,
                test_end,
                None,
                {
                    **common_diagnostic,
                    "model_backend": "not_run",
                    "model_fallback_used": False,
                    "status": "skipped_insufficient_classes",
                },
            )

        X_test = X.iloc[test_start:test_end].copy()
        head_metadata: list[dict[str, Any]] = []
        try:
            sideways_target = (y_train == 1).astype(int)
            if sideways_target.nunique() < 2:
                raise ValueError("Sideways head has one class")
            p_sideways, sideways_metadata = _fit_binary_probability(
                X_train,
                sideways_target,
                X_test,
                config,
                seed + test_start,
            )
            head_metadata.append(sideways_metadata)
            directional_mask = y_train != 1
            directional_target = (y_train.loc[directional_mask] == 2).astype(int)
            if directional_target.nunique() < 2:
                raise ValueError("Direction head has one class")
            p_bull_given_directional, directional_metadata = _fit_binary_probability(
                X_train.loc[directional_mask],
                directional_target,
                X_test,
                config,
                seed + 100_000 + test_start,
            )
            head_metadata.append(directional_metadata)
            raw = np.column_stack(
                [
                    (1.0 - p_sideways) * (1.0 - p_bull_given_directional),
                    p_sideways,
                    (1.0 - p_sideways) * p_bull_given_directional,
                ]
            )
            probability = normalize_probabilities(raw, floor=floor)
            _, valid_probability = _strict_probability_array(probability)
            if not valid_probability.all():
                raise RuntimeError("Hierarchical stacker returned invalid probabilities")
            backends = sorted({str(item["model_backend"]) for item in head_metadata})
            backend = backends[0] if len(backends) == 1 else f"mixed:{','.join(backends)}"
            return _RegimeFoldResult(
                test_start,
                test_end,
                probability,
                {
                    **common_diagnostic,
                    "directional_train_rows": int(directional_mask.sum()),
                    "model_backend": backend,
                    "model_fallback_used": any(
                        bool(item["model_fallback_used"]) for item in head_metadata
                    ),
                    "status": "ok",
                },
            )
        except Exception as exc:
            backends = sorted(
                {str(item["model_backend"]) for item in head_metadata}
            )
            return _RegimeFoldResult(
                test_start,
                test_end,
                None,
                {
                    **common_diagnostic,
                    "model_backend": backends[0] if len(backends) == 1 else "unknown",
                    "model_fallback_used": any(
                        bool(item["model_fallback_used"]) for item in head_metadata
                    ),
                    "status": "failed",
                    "error": str(exc),
                },
            )

    if effective_outer == 1:
        fold_results = [fit_fold(test_start) for test_start in test_starts]
    else:
        with ThreadPoolExecutor(
            max_workers=effective_outer,
            thread_name_prefix="pe-regime-fold",
        ) as executor:
            fold_results = list(executor.map(fit_fold, test_starts))

    for result in sorted(fold_results, key=lambda item: item.test_start):
        if result.probability is not None:
            output.iloc[result.test_start : result.test_end, :] = result.probability
        diagnostics.append(result.diagnostic)
    return output, diagnostics, labels


def gate_stacker_with_existing_ensemble(
    frame: pd.DataFrame,
    stacker: pd.DataFrame,
    proxy_labels: pd.Series,
    config: Mapping[str, Any],
    *,
    horizon: int,
) -> pd.DataFrame:
    base_frame = frame[["p_bear", "p_sideways", "p_bull"]].copy()
    stack_frame = stacker.reindex(frame.index).rename(
        columns={
            "stacker_p_bear": "p_bear",
            "stacker_p_sideways": "p_sideways",
            "stacker_p_bull": "p_bull",
        }
    )[["p_bear", "p_sideways", "p_bull"]]
    base, base_valid = _strict_probability_array(base_frame)
    stack, stack_valid = _strict_probability_array(stack_frame)
    strict_base = pd.DataFrame(
        base, index=frame.index, columns=["p_bear", "p_sideways", "p_bull"]
    )
    strict_stack = pd.DataFrame(
        stack, index=frame.index, columns=["p_bear", "p_sideways", "p_bull"]
    )
    aligned_labels = proxy_labels.reindex(frame.index)
    base_loss_row = probability_log_loss_rows(strict_base, aligned_labels)
    stack_loss_row = probability_log_loss_rows(strict_stack, aligned_labels)
    window = int(config.get("loss_window", 252))
    min_history = int(config.get("min_history", 63))
    # A proxy outcome at j becomes available only at j+horizon.  Both losses are
    # computed on exactly the same matured rows; disjoint coverage is not evidence.
    base_loss, stack_loss, paired_count = _paired_rolling_losses(
        base_loss_row,
        stack_loss_row,
        horizon=int(horizon),
        window=window,
        min_history=min_history,
    )
    gain = base_loss - stack_loss
    margin = float(config.get("improvement_margin", 0.005))
    temperature = max(float(config.get("temperature", 0.025)), 1e-6)
    max_weight = float(config.get("max_challenger_weight", 0.75))
    if not 0.0 <= max_weight <= 1.0:
        raise ValueError("max_challenger_weight must be in [0, 1]")
    mature = (
        (paired_count >= min_history)
        & base_loss.notna()
        & stack_loss.notna()
        & np.isfinite(gain.to_numpy(dtype=float))
    )
    both_current = pd.Series(base_valid & stack_valid, index=frame.index)
    accepted = mature & (gain > margin) & both_current
    weight = pd.Series(0.0, index=frame.index, dtype=float)
    proposed_weight = max_weight * sigmoid((gain - margin) / temperature)
    weight.loc[accepted] = proposed_weight[accepted.to_numpy(dtype=bool)]

    combined = np.full_like(base, np.nan)
    combined[base_valid] = base[base_valid]
    challenger_only = ~base_valid & stack_valid
    combined[challenger_only] = stack[challenger_only]
    blend_rows = accepted.to_numpy(dtype=bool)
    if blend_rows.any():
        w = weight.to_numpy(dtype=float)[blend_rows, None]
        combined[blend_rows] = normalize_probabilities(
            (1.0 - w) * base[blend_rows] + w * stack[blend_rows],
            floor=0.0,
        )
    combined, combined_valid = _strict_probability_array(combined)
    available = base_valid | stack_valid
    if not combined_valid[available].all():
        raise RuntimeError("regime gate produced invalid probabilities")

    base_only = base_valid & ~stack_valid
    neither = ~base_valid & ~stack_valid
    fallback_source = np.full(len(frame), "none", dtype=object)
    fallback_source[base_only] = "incumbent_only"
    fallback_source[challenger_only] = "challenger_only"
    fallback_source[neither] = "unavailable"
    fallback_used = base_only | challenger_only
    hard = np.full(len(frame), "UNKNOWN", dtype=object)
    hard[combined_valid] = np.asarray(REGIME_ORDER, dtype=object)[
        np.argmax(combined[combined_valid], axis=1)
    ]
    entropy = normalized_entropy(combined)
    return pd.DataFrame(
        {
            "v04_p_bear": combined[:, 0],
            "v04_p_sideways": combined[:, 1],
            "v04_p_bull": combined[:, 2],
            "v04_market_regime": hard,
            "v04_regime_entropy": entropy,
            "v04_regime_confidence": 1.0 - entropy,
            "v04_regime_stacker_weight": weight,
            "v04_regime_proxy_base_log_loss": base_loss,
            "v04_regime_proxy_stacker_log_loss": stack_loss,
            "v04_regime_proxy_oos_gain": gain,
            "v04_regime_proxy_paired_count": paired_count,
            "v04_regime_stacker_accepted": accepted,
            "v04_regime_stacker_fallback_used": fallback_used,
            "v04_regime_stacker_fallback_source": fallback_source,
        },
        index=frame.index,
    )


def classwise_gate_stacker_with_existing_ensemble(
    frame: pd.DataFrame,
    stacker: pd.DataFrame,
    proxy_labels: pd.Series,
    config: Mapping[str, Any],
    *,
    horizon: int,
) -> pd.DataFrame:
    """One-vs-rest causal Brier gate for balanced hard-state diagnostics.

    The log-loss gate remains the preferred soft probability path for valuation.
    This diagnostic path allows a strong SIDEWAYS head to contribute without
    forcing the same challenger weight onto BEAR and BULL.
    """
    base, base_valid = _strict_probability_array(
        frame[["p_bear", "p_sideways", "p_bull"]]
    )
    stack, stack_valid = _strict_probability_array(
        stacker.reindex(frame.index)[
            ["stacker_p_bear", "stacker_p_sideways", "stacker_p_bull"]
        ]
    )
    y = pd.to_numeric(proxy_labels.reindex(frame.index), errors="coerce").to_numpy(
        dtype=float
    )
    window = int(config.get("loss_window", 126))
    min_history = int(config.get("min_history", 63))
    margin = float(config.get("improvement_margin", 0.0))
    temperature = max(float(config.get("temperature", 0.02)), 1e-6)
    max_weight = float(config.get("max_challenger_weight", 1.0))
    if not 0.0 <= max_weight <= 1.0:
        raise ValueError("max_challenger_weight must be in [0, 1]")
    weights = np.zeros((len(frame), 3), dtype=float)
    gains = np.full((len(frame), 3), np.nan, dtype=float)
    base_losses = np.full((len(frame), 3), np.nan, dtype=float)
    stack_losses = np.full((len(frame), 3), np.nan, dtype=float)
    paired_counts = np.zeros((len(frame), 3), dtype=np.int64)
    accepted_by_class = np.zeros((len(frame), 3), dtype=bool)
    valid_label = (
        np.isfinite(y)
        & (y >= 0.0)
        & (y <= 2.0)
        & np.isclose(y, np.rint(y), atol=0.0, rtol=0.0)
    )
    both_current = base_valid & stack_valid

    for class_index in range(3):
        target = np.full(len(frame), np.nan, dtype=float)
        target[valid_label] = (y[valid_label].astype(int) == class_index).astype(float)
        common = valid_label & base_valid & stack_valid
        base_row = np.full(len(frame), np.nan, dtype=float)
        stack_row = np.full(len(frame), np.nan, dtype=float)
        base_row[common] = (base[common, class_index] - target[common]) ** 2
        stack_row[common] = (stack[common, class_index] - target[common]) ** 2
        base_loss, stack_loss, paired_count = _paired_rolling_losses(
            pd.Series(base_row, index=frame.index),
            pd.Series(stack_row, index=frame.index),
            horizon=int(horizon),
            window=window,
            min_history=min_history,
        )
        gain = base_loss - stack_loss
        mature = (
            (paired_count >= min_history)
            & base_loss.notna()
            & stack_loss.notna()
            & np.isfinite(gain.to_numpy(dtype=float))
        )
        accepted = mature & (gain > margin) & pd.Series(
            both_current, index=frame.index
        )
        weight = pd.Series(0.0, index=frame.index, dtype=float)
        proposed = max_weight * sigmoid((gain - margin) / temperature)
        weight.loc[accepted] = proposed[accepted.to_numpy(dtype=bool)]
        weights[:, class_index] = weight.to_numpy(dtype=float)
        gains[:, class_index] = gain.to_numpy(dtype=float)
        base_losses[:, class_index] = base_loss.to_numpy(dtype=float)
        stack_losses[:, class_index] = stack_loss.to_numpy(dtype=float)
        paired_counts[:, class_index] = paired_count.to_numpy(dtype=np.int64)
        accepted_by_class[:, class_index] = accepted.to_numpy(dtype=bool)

    combined = np.full_like(base, np.nan)
    combined[base_valid] = base[base_valid]
    challenger_only = ~base_valid & stack_valid
    combined[challenger_only] = stack[challenger_only]
    if both_current.any():
        mixed = (
            (1.0 - weights[both_current]) * base[both_current]
            + weights[both_current] * stack[both_current]
        )
        combined[both_current] = normalize_probabilities(mixed, floor=0.0)
    combined, combined_valid = _strict_probability_array(combined)
    available = base_valid | stack_valid
    if not combined_valid[available].all():
        raise RuntimeError("classwise regime gate produced invalid probabilities")

    base_only = base_valid & ~stack_valid
    neither = ~base_valid & ~stack_valid
    fallback_source = np.full(len(frame), "none", dtype=object)
    fallback_source[base_only] = "incumbent_only"
    fallback_source[challenger_only] = "challenger_only"
    fallback_source[neither] = "unavailable"
    hard = np.full(len(frame), "UNKNOWN", dtype=object)
    hard[combined_valid] = np.asarray(REGIME_ORDER, dtype=object)[
        np.argmax(combined[combined_valid], axis=1)
    ]
    entropy = normalized_entropy(combined)
    output = pd.DataFrame(
        {
            "v04_balanced_p_bear": combined[:, 0],
            "v04_balanced_p_sideways": combined[:, 1],
            "v04_balanced_p_bull": combined[:, 2],
            "v04_balanced_market_regime": hard,
            "v04_balanced_regime_entropy": entropy,
            "v04_balanced_regime_confidence": 1.0 - entropy,
            "v04_balanced_fallback_used": base_only | challenger_only,
            "v04_balanced_fallback_source": fallback_source,
        },
        index=frame.index,
    )
    for class_index, class_name in enumerate(("bear", "sideways", "bull")):
        output[f"v04_balanced_weight_{class_name}"] = weights[:, class_index]
        output[f"v04_balanced_gain_{class_name}"] = gains[:, class_index]
        output[f"v04_balanced_base_brier_{class_name}"] = base_losses[:, class_index]
        output[f"v04_balanced_stacker_brier_{class_name}"] = stack_losses[:, class_index]
        output[f"v04_balanced_paired_count_{class_name}"] = paired_counts[:, class_index]
        output[f"v04_balanced_accepted_{class_name}"] = accepted_by_class[:, class_index]
    return output
