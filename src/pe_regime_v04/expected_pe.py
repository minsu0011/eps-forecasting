from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

_COMMON_FEATURES = [
    "benchmark_return_21",
    "benchmark_return_63",
    "benchmark_return_126",
    "benchmark_return_252",
    "benchmark_realized_vol_20",
    "benchmark_realized_vol_63",
    "benchmark_drawdown_252",
    "benchmark_price_vs_sma_50",
    "benchmark_price_vs_sma_200",
    "benchmark_sma_50_vs_200",
    "benchmark_sma_200_slope_20",
    "benchmark_trend_efficiency_63",
    "benchmark_volume_log_z_63",
    "stock_return_63",
    "stock_return_126",
    "eps_ttm_growth_126",
    "eps_ttm_growth_252",
    "eps_staleness_days",
    "eps_period_age_days",
    "eps_confidence",
    "eps_disagreement",
    "eps_approximation_flag",
    "pe_median_252_lag",
    "pe_median_756_lag",
]

_CURRENT_REGIME_FEATURES = [
    # Contemporaneous state detector from v0.3. These columns never contain a
    # forward-return mixture.
    "v04_current_p_bear",
    "v04_current_p_sideways",
    "v04_current_p_bull",
    # Existing v0.3 future-detected-regime model remains a separate feature block.
    "forecast_p_bear",
    "forecast_p_sideways",
    "forecast_p_bull",
    "pe_bear_median",
    "pe_sideways_median",
    "pe_bull_median",
    "pe_bear_effective_history",
    "pe_sideways_effective_history",
    "pe_bull_effective_history",
]

_RETURN_FORECAST_FEATURES = [
    # Independently trained forward-return classifier. Its semantics remain
    # explicit and it is never averaged with the current-state probabilities.
    "v04_return_forecast_p_bear",
    "v04_return_forecast_p_sideways",
    "v04_return_forecast_p_bull",
    "v04_return_forecast_entropy",
    "v04_return_forecast_confidence",
]

_FORBIDDEN_FEATURES = {
    "observed_pe",
    "close",
    "eps_ttm",
    "earnings_yield",
    "pe_change_5",
    "pe_change_20",
    "pe_log_change_5",
    "pe_log_change_20",
    "pe_lag_21",
    "log_pe_lag_21",
    "regime_conditional_pe_percentile",
    "regime_conditional_pe_log_z",
}


def _numeric_float_series(values: pd.Series, *, name: str | None = None) -> pd.Series:
    """Normalize pandas nullable numerics to ordinary float64/NaN.

    Pandas extension dtypes propagate ``pd.NA`` through NumPy ufuncs and can
    turn an eligibility mask into a three-valued nullable boolean.  Model and
    causality masks must instead have the unambiguous finite-or-NaN contract.
    """

    coerced = pd.to_numeric(values, errors="coerce")
    array = coerced.to_numpy(dtype=float, na_value=np.nan)
    return pd.Series(array, index=values.index, name=name or values.name)


def expected_pe_feature_columns(
    frame: pd.DataFrame,
    *,
    include_regime: bool,
    include_return_forecast: bool = False,
) -> list[str]:
    candidates = list(_COMMON_FEATURES)
    if include_regime:
        candidates.extend(_CURRENT_REGIME_FEATURES)
        if include_return_forecast:
            candidates.extend(_RETURN_FORECAST_FEATURES)
    columns = [column for column in candidates if column in frame.columns]
    forbidden = _FORBIDDEN_FEATURES.intersection(columns)
    if forbidden:
        raise AssertionError(f"Expected-P/E feature leakage allowlist error: {forbidden}")
    return columns


def global_statistical_expected_pe(
    observed_pe: pd.Series,
    config: Mapping[str, Any],
) -> pd.Series:
    observed = _numeric_float_series(observed_pe)
    valid = np.isfinite(observed.to_numpy()) & (observed.to_numpy() > 0.0)
    log_values = np.full(len(observed), np.nan, dtype=float)
    log_values[valid] = np.log(observed.to_numpy()[valid])
    log_pe = pd.Series(log_values, index=observed.index)
    lookback = int(config.get("global_lookback", 756))
    min_history = int(config.get("global_min_history", 126))
    method = str(config.get("global_method", "median")).lower()
    history = log_pe.shift(1).rolling(lookback, min_periods=min_history)
    expected_log = history.mean() if method == "mean" else history.median()
    return np.exp(expected_log).rename("v04_global_statistical_expected_pe")


def _resolve_backend() -> tuple[str, str | None]:
    try:
        from lightgbm import LGBMRegressor  # noqa: F401

        return "lightgbm", None
    except ImportError as exc:
        return "sklearn_hist_gradient_boosting", f"{type(exc).__name__}: {exc}"


def _model_parameters(
    config: Mapping[str, Any],
    *,
    backend: str,
    seed: int,
) -> dict[str, Any]:
    if backend == "lightgbm":
        subsample = float(config.get("subsample", 0.90))
        configured_subsample_freq = int(
            config.get("subsample_freq", 1 if subsample < 1.0 else 0)
        )
        subsample_freq = (
            max(configured_subsample_freq, 1)
            if subsample < 1.0
            else configured_subsample_freq
        )
        return {
            "objective": "regression_l1",
            "n_estimators": int(config.get("n_estimators", 140)),
            "learning_rate": float(config.get("learning_rate", 0.03)),
            "num_leaves": int(config.get("num_leaves", 11)),
            "max_depth": int(config.get("max_depth", -1)),
            "min_child_samples": int(config.get("min_child_samples", 20)),
            "subsample": subsample,
            "subsample_freq": subsample_freq,
            "colsample_bytree": float(config.get("colsample_bytree", 0.85)),
            "reg_alpha": float(config.get("reg_alpha", 0.0)),
            "reg_lambda": float(config.get("reg_lambda", 0.2)),
            "random_state": seed,
            "n_jobs": 1,
            "deterministic": bool(config.get("deterministic", True)),
            "force_col_wise": bool(config.get("force_col_wise", True)),
            "verbosity": int(config.get("verbosity", -1)),
        }
    return {
        "loss": "absolute_error",
        "max_iter": int(config.get("fallback_max_iter", 180)),
        "learning_rate": float(config.get("fallback_learning_rate", 0.04)),
        "max_leaf_nodes": int(config.get("fallback_max_leaf_nodes", 15)),
        "min_samples_leaf": int(config.get("min_child_samples", 20)),
        "random_state": seed,
    }


def _make_regressor(
    config: Mapping[str, Any],
    *,
    backend: str,
    seed: int,
):
    parameters = _model_parameters(config, backend=backend, seed=seed)
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**parameters), parameters
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(**parameters), parameters


def _stable_row_hash(index: pd.Index, positions: np.ndarray) -> str:
    normalized_positions = np.asarray(positions, dtype="<i8")
    selected_index = index.take(normalized_positions)
    index_hash = pd.util.hash_pandas_object(selected_index, index=False).to_numpy(
        dtype="<u8"
    )
    digest = hashlib.sha256()
    digest.update(normalized_positions.tobytes())
    digest.update(index_hash.tobytes())
    return digest.hexdigest()


def _fit_candidate(
    X: pd.DataFrame,
    target: pd.Series,
    frame: pd.DataFrame,
    train_positions: np.ndarray,
    test_start: int,
    test_end: int,
    config: Mapping[str, Any],
    *,
    backend: str,
    seed: int,
    enabled: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not enabled:
        return None, {"status": "disabled", "active_features": []}

    X_train_raw = X.iloc[train_positions].copy()
    X_test_raw = X.iloc[test_start:test_end].copy()
    active = X_train_raw.columns[X_train_raw.notna().any(axis=0)].tolist()
    if not active:
        return None, {"status": "skipped_no_active_features", "active_features": []}

    try:
        imputer = SimpleImputer(strategy="median")
        train_array = imputer.fit_transform(X_train_raw[active])
        test_array = imputer.transform(X_test_raw[active])
        X_train = pd.DataFrame(train_array, index=X_train_raw.index, columns=active)
        X_test = pd.DataFrame(test_array, index=X_test_raw.index, columns=active)
        y_train = target.iloc[train_positions]
        sample_weight = None
        if "eps_confidence" in frame.columns:
            sample_weight = (
                pd.to_numeric(
                    frame.iloc[train_positions]["eps_confidence"], errors="coerce"
                )
                .fillna(50.0)
                .clip(5.0, 100.0)
                .to_numpy(dtype=float)
                / 100.0
            )
        model, parameters = _make_regressor(
            config,
            backend=backend,
            seed=seed,
        )
        if sample_weight is None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        predicted_log = np.asarray(model.predict(X_test), dtype=float)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            predicted = np.exp(predicted_log)
        invalid = ~np.isfinite(predicted) | (predicted <= 0.0)
        invalid_count = int(invalid.sum())
        predicted[invalid] = np.nan
        return predicted, {
            "status": "ok",
            "active_features": active,
            "active_feature_count": int(len(active)),
            "model_parameters": parameters,
            "sample_weight_mean": (
                float(np.mean(sample_weight)) if sample_weight is not None else None
            ),
            "nonfinite_or_nonpositive_prediction_count": invalid_count,
        }
    except Exception as exc:
        return None, {
            "status": "failed",
            "active_features": active,
            "error": f"{type(exc).__name__}: {exc}",
        }


def walk_forward_expected_pe_pair(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    seed: int,
    no_regime_output_name: str = "v04_ml_expected_pe_no_regime",
    with_regime_output_name: str = "v04_ml_expected_pe_with_regime",
) -> tuple[pd.Series, pd.Series, list[dict[str, Any]], list[str], list[str]]:
    """Train a matched Expected-P/E pair on one immutable fold schedule."""

    include_return_forecast = bool(
        config.get("include_return_forecast_features", False)
    )
    no_features = expected_pe_feature_columns(
        frame,
        include_regime=False,
        include_return_forecast=False,
    )
    with_features = expected_pe_feature_columns(
        frame,
        include_regime=True,
        include_return_forecast=include_return_forecast,
    )
    no_output = pd.Series(np.nan, index=frame.index, name=no_regime_output_name)
    with_output = pd.Series(np.nan, index=frame.index, name=with_regime_output_name)
    diagnostics: list[dict[str, Any]] = []
    if not no_features:
        diagnostics.append(
            {
                "status": "disabled_no_common_features",
                "backend": None,
                "fallback_reason": None,
            }
        )
        return no_output, with_output, diagnostics, no_features, with_features

    X_no = frame[no_features].replace([np.inf, -np.inf], np.nan)
    X_with = frame[with_features].replace([np.inf, -np.inf], np.nan)
    observed = _numeric_float_series(frame["observed_pe"])
    observed_values = observed.to_numpy()
    valid_observed = np.isfinite(observed_values) & (observed_values > 0.0)
    target_values = np.full(len(observed), np.nan, dtype=float)
    target_values[valid_observed] = np.log(observed_values[valid_observed])
    target = pd.Series(target_values, index=frame.index)
    common_eligible = pd.Series(
        np.isfinite(target_values)
        & X_no.notna().any(axis=1).to_numpy(dtype=bool),
        index=frame.index,
    )

    train_window = int(config.get("train_window", 1008))
    min_train = int(config.get("min_train", 252))
    refit_every = int(config.get("refit_every", 21))
    outer_n_jobs = int(config.get("outer_n_jobs", 1))
    parallel_backend = str(config.get("parallel_backend", "thread")).lower()
    if train_window <= 0 or min_train <= 0 or refit_every <= 0:
        raise ValueError("Expected-P/E train_window/min_train/refit_every must be positive")
    if train_window < min_train:
        raise ValueError("Expected-P/E train_window must be at least min_train")
    if outer_n_jobs <= 0:
        raise ValueError("Expected-P/E outer_n_jobs must be a positive integer")
    if parallel_backend not in {"serial", "thread"}:
        raise ValueError("Expected-P/E parallel_backend must be serial or thread")

    enabled_no = bool(config.get("train_no_regime", True))
    enabled_with = bool(config.get("train_with_regime", True))
    backend, fallback_reason = _resolve_backend()
    fold_starts = list(range(min_train, len(frame), refit_every))
    if not fold_starts:
        effective_workers = 0
    elif parallel_backend == "serial":
        effective_workers = 1
    else:
        effective_workers = min(outer_n_jobs, len(fold_starts))
    requested_inner_n_jobs = int(config.get("n_jobs", 1))

    def fit_fold(
        test_start: int,
    ) -> tuple[int, int, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        test_end = min(len(frame), test_start + refit_every)
        train_start = max(0, test_start - train_window)
        candidate_positions = np.arange(train_start, test_start, dtype=np.int64)
        eligible = common_eligible.iloc[train_start:test_start].to_numpy(dtype=bool)
        train_positions = candidate_positions[eligible]
        test_positions = np.arange(test_start, test_end, dtype=np.int64)
        fold_seed = seed + test_start
        diagnostic: dict[str, Any] = {
            "test_start_position": int(test_start),
            "test_end_exclusive_position": int(test_end),
            "train_window_start_position": int(train_start),
            "train_window_end_exclusive_position": int(test_start),
            "train_rows": int(len(train_positions)),
            "required_train_rows": int(min_train),
            "test_rows": int(len(test_positions)),
            "train_row_hash": _stable_row_hash(frame.index, train_positions),
            "test_row_hash": _stable_row_hash(frame.index, test_positions),
            "seed": int(fold_seed),
            "backend": backend,
            "fallback_reason": fallback_reason,
            "outer_n_jobs_requested": int(outer_n_jobs),
            "outer_n_jobs_effective": int(effective_workers),
            "outer_backend_requested": parallel_backend,
            "outer_backend": "thread" if effective_workers > 1 else "serial",
            "inner_n_jobs_requested": int(requested_inner_n_jobs),
            "inner_n_jobs_effective": 1,
        }
        if len(train_positions) < min_train:
            skipped = {"status": "skipped_insufficient_common_train_rows"}
            diagnostic.update(
                {
                    "status": "skipped_insufficient_common_train_rows",
                    "no_regime": skipped,
                    "with_regime": dict(skipped),
                }
            )
            return test_start, test_end, None, None, diagnostic

        no_prediction, no_diagnostic = _fit_candidate(
            X_no,
            target,
            frame,
            train_positions,
            test_start,
            test_end,
            config,
            backend=backend,
            seed=fold_seed,
            enabled=enabled_no,
        )
        with_prediction, with_diagnostic = _fit_candidate(
            X_with,
            target,
            frame,
            train_positions,
            test_start,
            test_end,
            config,
            backend=backend,
            seed=fold_seed,
            enabled=enabled_with,
        )
        if enabled_no and enabled_with:
            if (
                no_diagnostic["status"] != "ok"
                or with_diagnostic["status"] != "ok"
                or no_prediction is None
                or with_prediction is None
            ):
                status = "paired_failure"
                diagnostic["paired_failure_reason"] = (
                    "both matched candidates must fit successfully on the same fold"
                )
                no_prediction = None
                with_prediction = None
            else:
                paired_valid = (
                    np.isfinite(no_prediction)
                    & np.isfinite(with_prediction)
                    & (no_prediction > 0.0)
                    & (with_prediction > 0.0)
                )
                dropped = int((~paired_valid).sum())
                no_prediction = no_prediction.copy()
                with_prediction = with_prediction.copy()
                no_prediction[~paired_valid] = np.nan
                with_prediction[~paired_valid] = np.nan
                diagnostic["paired_prediction_rows"] = int(paired_valid.sum())
                diagnostic["paired_prediction_dropped_rows"] = dropped
                status = "ok"
        else:
            active_diagnostic = no_diagnostic if enabled_no else with_diagnostic
            status = str(active_diagnostic["status"])
        diagnostic.update(
            {
                "status": status,
                "no_regime": no_diagnostic,
                "with_regime": with_diagnostic,
            }
        )
        return test_start, test_end, no_prediction, with_prediction, diagnostic

    if effective_workers > 1:
        with ThreadPoolExecutor(
            max_workers=effective_workers,
            thread_name_prefix="v04-expected-pe",
        ) as executor:
            fold_results = list(executor.map(fit_fold, fold_starts))
    else:
        fold_results = [fit_fold(test_start) for test_start in fold_starts]

    for test_start, test_end, no_prediction, with_prediction, diagnostic in fold_results:
        if no_prediction is not None:
            no_output.iloc[test_start:test_end] = no_prediction
        if with_prediction is not None:
            with_output.iloc[test_start:test_end] = with_prediction
        diagnostics.append(diagnostic)
    return no_output, with_output, diagnostics, no_features, with_features


def walk_forward_expected_pe(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    include_regime: bool,
    seed: int,
    output_name: str,
) -> tuple[pd.Series, list[dict[str, Any]], list[str]]:
    """Backward-compatible single-candidate wrapper around the paired walker."""

    pair_config = dict(config)
    pair_config["train_no_regime"] = not include_regime
    pair_config["train_with_regime"] = include_regime
    no_output, with_output, diagnostics, no_features, with_features = (
        walk_forward_expected_pe_pair(
            frame,
            pair_config,
            seed=seed,
            no_regime_output_name=output_name if not include_regime else "unused_no_regime",
            with_regime_output_name=(
                output_name if include_regime else "unused_with_regime"
            ),
        )
    )
    if include_regime:
        return with_output.rename(output_name), diagnostics, with_features
    return no_output.rename(output_name), diagnostics, no_features
