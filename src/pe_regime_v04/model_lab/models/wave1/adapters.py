"""Frozen sklearn/CatBoost/XGBoost/statsmodels Wave-1 adapters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from ...contracts import (
    ContractError,
    FitContext,
    ModelMetadata,
    PredictContext,
    validate_model_call,
)
from .spec import MODEL_BY_ID, SPLINE_COLUMNS, Wave1ModelDefinition, feature_metadata


class Wave1ModelError(ContractError):
    """Raised for an invalid model fit/prediction; callers emit NaN without retry."""


@dataclass(frozen=True)
class AdapterDiagnostics:
    active_features: tuple[str, ...]
    dropped_all_missing_features: tuple[str, ...]
    dropped_constant_spline_features: tuple[str, ...]
    sample_weight_min: float | None
    sample_weight_max: float | None
    sample_weight_mean: float | None
    resolved_parameters: dict[str, Any]


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    return numeric.replace([np.inf, -np.inf], np.nan)


def training_sample_weight(frame: pd.DataFrame) -> np.ndarray:
    if "eps_confidence" not in frame:
        raise Wave1ModelError("supervised Wave1 models require eps_confidence for sample weights")
    confidence = pd.to_numeric(frame["eps_confidence"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    confidence[~np.isfinite(confidence)] = 50.0
    return np.clip(confidence, 5.0, 100.0) / 100.0


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "infinity" if value > 0.0 else "-infinity"}
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "get_params"):
        return {"estimator_class": f"{value.__class__.__module__}.{value.__class__.__qualname__}"}
    return {"python_type": f"{value.__class__.__module__}.{value.__class__.__qualname__}", "repr": repr(value)}


class Wave1SupervisedAdapter:
    def __init__(self, definition: Wave1ModelDefinition, *, fold_seed: int) -> None:
        if definition.family == "state_space_local_linear_trend":
            raise Wave1ModelError("state-space requires its dedicated adapter")
        self.definition = definition
        self.fold_seed = int(fold_seed)
        self._pipeline: Any | None = None
        self._active_features: tuple[str, ...] = ()
        self._dropped_all_missing: tuple[str, ...] = ()
        self._dropped_constant_spline: tuple[str, ...] = ()
        self._weight_summary: tuple[float, float, float] | None = None
        self._resolved_parameters: dict[str, Any] = {}

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.definition.model_id,
            model_version="wave1-b3a190-v1",
            family=self.definition.family,
            feature_ids=tuple(self.definition.feature_columns),
            training_target="observed_pe",
            prediction_name="expected_pe",
            output_semantics="positive market-conditioned expected P/E; research-only cheap screen",
            deterministic=True,
        )

    def _build_pipeline(
        self,
        active: tuple[str, ...],
        spline_active: tuple[str, ...],
    ) -> tuple[Any, str]:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import SplineTransformer, StandardScaler

        family = self.definition.family
        parameters = dict(self.definition.parameters)
        imputer = SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=False)
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

        if family == "ridge_svd":
            estimator = Ridge(
                alpha=parameters["alpha"],
                fit_intercept=parameters["fit_intercept"],
                solver=parameters["solver"],
                copy_X=True,
                max_iter=None,
                tol=1.0e-4,
                positive=False,
            )
            return Pipeline([("imputer", imputer), ("scaler", scaler), ("estimator", estimator)]), "estimator"
        if family == "elastic_net_cyclic":
            estimator = ElasticNet(
                alpha=parameters["alpha"],
                l1_ratio=parameters["l1_ratio"],
                fit_intercept=True,
                precompute=False,
                max_iter=parameters["max_iter"],
                copy_X=True,
                tol=parameters["tol"],
                warm_start=False,
                positive=False,
                random_state=self.fold_seed,
                selection=parameters["selection"],
            )
            return Pipeline([("imputer", imputer), ("scaler", scaler), ("estimator", estimator)]), "estimator"
        if family == "huber":
            estimator = HuberRegressor(
                epsilon=parameters["epsilon"],
                max_iter=parameters["max_iter"],
                alpha=parameters["alpha"],
                warm_start=False,
                fit_intercept=True,
                tol=parameters["tol"],
            )
            return Pipeline([("imputer", imputer), ("scaler", scaler), ("estimator", estimator)]), "estimator"
        if family == "spline_ridge":
            linear_columns = tuple(column for column in active if column not in spline_active)
            transformers: list[tuple[str, Any, list[str]]] = []
            if spline_active:
                spline = Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median", add_indicator=False, keep_empty_features=False
                            ),
                        ),
                        (
                            "spline",
                            SplineTransformer(
                                n_knots=parameters["n_knots"],
                                degree=parameters["degree"],
                                knots=parameters["knots"],
                                extrapolation=parameters["extrapolation"],
                                include_bias=parameters["include_bias"],
                                order=parameters["order"],
                                sparse_output=parameters["sparse_output"],
                            ),
                        ),
                    ]
                )
                transformers.append(("spline", spline, list(spline_active)))
            if linear_columns:
                transformers.append(
                    (
                        "linear",
                        SimpleImputer(
                            strategy="median", add_indicator=False, keep_empty_features=False
                        ),
                        list(linear_columns),
                    )
                )
            features = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                sparse_threshold=0.0,
                n_jobs=1,
                transformer_weights=None,
                verbose=False,
                verbose_feature_names_out=True,
                force_int_remainder_cols="deprecated",
            )
            estimator = Ridge(
                alpha=parameters["ridge_alpha"],
                fit_intercept=True,
                solver=parameters["ridge_solver"],
                copy_X=True,
                max_iter=None,
                tol=1.0e-4,
                positive=False,
            )
            return Pipeline([("features", features), ("scaler", scaler), ("estimator", estimator)]), "estimator"
        if family == "extra_trees":
            estimator = ExtraTreesRegressor(
                n_estimators=parameters["n_estimators"],
                criterion=parameters["criterion"],
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=parameters["min_samples_leaf"],
                min_weight_fraction_leaf=0.0,
                max_features=parameters["max_features"],
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=parameters["bootstrap"],
                oob_score=False,
                n_jobs=parameters["n_jobs"],
                random_state=self.fold_seed,
                verbose=0,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
                monotonic_cst=None,
            )
            return Pipeline([("imputer", imputer), ("estimator", estimator)]), "estimator"
        if family == "hist_gradient_boosting":
            estimator = HistGradientBoostingRegressor(
                loss=parameters["loss"],
                quantile=None,
                learning_rate=parameters["learning_rate"],
                max_iter=parameters["max_iter"],
                max_leaf_nodes=parameters["max_leaf_nodes"],
                max_depth=None,
                min_samples_leaf=parameters["min_samples_leaf"],
                l2_regularization=parameters["l2_regularization"],
                max_features=1.0,
                max_bins=255,
                categorical_features="from_dtype",
                monotonic_cst=None,
                interaction_cst=None,
                warm_start=False,
                early_stopping=parameters["early_stopping"],
                scoring="loss",
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1.0e-7,
                verbose=0,
                random_state=self.fold_seed,
            )
            return Pipeline([("imputer", imputer), ("estimator", estimator)]), "estimator"
        if family == "catboost_cpu":
            from catboost import CatBoostRegressor

            estimator = CatBoostRegressor(**parameters, random_seed=self.fold_seed)
            return Pipeline([("imputer", imputer), ("estimator", estimator)]), "estimator"
        if family == "xgboost_cpu":
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                **parameters,
                random_state=self.fold_seed,
                booster="gbtree",
                verbosity=0,
                validate_parameters=True,
            )
            return Pipeline([("imputer", imputer), ("estimator", estimator)]), "estimator"
        raise Wave1ModelError(f"unsupported Wave1 family: {family}")

    def fit(self, x: pd.DataFrame, y: pd.Series, *, context: FitContext) -> Wave1SupervisedAdapter:
        metadata = self.metadata()
        features = feature_metadata(tuple(self.definition.feature_columns))
        validate_model_call(
            metadata,
            x,
            features,
            phase="fit",
            target_name=context.target_name,
        )
        if context.seed != self.fold_seed:
            raise Wave1ModelError("FitContext seed must equal the locked fold seed")
        numeric = _numeric_frame(x)
        target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        if len(target) != len(numeric) or not (np.isfinite(target) & (target > 0.0)).all():
            raise Wave1ModelError("adapter fit target must be entirely positive and finite")
        weights = training_sample_weight(numeric)
        all_missing = tuple(column for column in numeric if numeric[column].isna().all())
        active = tuple(column for column in numeric if column not in all_missing)
        if not active:
            raise Wave1ModelError("all selected features are missing in this training fold")

        constant_spline: tuple[str, ...] = ()
        if self.definition.family == "spline_ridge":
            locked_spline = tuple(column for column in SPLINE_COLUMNS if column in active)
            constant_spline = tuple(
                column for column in locked_spline if numeric[column].dropna().nunique() <= 1
            )
            active = tuple(column for column in active if column not in constant_spline)
            if not active:
                raise Wave1ModelError("constant-spline removal left no active features")
            spline_active = tuple(
                column
                for column in SPLINE_COLUMNS
                if column in active and column not in constant_spline
            )
        else:
            spline_active = ()

        pipeline, sample_weight_step = self._build_pipeline(active, spline_active)
        pipeline.fit(
            numeric.loc[:, list(active)],
            np.log(target),
            **{f"{sample_weight_step}__sample_weight": weights},
        )
        self._pipeline = pipeline
        self._active_features = active
        self._dropped_all_missing = all_missing
        self._dropped_constant_spline = constant_spline
        self._weight_summary = (
            float(np.min(weights)),
            float(np.max(weights)),
            float(np.mean(weights)),
        )
        parameters = pipeline.get_params(deep=True)
        final_estimator = pipeline.named_steps["estimator"]
        if self.definition.family == "catboost_cpu" and hasattr(final_estimator, "get_all_params"):
            parameters = {**parameters, "estimator__resolved_after_fit": final_estimator.get_all_params()}
        self._resolved_parameters = _jsonable(parameters)
        return self

    def predict(self, x: pd.DataFrame, *, context: PredictContext) -> pd.Series:
        if self._pipeline is None:
            raise Wave1ModelError("adapter must be fit before predict")
        features = feature_metadata(tuple(self.definition.feature_columns))
        validate_model_call(self.metadata(), x, features, phase="predict")
        if context.seed != self.fold_seed:
            raise Wave1ModelError("PredictContext seed must equal the locked fold seed")
        numeric = _numeric_frame(x)
        log_prediction = np.asarray(
            self._pipeline.predict(numeric.loc[:, list(self._active_features)]), dtype=np.float64
        )
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = np.exp(log_prediction)
        if len(prediction) != len(x) or not (np.isfinite(prediction) & (prediction > 0.0)).all():
            raise Wave1ModelError("adapter produced a non-positive or non-finite prediction")
        return pd.Series(prediction, index=x.index, name="prediction")

    def diagnostics(self) -> AdapterDiagnostics:
        if self._pipeline is None or self._weight_summary is None:
            raise Wave1ModelError("adapter diagnostics require a completed fit")
        return AdapterDiagnostics(
            active_features=self._active_features,
            dropped_all_missing_features=self._dropped_all_missing,
            dropped_constant_spline_features=self._dropped_constant_spline,
            sample_weight_min=self._weight_summary[0],
            sample_weight_max=self._weight_summary[1],
            sample_weight_mean=self._weight_summary[2],
            resolved_parameters=self._resolved_parameters,
        )


class Wave1StateSpaceAdapter:
    _ALLOWED_WARNING = (
        "Value of `irregular` may be overridden when the trend component is specified using a model string."
    )

    def __init__(self, definition: Wave1ModelDefinition, *, fold_seed: int) -> None:
        self.definition = definition
        self.fold_seed = int(fold_seed)
        self._result: Any | None = None
        self._resolved_parameters: dict[str, Any] = {}

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id=self.definition.model_id,
            model_version="wave1-b3a190-v1",
            family=self.definition.family,
            feature_ids=(),
            training_target="observed_pe",
            prediction_name="expected_pe",
            output_semantics="fixed-origin state-space forecast of positive market-conditioned P/E",
            deterministic=True,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series, *, context: FitContext) -> Wave1StateSpaceAdapter:
        if len(x.columns) != 0 or len(x) != len(y):
            raise Wave1ModelError("state-space fit accepts an empty feature frame aligned to y")
        if context.seed != self.fold_seed or context.target_name != "observed_pe":
            raise Wave1ModelError("state-space FitContext differs from the locked contract")
        target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        if not (np.isfinite(target) & (target > 0.0)).all():
            raise Wave1ModelError("state-space target must be entirely positive and finite")
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        model = UnobservedComponents(
            np.log(target),
            level="local linear trend",
            irregular=True,
        )
        if not (
            model.irregular
            and model.level
            and model.trend
            and model.stochastic_level
            and model.stochastic_trend
        ):
            raise Wave1ModelError("statsmodels resolved a different structural specification")
        result = model.fit(method="lbfgs", maxiter=100, disp=False)
        if not bool(result.mle_retvals.get("converged", False)):
            raise Wave1ModelError("state-space optimizer did not converge")
        if not np.isfinite(np.asarray(result.params, dtype=np.float64)).all():
            raise Wave1ModelError("state-space fitted parameters are non-finite")
        self._result = result
        self._resolved_parameters = {
            "constructor": {"level": "local linear trend", "irregular": True},
            "fit": {"method": "lbfgs", "maxiter": 100, "disp": False},
            "model_flags": {
                "irregular": bool(model.irregular),
                "level": bool(model.level),
                "trend": bool(model.trend),
                "stochastic_level": bool(model.stochastic_level),
                "stochastic_trend": bool(model.stochastic_trend),
            },
            "param_names": list(model.param_names),
        }
        return self

    def predict(self, x: pd.DataFrame, *, context: PredictContext) -> pd.Series:
        if self._result is None:
            raise Wave1ModelError("state-space adapter must be fit before predict")
        if len(x.columns) != 0 or context.seed != self.fold_seed:
            raise Wave1ModelError("state-space predict accepts only an empty aligned frame")
        log_prediction = np.asarray(self._result.forecast(steps=len(x)), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = np.exp(log_prediction)
        if len(prediction) != len(x) or not (np.isfinite(prediction) & (prediction > 0.0)).all():
            raise Wave1ModelError("state-space forecast is non-positive or non-finite")
        return pd.Series(prediction, index=x.index, name="prediction")

    def diagnostics(self) -> AdapterDiagnostics:
        if self._result is None:
            raise Wave1ModelError("state-space diagnostics require a completed fit")
        return AdapterDiagnostics(
            active_features=(),
            dropped_all_missing_features=(),
            dropped_constant_spline_features=(),
            sample_weight_min=None,
            sample_weight_max=None,
            sample_weight_mean=None,
            resolved_parameters=dict(self._resolved_parameters),
        )


def build_wave1_model(model_id: str, *, fold_seed: int) -> Any:
    try:
        definition = MODEL_BY_ID[model_id]
    except KeyError as exc:
        raise Wave1ModelError(f"unknown Wave1 model_id: {model_id!r}") from exc
    if definition.family == "state_space_local_linear_trend":
        return Wave1StateSpaceAdapter(definition, fold_seed=fold_seed)
    return Wave1SupervisedAdapter(definition, fold_seed=fold_seed)


def resolved_parameter_manifest(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    fold_seed: int,
) -> dict[str, Any]:
    """Resolve every estimator on synthetic data only; never accepts evaluation truth."""

    if "true_fair_pe" in frame.columns:
        raise Wave1ModelError("resolved-parameter dry run rejects evaluation truth")
    output: dict[str, Any] = {}
    dates = pd.date_range("2000-01-03", periods=len(frame), freq="B")
    for model_id, definition in MODEL_BY_ID.items():
        model = build_wave1_model(model_id, fold_seed=fold_seed)
        if definition.feature_columns:
            x = frame.loc[:, list(definition.feature_columns)].copy()
        else:
            x = pd.DataFrame(index=frame.index)
        fit_context = FitContext(
            experiment_id="wave1-parameter-dry-run",
            fold_id="fold_000",
            seed=fold_seed,
            train_end=dates[-1],
            target_name="observed_pe",
            feature_metadata=feature_metadata(tuple(definition.feature_columns)),
        )
        model.fit(x, target, context=fit_context)
        diagnostics = model.diagnostics()
        output[model_id] = {
            "fold_seed": fold_seed,
            "active_features": list(diagnostics.active_features),
            "dropped_all_missing_features": list(diagnostics.dropped_all_missing_features),
            "dropped_constant_spline_features": list(
                diagnostics.dropped_constant_spline_features
            ),
            "sample_weight_min": diagnostics.sample_weight_min,
            "sample_weight_max": diagnostics.sample_weight_max,
            "sample_weight_mean": diagnostics.sample_weight_mean,
            "resolved_parameters": diagnostics.resolved_parameters,
        }
    return output
