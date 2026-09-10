"""Three fixed, score-free probabilistic candidate adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ..contracts import FitContext, PredictContext, validate_feature_frame
from .contracts import (
    EVALUATION_ONLY_COLUMN,
    NORMAL_QUANTILE_Z,
    QuantilePredictionBatch,
    QUANTILE_LEVELS,
    ProbabilisticContractError,
    canonical_json_bytes,
    require_no_evaluation_truth,
    seal_payload,
    sha256_bytes,
)
from .crossing import make_prediction_batch
from .spec import MODEL_BY_ID, ProbabilisticModelDefinition, feature_metadata


OBSERVED_TARGET = "observed_pe"


class AdapterUnavailable(ProbabilisticContractError):
    """Raised when an optional candidate cannot run in the active environment."""


class AdapterExecutionError(ProbabilisticContractError):
    """Raised on the one permitted fit/predict attempt; retry is forbidden."""


class AdapterWarningError(AdapterExecutionError):
    """Raised because candidate warnings are evidence, never silently suppressed."""


def deterministic_random_state(
    *, seed: int, fold_id: str, model_id: str, quantile: float | None = None
) -> int:
    payload = f"{seed}|{fold_id}|{model_id}|{quantile!r}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


def _finite_numeric_frame(frame: pd.DataFrame, *, phase: str) -> pd.DataFrame:
    require_no_evaluation_truth(frame.columns, context=f"probabilistic {phase}")
    if OBSERVED_TARGET.casefold() in {str(value).casefold() for value in frame.columns}:
        raise ProbabilisticContractError("same-row observed_pe is forbidden in model inputs")
    validate_feature_frame(frame, feature_metadata(), phase=phase)  # type: ignore[arg-type]
    try:
        output = frame.astype(np.float64).replace([np.inf, -np.inf], np.nan)
    except (TypeError, ValueError) as exc:
        raise ProbabilisticContractError("probabilistic features must be numeric") from exc
    return output


def _validated_target(y: pd.Series, *, rows: int, context: FitContext) -> np.ndarray:
    if context.target_name != OBSERVED_TARGET or y.name != OBSERVED_TARGET:
        raise ProbabilisticContractError("training target must be named observed_pe")
    if EVALUATION_ONLY_COLUMN.casefold() == str(y.name).casefold():
        raise ProbabilisticContractError("evaluation truth cannot be a training target")
    values = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64)
    if values.shape != (rows,) or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ProbabilisticContractError("observed_pe target must be row-aligned, positive, finite")
    return np.log(values)


def _fit_once(estimator: Any, x: np.ndarray, y: np.ndarray, *, model_id: str) -> Any:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(x, y)
    except Exception as exc:  # third-party exceptions are normalized at this boundary
        raise AdapterExecutionError(f"{model_id} fit failed; retry/fallback forbidden") from exc
    if caught:
        detail = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught[:5])
        raise AdapterWarningError(f"{model_id} fit emitted warning(s): {detail}")
    return estimator


def _predict_once(estimator: Any, x: np.ndarray, *, model_id: str) -> np.ndarray:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values = np.asarray(estimator.predict(x), dtype=np.float64)
    except Exception as exc:
        raise AdapterExecutionError(f"{model_id} predict failed; retry/fallback forbidden") from exc
    if caught:
        detail = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught[:5])
        raise AdapterWarningError(f"{model_id} predict emitted warning(s): {detail}")
    if values.shape != (x.shape[0],) or not np.isfinite(values).all():
        raise AdapterExecutionError(f"{model_id} produced invalid log quantiles")
    return values


class ProbabilisticAdapter(ABC):
    """Fit once on chronological training data, then predict one future block."""

    def __init__(self, definition: ProbabilisticModelDefinition) -> None:
        self.definition = definition
        self._fit_attempts = 0
        self._fitted = False
        self._train_end: pd.Timestamp | None = None
        self._fold_id: str | None = None
        self._experiment_id: str | None = None
        self._seed: int | None = None
        self._active_columns: tuple[str, ...] = ()

    def metadata(self):
        return self.definition.metadata()

    @property
    def fit_attempts(self) -> int:
        return self._fit_attempts

    @property
    def active_feature_manifest(self) -> tuple[str, ...]:
        if not self._fitted:
            raise AdapterExecutionError("active feature manifest is unavailable before fit")
        return self._active_columns

    def fit(self, x: pd.DataFrame, y: pd.Series, *, context: FitContext) -> "ProbabilisticAdapter":
        if self._fit_attempts != 0:
            raise AdapterExecutionError("one fit attempt per adapter/fold; retries are forbidden")
        self._fit_attempts += 1
        if context.seed < 0:
            raise ProbabilisticContractError("seed must be non-negative")
        if context.feature_metadata != feature_metadata():
            raise ProbabilisticContractError("FitContext feature sidecar differs from lock")
        frame = _finite_numeric_frame(x, phase="fit")
        target = _validated_target(y, rows=len(frame), context=context)
        active = tuple(column for column in frame if not frame[column].isna().all())
        if not active:
            raise ProbabilisticContractError("all training feature columns are missing")
        self._active_columns = active
        self._fit_impl(frame.loc[:, list(active)], target, context=context)
        self._train_end = pd.Timestamp(context.train_end)
        self._fold_id = context.fold_id
        self._experiment_id = context.experiment_id
        self._seed = context.seed
        self._fitted = True
        return self

    def predict(self, x: pd.DataFrame, *, context: PredictContext) -> QuantilePredictionBatch:
        if not self._fitted or self._train_end is None:
            raise AdapterExecutionError("adapter must be fitted before predict")
        if context.fold_id != self._fold_id:
            raise ProbabilisticContractError("fit and predict fold ids differ")
        if context.experiment_id != self._experiment_id or context.seed != self._seed:
            raise ProbabilisticContractError("fit and predict experiment/seed bindings differ")
        if context.feature_metadata != feature_metadata():
            raise ProbabilisticContractError("PredictContext feature sidecar differs from lock")
        if pd.Timestamp(context.prediction_start) <= self._train_end:
            raise ProbabilisticContractError("prediction block must start after training ends")
        frame = _finite_numeric_frame(x, phase="predict")
        return self._predict_impl(frame.loc[:, list(self._active_columns)], context=context)

    @abstractmethod
    def _fit_impl(self, x: pd.DataFrame, y_log: np.ndarray, *, context: FitContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict_impl(self, x: pd.DataFrame, *, context: PredictContext) -> QuantilePredictionBatch:
        raise NotImplementedError


class QuantileLinearAdapter(ProbabilisticAdapter):
    def __init__(self) -> None:
        super().__init__(MODEL_BY_ID["qlinear_l1_with_regime_v1"])
        self._imputer: SimpleImputer | None = None
        self._scaler: StandardScaler | None = None
        self._estimators: list[QuantileRegressor] = []

    def _fit_impl(self, x: pd.DataFrame, y_log: np.ndarray, *, context: FitContext) -> None:
        self._imputer = SimpleImputer(strategy="median")
        self._scaler = StandardScaler()
        matrix = self._scaler.fit_transform(self._imputer.fit_transform(x))
        for quantile in QUANTILE_LEVELS:
            estimator = QuantileRegressor(
                quantile=quantile,
                alpha=0.001,
                fit_intercept=True,
                solver="highs",
                solver_options=None,
            )
            self._estimators.append(
                _fit_once(estimator, matrix, y_log, model_id=self.definition.model_id)
            )

    def _predict_impl(self, x: pd.DataFrame, *, context: PredictContext) -> QuantilePredictionBatch:
        assert self._imputer is not None and self._scaler is not None
        matrix = self._scaler.transform(self._imputer.transform(x))
        raw = np.column_stack(
            [
                _predict_once(item, matrix, model_id=self.definition.model_id)
                for item in self._estimators
            ]
        )
        return make_prediction_batch(model_id=self.definition.model_id, raw_log_quantiles=raw)


class QuantileHistGBAdapter(ProbabilisticAdapter):
    def __init__(self) -> None:
        super().__init__(MODEL_BY_ID["qhistgb_with_regime_v1"])
        self._estimators: list[HistGradientBoostingRegressor] = []

    def _fit_impl(self, x: pd.DataFrame, y_log: np.ndarray, *, context: FitContext) -> None:
        matrix = x.to_numpy(dtype=np.float64)
        for quantile in QUANTILE_LEVELS:
            estimator = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_iter=180,
                learning_rate=0.04,
                max_leaf_nodes=15,
                max_depth=None,
                min_samples_leaf=20,
                l2_regularization=0.2,
                max_bins=255,
                max_features=1.0,
                early_stopping=False,
                warm_start=False,
                random_state=deterministic_random_state(
                    seed=context.seed,
                    fold_id=context.fold_id,
                    model_id=self.definition.model_id,
                    quantile=quantile,
                ),
            )
            self._estimators.append(
                _fit_once(estimator, matrix, y_log, model_id=self.definition.model_id)
            )

    def _predict_impl(self, x: pd.DataFrame, *, context: PredictContext) -> QuantilePredictionBatch:
        matrix = x.to_numpy(dtype=np.float64)
        raw = np.column_stack(
            [
                _predict_once(item, matrix, model_id=self.definition.model_id)
                for item in self._estimators
            ]
        )
        return make_prediction_batch(model_id=self.definition.model_id, raw_log_quantiles=raw)


class NGBoostNormalAdapter(ProbabilisticAdapter):
    def __init__(self) -> None:
        super().__init__(MODEL_BY_ID["ngboost_normal_crps_with_regime_v1"])
        self._imputer: SimpleImputer | None = None
        self._estimator: Any | None = None

    @staticmethod
    def _ngboost_types() -> tuple[Any, Any, Any]:
        try:
            ngboost = importlib.import_module("ngboost")
            distns = importlib.import_module("ngboost.distns")
            scores = importlib.import_module("ngboost.scores")
        except ImportError as exc:
            raise AdapterUnavailable(
                "ngboost==0.5.11 is unavailable; use the dedicated pinned environment"
            ) from exc
        version = getattr(ngboost, "__version__", None)
        if version != "0.5.11":
            raise AdapterUnavailable(f"NGBoost environment drift: expected 0.5.11, got {version!r}")
        return ngboost.NGBRegressor, distns.Normal, scores.CRPScore

    def _fit_impl(self, x: pd.DataFrame, y_log: np.ndarray, *, context: FitContext) -> None:
        NGBRegressor, Normal, CRPScore = self._ngboost_types()
        self._imputer = SimpleImputer(strategy="median")
        matrix = self._imputer.fit_transform(x)
        random_state = deterministic_random_state(
            seed=context.seed,
            fold_id=context.fold_id,
            model_id=self.definition.model_id,
        )
        base = DecisionTreeRegressor(
            criterion="friedman_mse",
            splitter="best",
            max_depth=2,
            min_samples_leaf=20,
            max_features=None,
            random_state=random_state,
        )
        estimator = NGBRegressor(
            Dist=Normal,
            Score=CRPScore,
            Base=base,
            natural_gradient=True,
            n_estimators=300,
            learning_rate=0.03,
            minibatch_frac=1.0,
            col_sample=1.0,
            tol=0.0001,
            random_state=random_state,
            verbose=False,
            early_stopping_rounds=None,
        )
        self._estimator = _fit_once(estimator, matrix, y_log, model_id=self.definition.model_id)

    def _predict_impl(self, x: pd.DataFrame, *, context: PredictContext) -> QuantilePredictionBatch:
        assert self._imputer is not None and self._estimator is not None
        matrix = self._imputer.transform(x)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                distribution = self._estimator.pred_dist(matrix)
        except Exception as exc:
            raise AdapterExecutionError(
                f"{self.definition.model_id} predict failed; retry/fallback forbidden"
            ) from exc
        if caught:
            detail = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught[:5])
            raise AdapterWarningError(
                f"{self.definition.model_id} predict emitted warning(s): {detail}"
            )
        params = distribution.params
        loc = np.asarray(params["loc"], dtype=np.float64).reshape(-1)
        scale = np.asarray(params["scale"], dtype=np.float64).reshape(-1)
        if (
            loc.shape != (len(x),)
            or scale.shape != (len(x),)
            or not np.isfinite(loc).all()
            or not np.isfinite(scale).all()
            or (scale <= 0.0).any()
        ):
            raise AdapterExecutionError("NGBoost returned invalid Normal parameters")
        raw = loc[:, None] + scale[:, None] * NORMAL_QUANTILE_Z[None, :]
        return make_prediction_batch(
            model_id=self.definition.model_id,
            raw_log_quantiles=raw,
            density_parameters={"loc": loc, "scale": scale},
        )


def create_adapter(model_id: str) -> ProbabilisticAdapter:
    if model_id == "qlinear_l1_with_regime_v1":
        return QuantileLinearAdapter()
    if model_id == "qhistgb_with_regime_v1":
        return QuantileHistGBAdapter()
    if model_id == "ngboost_normal_crps_with_regime_v1":
        return NGBoostNormalAdapter()
    raise ProbabilisticContractError(f"unknown probabilistic candidate: {model_id!r}")


_ADAPTER_CLASS_BY_ID = {
    "qlinear_l1_with_regime_v1": QuantileLinearAdapter,
    "qhistgb_with_regime_v1": QuantileHistGBAdapter,
    "ngboost_normal_crps_with_regime_v1": NGBoostNormalAdapter,
}


def _installed_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def adapter_binding_payload(model_id: str) -> dict[str, Any]:
    """Bind factory class, complete source bytes, fixed config, and runtime packages."""

    if model_id not in _ADAPTER_CLASS_BY_ID:
        raise ProbabilisticContractError(f"unknown probabilistic candidate: {model_id!r}")
    source_bytes = Path(__file__).read_bytes()
    definition = MODEL_BY_ID[model_id]
    return seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.probabilistic_adapter_binding.v2",
            "model_id": model_id,
            "factory": "pe_regime_v04.model_lab.probabilistic.adapters.create_adapter",
            "adapter_class": (
                f"{_ADAPTER_CLASS_BY_ID[model_id].__module__}."
                f"{_ADAPTER_CLASS_BY_ID[model_id].__qualname__}"
            ),
            "adapter_source_sha256": sha256_bytes(source_bytes),
            "definition": {
                "family": definition.family,
                "variant": definition.variant,
                "package": definition.package,
                "parameters": json.loads(canonical_json_bytes(definition.parameters)),
                "feature_columns": list(definition.feature_columns),
            },
            "required_runtime": candidate_environment_payload(model_id),
        }
    )


def adapter_binding_sha256_by_id() -> dict[str, str]:
    return {
        model_id: sha256_bytes(canonical_json_bytes(adapter_binding_payload(model_id)))
        for model_id in _ADAPTER_CLASS_BY_ID
    }


_COMMON_RUNTIME = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
}


def candidate_environment_payload(model_id: str) -> dict[str, Any]:
    if model_id not in _ADAPTER_CLASS_BY_ID:
        raise ProbabilisticContractError(f"unknown probabilistic candidate: {model_id!r}")
    distributions = dict(_COMMON_RUNTIME)
    if model_id == "ngboost_normal_crps_with_regime_v1":
        distributions["ngboost"] = "0.5.11"
    return seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.probabilistic_candidate_environment.v2",
            "model_id": model_id,
            "python_major_minor": "3.10",
            "implementation": "CPython",
            "distributions": distributions,
            "inner_threads": 1,
            "gpu": "OFF",
        }
    )


def candidate_environment_sha256_by_id() -> dict[str, str]:
    return {
        model_id: sha256_bytes(canonical_json_bytes(candidate_environment_payload(model_id)))
        for model_id in _ADAPTER_CLASS_BY_ID
    }


def verify_candidate_runtime_environment(model_id: str) -> str:
    """Fail before fit if the active interpreter differs from candidate-specific pins."""

    payload = candidate_environment_payload(model_id)
    if ".".join(platform.python_version_tuple()[:2]) != payload["python_major_minor"]:
        raise AdapterUnavailable("candidate requires the pinned Python 3.10 runtime")
    if platform.python_implementation() != payload["implementation"]:
        raise AdapterUnavailable("candidate requires CPython")
    drift = {
        name: {"expected": expected, "observed": _installed_version(name)}
        for name, expected in payload["distributions"].items()
        if _installed_version(name) != expected
    }
    if drift:
        raise AdapterUnavailable(f"candidate runtime environment drift: {drift}")
    return sha256_bytes(canonical_json_bytes(payload))
