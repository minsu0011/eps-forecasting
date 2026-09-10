"""Dependency-light reference models used to validate Model Lab plumbing."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contracts import (
    ContractError,
    FitContext,
    ModelMetadata,
    PredictContext,
    validate_model_call,
)


class HistoricalGeometricMeanPE:
    """Constant expanding-history baseline; intentionally not a promoted model."""

    def __init__(self) -> None:
        self._prediction: float | None = None
        self._metadata = ModelMetadata(
            model_id="historical_geometric_mean_pe",
            model_version="1",
            family="baseline",
            feature_ids=(),
            training_target="observed_pe",
            prediction_name="expected_pe",
            output_semantics="constant geometric mean of positive finite PIT observed P/E",
            deterministic=True,
        )

    def metadata(self) -> ModelMetadata:
        return self._metadata

    def fit(
        self,
        x: pd.DataFrame,
        y: pd.Series,
        *,
        context: FitContext,
    ) -> HistoricalGeometricMeanPE:
        validate_model_call(
            self._metadata,
            x,
            context.feature_metadata,
            phase="fit",
            target_name=context.target_name,
        )
        if not y.index.equals(x.index):
            raise ContractError("training target index must exactly match model input")
        numeric = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        valid = np.isfinite(numeric) & (numeric > 0.0)
        if not valid.any():
            raise ContractError("baseline fit requires at least one positive finite target")
        prediction = float(np.exp(np.mean(np.log(numeric[valid]))))
        if not math.isfinite(prediction) or prediction <= 0.0:
            raise ContractError("baseline fit produced an invalid prediction")
        self._prediction = prediction
        return self

    def predict(self, x: pd.DataFrame, *, context: PredictContext) -> pd.Series:
        validate_model_call(
            self._metadata,
            x,
            context.feature_metadata,
            phase="predict",
        )
        if self._prediction is None:
            raise RuntimeError("model must be fitted before predict")
        return pd.Series(self._prediction, index=x.index, name=self._metadata.prediction_name)
