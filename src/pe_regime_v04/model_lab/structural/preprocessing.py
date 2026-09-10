"""Deterministic train-only numeric preprocessing for structural kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .contracts import StructuralContractError, require_no_evaluation_truth, require_unique_columns


@dataclass(frozen=True)
class NumericPreprocessorFit:
    requested_columns: tuple[str, ...]
    active_columns: tuple[str, ...]
    dropped_all_missing_columns: tuple[str, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(set(self.requested_columns)) != len(self.requested_columns):
            raise StructuralContractError("requested preprocessing columns must be unique")
        if set(self.active_columns).union(self.dropped_all_missing_columns) != set(
            self.requested_columns
        ):
            raise StructuralContractError("preprocessor active/drop partition is incomplete")
        if len(self.active_columns) != len(self.medians) or len(self.medians) != len(self.scales):
            raise StructuralContractError("preprocessor parameter lengths differ")
        if not self.active_columns:
            raise StructuralContractError("preprocessor cannot drop every requested column")
        if not np.isfinite(np.asarray(self.medians, dtype=np.float64)).all():
            raise StructuralContractError("preprocessor medians must be finite")
        scales = np.asarray(self.scales, dtype=np.float64)
        if not np.isfinite(scales).all() or (scales <= 0.0).any():
            raise StructuralContractError("preprocessor scales must be positive and finite")


def _numeric(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> pd.DataFrame:
    require_unique_columns(frame, context=context)
    require_no_evaluation_truth(frame.columns, context=context)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise StructuralContractError(f"{context} is missing columns: {missing}")
    output = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    return output.replace([np.inf, -np.inf], np.nan)


def fit_numeric_preprocessor(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
) -> tuple[NumericPreprocessorFit, np.ndarray]:
    requested = tuple(columns)
    if not requested or len(set(requested)) != len(requested):
        raise StructuralContractError(f"{context} requested columns must be non-empty and unique")
    numeric = _numeric(frame, requested, context=context)
    active = tuple(column for column in requested if not numeric[column].isna().all())
    dropped = tuple(column for column in requested if column not in active)
    if not active:
        raise StructuralContractError(f"{context} has no active numeric columns")
    medians = numeric.loc[:, list(active)].median(axis=0, skipna=True).to_numpy(dtype=np.float64)
    values = numeric.loc[:, list(active)].to_numpy(dtype=np.float64, na_value=np.nan)
    missing = np.isnan(values)
    if missing.any():
        values[missing] = medians[np.nonzero(missing)[1]]
    scales = np.std(values, axis=0, ddof=0)
    scales[(~np.isfinite(scales)) | (scales == 0.0)] = 1.0
    standardized = (values - medians) / scales
    if not np.isfinite(standardized).all():
        raise StructuralContractError(f"{context} standardized values are non-finite")
    fit = NumericPreprocessorFit(
        requested_columns=requested,
        active_columns=active,
        dropped_all_missing_columns=dropped,
        medians=tuple(float(value) for value in medians),
        scales=tuple(float(value) for value in scales),
    )
    return fit, standardized


def transform_numeric_preprocessor(
    frame: pd.DataFrame,
    fit: NumericPreprocessorFit,
    *,
    context: str,
) -> np.ndarray:
    numeric = _numeric(frame, fit.requested_columns, context=context)
    values = numeric.loc[:, list(fit.active_columns)].to_numpy(dtype=np.float64, na_value=np.nan)
    medians = np.asarray(fit.medians, dtype=np.float64)
    missing = np.isnan(values)
    if missing.any():
        values[missing] = medians[np.nonzero(missing)[1]]
    output = (values - medians) / np.asarray(fit.scales, dtype=np.float64)
    if not np.isfinite(output).all():
        raise StructuralContractError(f"{context} transformed values are non-finite")
    return output
