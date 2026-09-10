"""Locked zero-intercept AR(1) estimation and fixed-origin forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import StructuralContractError


AR_MIN_CONSECUTIVE_PAIRS = 63
AR_RHO_MIN = -0.95
AR_RHO_MAX = 0.95


@dataclass(frozen=True)
class FixedOriginAR1Fit:
    rho: float
    last_residual: float
    consecutive_pair_count: int
    minimum_pair_count: int = AR_MIN_CONSECUTIVE_PAIRS

    def __post_init__(self) -> None:
        if not np.isfinite(self.rho) or not AR_RHO_MIN <= self.rho <= AR_RHO_MAX:
            raise StructuralContractError("AR(1) rho is outside the frozen bounds")
        if not np.isfinite(self.last_residual):
            raise StructuralContractError("AR(1) origin residual must be finite")
        if self.minimum_pair_count != AR_MIN_CONSECUTIVE_PAIRS:
            raise StructuralContractError("AR(1) minimum pair count differs from the design")
        if self.consecutive_pair_count < 0:
            raise StructuralContractError("AR(1) pair count must be non-negative")
        if self.consecutive_pair_count < self.minimum_pair_count and self.rho != 0.0:
            raise StructuralContractError("AR(1) must use rho=0 below the minimum pair count")

    def forecast(self, horizon_count: int) -> np.ndarray:
        if (
            not isinstance(horizon_count, int)
            or isinstance(horizon_count, bool)
            or horizon_count < 1
        ):
            raise StructuralContractError("AR(1) horizon_count must be a positive integer")
        horizons = np.arange(1, horizon_count + 1, dtype=np.int64)
        output = np.power(self.rho, horizons, dtype=np.float64) * self.last_residual
        if not np.isfinite(output).all():
            raise StructuralContractError("AR(1) fixed-origin forecast is non-finite")
        return output


def fit_zero_intercept_ar1(
    residuals: Sequence[float] | np.ndarray,
    *,
    session_positions: Sequence[int] | np.ndarray | None = None,
    segment_ids: Sequence[object] | np.ndarray | None = None,
) -> FixedOriginAR1Fit:
    """Fit only pairs adjacent in the declared session order and segment."""

    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise StructuralContractError("AR(1) residuals must be a non-empty finite vector")
    if session_positions is None:
        positions = np.arange(values.size, dtype=np.int64)
    else:
        positions_float = np.asarray(session_positions, dtype=np.float64)
        if (
            positions_float.shape != values.shape
            or not np.isfinite(positions_float).all()
            or not np.equal(positions_float, np.floor(positions_float)).all()
        ):
            raise StructuralContractError("AR(1) session positions must be finite integers")
        positions = positions_float.astype(np.int64)
    if segment_ids is None:
        segments = np.zeros(values.size, dtype=np.int64)
    else:
        segments = np.asarray(segment_ids, dtype=object)
        if segments.shape != values.shape or any(value is None for value in segments):
            raise StructuralContractError("AR(1) segment identifiers are invalid")
    if values.size > 1 and (np.diff(positions) <= 0).any():
        raise StructuralContractError("AR(1) rows must be in strictly increasing session order")

    pair_mask = (positions[1:] == positions[:-1] + 1) & (segments[1:] == segments[:-1])
    pair_count = int(pair_mask.sum())
    rho = 0.0
    if pair_count >= AR_MIN_CONSECUTIVE_PAIRS:
        previous = values[:-1][pair_mask]
        current = values[1:][pair_mask]
        denominator = float(np.dot(previous, previous))
        if denominator > 1e-12:
            rho = float(np.dot(previous, current) / denominator)
            rho = float(np.clip(rho, AR_RHO_MIN, AR_RHO_MAX))
    return FixedOriginAR1Fit(
        rho=rho,
        last_residual=float(values[-1]),
        consecutive_pair_count=pair_count,
    )
