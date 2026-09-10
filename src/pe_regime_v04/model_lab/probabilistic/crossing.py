"""Truth-blind deterministic quantile crossing diagnostics and repair."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .contracts import (
    CrossingDiagnostics,
    QuantilePredictionBatch,
    QUANTILE_LEVELS,
    ProbabilisticContractError,
)


MAXIMUM_RAW_CROSSING_ROW_RATE = 0.05
MAXIMUM_P95_CROSSING_MAGNITUDE = 0.02


def stable_monotone_rearrangement(
    raw_log_quantiles: object,
) -> tuple[np.ndarray, CrossingDiagnostics]:
    """Stable-sort five log quantiles per row and retain raw diagnostics.

    ``kind="stable"`` is part of the locked policy. Ties are allowed and a
    collapsed p10--p90 interval is reported rather than silently widened.
    """

    raw = np.asarray(raw_log_quantiles, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(QUANTILE_LEVELS) or raw.shape[0] < 1:
        raise ProbabilisticContractError("raw log quantiles must have shape (n,5)")
    if not np.isfinite(raw).all():
        raise ProbabilisticContractError("raw log quantiles must be finite")
    adjacent = np.diff(raw, axis=1)
    negative = np.maximum(-adjacent, 0.0)
    pair_counts = (adjacent < 0.0).sum(axis=1)
    crossing = pair_counts > 0
    repaired = np.sort(raw, axis=1, kind="stable")
    post_crossing = (np.diff(repaired, axis=1) < 0.0).any(axis=1)
    collapsed = repaired[:, -1] == repaired[:, 0]
    diagnostics = CrossingDiagnostics(
        rows=int(raw.shape[0]),
        crossing_rows=int(crossing.sum()),
        crossing_row_rate=float(crossing.mean()),
        maximum_crossing_pair_count=int(pair_counts.max(initial=0)),
        p95_total_negative_adjacent_gap=float(
            np.quantile(negative.sum(axis=1), 0.95, method="linear")
        ),
        post_repair_crossing_rows=int(post_crossing.sum()),
        interval_collapse_rows=int(collapsed.sum()),
    )
    return repaired, diagnostics


def crossing_screen_status(diagnostics: CrossingDiagnostics) -> str:
    """Return the immutable cheap-screen crossing decision without retry."""

    if diagnostics.post_repair_crossing_rows != 0:
        return "FAIL_POST_REPAIR_CROSSING"
    if diagnostics.crossing_row_rate > MAXIMUM_RAW_CROSSING_ROW_RATE:
        return "FAIL_RAW_CROSSING_RATE"
    if diagnostics.p95_total_negative_adjacent_gap > MAXIMUM_P95_CROSSING_MAGNITUDE:
        return "FAIL_RAW_CROSSING_MAGNITUDE"
    return "PASS"


def make_prediction_batch(
    *,
    model_id: str,
    raw_log_quantiles: object,
    density_parameters: Mapping[str, np.ndarray] | None = None,
) -> QuantilePredictionBatch:
    repaired, diagnostics = stable_monotone_rearrangement(raw_log_quantiles)
    pe_quantiles = np.exp(repaired)
    if not np.isfinite(pe_quantiles).all() or (pe_quantiles <= 0.0).any():
        raise ProbabilisticContractError("exponentiated quantiles overflow or are invalid")
    return QuantilePredictionBatch(
        model_id=model_id,
        raw_log_quantiles=np.asarray(raw_log_quantiles, dtype=np.float64),
        repaired_log_quantiles=repaired,
        pe_quantiles=pe_quantiles,
        diagnostics=diagnostics,
        density_parameters=density_parameters,
    )
