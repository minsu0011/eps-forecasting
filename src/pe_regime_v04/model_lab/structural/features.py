"""Track-A groupwise lag-one features and score-free intervention audits."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Sequence

import numpy as np
import pandas as pd

from .contracts import (
    STRUCTURAL_DESIGN_SHA256,
    StructuralContractError,
    logical_frame_sha256,
    normalize_dates,
    require_no_evaluation_truth,
    require_sha256,
    require_unique_columns,
)


FUNDAMENTAL_CURRENT_PIT = (
    "eps_ttm_growth_126",
    "eps_ttm_growth_252",
    "eps_staleness_days",
    "eps_period_age_days",
    "eps_confidence",
    "eps_disagreement",
    "eps_approximation_flag",
)
VALUATION_LAGGED = ("pe_median_252_lag", "pe_median_756_lag")
MARKET_CURRENT = (
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
)
REGIME_CURRENT = (
    "v04_current_p_bear",
    "v04_current_p_sideways",
    "v04_current_p_bull",
    "forecast_p_bear",
    "forecast_p_sideways",
    "forecast_p_bull",
    "pe_bear_median",
    "pe_sideways_median",
    "pe_bull_median",
    "pe_bear_effective_history",
    "pe_sideways_effective_history",
    "pe_bull_effective_history",
)
TRACK_A_LAG_SOURCE_COLUMNS = MARKET_CURRENT + REGIME_CURRENT
TRACK_A_GROUP_COLUMNS = ("seed", "entity_id")
TRACK_A_SORT_COLUMN = "date"
REQUIRED_UPSTREAM_AUDITS = (
    "prefix_invariance",
    "future_intervention",
    "same_row_target_leakage",
    "pit_availability",
    "publication_date",
    "restatement_availability",
)


def lag1_feature_name(column: str) -> str:
    return f"{column}__lag1_by_seed_entity"


TRACK_A_LAGGED_COLUMNS = tuple(lag1_feature_name(column) for column in TRACK_A_LAG_SOURCE_COLUMNS)
TRACK_A_FEATURE_COLUMNS = FUNDAMENTAL_CURRENT_PIT + VALUATION_LAGGED + TRACK_A_LAGGED_COLUMNS
TRACK_C_FEATURE_COLUMNS = (
    FUNDAMENTAL_CURRENT_PIT + VALUATION_LAGGED + MARKET_CURRENT + REGIME_CURRENT
)


@dataclass(frozen=True)
class TrackASourceAuditBinding:
    """Immutable reference to the upstream PIT/audit evidence for source columns."""

    audit_sha256: str
    passed_audits: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(self.audit_sha256, field="audit_sha256")
        if self.passed_audits != REQUIRED_UPSTREAM_AUDITS:
            raise StructuralContractError(
                "Track-A lag source requires the exact six frozen upstream audits in order"
            )


@dataclass(frozen=True)
class TrackALag1Artifact:
    """Immutable logical snapshot of registered Track-A derived features."""

    csv_text: str
    row_count: int
    source_identity_sha256: str
    source_content_sha256: str
    upstream_audit_sha256: str
    content_sha256: str
    design_sha256: str = STRUCTURAL_DESIGN_SHA256

    def __post_init__(self) -> None:
        for field_name in (
            "source_identity_sha256",
            "source_content_sha256",
            "upstream_audit_sha256",
            "content_sha256",
            "design_sha256",
        ):
            require_sha256(getattr(self, field_name), field=field_name)
        if self.design_sha256 != STRUCTURAL_DESIGN_SHA256:
            raise StructuralContractError("Track-A artifact is bound to a different design")
        if (
            not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or self.row_count < 1
        ):
            raise StructuralContractError("Track-A artifact row_count must be positive")
        frame = self.to_frame()
        if len(frame) != self.row_count:
            raise StructuralContractError("Track-A artifact row count changed")
        if logical_frame_sha256(frame) != self.content_sha256:
            raise StructuralContractError("Track-A artifact content hash changed")

    def to_frame(self) -> pd.DataFrame:
        frame = pd.read_csv(
            StringIO(self.csv_text),
            float_precision="round_trip",
            dtype={"entity_id": "string", "date": "string"},
        )
        expected = ("seed", "entity_id", "date", *TRACK_A_LAGGED_COLUMNS)
        if tuple(frame.columns) != expected:
            raise StructuralContractError("Track-A artifact schema changed")
        frame["date"] = normalize_dates(frame["date"], context="Track-A artifact date")
        return frame


def _normalized_lag_source(frame: pd.DataFrame) -> pd.DataFrame:
    require_unique_columns(frame, context="Track-A lag source")
    require_no_evaluation_truth(frame.columns, context="Track-A lag source")
    required = (*TRACK_A_GROUP_COLUMNS, TRACK_A_SORT_COLUMN, *TRACK_A_LAG_SOURCE_COLUMNS)
    missing = [column for column in required if column not in frame]
    if missing:
        raise StructuralContractError(f"Track-A lag source is missing columns: {missing}")
    output = frame.loc[:, list(required)].copy()
    output["date"] = normalize_dates(output["date"], context="Track-A lag source date").to_numpy(
        copy=True
    )
    if output.loc[:, list(TRACK_A_GROUP_COLUMNS)].isna().any().any():
        raise StructuralContractError("Track-A group identity contains missing values")
    output["entity_id"] = output["entity_id"].astype(str)
    seed_numeric = pd.to_numeric(output["seed"], errors="coerce")
    if seed_numeric.isna().any() or (seed_numeric % 1.0 != 0.0).any():
        raise StructuralContractError("Track-A seed must contain integral values")
    output["seed"] = seed_numeric.astype("int64")
    if output.duplicated([*TRACK_A_GROUP_COLUMNS, TRACK_A_SORT_COLUMN]).any():
        raise StructuralContractError("Track-A lag source identities must be one-to-one")
    output = output.sort_values(
        [*TRACK_A_GROUP_COLUMNS, TRACK_A_SORT_COLUMN], kind="mergesort"
    ).reset_index(drop=True)
    for column in TRACK_A_LAG_SOURCE_COLUMNS:
        numeric = pd.to_numeric(output[column], errors="coerce").astype("float64")
        output[column] = numeric.mask(~np.isfinite(numeric.to_numpy(dtype=np.float64)))
    return output


def build_track_a_lag1_artifact(
    frame: pd.DataFrame,
    *,
    source_audit: TrackASourceAuditBinding,
) -> TrackALag1Artifact:
    """Shift market/regime values exactly one session inside each seed/entity group."""

    source = _normalized_lag_source(frame)
    identity = source.loc[:, [*TRACK_A_GROUP_COLUMNS, TRACK_A_SORT_COLUMN]].copy()
    shifted = source.groupby(list(TRACK_A_GROUP_COLUMNS), sort=False, observed=True)[
        list(TRACK_A_LAG_SOURCE_COLUMNS)
    ].shift(1)
    shifted.columns = list(TRACK_A_LAGGED_COLUMNS)
    output = pd.concat([identity, shifted], axis=1)
    output = output.loc[:, ["seed", "entity_id", "date", *TRACK_A_LAGGED_COLUMNS]]
    csv_text = output.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )
    round_trip = pd.read_csv(StringIO(csv_text), float_precision="round_trip")
    round_trip["date"] = normalize_dates(round_trip["date"], context="Track-A round trip date")
    return TrackALag1Artifact(
        csv_text=csv_text,
        row_count=len(output),
        source_identity_sha256=logical_frame_sha256(identity),
        source_content_sha256=logical_frame_sha256(source),
        upstream_audit_sha256=source_audit.audit_sha256,
        content_sha256=logical_frame_sha256(round_trip),
    )


def verify_track_a_lag1_artifact(
    frame: pd.DataFrame,
    artifact: TrackALag1Artifact,
    *,
    source_audit: TrackASourceAuditBinding,
) -> None:
    recomputed = build_track_a_lag1_artifact(frame, source_audit=source_audit)
    if recomputed != artifact:
        raise StructuralContractError(
            "Track-A lag artifact differs from deterministic recomputation"
        )


@dataclass(frozen=True)
class TrackAInterventionAudit:
    prefix_invariance: bool
    future_intervention: bool
    same_row_source_is_lagged: bool
    same_row_target_ignored: bool
    group_boundary_isolation: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.prefix_invariance,
                self.future_intervention,
                self.same_row_source_is_lagged,
                self.same_row_target_ignored,
                self.group_boundary_isolation,
            )
        )


def audit_track_a_lag_transform(
    frame: pd.DataFrame,
    *,
    source_audit: TrackASourceAuditBinding,
    perturb_column: str = MARKET_CURRENT[0],
) -> TrackAInterventionAudit:
    """Run deterministic intervention checks without using a candidate target or score."""

    if perturb_column not in TRACK_A_LAG_SOURCE_COLUMNS:
        raise StructuralContractError("perturb_column is not a lag source")
    normalized = _normalized_lag_source(frame)
    groups = normalized.groupby(list(TRACK_A_GROUP_COLUMNS), sort=False, observed=True)
    eligible = [group.index.to_list() for _, group in groups if len(group) >= 3]
    if not eligible:
        raise StructuralContractError("Track-A intervention audit needs a group with >=3 rows")
    indices = eligible[0]
    pivot = indices[-2]
    prefix_end = normalized.loc[pivot, "date"]

    full = build_track_a_lag1_artifact(normalized, source_audit=source_audit).to_frame()
    prefix_source = normalized.loc[normalized["date"] <= prefix_end].copy()
    prefix = build_track_a_lag1_artifact(prefix_source, source_audit=source_audit).to_frame()
    common_full = full.loc[full["date"] <= prefix_end].reset_index(drop=True)
    prefix_invariance = common_full.equals(prefix.reset_index(drop=True))

    future_changed = normalized.copy()
    future_mask = future_changed["date"] > prefix_end
    future_changed.loc[future_mask, perturb_column] = (
        pd.to_numeric(future_changed.loc[future_mask, perturb_column], errors="coerce") + 12345.0
    )
    future_artifact = build_track_a_lag1_artifact(
        future_changed, source_audit=source_audit
    ).to_frame()
    future_intervention = (
        full.loc[full["date"] <= prefix_end]
        .reset_index(drop=True)
        .equals(future_artifact.loc[future_artifact["date"] <= prefix_end].reset_index(drop=True))
    )

    same_changed = normalized.copy()
    original = float(pd.to_numeric(pd.Series([same_changed.loc[pivot, perturb_column]])).iloc[0])
    same_changed.loc[pivot, perturb_column] = original + 777.0
    same_artifact = build_track_a_lag1_artifact(same_changed, source_audit=source_audit).to_frame()
    derived = lag1_feature_name(perturb_column)
    same_row_source_is_lagged = bool(
        pd.isna(full.loc[pivot, derived])
        and pd.isna(same_artifact.loc[pivot, derived])
        or full.loc[pivot, derived] == same_artifact.loc[pivot, derived]
    ) and bool(full.loc[pivot + 1, derived] != same_artifact.loc[pivot + 1, derived])

    with_target = normalized.copy()
    with_target["observed_pe"] = np.arange(len(with_target), dtype=np.float64) + 10.0
    target_a = build_track_a_lag1_artifact(with_target, source_audit=source_audit)
    with_target["observed_pe"] = with_target["observed_pe"] * 1000.0
    target_b = build_track_a_lag1_artifact(with_target, source_audit=source_audit)
    same_row_target_ignored = target_a == target_b

    group_boundary_isolation = True
    for _, group in full.groupby(list(TRACK_A_GROUP_COLUMNS), sort=False, observed=True):
        first = group.iloc[0]
        if not first.loc[list(TRACK_A_LAGGED_COLUMNS)].isna().all():
            group_boundary_isolation = False
            break

    return TrackAInterventionAudit(
        prefix_invariance=prefix_invariance,
        future_intervention=future_intervention,
        same_row_source_is_lagged=same_row_source_is_lagged,
        same_row_target_ignored=same_row_target_ignored,
        group_boundary_isolation=group_boundary_isolation,
    )


def require_exact_feature_columns(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    identity_columns: Sequence[str] = (),
    context: str,
) -> None:
    require_unique_columns(frame, context=context)
    require_no_evaluation_truth(frame.columns, context=context)
    expected = (*identity_columns, *feature_columns)
    if tuple(frame.columns) != expected:
        raise StructuralContractError(
            f"{context} schema must be exact: expected={expected}, actual={tuple(frame.columns)}"
        )
