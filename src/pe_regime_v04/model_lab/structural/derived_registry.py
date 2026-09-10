"""Isolated definitions for Track-A lag-one features (no frozen registry edit)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import STRUCTURAL_DESIGN_SHA256, StructuralContractError
from .features import MARKET_CURRENT, REGIME_CURRENT, lag1_feature_name


@dataclass(frozen=True)
class TrackADerivedFeatureDefinition:
    feature_id: str
    column_name: str
    source_column: str
    source_block: str
    transform: str
    group_keys: tuple[str, str]
    sort_key: str
    availability_lag_sessions: int
    uses_same_row_price: bool
    crosses_seed_or_entity_boundary: bool
    algorithmic_audits: tuple[str, ...]
    inherited_source_audits_required: tuple[str, ...]
    execution_status: str
    design_sha256: str = STRUCTURAL_DESIGN_SHA256

    def __post_init__(self) -> None:
        if self.column_name != lag1_feature_name(self.source_column):
            raise StructuralContractError("Track-A derived feature name/source differs")
        if self.source_block not in {"market_current", "regime_current"}:
            raise StructuralContractError("Track-A derived source block is invalid")
        if (
            self.transform != "groupwise_shift_one_session"
            or self.group_keys != ("seed", "entity_id")
            or self.sort_key != "date"
            or self.availability_lag_sessions != 1
            or self.uses_same_row_price
            or self.crosses_seed_or_entity_boundary
        ):
            raise StructuralContractError("Track-A derived semantics differ from the design")
        if self.execution_status != "DATA_BOUND_SIX_AUDITS_PASS_5_OF_5":
            raise StructuralContractError("Track-A derived lifecycle status changed")

    def as_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["group_keys"] = list(self.group_keys)
        output["algorithmic_audits"] = list(self.algorithmic_audits)
        output["inherited_source_audits_required"] = list(self.inherited_source_audits_required)
        return output


def track_a_derived_feature_definitions() -> tuple[TrackADerivedFeatureDefinition, ...]:
    definitions: list[TrackADerivedFeatureDefinition] = []
    for block, columns in (
        ("market_current", MARKET_CURRENT),
        ("regime_current", REGIME_CURRENT),
    ):
        for source in columns:
            definitions.append(
                TrackADerivedFeatureDefinition(
                    feature_id=f"structural.track_a.lag1.{source}",
                    column_name=lag1_feature_name(source),
                    source_column=source,
                    source_block=block,
                    transform="groupwise_shift_one_session",
                    group_keys=("seed", "entity_id"),
                    sort_key="date",
                    availability_lag_sessions=1,
                    uses_same_row_price=False,
                    crosses_seed_or_entity_boundary=False,
                    algorithmic_audits=(
                        "prefix_invariance",
                        "future_intervention",
                        "same_row_source_lag",
                        "same_row_target_ignored",
                        "group_boundary_isolation",
                    ),
                    inherited_source_audits_required=(
                        "pit_availability",
                        "publication_date",
                        "restatement_availability",
                    ),
                    execution_status="DATA_BOUND_SIX_AUDITS_PASS_5_OF_5",
                )
            )
    output = tuple(definitions)
    if len(output) != 27 or len({item.feature_id for item in output}) != 27:
        raise StructuralContractError("Track-A derived registry must contain exactly 27 features")
    return output
