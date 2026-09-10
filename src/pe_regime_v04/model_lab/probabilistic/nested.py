"""Locked chronological outer/inner fold infrastructure."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..contracts import ContractError
from ..folds import PITFold, PITFoldSpec, generate_pit_folds
from .contracts import ProbabilisticContractError, require_no_evaluation_truth


OUTER_FOLD_SPEC = PITFoldSpec(
    min_train_sessions=252,
    max_train_sessions=1008,
    test_sessions=21,
    step_sessions=21,
    embargo_sessions=0,
    allow_partial_final_test=True,
)
INNER_FOLD_SPEC = PITFoldSpec(
    min_train_sessions=252,
    max_train_sessions=1008,
    test_sessions=21,
    step_sessions=21,
    embargo_sessions=0,
    allow_partial_final_test=False,
)
FORMAL_SCORE_START_POSITION = 504
FORMAL_SESSIONS_PER_SEED = 1800
FORMAL_EVALUATION_ROWS_PER_SEED = 1296


@dataclass(frozen=True)
class NestedFoldPlan:
    outer: PITFold
    inner: tuple[PITFold, ...]

    def __post_init__(self) -> None:
        for fold in self.inner:
            if fold.test_end >= self.outer.test_start:
                raise ProbabilisticContractError(
                    "every inner test must end strictly before the outer test starts"
                )


def build_outer_folds(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    label_available_at_column: str | None = None,
    require_formal_shape: bool = False,
) -> tuple[PITFold, ...]:
    require_no_evaluation_truth(frame.columns, context="fold scheduler")
    prepared = frame.copy()
    prepared[date_column] = pd.to_datetime(
        prepared[date_column], errors="coerce", utc=True
    ).dt.tz_convert(None)
    if label_available_at_column is not None:
        prepared[label_available_at_column] = pd.to_datetime(
            prepared[label_available_at_column], errors="coerce", utc=True
        ).dt.tz_convert(None)
    if require_formal_shape:
        sessions = prepared[date_column].nunique()
        if len(prepared) != FORMAL_SESSIONS_PER_SEED or sessions != FORMAL_SESSIONS_PER_SEED:
            raise ProbabilisticContractError("formal input must be one row per 1,800 sessions")
    folds = generate_pit_folds(
        prepared,
        OUTER_FOLD_SPEC,
        date_column=date_column,
        label_available_at_column=label_available_at_column,
    )
    _verify_fold_chronology(folds)
    return folds


def build_inner_folds(
    frame: pd.DataFrame,
    outer: PITFold,
    *,
    date_column: str = "date",
    label_available_at_column: str | None = None,
) -> NestedFoldPlan:
    """Build inner PIT folds solely inside an outer training prefix.

    Earliest outer blocks can legitimately have no inner validation fold. The
    fixed cheap screen does not use inner selection; callers requesting an
    inner-dependent operation must reject an empty plan.
    """

    require_no_evaluation_truth(frame.columns, context="nested fold scheduler")
    training = frame.iloc[list(outer.train_positions)].copy().reset_index(drop=True)
    training[date_column] = pd.to_datetime(
        training[date_column], errors="coerce", utc=True
    ).dt.tz_convert(None)
    if label_available_at_column is not None:
        training[label_available_at_column] = pd.to_datetime(
            training[label_available_at_column], errors="coerce", utc=True
        ).dt.tz_convert(None)
    try:
        inner = generate_pit_folds(
            training,
            INNER_FOLD_SPEC,
            date_column=date_column,
            label_available_at_column=label_available_at_column,
        )
    except ContractError as exc:
        if "produced no PIT-safe folds" not in str(exc):
            raise
        inner = ()
    _verify_fold_chronology(inner)
    return NestedFoldPlan(outer=outer, inner=inner)


def require_inner_folds(plan: NestedFoldPlan, *, use: str) -> tuple[PITFold, ...]:
    allowed = {
        "fit_derived_calibration",
        "residual_reference",
        "chronological_early_stopping",
        "formal_hyperparameter_choice",
    }
    if use not in allowed:
        raise ProbabilisticContractError("undeclared nested-fold use")
    if not plan.inner:
        raise ProbabilisticContractError(f"{use} requires at least one chronological inner fold")
    return plan.inner


def _verify_fold_chronology(folds: tuple[PITFold, ...]) -> None:
    previous_test_end: pd.Timestamp | None = None
    for fold in folds:
        if fold.train_end >= fold.test_start or fold.label_information_cutoff >= fold.test_start:
            raise ProbabilisticContractError("fold leaks future information")
        if previous_test_end is not None and fold.test_start <= previous_test_end:
            raise ProbabilisticContractError("out-of-sample fold windows overlap")
        previous_test_end = fold.test_end
