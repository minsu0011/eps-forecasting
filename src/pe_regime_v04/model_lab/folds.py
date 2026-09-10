"""Shared chronological point-in-time fold generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from .contracts import ContractError


@dataclass(frozen=True)
class PITFoldSpec:
    min_train_sessions: int
    test_sessions: int
    step_sessions: int
    embargo_sessions: int = 0
    max_train_sessions: int | None = None
    allow_partial_final_test: bool = False

    def __post_init__(self) -> None:
        integer_fields = {
            "min_train_sessions": self.min_train_sessions,
            "test_sessions": self.test_sessions,
            "step_sessions": self.step_sessions,
            "embargo_sessions": self.embargo_sessions,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise ContractError(f"{name} must be an integer")
        if self.min_train_sessions < 1 or self.test_sessions < 1 or self.step_sessions < 1:
            raise ContractError("train, test, and step sessions must be positive")
        if self.step_sessions < self.test_sessions:
            raise ContractError("step_sessions must prevent overlapping OOS test windows")
        if self.embargo_sessions < 0:
            raise ContractError("embargo_sessions must be non-negative")
        if not isinstance(self.allow_partial_final_test, bool):
            raise ContractError("allow_partial_final_test must be a boolean")
        if self.max_train_sessions is not None:
            if (
                not isinstance(self.max_train_sessions, int)
                or isinstance(self.max_train_sessions, bool)
                or self.max_train_sessions < self.min_train_sessions
            ):
                raise ContractError("max_train_sessions must be at least min_train_sessions")


@dataclass(frozen=True)
class PITFold:
    fold_id: str
    train_positions: tuple[int, ...]
    test_positions: tuple[int, ...]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    label_information_cutoff: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.train_positions or not self.test_positions:
            raise ContractError("PIT folds require non-empty train and test positions")
        if self.train_end >= self.test_start:
            raise ContractError("training decisions must precede the test window")
        if self.label_information_cutoff >= self.test_start:
            raise ContractError("training label information must precede the test window")


def generate_pit_folds(
    frame: pd.DataFrame,
    spec: PITFoldSpec,
    *,
    date_column: str = "date",
    label_available_at_column: str | None = None,
) -> tuple[PITFold, ...]:
    """Create deterministic folds on unique sessions, never row offsets alone."""

    if date_column not in frame:
        raise ContractError(f"missing date column: {date_column}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ContractError("date column contains invalid timestamps")
    if dates.empty:
        raise ContractError("cannot build folds from an empty frame")
    if label_available_at_column is None:
        available = dates.copy()
    else:
        if label_available_at_column not in frame:
            raise ContractError(f"missing label availability column: {label_available_at_column}")
        available = pd.to_datetime(frame[label_available_at_column], errors="coerce")
        if available.isna().any():
            raise ContractError("label availability contains invalid timestamps")
        if (available < dates).any():
            raise ContractError("a label cannot be available before its decision row")

    sessions = pd.Index(dates.drop_duplicates().sort_values(kind="mergesort"))
    first_test_position = spec.min_train_sessions + spec.embargo_sessions
    folds: list[PITFold] = []
    candidate_start = first_test_position
    while candidate_start < len(sessions):
        test_sessions = sessions[candidate_start : candidate_start + spec.test_sessions]
        if len(test_sessions) < spec.test_sessions and not spec.allow_partial_final_test:
            break
        train_session_end_position = candidate_start - spec.embargo_sessions - 1
        train_session_start_position = 0
        if spec.max_train_sessions is not None:
            train_session_start_position = max(
                0, train_session_end_position - spec.max_train_sessions + 1
            )
        train_sessions = sessions[train_session_start_position : train_session_end_position + 1]
        information_cutoff = pd.Timestamp(train_sessions[-1])
        train_mask = (
            dates.isin(train_sessions)
            & (available <= information_cutoff)
            & (available < pd.Timestamp(test_sessions[0]))
        )
        test_mask = dates.isin(test_sessions)
        eligible_train_sessions = dates.loc[train_mask].nunique()
        if eligible_train_sessions >= spec.min_train_sessions and test_mask.any():
            folds.append(
                PITFold(
                    fold_id=f"fold_{len(folds):03d}",
                    train_positions=tuple(
                        int(position) for position in train_mask.to_numpy().nonzero()[0]
                    ),
                    test_positions=tuple(
                        int(position) for position in test_mask.to_numpy().nonzero()[0]
                    ),
                    train_start=pd.Timestamp(dates.loc[train_mask].min()),
                    train_end=pd.Timestamp(dates.loc[train_mask].max()),
                    test_start=pd.Timestamp(test_sessions[0]),
                    test_end=pd.Timestamp(test_sessions[-1]),
                    label_information_cutoff=pd.Timestamp(available.loc[train_mask].max()),
                )
            )
        candidate_start += spec.step_sessions

    if not folds:
        raise ContractError("fold specification produced no PIT-safe folds")
    return tuple(folds)


def iter_fold_frames(
    frame: pd.DataFrame,
    folds: tuple[PITFold, ...],
) -> Iterator[tuple[PITFold, pd.DataFrame, pd.DataFrame]]:
    for fold in folds:
        yield (
            fold,
            frame.iloc[list(fold.train_positions)].copy(),
            frame.iloc[list(fold.test_positions)].copy(),
        )


def fold_manifest(folds: tuple[PITFold, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "label_information_cutoff": fold.label_information_cutoff,
                "train_rows": len(fold.train_positions),
                "test_rows": len(fold.test_positions),
            }
            for fold in folds
        ]
    )
