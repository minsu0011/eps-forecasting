"""Natural-coverage and exact-identity guards; no evaluation metrics live here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .contracts import (
    StructuralContractError,
    identity_sha256,
    normalized_identity_frame,
    require_no_evaluation_truth,
)


@dataclass(frozen=True)
class NaturalCoverageAudit:
    required_rows: int
    supplied_rows: int
    unique_supplied_rows: int
    positive_finite_rows: int
    missing_identity_rows: int
    extra_identity_rows: int
    duplicate_identity_rows: int
    required_identity_sha256: str
    supplied_identity_sha256: str | None

    @property
    def natural_coverage(self) -> float:
        if self.required_rows == 0:
            return 0.0
        return self.positive_finite_rows / self.required_rows

    @property
    def exact_full_coverage(self) -> bool:
        return (
            self.required_rows > 0
            and self.supplied_rows == self.required_rows
            and self.unique_supplied_rows == self.required_rows
            and self.positive_finite_rows == self.required_rows
            and self.missing_identity_rows == 0
            and self.extra_identity_rows == 0
            and self.duplicate_identity_rows == 0
            and self.required_identity_sha256 == self.supplied_identity_sha256
        )


def audit_natural_coverage(
    required_identity: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    prediction_column: str = "expected_pe",
) -> NaturalCoverageAudit:
    require_no_evaluation_truth(required_identity.columns, context="required identity")
    require_no_evaluation_truth(candidate.columns, context="candidate natural coverage")
    required = normalized_identity_frame(
        required_identity,
        columns=identity_columns,
        context="required identity",
        sort=False,
    )
    missing_columns = [column for column in identity_columns if column not in candidate]
    if prediction_column not in candidate:
        missing_columns.append(prediction_column)
    if missing_columns:
        raise StructuralContractError(
            f"candidate coverage frame is missing columns: {missing_columns}"
        )
    supplied_raw = candidate.loc[:, list(identity_columns)].copy()
    if "date" in supplied_raw:
        supplied_raw["date"] = (
            pd.to_datetime(supplied_raw["date"], errors="coerce", utc=True)
            .dt.tz_convert(None)
            .to_numpy(copy=True)
        )
    if supplied_raw.isna().any().any():
        raise StructuralContractError("candidate identity contains missing/invalid values")
    duplicate_count = int(supplied_raw.duplicated(list(identity_columns)).sum())
    supplied_unique = supplied_raw.drop_duplicates(list(identity_columns), keep="first")
    required_keys = {
        tuple(row)
        for row in required.loc[:, list(identity_columns)].itertuples(index=False, name=None)
    }
    supplied_keys = {
        tuple(row)
        for row in supplied_unique.loc[:, list(identity_columns)].itertuples(index=False, name=None)
    }
    values = pd.to_numeric(candidate[prediction_column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    positive_count = int((np.isfinite(values) & (values > 0.0)).sum())
    supplied_hash: str | None = None
    if duplicate_count == 0:
        try:
            supplied_hash = identity_sha256(candidate, columns=identity_columns)
        except StructuralContractError:
            supplied_hash = None
    return NaturalCoverageAudit(
        required_rows=len(required),
        supplied_rows=len(candidate),
        unique_supplied_rows=len(supplied_unique),
        positive_finite_rows=positive_count,
        missing_identity_rows=len(required_keys.difference(supplied_keys)),
        extra_identity_rows=len(supplied_keys.difference(required_keys)),
        duplicate_identity_rows=duplicate_count,
        required_identity_sha256=identity_sha256(required, columns=identity_columns),
        supplied_identity_sha256=supplied_hash,
    )


def require_full_natural_coverage(
    required_identity: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    identity_columns: Sequence[str],
    prediction_column: str = "expected_pe",
) -> NaturalCoverageAudit:
    """Fail instead of shrinking a common mask or filling a missing prediction."""

    audit = audit_natural_coverage(
        required_identity,
        candidate,
        identity_columns=identity_columns,
        prediction_column=prediction_column,
    )
    if not audit.exact_full_coverage:
        raise StructuralContractError(
            "candidate is BROKEN: exact natural coverage is below 1.0 or identity/order differs"
        )
    required_order = normalized_identity_frame(
        required_identity,
        columns=identity_columns,
        context="required identity order",
        sort=False,
    )
    candidate_order = normalized_identity_frame(
        candidate,
        columns=identity_columns,
        context="candidate identity order",
        sort=False,
    )
    if not required_order.equals(candidate_order):
        raise StructuralContractError("candidate is BROKEN: identity row order differs")
    return audit
