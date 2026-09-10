"""Canonical OOS long/wide matrices with mandatory CSV output."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .contracts import EVALUATION_ONLY_TRUTH_COLUMN, ContractError
from .dataset import DEFAULT_IDENTITY_COLUMNS


ParquetMode = Literal["off", "required"]


class ParquetUnavailableError(RuntimeError):
    """Raised before any output is written when required parquet is unavailable."""


@dataclass(frozen=True)
class OOSMatrices:
    long: pd.DataFrame
    wide: pd.DataFrame


@dataclass(frozen=True)
class MatrixArtifacts:
    long_csv: Path
    wide_csv: Path
    long_csv_sha256: str
    wide_csv_sha256: str
    long_parquet: Path | None
    wide_parquet: Path | None


def canonicalize_oos_matrices(
    prediction_frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
) -> OOSMatrices:
    keys = tuple(identity_columns)
    required = {*keys, "model_id", "prediction", EVALUATION_ONLY_TRUTH_COLUMN}
    missing = sorted(required.difference(prediction_frame.columns))
    if missing:
        raise ContractError(f"OOS frame is missing columns: {missing}")
    if prediction_frame.duplicated([*keys, "model_id"]).any():
        raise ContractError("OOS predictions must be unique per identity and model")
    frame = prediction_frame.copy()
    if "date" in keys:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ContractError("OOS dates must be valid")
    prediction = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    truth = pd.to_numeric(frame[EVALUATION_ONLY_TRUTH_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    frame["prediction"] = prediction
    frame[EVALUATION_ONLY_TRUTH_COLUMN] = truth
    truth_counts = frame.groupby(list(keys), sort=False, dropna=False)[
        EVALUATION_ONLY_TRUTH_COLUMN
    ].nunique(dropna=False)
    if (truth_counts != 1).any():
        raise ContractError("true_fair_pe must be identical across models for each identity")

    optional = [column for column in ("fold_id",) if column in frame]
    for column in optional:
        metadata_counts = frame.groupby(list(keys), sort=False, dropna=False)[column].nunique(
            dropna=False
        )
        if (metadata_counts != 1).any():
            raise ContractError(f"{column} must be identical across models for each identity")
    long_columns = [*keys, *optional, "model_id", "prediction", EVALUATION_ONLY_TRUTH_COLUMN]
    long = (
        frame.loc[:, long_columns]
        .sort_values([*keys, "model_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    truth_frame = long.loc[:, [*keys, *optional, EVALUATION_ONLY_TRUTH_COLUMN]].drop_duplicates(
        keys
    )
    prediction_wide = long.pivot(index=list(keys), columns="model_id", values="prediction")
    prediction_wide = prediction_wide.sort_index(axis=1)
    prediction_wide.columns = [f"prediction__{column}" for column in prediction_wide.columns]
    prediction_wide = prediction_wide.reset_index()
    wide = (
        truth_frame.merge(
            prediction_wide,
            how="outer",
            on=list(keys),
            validate="one_to_one",
            sort=True,
        )
        .sort_values(list(keys), kind="mergesort")
        .reset_index(drop=True)
    )
    ordered_predictions = sorted(
        column for column in wide.columns if column.startswith("prediction__")
    )
    wide = wide.loc[:, [*keys, *optional, EVALUATION_ONLY_TRUTH_COLUMN, *ordered_predictions]]
    return OOSMatrices(long=long, wide=wide)


def _resolve_parquet_engine(engine: str | None) -> str:
    if engine is not None:
        if engine not in {"pyarrow", "fastparquet"}:
            raise ParquetUnavailableError(f"unsupported parquet engine: {engine!r}")
        if importlib.util.find_spec(engine) is None:
            raise ParquetUnavailableError(f"required parquet engine {engine!r} is unavailable")
        return engine
    for candidate in ("pyarrow", "fastparquet"):
        if importlib.util.find_spec(candidate) is not None:
            return candidate
    raise ParquetUnavailableError("parquet requested but pyarrow and fastparquet are unavailable")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def write_oos_matrices(
    prediction_frame: pd.DataFrame,
    output_directory: Path,
    *,
    stem: str = "oos_predictions",
    identity_columns: Sequence[str] = DEFAULT_IDENTITY_COLUMNS,
    parquet: ParquetMode = "off",
    parquet_engine: str | None = None,
) -> MatrixArtifacts:
    if not stem or any(character in stem for character in "\\/:"):
        raise ContractError("stem must be a safe non-empty filename component")
    if parquet not in {"off", "required"}:
        raise ContractError(f"unsupported parquet mode: {parquet!r}")
    resolved_engine = _resolve_parquet_engine(parquet_engine) if parquet == "required" else None
    matrices = canonicalize_oos_matrices(prediction_frame, identity_columns=identity_columns)

    output_directory = Path(output_directory)
    long_csv = output_directory / f"{stem}.long.csv"
    wide_csv = output_directory / f"{stem}.wide.csv"
    long_parquet = output_directory / f"{stem}.long.parquet" if resolved_engine else None
    wide_parquet = output_directory / f"{stem}.wide.parquet" if resolved_engine else None
    finals = [long_csv, wide_csv]
    if long_parquet is not None and wide_parquet is not None:
        finals.extend([long_parquet, wide_parquet])
    if any(path.exists() for path in finals):
        raise FileExistsError("OOS artifacts are immutable; choose a fresh output stem")

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_paths = [_temporary(path) for path in finals]
    if any(path.exists() for path in temporary_paths):
        raise FileExistsError("an OOS temporary artifact already exists")
    try:
        matrices.long.to_csv(temporary_paths[0], index=False, lineterminator="\n")
        matrices.wide.to_csv(temporary_paths[1], index=False, lineterminator="\n")
        if resolved_engine is not None:
            matrices.long.to_parquet(temporary_paths[2], index=False, engine=resolved_engine)
            matrices.wide.to_parquet(temporary_paths[3], index=False, engine=resolved_engine)
        for temporary, final in zip(temporary_paths, finals):
            os.replace(temporary, final)
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()

    return MatrixArtifacts(
        long_csv=long_csv,
        wide_csv=wide_csv,
        long_csv_sha256=_sha256(long_csv),
        wide_csv_sha256=_sha256(wide_csv),
        long_parquet=long_parquet,
        wide_parquet=wide_parquet,
    )
