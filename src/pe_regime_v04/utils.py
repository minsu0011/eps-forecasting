from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REGIME_ORDER = ("BEAR", "SIDEWAYS", "BULL")
PROBABILITY_COLUMNS = ("p_bear", "p_sideways", "p_bull")


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str], *, context: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{context}: required columns missing: {missing}")


def safe_log(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan, dtype=float)
    valid = np.isfinite(array) & (array > 0)
    result[valid] = np.log(array[valid])
    return result


def normalize_probabilities(values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).copy()
    valid = np.isfinite(probabilities).all(axis=1)
    output = np.full_like(probabilities, np.nan)
    if not valid.any():
        return output
    clipped = np.clip(probabilities[valid], max(float(floor), 0.0), np.inf)
    row_sum = clipped.sum(axis=1, keepdims=True)
    output[valid] = np.divide(
        clipped,
        row_sum,
        out=np.full_like(clipped, np.nan),
        where=row_sum > 0,
    )
    return output


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = normalize_probabilities(probabilities)
    output = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p).all(axis=1)
    if valid.any():
        q = np.clip(p[valid], 1e-12, 1.0)
        output[valid] = -(q * np.log(q)).sum(axis=1) / math.log(q.shape[1])
    return output


def sigmoid(values: pd.Series | np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -40.0, 40.0)))


def geometric_blend(
    baseline: pd.Series,
    challenger: pd.Series,
    challenger_weight: pd.Series,
) -> pd.Series:
    base = pd.to_numeric(baseline, errors="coerce")
    challenge = pd.to_numeric(challenger, errors="coerce")
    weight = pd.to_numeric(challenger_weight, errors="coerce").clip(0.0, 1.0)
    output = pd.Series(np.nan, index=baseline.index, dtype=float)
    both = (base > 0) & (challenge > 0) & weight.notna()
    output.loc[both] = np.exp(
        (1.0 - weight.loc[both]) * np.log(base.loc[both])
        + weight.loc[both] * np.log(challenge.loc[both])
    )
    output.loc[(base > 0) & ~both] = base.loc[(base > 0) & ~both]
    output.loc[(challenge > 0) & base.isna()] = challenge.loc[(challenge > 0) & base.isna()]
    return output


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
