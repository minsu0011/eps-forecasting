from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss

from .utils import REGIME_ORDER, normalize_probabilities


def _numeric_float_series(values: pd.Series) -> pd.Series:
    """Coerce pandas nullable/object numeric data to plain float64 plus np.nan."""

    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
    return pd.Series(array, index=values.index, name=values.name, dtype=float)


def _numeric_float_frame(values: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: _numeric_float_series(values[column])
            for column in values.columns
        },
        index=values.index,
    )


def candidate_metrics(
    candidate: pd.Series,
    observed_pe: pd.Series,
    fair_pe: pd.Series | None = None,
) -> dict[str, Any]:
    prediction = _numeric_float_series(candidate)
    observed = _numeric_float_series(observed_pe)
    valid_observed = (
        np.isfinite(prediction)
        & np.isfinite(observed)
        & (prediction > 0)
        & (observed > 0)
    )
    payload: dict[str, Any] = {
        "eligible_rows": int(len(prediction)),
        "coverage_rows": int(valid_observed.sum()),
        "coverage": float(valid_observed.mean()),
        "observed_target_log_mae": None,
        "observed_target_log_rmse": None,
    }
    if valid_observed.any():
        error = np.log(prediction.loc[valid_observed] / observed.loc[valid_observed])
        payload["observed_target_log_mae"] = float(np.abs(error).mean())
        payload["observed_target_log_rmse"] = float(np.sqrt(np.mean(error * error)))
    if fair_pe is not None:
        fair = _numeric_float_series(fair_pe)
        valid_fair = (
            np.isfinite(prediction)
            & np.isfinite(fair)
            & (prediction > 0)
            & (fair > 0)
        )
        payload["fair_coverage_rows"] = int(valid_fair.sum())
        payload["fair_coverage"] = float(valid_fair.mean())
        if valid_fair.any():
            error = np.log(prediction.loc[valid_fair] / fair.loc[valid_fair])
            payload["fair_log_mae"] = float(np.abs(error).mean())
            payload["fair_log_rmse"] = float(np.sqrt(np.mean(error * error)))
        else:
            payload["fair_log_mae"] = None
            payload["fair_log_rmse"] = None
    return payload


def paired_candidate_metrics(
    baseline: pd.Series,
    challenger: pd.Series,
    observed_pe: pd.Series,
    fair_pe: pd.Series | None = None,
) -> dict[str, Any]:
    """Compare candidates only where target and both predictions coexist."""

    base = _numeric_float_series(baseline)
    challenge = _numeric_float_series(challenger)
    observed = _numeric_float_series(observed_pe)
    common = (
        np.isfinite(base)
        & np.isfinite(challenge)
        & np.isfinite(observed)
        & (base > 0)
        & (challenge > 0)
        & (observed > 0)
    )
    payload: dict[str, Any] = {
        "eligible_rows": int(len(base)),
        "paired_observed_rows": int(common.sum()),
        "paired_observed_coverage": float(common.mean()),
        "baseline_observed_log_mae": None,
        "challenger_observed_log_mae": None,
        "challenger_observed_log_mae_gain": None,
    }
    if common.any():
        base_error = np.abs(np.log(base.loc[common] / observed.loc[common]))
        challenge_error = np.abs(
            np.log(challenge.loc[common] / observed.loc[common])
        )
        base_mae = float(base_error.mean())
        challenge_mae = float(challenge_error.mean())
        payload.update(
            {
                "baseline_observed_log_mae": base_mae,
                "challenger_observed_log_mae": challenge_mae,
                "challenger_observed_log_mae_gain": base_mae - challenge_mae,
            }
        )
    if fair_pe is not None:
        fair = _numeric_float_series(fair_pe)
        fair_common = (
            np.isfinite(base)
            & np.isfinite(challenge)
            & np.isfinite(fair)
            & (base > 0)
            & (challenge > 0)
            & (fair > 0)
        )
        payload["paired_fair_rows"] = int(fair_common.sum())
        payload["paired_fair_coverage"] = float(fair_common.mean())
        payload["baseline_fair_log_mae"] = None
        payload["challenger_fair_log_mae"] = None
        payload["challenger_fair_log_mae_gain"] = None
        if fair_common.any():
            base_error = np.abs(np.log(base.loc[fair_common] / fair.loc[fair_common]))
            challenge_error = np.abs(
                np.log(challenge.loc[fair_common] / fair.loc[fair_common])
            )
            base_mae = float(base_error.mean())
            challenge_mae = float(challenge_error.mean())
            payload.update(
                {
                    "baseline_fair_log_mae": base_mae,
                    "challenger_fair_log_mae": challenge_mae,
                    "challenger_fair_log_mae_gain": base_mae - challenge_mae,
                }
            )
    return payload


def regime_metrics(
    probabilities: pd.DataFrame,
    true_regime: pd.Series,
) -> dict[str, Any]:
    mapping = {name: index for index, name in enumerate(REGIME_ORDER)}
    truth = true_regime.map(mapping)
    p = _numeric_float_frame(probabilities).to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    finite = np.isfinite(p).all(axis=1)
    in_range = ((p >= 0.0) & (p <= 1.0)).all(axis=1)
    positive_sum = np.sum(np.where(finite[:, None], p, 0.0), axis=1) > 0.0
    valid = truth.notna().to_numpy() & finite & in_range & positive_sum
    if not valid.any():
        return {
            "evaluated_rows": 0,
            "invalid_probability_rows": int((~finite | ~in_range | ~positive_sum).sum()),
        }
    normalized = normalize_probabilities(p[valid])
    y = truth.to_numpy(dtype=np.float64, na_value=np.nan)[valid].astype(int)
    predicted = np.argmax(normalized, axis=1)
    confidence = np.max(normalized, axis=1)
    correct = (predicted == y).astype(float)
    recall = {}
    for class_index, class_name in enumerate(REGIME_ORDER):
        mask = y == class_index
        recall[class_name] = float(np.mean(predicted[mask] == class_index)) if mask.any() else None
    brier = float(
        np.mean(
            np.sum(
                (
                    normalized
                    - np.eye(3, dtype=float)[y]
                )
                ** 2,
                axis=1,
            )
        )
    )
    # Equal-width expected calibration error.  This is diagnostic only and does
    # not replace log-loss/Brier for promotion decisions.
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    matrix = confusion_matrix(y, predicted, labels=[0, 1, 2])
    return {
        "evaluated_rows": int(valid.sum()),
        "invalid_probability_rows": int((~finite | ~in_range | ~positive_sum).sum()),
        "coverage": float(valid.mean()),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "log_loss": float(log_loss(y, normalized, labels=[0, 1, 2])),
        "brier": brier,
        "ece_10_bin": float(ece),
        "mean_confidence": float(confidence.mean()),
        "confidence_accuracy_gap": float(confidence.mean() - correct.mean()),
        "class_recall": recall,
        "confusion_matrix": {
            true_name: {
                predicted_name: int(matrix[true_index, predicted_index])
                for predicted_index, predicted_name in enumerate(REGIME_ORDER)
            }
            for true_index, true_name in enumerate(REGIME_ORDER)
        },
    }


def _precision_recall_f1(
    true_mask: pd.Series | np.ndarray,
    predicted_mask: pd.Series | np.ndarray,
) -> dict[str, Any]:
    truth = np.asarray(true_mask, dtype=bool)
    predicted = np.asarray(predicted_mask, dtype=bool)
    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else None
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "support": int(truth.sum()),
        "predicted_rows": int(predicted.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def valuation_direction_metrics(
    observed_pe: pd.Series,
    expected_pe: pd.Series,
    fair_pe: pd.Series,
    *,
    threshold: float = 0.10,
) -> dict[str, Any]:
    observed = _numeric_float_series(observed_pe)
    expected = _numeric_float_series(expected_pe)
    fair = _numeric_float_series(fair_pe)
    valid = (
        np.isfinite(observed)
        & np.isfinite(expected)
        & np.isfinite(fair)
        & (observed > 0)
        & (expected > 0)
        & (fair > 0)
    )
    if not valid.any():
        return {"evaluated_rows": 0}
    true_gap = observed.loc[valid] / fair.loc[valid] - 1.0
    predicted_gap = observed.loc[valid] / expected.loc[valid] - 1.0
    correlation = None
    if len(true_gap) >= 2 and true_gap.nunique() > 1 and predicted_gap.nunique() > 1:
        candidate_correlation = float(true_gap.corr(predicted_gap))
        if math.isfinite(candidate_correlation):
            correlation = candidate_correlation
    direction_accuracy = float(np.mean(np.sign(true_gap) == np.sign(predicted_gap)))
    true_cheap = true_gap <= -threshold
    true_expensive = true_gap >= threshold
    predicted_cheap = predicted_gap <= -threshold
    predicted_expensive = predicted_gap >= threshold
    cheap = _precision_recall_f1(true_cheap, predicted_cheap)
    expensive = _precision_recall_f1(true_expensive, predicted_expensive)
    return {
        "evaluated_rows": int(valid.sum()),
        "gap_correlation": correlation,
        "gap_direction_accuracy": direction_accuracy,
        "threshold": float(threshold),
        "cheap": cheap,
        "expensive": expensive,
    }


def valuation_state_metrics(
    state: pd.Series,
    observed_pe: pd.Series,
    fair_pe: pd.Series,
    *,
    threshold: float = 0.10,
) -> dict[str, Any]:
    observed = _numeric_float_series(observed_pe)
    fair = _numeric_float_series(fair_pe)
    state_text = state.astype("string")
    signal_state = ~state_text.isin(["INSUFFICIENT_DATA", "NO_MEANINGFUL_PE"])
    valid = (
        np.isfinite(observed)
        & np.isfinite(fair)
        & (observed > 0)
        & (fair > 0)
        & state_text.notna()
        & signal_state
    )
    if not valid.any():
        return {
            "evaluated_rows": 0,
            "excluded_non_signal_rows": int((state_text.notna() & ~signal_state).sum()),
        }
    true_gap = observed.loc[valid] / fair.loc[valid] - 1.0
    state_valid = state_text.loc[valid]
    true_cheap = true_gap <= -float(threshold)
    true_expensive = true_gap >= float(threshold)
    predicted_cheap = state_valid.str.startswith("CHEAP", na=False)
    predicted_expensive = state_valid.str.startswith("EXPENSIVE", na=False)
    return {
        "evaluated_rows": int(valid.sum()),
        "excluded_non_signal_rows": int((state_text.notna() & ~signal_state).sum()),
        "threshold": float(threshold),
        "state_counts": {
            str(key): int(value)
            for key, value in state_valid.value_counts(dropna=False).to_dict().items()
        },
        "cheap": _precision_recall_f1(true_cheap, predicted_cheap),
        "expensive": _precision_recall_f1(true_expensive, predicted_expensive),
    }
