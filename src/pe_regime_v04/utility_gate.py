from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .utils import geometric_blend


def _positive_finite(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric) & numeric.gt(0.0))


def _finite_config_float(value: Any, *, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if np.isfinite(numeric) else float(default)


def _shifted_paired_losses(
    target_log: pd.Series,
    first_log: pd.Series,
    second_log: pd.Series,
    *,
    shift: int,
    window: int,
    min_history: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return losses computed on exactly the same mature OOS observations."""

    paired = target_log.notna() & first_log.notna() & second_log.notna()
    shifted_pair = paired.astype(float).shift(shift, fill_value=0.0)
    paired_count = shifted_pair.rolling(window, min_periods=1).sum().astype("int64")
    mature = paired_count.ge(min_history)

    first_error = (target_log - first_log).abs().where(paired)
    second_error = (target_log - second_log).abs().where(paired)
    first_loss = (
        first_error.shift(shift).rolling(window, min_periods=1).mean().where(mature)
    )
    second_loss = (
        second_error.shift(shift).rolling(window, min_periods=1).mean().where(mature)
    )
    return first_loss, second_loss, paired_count


def shifted_candidate_gate(
    target: pd.Series,
    baseline: pd.Series,
    challenger: pd.Series,
    config: Mapping[str, Any],
    *,
    availability_shift: int = 1,
    prefix: str,
) -> pd.DataFrame:
    """Causal candidate gate.

    Current-row candidate errors are never used for the current-row weight.  Error is
    shifted by ``availability_shift`` before rolling aggregation.
    """
    target_value = _positive_finite(target)
    baseline_value = _positive_finite(baseline)
    challenger_value = _positive_finite(challenger)
    target_log = np.log(target_value)
    baseline_log = np.log(baseline_value)
    challenger_log = np.log(challenger_value)
    shift = max(int(availability_shift), 1)
    window = max(int(config.get("loss_window", 252)), 1)
    min_history = max(int(config.get("min_history", 63)), 1)
    baseline_loss, challenger_loss, paired_count = _shifted_paired_losses(
        target_log,
        baseline_log,
        challenger_log,
        shift=shift,
        window=window,
        min_history=min_history,
    )
    gain = (baseline_loss - challenger_loss).where(
        np.isfinite(baseline_loss) & np.isfinite(challenger_loss)
    )
    margin = max(
        _finite_config_float(config.get("improvement_margin", 0.0), default=0.0),
        0.0,
    )
    temperature = max(
        _finite_config_float(config.get("temperature", 0.01), default=0.01),
        1e-6,
    )
    max_weight = float(
        np.clip(
            _finite_config_float(config.get("max_challenger_weight", 1.0), default=0.0),
            0.0,
            1.0,
        )
    )
    mature = baseline_loss.notna() & challenger_loss.notna()
    current_pair = baseline_value.notna() & challenger_value.notna()
    has_evidence = mature & gain.gt(margin) & current_pair

    # This ramp is exactly zero at the promotion margin.  Unlike a raw sigmoid,
    # it cannot give an unaccepted or historically inferior challenger any weight.
    excess_gain = (gain - margin).where(has_evidence, 0.0).clip(lower=0.0)
    soft_weight = max_weight * (-np.expm1(-excess_gain / temperature))
    weight = pd.Series(soft_weight, index=target.index, dtype=float).where(
        has_evidence, 0.0
    )
    decision_mode = str(config.get("decision_mode", "soft")).lower()
    if decision_mode == "winner_take_most":
        decisive = max(
            _finite_config_float(config.get("decisive_margin", margin), default=margin),
            margin,
        )
        weight.loc[has_evidence & gain.ge(decisive)] = max_weight
    weight = weight.where(np.isfinite(weight), 0.0).clip(0.0, max_weight)
    accepted = weight.gt(0.0)

    # Availability fallback is not evidence-based promotion.  Keep its weight and
    # accepted flag at zero/False and expose it independently for downstream audit.
    fallback_used = baseline_value.isna() & challenger_value.notna()
    blended = geometric_blend(baseline_value, challenger_value, weight)
    blended.loc[baseline_value.notna() & weight.eq(0.0)] = baseline_value
    return pd.DataFrame(
        {
            f"{prefix}_baseline_oos_log_mae": baseline_loss,
            f"{prefix}_challenger_oos_log_mae": challenger_loss,
            f"{prefix}_paired_oos_count": paired_count,
            f"{prefix}_oos_gain": gain,
            f"{prefix}_challenger_weight": weight,
            f"{prefix}_accepted": accepted,
            f"{prefix}_fallback_used": fallback_used,
            f"{prefix}_blended": blended,
        },
        index=target.index,
    )


def shifted_two_candidate_prior_blend(
    target: pd.Series,
    statistical: pd.Series,
    machine_learning: pd.Series,
    config: Mapping[str, Any],
    *,
    prefix: str,
) -> pd.DataFrame:
    target_value = _positive_finite(target)
    stat_value = _positive_finite(statistical)
    ml_value = _positive_finite(machine_learning)
    target_log = np.log(target_value)
    stat_log = np.log(stat_value)
    ml_log = np.log(ml_value)
    window = max(int(config.get("loss_window", 252)), 1)
    min_history = max(int(config.get("min_history", 63)), 1)
    temperature = max(
        _finite_config_float(config.get("temperature", 0.012), default=0.012),
        1e-6,
    )
    stat_prior = max(
        _finite_config_float(config.get("stat_prior", 0.10), default=0.10),
        1e-8,
    )
    ml_prior = max(
        _finite_config_float(config.get("ml_prior", 0.90), default=0.90),
        1e-8,
    )
    stat_loss, ml_loss, paired_count = _shifted_paired_losses(
        target_log,
        stat_log,
        ml_log,
        shift=1,
        window=window,
        min_history=min_history,
    )
    stat_score = np.log(stat_prior) - stat_loss / temperature
    ml_score = np.log(ml_prior) - ml_loss / temperature
    max_score = pd.concat([stat_score, ml_score], axis=1).max(axis=1)
    stat_exp = np.exp(stat_score - max_score)
    ml_exp = np.exp(ml_score - max_score)
    denominator = stat_exp + ml_exp
    adaptive_stat_weight = stat_exp / denominator
    adaptive_ml_weight = ml_exp / denominator
    immature = stat_loss.isna() | ml_loss.isna()
    prior_scale = max(stat_prior, ml_prior)
    scaled_stat_prior = stat_prior / prior_scale
    scaled_ml_prior = ml_prior / prior_scale
    scaled_prior_total = scaled_stat_prior + scaled_ml_prior
    stat_prior_weight = scaled_stat_prior / scaled_prior_total
    ml_prior_weight = scaled_ml_prior / scaled_prior_total
    adaptive_stat_weight.loc[immature] = stat_prior_weight
    adaptive_ml_weight.loc[immature] = ml_prior_weight

    stat_valid = stat_value.notna()
    ml_valid = ml_value.notna()
    both = stat_valid & ml_valid
    stat_weight = pd.Series(np.nan, index=target.index, dtype=float)
    ml_weight = pd.Series(np.nan, index=target.index, dtype=float)
    stat_weight.loc[both] = adaptive_stat_weight.loc[both]
    ml_weight.loc[both] = adaptive_ml_weight.loc[both]
    stat_only = stat_valid & ~ml_valid
    ml_only = ml_valid & ~stat_valid
    stat_weight.loc[stat_only] = 1.0
    ml_weight.loc[stat_only] = 0.0
    stat_weight.loc[ml_only] = 0.0
    ml_weight.loc[ml_only] = 1.0
    invalid_weights = ~np.isfinite(stat_weight) | ~np.isfinite(ml_weight)
    stat_weight.loc[both & invalid_weights] = stat_prior_weight
    ml_weight.loc[both & invalid_weights] = ml_prior_weight

    output = pd.Series(np.nan, index=target.index, dtype=float)
    output.loc[both] = np.exp(
        stat_weight.loc[both] * stat_log.loc[both]
        + ml_weight.loc[both] * ml_log.loc[both]
    )
    output.loc[stat_only] = stat_value.loc[stat_only]
    output.loc[ml_only] = ml_value.loc[ml_only]
    fallback_used = stat_only | ml_only
    return pd.DataFrame(
        {
            f"{prefix}_stat_oos_log_mae": stat_loss,
            f"{prefix}_ml_oos_log_mae": ml_loss,
            f"{prefix}_paired_oos_count": paired_count,
            f"{prefix}_stat_weight": stat_weight,
            f"{prefix}_ml_weight": ml_weight,
            f"{prefix}_fallback_used": fallback_used,
            f"{prefix}_expected_pe": output,
        },
        index=target.index,
    )
