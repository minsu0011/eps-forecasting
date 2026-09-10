from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import yaml

from .matured_proxy_gate import LOCKED_MATURED_PROXY_GATE_CONFIG

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "random_seed": 42,
        "allow_existing_v04_columns": False,
    },
    "regime_stacker": {
        "enabled": True,
        "horizon": 21,
        "bull_return_threshold": 0.03,
        "bear_return_threshold": -0.03,
        "train_window": 1008,
        "min_train": 252,
        "refit_every": 21,
        "n_estimators": 100,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_child_samples": 25,
        "subsample": 0.90,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_lambda": 1.0,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "probability_floor": 0.01,
        "class_weight": None,
        "outer_n_jobs": 16,
        "parallel_backend": "thread",
        "n_jobs": 1,
    },
    "statistics": {
        "global_lookback": 756,
        "global_min_history": 126,
        "global_method": "median",
    },
    "expected_pe": {
        "train_no_regime": True,
        # The matched pair is intentionally trained through the same function,
        # folds and hyper-parameters.  This is the only defensible way to measure
        # the incremental value of regime features.
        "train_with_regime": True,
        # The new forward-return stacker remains an explicit challenger until it
        # wins locked multi-seed tests against the current/v0.3 regime block.
        "include_return_forecast_features": False,
        "reuse_v03_ml_with_regime": True,
        "train_window": 1890,
        "min_train": 378,
        "refit_every": 1,
        "n_estimators": 200,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": -1,
        "min_child_samples": 15,
        "subsample": 0.90,
        "subsample_freq": 1,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.0,
        "reg_lambda": 0.2,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "fallback_max_iter": 180,
        "fallback_learning_rate": 0.04,
        "fallback_max_leaf_nodes": 15,
        "outer_n_jobs": 32,
        "parallel_backend": "thread",
        "n_jobs": 1,
    },
    "no_regime_shrinkage": {
        # Fixed, causal challenger blend.  This is published for evaluation only
        # and does not feed the production v04_expected_pe path.
        "weight": 0.50,
    },
    "market_conditioned_diagnostic": {
        # Exploration-locked kalman_qr0p010_clip3__blend025.  This consumes the
        # same-row observed P/E and must remain disconnected from production.
        "enabled": True,
        "q_over_r": 0.01,
        "clip_sigma": 3.0,
        "scale_window": 252,
        "scale_min_history": 126,
        "scale_floor": 0.01,
        "latent_weight": 0.25,
    },
    "fundamental_vintage": {
        # Price-free-at-prediction challenger. Historical observed P/E is consumed
        # only after an EPS information vintage has completely ended. Default-off;
        # the locked research candidate must opt in through its explicit override.
        "enabled": False,
        "min_completed_target_vintages": 4,
        "min_usable_vintages": 8,
        "max_train_vintages": 20,
        "quantile": 0.50,
        "alpha": 0.05,
        "solver": "highs",
        "growth_winsor_lower": 0.05,
        "growth_winsor_upper": 0.95,
        "growth_iqr_floor": 0.01,
        "target_clip_lower": 0.10,
        "target_clip_upper": 0.90,
        "confidence_floor": 25.0,
        "confidence_ceiling": 100.0,
        "approximation_weight": 0.75,
    },
    "lagged_market_conditioned": {
        # Default-off research candidate. The local-level state is emitted before
        # the row observation update, then routed by a dedicated shifted loss gate.
        # Its row-t incumbent can contain price-derived features, so this is EOD-only.
        "enabled": False,
        "q_over_r": 0.01,
        "clip_sigma": 3.0,
        "scale_window": 252,
        "scale_min_history": 126,
        "scale_floor": 0.01,
        "latent_weight": 0.25,
        "incumbent_weight": 0.75,
        "loss_window": 252,
        "min_history": 63,
        "improvement_margin": 0.001,
        "decisive_margin": 0.002,
        "temperature": 0.004,
        "max_challenger_weight": 1.0,
        "decision_mode": "winner_take_most",
        "availability_shift": 1,
    },
    "ml_incumbent_weekly_median_shrinkage": {
        # Default-off, fixed retrospective winner. This uses only the code-owned
        # row-t v0.3 ML incumbent and its four preceding adjacent input rows.
        # Because the current incumbent is included, publication is EOD-only.
        "enabled": False,
        "window_sessions": 5,
        "incumbent_weight": 0.75,
        "anchor_weight": 0.25,
        "method": "log_median",
    },
    "matured_proxy_gate": {
        # Default-off parallel EOD research surface.  The routing target is a
        # fully matured 21-session forward median of observed log P/E and the
        # raw v0.3 ML estimate remains the mandatory safety anchor.
        "enabled": False,
        **LOCKED_MATURED_PROXY_GATE_CONFIG,
    },
    "statistical_gate": {
        "loss_window": 252,
        "min_history": 63,
        "improvement_margin": 0.0,
        "decisive_margin": 0.002,
        "temperature": 0.004,
        "max_challenger_weight": 1.0,
        "decision_mode": "winner_take_most",
    },
    "ml_no_regime_gate": {
        "loss_window": 252,
        "min_history": 63,
        "improvement_margin": 0.001,
        "decisive_margin": 0.002,
        "temperature": 0.004,
        "max_challenger_weight": 1.0,
        "decision_mode": "winner_take_most",
    },
    "ml_matched_regime_gate": {
        "loss_window": 252,
        "min_history": 63,
        "improvement_margin": 0.001,
        "decisive_margin": 0.002,
        "temperature": 0.004,
        "max_challenger_weight": 1.0,
        "decision_mode": "winner_take_most",
    },
    "ml_incumbent_gate": {
        "loss_window": 252,
        "min_history": 63,
        "improvement_margin": 0.001,
        "decisive_margin": 0.002,
        "temperature": 0.004,
        "max_challenger_weight": 1.0,
        "decision_mode": "winner_take_most",
    },
    "final_blend": {
        "loss_window": 252,
        "min_history": 63,
        "temperature": 0.012,
        "stat_prior": 0.10,
        "ml_prior": 0.90,
    },
    # The dynamic blend is retained as a diagnostic.  Production promotion is
    # guarded separately so the weak statistical candidate cannot degrade the
    # stronger ML incumbent merely because it receives a non-zero prior.
    "final_guard": {
        "loss_window": 252,
        "min_history": 63,
        "improvement_margin": 0.001,
        "decisive_margin": 0.002,
        "temperature": 0.004,
        "max_challenger_weight": 1.0,
        "decision_mode": "winner_take_most",
    },
    "valuation": {
        "rolling_window": 756,
        "min_history": 126,
        "cheap_quantile": 0.15,
        "expensive_quantile": 0.85,
        "minimum_absolute_gap": 0.03,
        "confirmed_cheap_quantile": 0.10,
        "confirmed_expensive_quantile": 0.90,
        "confirmed_minimum_absolute_gap": 0.05,
        "confirmed_robust_z": 1.0,
        "absolute_cheap_gap": -0.10,
        "absolute_expensive_gap": 0.10,
    },
}


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _reject_unknown_keys(
    supplied: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    path: str = "",
) -> None:
    unknown = sorted(set(supplied) - set(reference))
    if unknown:
        location = path or "config"
        raise ValueError(f"{location}: unknown keys: {unknown}")
    for key, value in supplied.items():
        reference_value = reference[key]
        if isinstance(value, Mapping):
            if not isinstance(reference_value, Mapping):
                raise ValueError(f"{path + '.' if path else ''}{key} must not be a mapping")
            _reject_unknown_keys(
                value,
                reference_value,
                path=f"{path + '.' if path else ''}{key}",
            )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("Config root must be a mapping")
        _reject_unknown_keys(payload, DEFAULT_CONFIG)
        _deep_update(config, payload)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _reject_unknown_keys(config, DEFAULT_CONFIG)
    project = config["project"]
    if (
        isinstance(project["random_seed"], bool)
        or not isinstance(project["random_seed"], int)
        or int(project["random_seed"]) < 0
    ):
        raise ValueError("project.random_seed must be a non-negative integer")
    if not isinstance(project["allow_existing_v04_columns"], bool):
        raise ValueError("project.allow_existing_v04_columns must be boolean")
    positive_integer_fields = {
        "regime_stacker": (
            "horizon",
            "train_window",
            "min_train",
            "refit_every",
            "n_estimators",
            "num_leaves",
            "min_child_samples",
            "subsample_freq",
            "outer_n_jobs",
            "n_jobs",
        ),
        "statistics": ("global_lookback", "global_min_history"),
        "market_conditioned_diagnostic": (
            "scale_window",
            "scale_min_history",
        ),
        "fundamental_vintage": (
            "min_completed_target_vintages",
            "min_usable_vintages",
            "max_train_vintages",
        ),
        "lagged_market_conditioned": (
            "scale_window",
            "scale_min_history",
            "loss_window",
            "min_history",
            "availability_shift",
        ),
        "ml_incumbent_weekly_median_shrinkage": ("window_sessions",),
        "matured_proxy_gate": (
            "target_horizon_sessions",
            "maturity_shift_sessions",
            "loss_window",
            "min_history",
        ),
        "expected_pe": (
            "train_window",
            "min_train",
            "refit_every",
            "n_estimators",
            "num_leaves",
            "min_child_samples",
            "subsample_freq",
            "outer_n_jobs",
            "n_jobs",
        ),
        "statistical_gate": ("loss_window", "min_history"),
        "ml_no_regime_gate": ("loss_window", "min_history"),
        "ml_matched_regime_gate": ("loss_window", "min_history"),
        "ml_incumbent_gate": ("loss_window", "min_history"),
        "final_blend": ("loss_window", "min_history"),
        "final_guard": ("loss_window", "min_history"),
        "valuation": ("rolling_window", "min_history"),
    }
    for section, fields in positive_integer_fields.items():
        values = config.get(section, {})
        for field in fields:
            value = values.get(field, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{section}.{field} must be a positive integer")

    for section in ("regime_stacker", "expected_pe"):
        values = config[section]
        if int(values["train_window"]) < int(values["min_train"]):
            raise ValueError(f"{section}.train_window must be >= min_train")
        backend = str(values.get("parallel_backend", "thread")).lower()
        if backend not in {"serial", "thread"}:
            raise ValueError(f"{section}.parallel_backend must be serial or thread")
        if int(values["outer_n_jobs"]) > 1 and int(values["n_jobs"]) != 1:
            raise ValueError(f"{section}.n_jobs must equal 1 when outer_n_jobs > 1")

    for field in ("enabled", "deterministic", "force_col_wise"):
        if not isinstance(config["regime_stacker"][field], bool):
            raise ValueError(f"regime_stacker.{field} must be boolean")
    for field in (
        "train_no_regime",
        "train_with_regime",
        "include_return_forecast_features",
        "reuse_v03_ml_with_regime",
        "deterministic",
        "force_col_wise",
    ):
        if not isinstance(config["expected_pe"][field], bool):
            raise ValueError(f"expected_pe.{field} must be boolean")

    diagnostic = config["market_conditioned_diagnostic"]
    if not isinstance(diagnostic["enabled"], bool):
        raise ValueError("market_conditioned_diagnostic.enabled must be boolean")
    locked_diagnostic = {
        "q_over_r": 0.01,
        "clip_sigma": 3.0,
        "scale_window": 252,
        "scale_min_history": 126,
        "scale_floor": 0.01,
        "latent_weight": 0.25,
    }
    for field, locked_value in locked_diagnostic.items():
        value = diagnostic[field]
        if isinstance(locked_value, int):
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, Real) and not isinstance(value, bool)
        if not valid_type or not math.isfinite(float(value)) or float(value) != float(locked_value):
            raise ValueError(
                f"market_conditioned_diagnostic.{field} must equal locked value {locked_value}"
            )

    fundamental = config["fundamental_vintage"]
    if not isinstance(fundamental["enabled"], bool):
        raise ValueError("fundamental_vintage.enabled must be boolean")
    locked_fundamental = {
        "min_completed_target_vintages": 4,
        "min_usable_vintages": 8,
        "max_train_vintages": 20,
        "quantile": 0.50,
        "alpha": 0.05,
        "solver": "highs",
        "growth_winsor_lower": 0.05,
        "growth_winsor_upper": 0.95,
        "growth_iqr_floor": 0.01,
        "target_clip_lower": 0.10,
        "target_clip_upper": 0.90,
        "confidence_floor": 25.0,
        "confidence_ceiling": 100.0,
        "approximation_weight": 0.75,
    }
    for field, locked_value in locked_fundamental.items():
        value = fundamental[field]
        if isinstance(locked_value, int):
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(locked_value, float):
            valid_type = isinstance(value, Real) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, str)
        if not valid_type:
            raise ValueError(
                f"fundamental_vintage.{field} must equal locked value {locked_value!r}"
            )
        if isinstance(locked_value, (int, float)):
            valid_value = math.isfinite(float(value)) and float(value) == float(locked_value)
        else:
            valid_value = value == locked_value
        if not valid_value:
            raise ValueError(
                f"fundamental_vintage.{field} must equal locked value {locked_value!r}"
            )

    lagged = config["lagged_market_conditioned"]
    if not isinstance(lagged["enabled"], bool):
        raise ValueError("lagged_market_conditioned.enabled must be boolean")
    locked_lagged = {
        "q_over_r": 0.01,
        "clip_sigma": 3.0,
        "scale_window": 252,
        "scale_min_history": 126,
        "scale_floor": 0.01,
        "latent_weight": 0.25,
        "incumbent_weight": 0.75,
        "loss_window": 252,
        "min_history": 63,
        "improvement_margin": 0.001,
        "decisive_margin": 0.002,
        "temperature": 0.004,
        "max_challenger_weight": 1.0,
        "decision_mode": "winner_take_most",
        "availability_shift": 1,
    }
    for field, locked_value in locked_lagged.items():
        value = lagged[field]
        if isinstance(locked_value, int):
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(locked_value, float):
            valid_type = isinstance(value, Real) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, str)
        if not valid_type:
            raise ValueError(
                f"lagged_market_conditioned.{field} must equal locked value {locked_value!r}"
            )
        if isinstance(locked_value, (int, float)):
            valid_value = math.isfinite(float(value)) and float(value) == float(locked_value)
        else:
            valid_value = value == locked_value
        if not valid_value:
            raise ValueError(
                f"lagged_market_conditioned.{field} must equal locked value {locked_value!r}"
            )

    smoothing = config["ml_incumbent_weekly_median_shrinkage"]
    if not isinstance(smoothing["enabled"], bool):
        raise ValueError("ml_incumbent_weekly_median_shrinkage.enabled must be boolean")
    locked_smoothing = {
        "window_sessions": 5,
        "incumbent_weight": 0.75,
        "anchor_weight": 0.25,
        "method": "log_median",
    }
    for field, locked_value in locked_smoothing.items():
        value = smoothing[field]
        if isinstance(locked_value, int):
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(locked_value, float):
            valid_type = isinstance(value, Real) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, str)
        if not valid_type:
            raise ValueError(
                "ml_incumbent_weekly_median_shrinkage."
                f"{field} must equal locked value {locked_value!r}"
            )
        if isinstance(locked_value, (int, float)):
            valid_value = math.isfinite(float(value)) and float(value) == float(locked_value)
        else:
            valid_value = value == locked_value
        if not valid_value:
            raise ValueError(
                "ml_incumbent_weekly_median_shrinkage."
                f"{field} must equal locked value {locked_value!r}"
            )

    matured_proxy = config["matured_proxy_gate"]
    if not isinstance(matured_proxy["enabled"], bool):
        raise ValueError("matured_proxy_gate.enabled must be boolean")
    for field, locked_value in LOCKED_MATURED_PROXY_GATE_CONFIG.items():
        value = matured_proxy[field]
        if isinstance(locked_value, int):
            valid_type = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(locked_value, float):
            valid_type = isinstance(value, Real) and not isinstance(value, bool)
        else:
            valid_type = isinstance(value, str)
        if not valid_type:
            raise ValueError(f"matured_proxy_gate.{field} must equal locked value {locked_value!r}")
        if isinstance(locked_value, (int, float)):
            valid_value = math.isfinite(float(value)) and float(value) == float(locked_value)
        else:
            valid_value = value == locked_value
        if not valid_value:
            raise ValueError(f"matured_proxy_gate.{field} must equal locked value {locked_value!r}")

    stacker = config["regime_stacker"]
    bear = float(stacker["bear_return_threshold"])
    bull = float(stacker["bull_return_threshold"])
    floor = float(stacker["probability_floor"])
    if not (math.isfinite(bear) and math.isfinite(bull) and bear < 0.0 < bull):
        raise ValueError("regime_stacker thresholds must satisfy bear < 0 < bull")
    if not math.isfinite(floor) or not 0.0 <= floor < (1.0 / 3.0):
        raise ValueError("regime_stacker.probability_floor must be in [0, 1/3)")
    class_weight = stacker.get("class_weight")
    if class_weight is not None and str(class_weight).strip().lower() not in {
        "none",
        "natural",
    }:
        raise ValueError("regime_stacker.class_weight must be null/natural")

    expected = config["expected_pe"]
    max_depth = expected["max_depth"]
    if (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, int)
        or (max_depth != -1 and max_depth <= 0)
    ):
        raise ValueError("expected_pe.max_depth must be -1 or a positive integer")
    for section in ("regime_stacker", "expected_pe"):
        verbosity = config[section]["verbosity"]
        if isinstance(verbosity, bool) or not isinstance(verbosity, int):
            raise ValueError(f"{section}.verbosity must be an integer")
    for field in ("fallback_max_iter", "fallback_max_leaf_nodes"):
        value = expected[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"expected_pe.{field} must be a positive integer")
    fallback_rate = float(expected["fallback_learning_rate"])
    if not math.isfinite(fallback_rate) or fallback_rate <= 0.0:
        raise ValueError("expected_pe.fallback_learning_rate must be positive and finite")
    for section in ("regime_stacker", "expected_pe"):
        for field in ("subsample", "colsample_bytree"):
            value = float(config[section][field])
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{section}.{field} must be in (0, 1]")
    for field in ("reg_alpha", "reg_lambda"):
        value = float(expected[field])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"expected_pe.{field} must be finite and non-negative")
    shrinkage_weight = config["no_regime_shrinkage"]["weight"]
    if isinstance(shrinkage_weight, bool):
        raise ValueError("no_regime_shrinkage.weight must be finite and in [0, 1]")
    try:
        shrinkage_weight = float(shrinkage_weight)
    except (TypeError, ValueError) as exc:
        raise ValueError("no_regime_shrinkage.weight must be finite and in [0, 1]") from exc
    if not math.isfinite(shrinkage_weight) or not 0.0 <= shrinkage_weight <= 1.0:
        raise ValueError("no_regime_shrinkage.weight must be finite and in [0, 1]")
    regime_lambda = float(config["regime_stacker"]["reg_lambda"])
    if not math.isfinite(regime_lambda) or regime_lambda < 0.0:
        raise ValueError("regime_stacker.reg_lambda must be finite and non-negative")
    for section in ("regime_stacker", "expected_pe"):
        rate = float(config[section]["learning_rate"])
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"{section}.learning_rate must be positive and finite")

    gate_sections = (
        "statistical_gate",
        "ml_no_regime_gate",
        "ml_matched_regime_gate",
        "ml_incumbent_gate",
        "final_guard",
    )
    for section in gate_sections:
        values = config[section]
        if int(values["min_history"]) > int(values["loss_window"]):
            raise ValueError(f"{section}.min_history must be <= loss_window")
        temperature = float(values["temperature"])
        margin = float(values["improvement_margin"])
        max_weight = float(values["max_challenger_weight"])
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"{section}.temperature must be positive and finite")
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError(f"{section}.improvement_margin must be finite and non-negative")
        if not math.isfinite(max_weight) or not 0.0 <= max_weight <= 1.0:
            raise ValueError(f"{section}.max_challenger_weight must be in [0, 1]")
        if "decisive_margin" in values:
            decisive = float(values["decisive_margin"])
            if not math.isfinite(decisive) or decisive < margin:
                raise ValueError(
                    f"{section}.decisive_margin must be finite and >= improvement_margin"
                )
        if "decision_mode" in values and values["decision_mode"] not in {
            "soft",
            "winner_take_most",
        }:
            raise ValueError(f"{section}.decision_mode is unsupported")

    blend = config["final_blend"]
    if int(blend["min_history"]) > int(blend["loss_window"]):
        raise ValueError("final_blend.min_history must be <= loss_window")
    if not math.isfinite(float(blend["temperature"])) or float(blend["temperature"]) <= 0:
        raise ValueError("final_blend.temperature must be positive and finite")
    priors = (float(blend["stat_prior"]), float(blend["ml_prior"]))
    if any(not math.isfinite(value) or value < 0.0 for value in priors) or sum(priors) <= 0:
        raise ValueError("final_blend priors must be finite, non-negative, and not both zero")

    method = str(config["statistics"]["global_method"]).lower()
    if method not in {"median", "mean"}:
        raise ValueError("statistics.global_method must be median or mean")

    valuation = config.get("valuation", {})
    cheap = float(valuation.get("cheap_quantile", 0.15))
    expensive = float(valuation.get("expensive_quantile", 0.85))
    confirmed_cheap = float(valuation.get("confirmed_cheap_quantile", 0.10))
    confirmed_expensive = float(valuation.get("confirmed_expensive_quantile", 0.90))
    if not 0.0 < cheap < expensive < 1.0:
        raise ValueError("valuation cheap/expensive quantiles are invalid")
    if not 0.0 < confirmed_cheap < confirmed_expensive < 1.0:
        raise ValueError("valuation confirmed quantiles are invalid")
    for field in (
        "minimum_absolute_gap",
        "confirmed_minimum_absolute_gap",
        "confirmed_robust_z",
    ):
        value = float(valuation[field])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"valuation.{field} must be finite and non-negative")
    absolute_cheap = float(valuation["absolute_cheap_gap"])
    absolute_expensive = float(valuation["absolute_expensive_gap"])
    if not (
        math.isfinite(absolute_cheap)
        and math.isfinite(absolute_expensive)
        and absolute_cheap < 0.0 < absolute_expensive
    ):
        raise ValueError(
            "valuation absolute gaps must satisfy absolute_cheap_gap < 0 < absolute_expensive_gap"
        )
