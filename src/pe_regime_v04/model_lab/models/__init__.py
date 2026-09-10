"""Model Lab candidate adapters isolated from the production overlay."""

from .wave1.adapters import build_wave1_model, resolved_parameter_manifest
from .wave1.spec import (
    BASELINE_MODEL_IDS,
    CANDIDATE_MODEL_IDS,
    COMMON_FEATURES,
    REGIME_EXTENSION,
    WAVE1_MODEL_DEFINITIONS,
)

__all__ = [
    "BASELINE_MODEL_IDS",
    "CANDIDATE_MODEL_IDS",
    "COMMON_FEATURES",
    "REGIME_EXTENSION",
    "WAVE1_MODEL_DEFINITIONS",
    "build_wave1_model",
    "resolved_parameter_manifest",
]
