"""Frozen Wave-1 cheap-screen implementation (no production connection)."""

from .adapters import build_wave1_model, resolved_parameter_manifest
from .spec import BASELINE_MODEL_IDS, CANDIDATE_MODEL_IDS, WAVE1_MODEL_DEFINITIONS

__all__ = [
    "BASELINE_MODEL_IDS",
    "CANDIDATE_MODEL_IDS",
    "WAVE1_MODEL_DEFINITIONS",
    "build_wave1_model",
    "resolved_parameter_manifest",
]
