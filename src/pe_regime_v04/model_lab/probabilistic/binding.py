"""Design and feature-sidecar byte bindings for the isolated wave."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    PROBABILISTIC_DESIGN_SHA256,
    ProbabilisticContractError,
    canonical_json_bytes,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
    verify_probabilistic_design,
)
from .spec import FEATURE_COLUMNS, feature_metadata


FEATURE_REGISTRY_SHA256 = "2ea4c225ed3bcd94d4bb13536a6f00735cdfadf32d1534d4b671ea75e6524d7d"


def feature_sidecar() -> dict[str, Any]:
    records = [asdict(value) for value in feature_metadata()]
    return seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.probabilistic_feature_sidecar.v1",
            "design_sha256": PROBABILISTIC_DESIGN_SHA256,
            "source_feature_registry_sha256": FEATURE_REGISTRY_SHA256,
            "feature_columns": list(FEATURE_COLUMNS),
            "features": records,
        }
    )


LOCKED_FEATURE_SIDECAR_SHA256 = sha256_bytes(canonical_json_bytes(feature_sidecar()))


def verify_feature_sidecar(payload: Mapping[str, Any]) -> None:
    verify_payload_seal(payload)
    if sha256_bytes(canonical_json_bytes(payload)) != LOCKED_FEATURE_SIDECAR_SHA256:
        raise ProbabilisticContractError("feature sidecar differs from the implementation lock")
    if tuple(payload.get("feature_columns", ())) != FEATURE_COLUMNS:
        raise ProbabilisticContractError("feature sidecar ordering changed")


def verify_implementation_binding(*, design_path: Path, sidecar: Mapping[str, Any]) -> None:
    verify_probabilistic_design(Path(design_path))
    verify_feature_sidecar(sidecar)
