"""Immutable score-free artifact writers and audit inventory helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from .contracts import (
    EVALUATION_ONLY_COLUMN,
    PROBABILISTIC_DESIGN_SHA256,
    ProbabilisticContractError,
    seal_payload,
    sha256_bytes,
    sha256_file,
)


SCREEN_SCHEMA = "expected_pe_model_zoo.probabilistic_wave_screen.v1"
FORBIDDEN_SCORE_WORDS = ("prediction.csv", "prediction.parquet", "scores.csv", "scores.parquet")


def immutable_write_bytes(path: Path, payload: bytes) -> str:
    """Create once, or verify byte identity; never replace differing evidence."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise ProbabilisticContractError(f"immutable artifact already differs: {path}")
        return sha256_bytes(existing)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".tmp.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_bytes(payload)


def immutable_write_text(path: Path, text: str) -> str:
    return immutable_write_bytes(Path(path), text.encode("utf-8"))


def immutable_write_json(path: Path, payload: dict[str, Any], *, sealed: bool = True) -> str:
    value = seal_payload(payload) if sealed else dict(payload)
    rendered = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return immutable_write_text(Path(path), rendered)


def file_record(path: Path, *, root: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProbabilisticContractError("audit artifact is outside repository root") from exc
    lowered = relative.casefold()
    if EVALUATION_ONLY_COLUMN in lowered or any(word in lowered for word in FORBIDDEN_SCORE_WORDS):
        raise ProbabilisticContractError("score/truth artifact cannot enter score-free inventory")
    return {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def checksum_manifest(paths: Iterable[Path], *, root: Path) -> str:
    records = sorted(
        (file_record(path, root=root) for path in paths), key=lambda value: value["path"]
    )
    return "".join(f"{record['sha256']}  {record['path']}\n" for record in records)


def base_artifact_payload(*, artifact_type: str) -> dict[str, Any]:
    return {
        "schema_version": SCREEN_SCHEMA,
        "artifact_type": artifact_type,
        "design_sha256": PROBABILISTIC_DESIGN_SHA256,
        "scope": "SCORE_FREE_SYNTHETIC_ONLY",
        "project_predictions_generated": False,
        "project_scores_generated_or_read": False,
        "fresh_seed_reserved_or_opened": False,
        "heldout_opened": False,
        "registry_modified": False,
    }
