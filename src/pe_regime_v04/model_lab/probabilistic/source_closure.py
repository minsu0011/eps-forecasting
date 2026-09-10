"""Content-addressed verification of the complete probabilistic execution closure."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import (
    PROBABILISTIC_DESIGN_SHA256,
    ProbabilisticContractError,
    require_sha256,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)


SOURCE_CLOSURE_SCHEMA = "expected_pe_model_zoo.probabilistic_source_closure.v5"
_SOURCE_CLOSURE_TOKEN = object()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def model_lab_initializer_import_paths(repo_root: Path) -> tuple[Path, ...]:
    """Derive the exact shared modules loaded by the bound package initializer."""

    package = repo_root / "src/pe_regime_v04/model_lab"
    initializer = package / "__init__.py"
    try:
        tree = ast.parse(initializer.read_text(encoding="utf-8"), filename=str(initializer))
    except (OSError, SyntaxError) as exc:
        raise ProbabilisticContractError(
            "model_lab package initializer cannot be inspected"
        ) from exc
    paths = {initializer}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
            continue
        relative = Path(*node.module.split("."))
        module_path = package / relative.with_suffix(".py")
        package_path = package / relative / "__init__.py"
        if module_path.is_file() == package_path.is_file():
            raise ProbabilisticContractError(
                f"model_lab initializer import is missing or ambiguous: {node.module}"
            )
        paths.add(module_path if module_path.is_file() else package_path)
    return tuple(sorted((path.resolve() for path in paths), key=lambda path: path.as_posix()))


def execution_closure_paths(repo_root: Path | None = None) -> tuple[Path, ...]:
    """Return the exact project-code/config closure executed by this wave."""

    root = (repo_root or repository_root()).resolve()
    probabilistic = root / "src/pe_regime_v04/model_lab/probabilistic"
    scripts = root / "scripts/model_lab/probabilistic"
    shared_model_lab = model_lab_initializer_import_paths(root)
    fixed = (
        root / "src/pe_regime_v04/__init__.py",
        root / "pyproject.toml",
        root / "outputs/model_zoo_probabilistic_wave_design_20260819/DESIGN.json",
        root / "outputs/model_zoo_probabilistic_wave_screen_20260819/"
        "MODEL_LAB_FULL_PACKAGE_FREEZE.txt",
        root / "outputs/model_zoo_probabilistic_wave_screen_20260819/"
        "NGBOOST_FULL_PACKAGE_FREEZE.txt",
        root / "outputs/model_zoo_probabilistic_wave_screen_20260819/"
        "NGBOOST_FULL_CLONE_EVIDENCE.json",
    )
    paths = {
        *probabilistic.glob("*.py"),
        *scripts.glob("*.py"),
        *shared_model_lab,
        *fixed,
    }
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        raise ProbabilisticContractError(
            f"probabilistic execution closure is incomplete: {[str(path) for path in missing]}"
        )
    return tuple(sorted((path.resolve() for path in paths), key=lambda path: path.as_posix()))


def build_source_closure_payload(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    records = []
    for path in execution_closure_paths(root):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ProbabilisticContractError("execution closure escaped repository root") from exc
        raw = path.read_bytes()
        records.append(
            {
                "path": relative,
                "sha256": sha256_bytes(raw),
                "bytes": len(raw),
            }
        )
    return seal_payload(
        {
            "schema_version": SOURCE_CLOSURE_SCHEMA,
            "design_sha256": PROBABILISTIC_DESIGN_SHA256,
            "closure_policy": "exact_initializer_derived_project_execution_closure",
            "records": records,
        }
    )


class VerifiedSourceClosure:
    """Opaque capability that continuously attests current execution bytes."""

    __slots__ = ("_payload", "_raw", "_raw_sha256", "_repo_root")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _SOURCE_CLOSURE_TOKEN:
            raise ProbabilisticContractError(
                "VerifiedSourceClosure must come from its content-addressed loader"
            )
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        payload: Mapping[str, Any],
        raw: bytes,
        repo_root: Path,
    ) -> None:
        if token is not _SOURCE_CLOSURE_TOKEN:
            raise ProbabilisticContractError("invalid source-closure factory token")
        object.__setattr__(self, "_payload", json.loads(json.dumps(payload)))
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))
        object.__setattr__(self, "_repo_root", repo_root.resolve())
        self.verify_integrity()

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedSourceClosure is immutable")

    @property
    def raw_sha256(self) -> str:
        return self._raw_sha256

    @property
    def payload(self) -> Mapping[str, Any]:
        self.verify_integrity()
        return MappingProxyType(json.loads(json.dumps(self._payload)))

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw) != self._raw_sha256:
            raise ProbabilisticContractError("source-closure snapshot bytes changed")
        try:
            decoded = json.loads(self._raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProbabilisticContractError("source-closure snapshot cannot be decoded") from exc
        verify_payload_seal(decoded)
        if decoded != self._payload or decoded.get("schema_version") != SOURCE_CLOSURE_SCHEMA:
            raise ProbabilisticContractError("source-closure snapshot payload changed")
        expected_paths = execution_closure_paths(self._repo_root)
        expected_relative = [
            path.relative_to(self._repo_root).as_posix() for path in expected_paths
        ]
        records = decoded.get("records")
        if (
            not isinstance(records, list)
            or [item.get("path") for item in records] != expected_relative
        ):
            raise ProbabilisticContractError("source-closure path set/order differs")
        for path, record in zip(expected_paths, records, strict=True):
            digest = record.get("sha256")
            require_sha256(digest, field=f"source_closure.{record.get('path')}")
            raw = path.read_bytes()
            if sha256_bytes(raw) != digest or len(raw) != record.get("bytes"):
                raise ProbabilisticContractError(
                    f"current execution source/config bytes drifted: {record.get('path')}"
                )


def load_verified_source_closure(
    path: Path,
    *,
    expected_raw_sha256: str,
    repo_root: Path | None = None,
) -> VerifiedSourceClosure:
    require_sha256(expected_raw_sha256, field="source_closure_raw_sha256")
    snapshot_path = Path(path)
    try:
        raw = snapshot_path.read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError("source-closure snapshot is unavailable") from exc
    if sha256_bytes(raw) != expected_raw_sha256:
        raise ProbabilisticContractError("source-closure snapshot differs from authorization")
    if expected_raw_sha256 not in snapshot_path.name.split("."):
        raise ProbabilisticContractError("source-closure snapshot is not content-addressed")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("source-closure snapshot cannot be decoded") from exc
    verify_payload_seal(payload)
    return VerifiedSourceClosure(
        _SOURCE_CLOSURE_TOKEN,
        payload=payload,
        raw=raw,
        repo_root=(repo_root or repository_root()),
    )
