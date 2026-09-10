"""Sealed, CSV-only Wave-1 artifacts and leakage-safe input loading."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ...contracts import ContractError
from .spec import (
    BASELINE_MODEL_IDS,
    COMMON_FEATURES,
    DESIGN_LOCK_SHA256,
    REGIME_EXTENSION,
)


PREDICTION_COLUMNS = (
    "seed",
    "date",
    "symbol",
    "fold_id",
    "test_start_position",
    "model_id",
    "prediction",
)
DIAGNOSTIC_COLUMNS = (
    "seed",
    "fold_id",
    "test_start_position",
    "test_rows",
    "model_id",
    "fold_seed",
    "status",
    "eligible_train_rows",
    "active_features_json",
    "dropped_all_missing_json",
    "dropped_constant_spline_json",
    "warnings_json",
    "exception_type",
    "exception_message",
    "runtime_seconds",
)


class Wave1ArtifactError(ContractError):
    """Raised for a missing, mutable, malformed, or leakage-prone artifact."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_file_record(record: Mapping[str, Any], *, context: str) -> Path:
    if set(record) != {"path", "bytes", "sha256"}:
        raise Wave1ArtifactError(f"{context} file record fields are invalid")
    path = Path(str(record["path"]))
    if not path.is_absolute() or not path.is_file():
        raise Wave1ArtifactError(f"{context} path must be an existing absolute file")
    if path.stat().st_size != int(record["bytes"]):
        raise Wave1ArtifactError(f"{context} byte count differs from the seal")
    if sha256_file(path) != str(record["sha256"]):
        raise Wave1ArtifactError(f"{context} SHA-256 differs from the seal")
    return path


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def seal_payload(payload: Mapping[str, Any], *, field: str = "manifest_sha256") -> dict[str, Any]:
    output = dict(payload)
    output.pop(field, None)
    output[field] = hashlib.sha256(canonical_json_bytes(output)).hexdigest()
    return output


def verify_payload_seal(payload: Mapping[str, Any], *, field: str = "manifest_sha256") -> None:
    recorded = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if recorded != expected:
        raise Wave1ArtifactError(f"{field} is missing or invalid")


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable JSON already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_immutable_text(path: Path, text: str) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable text already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_sealed_json(path: Path, *, expected_mode: str | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Wave1ArtifactError(f"cannot read sealed JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise Wave1ArtifactError("sealed JSON root must be an object")
    verify_payload_seal(payload)
    if expected_mode is not None and payload.get("mode") != expected_mode:
        raise Wave1ArtifactError(
            f"manifest mode mismatch: expected={expected_mode!r}, actual={payload.get('mode')!r}"
        )
    return payload


def _contains_evaluation_path(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if "truth" in lowered or "heldout" in lowered or "true_fair_pe" in lowered:
                return True
            if _contains_evaluation_path(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_evaluation_path(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower().replace("\\", "/")
        basename = lowered.rsplit("/", 1)[-1]
        return "truth" in basename or "heldout" in basename or "true_fair_pe" in lowered
    return False


def load_predict_inputs(path: Path) -> dict[str, Any]:
    """Load the only manifest accepted by predict mode; reject evaluation references."""

    payload = load_sealed_json(path, expected_mode="predict_inputs")
    if _contains_evaluation_path(payload):
        raise Wave1ArtifactError("predict mode manifest contains an evaluation-data reference")
    required = {
        "format_version",
        "mode",
        "design_lock_sha256",
        "execution_binding",
        "execution_precommit",
        "evaluation_data_excluded",
        "seeds",
        "manifest_sha256",
    }
    if set(payload) != required:
        raise Wave1ArtifactError("predict input manifest fields are invalid")
    if payload["design_lock_sha256"] != DESIGN_LOCK_SHA256:
        raise Wave1ArtifactError("predict input manifest is bound to a different design")
    if payload["evaluation_data_excluded"] is not True:
        raise Wave1ArtifactError("predict manifest must affirm evaluation-data exclusion")
    return payload


def _read_csv_round_trip(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, float_precision="round_trip")
    except (OSError, ValueError) as exc:
        raise Wave1ArtifactError(f"cannot read round-trip CSV: {path}") from exc


def inspect_predict_seed_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "seed",
        "source_record_bytes",
        "source_record_sha256",
        "canonical_csv",
        "output_csv",
        "expected_rows",
        "expected_symbol",
        "canonical_columns_sha256",
        "output_columns_sha256",
        "prefix150_logical_sha256",
        "identity_logical_sha256",
    }
    if set(entry) != required:
        raise Wave1ArtifactError("predict seed entry fields are invalid")
    canonical_path = verify_file_record(entry["canonical_csv"], context="canonical150")
    output_path = verify_file_record(entry["output_csv"], context="output254")
    canonical = _read_csv_round_trip(canonical_path)
    output = _read_csv_round_trip(output_path)
    if canonical.columns.has_duplicates or output.columns.has_duplicates:
        raise Wave1ArtifactError("canonical/output columns must be unique")
    if canonical.shape != (int(entry["expected_rows"]), 150):
        raise Wave1ArtifactError(f"canonical150 shape mismatch: {canonical.shape}")
    if output.shape != (int(entry["expected_rows"]), 254):
        raise Wave1ArtifactError(f"output254 shape mismatch: {output.shape}")
    if "true_fair_pe" in canonical or "true_fair_pe" in output:
        raise Wave1ArtifactError("fit/predict artifacts must not contain true_fair_pe")
    if tuple(output.columns[:150]) != tuple(canonical.columns):
        raise Wave1ArtifactError("output254 prefix column order differs from canonical150")
    try:
        assert_frame_equal(
            canonical,
            output.iloc[:, :150],
            check_dtype=True,
            check_exact=True,
            check_names=True,
        )
    except AssertionError as exc:
        raise Wave1ArtifactError("output254 prefix values differ from canonical150") from exc

    canonical_columns_hash = hashlib.sha256(
        canonical_json_bytes(list(canonical.columns))
    ).hexdigest()
    output_columns_hash = hashlib.sha256(canonical_json_bytes(list(output.columns))).hexdigest()
    if canonical_columns_hash != entry["canonical_columns_sha256"]:
        raise Wave1ArtifactError("canonical150 column hash differs")
    if output_columns_hash != entry["output_columns_sha256"]:
        raise Wave1ArtifactError("output254 column hash differs")

    required_columns = {
        "date",
        "symbol",
        "observed_pe",
        *COMMON_FEATURES,
        *REGIME_EXTENSION,
        *BASELINE_MODEL_IDS,
    }
    missing = sorted(required_columns.difference(output.columns))
    if missing:
        raise Wave1ArtifactError(f"output254 is missing locked Wave1 columns: {missing}")
    if sorted({*COMMON_FEATURES, "date", "symbol", "observed_pe"}.difference(canonical.columns)):
        raise Wave1ArtifactError("canonical150 is missing locked common inputs")

    dates = pd.to_datetime(canonical["date"], errors="coerce")
    output_dates = pd.to_datetime(output["date"], errors="coerce")
    if dates.isna().any() or output_dates.isna().any() or not dates.equals(output_dates):
        raise Wave1ArtifactError("canonical/output dates are invalid or reordered")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise Wave1ArtifactError("canonical dates must be strictly unique and increasing")
    symbols = canonical["symbol"].astype(str)
    output_symbols = output["symbol"].astype(str)
    if not symbols.equals(output_symbols) or symbols.nunique(dropna=False) != 1:
        raise Wave1ArtifactError("canonical/output symbol identity differs or is not single-issuer")
    if symbols.iloc[0] != str(entry["expected_symbol"]):
        raise Wave1ArtifactError("canonical symbol differs from the sealed expected symbol")

    prefix_hash = logical_frame_sha256(canonical)
    identity_hash = logical_frame_sha256(
        pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "symbol": symbols})
    )
    if prefix_hash != entry["prefix150_logical_sha256"]:
        raise Wave1ArtifactError("canonical150 logical frame hash differs")
    if identity_hash != entry["identity_logical_sha256"]:
        raise Wave1ArtifactError("canonical identity logical hash differs")
    return {
        "canonical": canonical,
        "output": output,
        "dates": dates,
        "symbols": symbols,
        "canonical_path": canonical_path,
        "output_path": output_path,
    }


def load_wave1_seed_frames(entry: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inspected = inspect_predict_seed_entry(entry)
    canonical = inspected["canonical"]
    output = inspected["output"]
    dates = inspected["dates"]
    symbols = inspected["symbols"]

    current_sidecar_columns = tuple(REGIME_EXTENSION[:3])
    canonical_regime_columns = tuple(REGIME_EXTENSION[3:])
    collision = sorted(set(current_sidecar_columns).intersection(canonical.columns))
    if collision:
        raise Wave1ArtifactError(
            f"current-regime sidecar columns collide with canonical150: {collision}"
        )
    missing_canonical_regime = sorted(set(canonical_regime_columns).difference(canonical.columns))
    if missing_canonical_regime:
        raise Wave1ArtifactError(
            f"canonical150 is missing locked regime-extension columns: {missing_canonical_regime}"
        )
    model_frame = canonical.loc[
        :, ["observed_pe", *COMMON_FEATURES, *canonical_regime_columns]
    ].copy()
    model_frame.insert(0, "symbol", symbols.to_numpy(copy=True))
    model_frame.insert(0, "date", dates.to_numpy(copy=True))
    sidecar = output.loc[:, ["date", "symbol", *current_sidecar_columns]].copy()
    sidecar["date"] = pd.to_datetime(sidecar["date"], errors="coerce")
    merged = model_frame.loc[:, ["date", "symbol"]].merge(
        sidecar,
        how="left",
        on=["date", "symbol"],
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise Wave1ArtifactError("current-regime sidecar date/symbol join is incomplete")
    for column in current_sidecar_columns:
        model_frame[column] = merged[column].to_numpy(copy=True)

    baselines = output.loc[:, ["date", "symbol", *BASELINE_MODEL_IDS]].copy()
    baselines["date"] = pd.to_datetime(baselines["date"], errors="coerce")
    if baselines.duplicated(["date", "symbol"]).any():
        raise Wave1ArtifactError("baseline output254 keys must be unique")
    return model_frame, baselines


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column]).dt.strftime("%Y-%m-%d")
    rendered = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _normalized_for_csv(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if tuple(frame.columns) != tuple(columns):
        raise Wave1ArtifactError(
            f"CSV schema mismatch: expected={tuple(columns)}, actual={tuple(frame.columns)}"
        )
    output = frame.copy()
    if "date" in output:
        dates = pd.to_datetime(output["date"], errors="coerce")
        if dates.isna().any():
            raise Wave1ArtifactError("CSV dates must be valid")
        output["date"] = dates.dt.strftime("%Y-%m-%d")
    return output


def write_round_trip_csv(frame: pd.DataFrame, path: Path, *, columns: Sequence[str]) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"immutable CSV already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalized_for_csv(frame, columns)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary CSV already exists: {temporary}")
    try:
        normalized.to_csv(
            temporary,
            index=False,
            lineterminator="\n",
            float_format="%.17g",
            na_rep="",
        )
        reread = _read_csv_round_trip(temporary)
        if tuple(reread.columns) != tuple(columns):
            raise Wave1ArtifactError("round-trip CSV column order changed")
        for column in normalized.columns:
            left = normalized[column]
            right = reread[column]
            if pd.api.types.is_numeric_dtype(left):
                left_values = pd.to_numeric(left, errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
                right_values = pd.to_numeric(right, errors="coerce").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
                if not np.array_equal(left_values, right_values, equal_nan=True):
                    raise Wave1ArtifactError(f"round-trip numeric values changed in {column}")
            else:
                left_values = left.fillna("").astype(str).to_numpy()
                right_values = right.fillna("").astype(str).to_numpy()
                if not np.array_equal(left_values, right_values):
                    raise Wave1ArtifactError(f"round-trip text values changed in {column}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_record(path)


def require_finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise Wave1ArtifactError("JSON evidence cannot contain non-finite values")
    if isinstance(value, Mapping):
        for item in value.values():
            require_finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            require_finite_json(item)
