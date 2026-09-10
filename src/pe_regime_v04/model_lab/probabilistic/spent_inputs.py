"""Exact Wave-1 spent-input reconstruction without exposing evaluator truth to prediction."""

from __future__ import annotations

import json
import io
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from .contracts import (
    IDENTITY_COLUMNS,
    ProbabilisticContractError,
    canonical_json_bytes,
    seal_payload,
    sha256_bytes,
)
from .nested import build_outer_folds
from .spec import FEATURE_COLUMNS
from .spent import ENTITY_ID, SPENT_SEEDS, UPSTREAM_INPUTS


WAVE1_DESIGN_SHA256 = "b3a190cbf80045b460551af4b9a53476d5c369e47f2cbf460a02d8779672c82d"
BASELINE_COLUMNS = ("v04_expected_pe", "ml_expected_pe")


def _verify_sealed_file_record(record: object, *, context: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ProbabilisticContractError(f"{context} file record differs")
    path = Path(str(record["path"]))
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError(f"{context} file is unavailable") from exc
    if len(raw) != int(record["bytes"]) or sha256_bytes(raw) != record["sha256"]:
        raise ProbabilisticContractError(f"{context} file bytes differ from seal")
    return path


def _load_manifest(repo_root: Path, *, role: str) -> tuple[bytes, dict[str, Any]]:
    record = UPSTREAM_INPUTS[role]
    path = Path(repo_root) / record["path"]
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError(f"spent {role} manifest is unavailable") from exc
    if (
        sha256_bytes(raw) != record["raw_sha256"]
        or not isinstance(payload, dict)
        or payload.get("manifest_sha256") != record["logical_sha256"]
    ):
        raise ProbabilisticContractError(f"spent {role} manifest differs from fixed pin")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    if sha256_bytes(canonical_json_bytes(unsigned)) != payload["manifest_sha256"]:
        raise ProbabilisticContractError(f"spent {role} manifest seal is invalid")
    return raw, payload


def _wave1_logical_sha256(frame: pd.DataFrame) -> str:
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
    return sha256_bytes(rendered)


def _read_round_trip(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, float_precision="round_trip")
    except (OSError, ValueError) as exc:
        raise ProbabilisticContractError(f"spent CSV cannot be decoded: {path}") from exc


def _read_sealed_csv_once(record: object, *, context: str) -> tuple[bytes, pd.DataFrame]:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ProbabilisticContractError(f"{context} file record differs")
    try:
        raw = Path(str(record["path"])).read_bytes()
        frame = pd.read_csv(io.BytesIO(raw), low_memory=False, float_precision="round_trip")
    except (OSError, ValueError) as exc:
        raise ProbabilisticContractError(f"{context} cannot be loaded once") from exc
    if len(raw) != int(record["bytes"]) or sha256_bytes(raw) != record["sha256"]:
        raise ProbabilisticContractError(f"{context} bytes differ from seal")
    return raw, frame


def _reject_evaluation_reference(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).casefold()
            if any(token in lowered for token in ("truth", "heldout", "true_fair_pe")):
                raise ProbabilisticContractError("prediction manifest exposes evaluation data")
            _reject_evaluation_reference(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_evaluation_reference(item)
    elif isinstance(value, str):
        lowered = value.casefold().replace("\\", "/")
        if (
            "true_fair_pe" in lowered
            or "heldout" in lowered
            or "truth" in lowered.rsplit("/", 1)[-1]
        ):
            raise ProbabilisticContractError("prediction manifest exposes evaluation data")


def assemble_prediction_inputs(repo_root: Path) -> dict[str, Any]:
    """Load exact prediction-safe bytes and return the seven formal custody frames."""

    manifest_raw, manifest = _load_manifest(repo_root, role="predict")
    _reject_evaluation_reference(manifest)
    if (
        manifest.get("mode") != "predict_inputs"
        or manifest.get("design_lock_sha256") != WAVE1_DESIGN_SHA256
        or manifest.get("evaluation_data_excluded") is not True
        or tuple(item.get("seed") for item in manifest.get("seeds", ())) != SPENT_SEEDS
    ):
        raise ProbabilisticContractError("prediction-safe spent manifest role differs")
    _verify_sealed_file_record(manifest.get("execution_binding"), context="execution binding")
    _verify_sealed_file_record(manifest.get("execution_precommit"), context="execution precommit")

    feature_parts: list[pd.DataFrame] = []
    label_parts: list[pd.DataFrame] = []
    identity_parts: list[pd.DataFrame] = []
    point_parts: list[pd.DataFrame] = []
    comparator_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in BASELINE_COLUMNS}
    source_records: list[dict[str, Any]] = []
    for entry in manifest["seeds"]:
        expected_fields = {
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
        if set(entry) != expected_fields:
            raise ProbabilisticContractError("spent prediction seed record schema differs")
        seed = int(entry["seed"])
        canonical_path = _verify_sealed_file_record(
            entry["canonical_csv"], context=f"seed {seed} canonical"
        )
        output_path = _verify_sealed_file_record(
            entry["output_csv"], context=f"seed {seed} Wave-1 output"
        )
        canonical = _read_round_trip(canonical_path)
        output = _read_round_trip(output_path)
        if canonical.shape != (1800, 150) or output.shape != (1800, 254):
            raise ProbabilisticContractError("spent canonical/output shape differs")
        if canonical.columns.has_duplicates or output.columns.has_duplicates:
            raise ProbabilisticContractError("spent input columns are duplicated")
        if tuple(output.columns[:150]) != tuple(canonical.columns):
            raise ProbabilisticContractError("Wave-1 output prefix schema differs")
        try:
            assert_frame_equal(
                canonical,
                output.iloc[:, :150],
                check_dtype=True,
                check_exact=True,
                check_names=True,
            )
        except AssertionError as exc:
            raise ProbabilisticContractError("Wave-1 output prefix bytes differ") from exc
        if (
            sha256_bytes(canonical_json_bytes(list(canonical.columns)))
            != entry["canonical_columns_sha256"]
            or sha256_bytes(canonical_json_bytes(list(output.columns)))
            != entry["output_columns_sha256"]
        ):
            raise ProbabilisticContractError("spent input column-order seal differs")
        if _wave1_logical_sha256(canonical) != entry["prefix150_logical_sha256"]:
            raise ProbabilisticContractError("spent canonical logical bytes differ")
        missing = sorted(
            {"date", "symbol", "observed_pe", *FEATURE_COLUMNS, *BASELINE_COLUMNS}.difference(
                output.columns
            )
        )
        if missing or "true_fair_pe" in output:
            raise ProbabilisticContractError(f"spent prediction surface differs: {missing}")
        dates = pd.to_datetime(canonical["date"], errors="coerce", utc=True).dt.tz_convert(None)
        symbols = canonical["symbol"].astype(str)
        if (
            dates.isna().any()
            or not dates.is_monotonic_increasing
            or dates.duplicated().any()
            or symbols.nunique(dropna=False) != 1
            or symbols.iloc[0] != ENTITY_ID
            or str(entry["expected_symbol"]) != ENTITY_ID
        ):
            raise ProbabilisticContractError("spent date/entity identity differs")
        wave_identity = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"), "symbol": symbols})
        if _wave1_logical_sha256(wave_identity) != entry["identity_logical_sha256"]:
            raise ProbabilisticContractError("spent date/entity logical seal differs")

        base_identity = pd.DataFrame(
            {
                "seed": seed,
                "entity_id": ENTITY_ID,
                "date": dates,
                "ordered_position": np.arange(1800, dtype=np.int64),
            }
        )
        features = base_identity.copy()
        for column in FEATURE_COLUMNS:
            features[column] = pd.to_numeric(output[column], errors="coerce").to_numpy()
        labels = base_identity.copy()
        labels["label_available_at"] = dates
        labels["observed_pe"] = pd.to_numeric(output["observed_pe"], errors="coerce").to_numpy()
        folds = build_outer_folds(
            labels, label_available_at_column="label_available_at", require_formal_shape=True
        )
        selected_folds = folds[12:]
        if len(selected_folds) != 62:
            raise ProbabilisticContractError("spent formal outer-fold count differs")
        records: list[dict[str, Any]] = []
        for fold in selected_folds:
            for position in fold.test_positions:
                records.append(
                    {
                        "seed": seed,
                        "entity_id": ENTITY_ID,
                        "date": dates.iloc[position],
                        "ordered_position": position,
                        "fold_id": fold.fold_id,
                    }
                )
        identity = pd.DataFrame(records, columns=IDENTITY_COLUMNS)
        if len(identity) != 1296 or not np.array_equal(
            identity["ordered_position"].to_numpy(), np.arange(504, 1800)
        ):
            raise ProbabilisticContractError("spent common identity differs")
        point = base_identity.copy()
        point["point_expected_pe"] = pd.to_numeric(
            output["v04_expected_pe"], errors="coerce"
        ).to_numpy()
        for comparator_id in BASELINE_COLUMNS:
            comparator = identity.copy()
            values = pd.to_numeric(output[comparator_id], errors="coerce").to_numpy()[504:]
            if not np.isfinite(values).all() or (values <= 0.0).any():
                raise ProbabilisticContractError(f"{comparator_id} common-mask values are invalid")
            comparator["expected_pe"] = values
            comparator_parts[comparator_id].append(comparator)
        feature_parts.append(features)
        label_parts.append(labels)
        identity_parts.append(identity)
        point_parts.append(point)
        source_records.append(
            {
                "seed": seed,
                "canonical_raw_sha256": entry["canonical_csv"]["sha256"],
                "output_raw_sha256": entry["output_csv"]["sha256"],
                "source_record_bytes": entry["source_record_bytes"],
                "source_record_sha256": entry["source_record_sha256"],
            }
        )
    return {
        "features": pd.concat(feature_parts, ignore_index=True),
        "labels": pd.concat(label_parts, ignore_index=True),
        "identity": pd.concat(identity_parts, ignore_index=True),
        "point_history": pd.concat(point_parts, ignore_index=True),
        "v04_expected_pe": pd.concat(comparator_parts["v04_expected_pe"], ignore_index=True),
        "ml_expected_pe": pd.concat(comparator_parts["ml_expected_pe"], ignore_index=True),
        "role": seal_payload(
            {
                "schema_version": "expected_pe_model_zoo.probabilistic_spent_predict_role.v1",
                "role": "TRUTH_BLIND_PREDICTION_INPUT_CUSTODY",
                "predict_manifest_raw_sha256": sha256_bytes(manifest_raw),
                "predict_manifest_logical_sha256": manifest["manifest_sha256"],
                "source_records": source_records,
                "spent_seeds": list(SPENT_SEEDS),
                "evaluation_input_opened": False,
                "fresh_seed_reserved_or_opened": False,
                "heldout_opened": False,
            }
        ),
    }


def assemble_detached_truth(
    repo_root: Path, identity: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any], bytes]:
    """Evaluator-only assembly of exact truth rows after prediction custody exists."""

    manifest_raw, manifest = _load_manifest(repo_root, role="evaluate")
    if (
        manifest.get("mode") != "evaluate_inputs"
        or manifest.get("design_lock_sha256") != WAVE1_DESIGN_SHA256
        or manifest.get("predict_process_must_not_receive_this_manifest") is not True
        or tuple(item.get("seed") for item in manifest.get("seeds", ())) != SPENT_SEEDS
    ):
        raise ProbabilisticContractError("evaluation-only spent manifest role differs")
    truth_parts: list[pd.DataFrame] = []
    source_records: list[dict[str, Any]] = []
    for entry in manifest["seeds"]:
        if set(entry) != {"seed", "truth_csv", "expected_rows"}:
            raise ProbabilisticContractError("spent truth record schema differs")
        seed = int(entry["seed"])
        _, truth = _read_sealed_csv_once(
            entry["truth_csv"], context=f"seed {seed} evaluation truth"
        )
        if len(truth) != 1800 or tuple(truth.columns).count("true_fair_pe") != 1:
            raise ProbabilisticContractError("spent truth schema/row count differs")
        dates = pd.to_datetime(truth["date"], errors="coerce", utc=True).dt.tz_convert(None)
        values = pd.to_numeric(truth["true_fair_pe"], errors="coerce").to_numpy(dtype=np.float64)
        expected = identity.loc[identity["seed"] == seed].reset_index(drop=True)
        if (
            dates.isna().any()
            or not np.array_equal(dates.iloc[504:].to_numpy(), expected["date"].to_numpy())
            or not np.isfinite(values[504:]).all()
            or (values[504:] <= 0.0).any()
        ):
            raise ProbabilisticContractError("spent truth common identity/values differ")
        selected = expected.copy()
        selected["true_fair_pe"] = values[504:]
        truth_parts.append(selected)
        source_records.append(
            {
                "seed": seed,
                "truth_raw_sha256": entry["truth_csv"]["sha256"],
                "truth_bytes": entry["truth_csv"]["bytes"],
            }
        )
    combined = pd.concat(truth_parts, ignore_index=True)
    if not combined.loc[:, list(IDENTITY_COLUMNS)].equals(identity.reset_index(drop=True)):
        raise ProbabilisticContractError("detached truth does not equal formal identity")
    role = seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.probabilistic_spent_truth_role.v1",
            "role": "EVALUATOR_ONLY_DETACHED_TRUTH_CUSTODY",
            "evaluate_manifest_raw_sha256": sha256_bytes(manifest_raw),
            "evaluate_manifest_logical_sha256": manifest["manifest_sha256"],
            "source_records": source_records,
            "spent_seeds": list(SPENT_SEEDS),
            "prediction_code_imported": False,
            "fresh_seed_reserved_or_opened": False,
            "heldout_opened": False,
        }
    )
    return combined, role, manifest_raw
