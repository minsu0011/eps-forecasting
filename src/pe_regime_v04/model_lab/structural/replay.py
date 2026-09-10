"""Exact, factory-controlled spent-base replay for nested Structural OOF plans.

This module intentionally has no API accepting caller predictions.  The only
formal inputs are a factory-created task and a live FORMAL_SPENT capability.
The capability is currently fail-closed until the repeat independent audit GO.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping
import warnings

import numpy as np
import pandas as pd

from ..contracts import FitContext, PredictContext
from ..models.wave1.adapters import build_wave1_model
from ..models.wave1.artifacts import (
    inspect_predict_seed_entry,
    load_predict_inputs,
    load_wave1_seed_frames,
)
from ..models.wave1.spec import MODEL_BY_ID, feature_metadata
from .authorization import EXPECTED_PAIR, StructuralExecutionAuthorization
from .contracts import (
    StructuralContractError,
    canonical_json_bytes,
    normalize_dates,
    seal_payload,
    sha256_bytes,
    sha256_file,
    verify_payload_seal,
)
from .nested import NestedOOFTaskPayload


PREDICT_INPUTS_RELATIVE = Path("outputs/model_zoo_wave1_screen_20260819/PREDICT_INPUTS.json")
V04_MODEL_ID = "v04_expected_pe"
V04_RAW_COLUMN = "v04_expected_pe"
V04_ALIGNMENT = "prediction_at_t_equals_raw_overlay_output_at_t_minus_1"
V04_CANDIDATE_ID = "causal_matured_forward_median_regularized_promotion_v1"
_REPLAY_RESULT_TOKEN = object()
_V04_SURFACE_CACHE: dict[tuple[str, int], tuple[np.ndarray, pd.Series, dict[str, Any]]] = {}


@dataclass(frozen=True, init=False)
class ExactBaseTaskReplay:
    """Opaque result produced only by an exact locked adapter execution."""

    predictions: tuple[float, ...]
    base_train_end_isos: tuple[str, ...]
    receipt: dict[str, Any]

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError("ExactBaseTaskReplay is factory-only")

    @classmethod
    def _mint(
        cls,
        *,
        token: object,
        predictions: np.ndarray,
        base_train_end_isos: tuple[str, ...],
        receipt: Mapping[str, Any],
    ) -> "ExactBaseTaskReplay":
        if token is not _REPLAY_RESULT_TOKEN:
            raise StructuralContractError("invalid exact-replay mint token")
        values = np.asarray(predictions, dtype=np.float64)
        if (
            values.ndim != 1
            or len(values) != len(base_train_end_isos)
            or not np.isfinite(values).all()
            or (values <= 0.0).any()
        ):
            raise StructuralContractError("exact replay did not produce full positive coverage")
        train_ends = normalize_dates(base_train_end_isos, context="exact replay train ends")
        verify_payload_seal(receipt, field="receipt_sha256")
        output = object.__new__(cls)
        object.__setattr__(output, "predictions", tuple(float(value) for value in values))
        object.__setattr__(
            output,
            "base_train_end_isos",
            tuple(pd.Timestamp(value).isoformat() for value in train_ends),
        )
        object.__setattr__(output, "receipt", dict(receipt))
        return output


def _prediction_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or not np.isfinite(array).all() or (array <= 0.0).any():
        raise StructuralContractError("exact replay prediction bytes are invalid")
    return array.tobytes(order="C")


def _load_base_lock(authorization: StructuralExecutionAuthorization) -> dict[str, Any]:
    try:
        value = json.loads(authorization.base_binding_lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralContractError("V3 base replay lock is unreadable") from exc
    if not isinstance(value, dict):
        raise StructuralContractError("V3 base replay lock must be an object")
    verify_payload_seal(value)
    return value


def _seed_entry(
    task: NestedOOFTaskPayload,
    authorization: StructuralExecutionAuthorization,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    manifest_path = (authorization.project_root / PREDICT_INPUTS_RELATIVE).resolve(strict=True)
    lock = _load_base_lock(authorization)
    evidence = lock.get("binding_evidence")
    if not isinstance(evidence, Mapping):
        raise StructuralContractError("base lock replay evidence is missing")
    input_record = evidence.get("predict_inputs")
    if (
        not isinstance(input_record, Mapping)
        or input_record.get("path") != PREDICT_INPUTS_RELATIVE.as_posix()
        or int(input_record.get("bytes", -1)) != manifest_path.stat().st_size
        or input_record.get("sha256") != sha256_file(manifest_path)
    ):
        raise StructuralContractError("frozen predict-input custody changed")
    manifest = load_predict_inputs(manifest_path)
    matches = [row for row in manifest["seeds"] if int(row["seed"]) == task.seed]
    if len(matches) != 1:
        raise StructuralContractError("task seed is outside the exact spent-input universe")
    surfaces = evidence.get("spent_surfaces")
    if not isinstance(surfaces, list):
        raise StructuralContractError("base lock spent-surface evidence is missing")
    bound = [row for row in surfaces if int(row.get("seed", -1)) == task.seed]
    if len(bound) != 1:
        raise StructuralContractError("task seed has no unique replay-surface binding")
    entry = matches[0]
    if (
        bound[0].get("canonical_csv") != entry["canonical_csv"]
        or bound[0].get("output_csv") != entry["output_csv"]
    ):
        raise StructuralContractError("spent replay-surface file custody changed")
    return entry, lock


def _require_task_surface_identity(
    task: NestedOOFTaskPayload,
    dates: pd.Series,
) -> None:
    for position, date_text in (*task.train_records, *task.test_records):
        if position < 0 or position >= len(dates):
            raise StructuralContractError("nested task position is outside the spent surface")
        if pd.Timestamp(dates.iloc[position]) != pd.Timestamp(date_text):
            raise StructuralContractError("nested task position/date differs from spent input")


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), Mapping):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def _verify_source_manifest(
    authorization: StructuralExecutionAuthorization,
    replay_evidence: Mapping[str, Any],
) -> None:
    rows = replay_evidence.get("source_inventory")
    if not isinstance(rows, list) or not rows:
        raise StructuralContractError("v04 replay source inventory is missing")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "bytes", "sha256"}:
            raise StructuralContractError("v04 replay source inventory schema changed")
        path = authorization.project_root / str(row["path"])
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise StructuralContractError(f"v04 replay source changed: {row['path']}")
        normalized.append(dict(row))
    combined = sha256_bytes(canonical_json_bytes(normalized))
    if combined != authorization.residual_base.source_sha256:
        raise StructuralContractError("v04 replay source manifest hash changed")


def _recompute_v04_surface(
    task: NestedOOFTaskPayload,
    authorization: StructuralExecutionAuthorization,
    inspected: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> tuple[np.ndarray, pd.Series, dict[str, Any]]:
    if task.base_model_id != V04_MODEL_ID:
        raise StructuralContractError("v04 replay received a substituted base")
    evidence = lock["binding_evidence"]
    replay_evidence = evidence.get("residual_v04_replay")
    if not isinstance(replay_evidence, Mapping):
        raise StructuralContractError("v04 exact replay evidence is missing")
    _verify_source_manifest(authorization, replay_evidence)
    config_record = replay_evidence.get("config")
    design_record = replay_evidence.get("design_lock")
    candidate_record = replay_evidence.get("candidate_overrides")
    for name, record in (
        ("config", config_record),
        ("design lock", design_record),
        ("candidate overrides", candidate_record),
    ):
        if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
            raise StructuralContractError(f"v04 {name} binding schema changed")
        path = authorization.project_root / str(record["path"])
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise StructuralContractError(f"v04 {name} bytes changed")
    if config_record["sha256"] != authorization.residual_base.config_sha256:
        raise StructuralContractError("v04 base config hash differs from authorization")

    from pe_regime_v04.config import load_config
    from pe_regime_v04.pipeline import apply_v04_layers

    config = load_config(authorization.project_root / str(config_record["path"]))
    candidate_payload = json.loads(
        (authorization.project_root / str(candidate_record["path"])).read_text(encoding="utf-8")
    )
    try:
        overrides = candidate_payload["candidates"][V04_CANDIDATE_ID]["overrides"]
    except (KeyError, TypeError) as exc:
        raise StructuralContractError("v04 candidate override contract changed") from exc
    config = _deep_merge(config, overrides)
    model_seed_by_seed = replay_evidence.get("model_seed_by_spent_seed")
    if not isinstance(model_seed_by_seed, Mapping) or str(task.seed) not in model_seed_by_seed:
        raise StructuralContractError("v04 model-seed replay binding is missing")
    config.setdefault("project", {})["random_seed"] = int(model_seed_by_seed[str(task.seed)])
    for section in ("regime_stacker", "expected_pe"):
        config.setdefault(section, {})["outer_n_jobs"] = 1
        config[section]["parallel_backend"] = "serial"
        config[section]["n_jobs"] = 1

    canonical = inspected["canonical"].copy(deep=True)
    before = canonical.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        replayed, _diagnostics = apply_v04_layers(canonical, config)
    pd.testing.assert_frame_equal(before, canonical, check_exact=True, check_dtype=True)
    custody_raw = pd.to_numeric(inspected["output"][V04_RAW_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    replayed_raw = pd.to_numeric(replayed[V04_RAW_COLUMN], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if not np.array_equal(custody_raw, replayed_raw, equal_nan=True):
        raise StructuralContractError("v04 live replay differs from frozen output254 bytes")
    dates = pd.to_datetime(inspected["dates"], errors="raise")
    aligned = np.roll(replayed_raw, 1)
    aligned[0] = np.nan
    return (
        aligned,
        dates,
        {
            "adapter": "pe_regime_v04.pipeline.apply_v04_layers",
            "alignment": V04_ALIGNMENT,
            "raw_column": V04_RAW_COLUMN,
            "canonical_csv_sha256": sha256_file(inspected["canonical_path"]),
            "output_csv_sha256": sha256_file(inspected["output_path"]),
            "raw_replay_bytes_sha256": sha256_bytes(
                np.asarray(replayed_raw, dtype="<f8").tobytes()
            ),
            "raw_custody_bytes_sha256": sha256_bytes(
                np.asarray(custody_raw, dtype="<f8").tobytes()
            ),
            "config_sha256": str(config_record["sha256"]),
            "design_lock_sha256": str(design_record["sha256"]),
            "candidate_overrides_sha256": str(candidate_record["sha256"]),
            "model_seed": int(model_seed_by_seed[str(task.seed)]),
            "inner_native_threads": 1,
            "truth_access": False,
        },
    )


def _v04_task_replay(
    task: NestedOOFTaskPayload,
    authorization: StructuralExecutionAuthorization,
) -> ExactBaseTaskReplay:
    entry, lock = _seed_entry(task, authorization)
    inspected = inspect_predict_seed_entry(entry)
    dates = pd.to_datetime(inspected["dates"], errors="raise")
    _require_task_surface_identity(task, dates)
    cache_key = (authorization.authorization_sha256, task.seed)
    cached = _V04_SURFACE_CACHE.get(cache_key)
    if cached is None:
        aligned, dates, adapter_evidence = _recompute_v04_surface(
            task, authorization, inspected, lock
        )
        aligned.setflags(write=False)
        _V04_SURFACE_CACHE[cache_key] = (aligned, dates.copy(deep=True), adapter_evidence)
    else:
        aligned, dates, adapter_evidence = cached
    positions = np.asarray(task.test_positions, dtype=np.int64)
    if (positions <= 0).any():
        raise StructuralContractError("t-minus-one v04 replay cannot predict position zero")
    predictions = aligned[positions]
    train_ends = tuple(pd.Timestamp(dates.iloc[position - 1]).isoformat() for position in positions)
    return _finalize_replay(task, authorization, predictions, train_ends, adapter_evidence)


def _wave1_task_replay(
    task: NestedOOFTaskPayload,
    authorization: StructuralExecutionAuthorization,
) -> ExactBaseTaskReplay:
    if task.base_model_id not in EXPECTED_PAIR:
        raise StructuralContractError("Wave1 replay received an unselected base")
    authorization.require_pair(
        task.candidate_id,
        authorization.pair[0].model_id,
        authorization.pair[1].model_id,
    )
    entry, _lock = _seed_entry(task, authorization)
    inspected = inspect_predict_seed_entry(entry)
    dates = pd.to_datetime(inspected["dates"], errors="raise")
    _require_task_surface_identity(task, dates)
    model_frame, _baselines = load_wave1_seed_frames(entry)
    definition = MODEL_BY_ID[task.base_model_id]
    train = model_frame.iloc[list(task.train_positions)]
    test = model_frame.iloc[list(task.test_positions)]
    target = pd.to_numeric(train["observed_pe"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    eligible = np.isfinite(target) & (target > 0.0)
    if int(eligible.sum()) < 252:
        raise StructuralContractError("exact Wave1 nested replay has insufficient train labels")
    train = train.loc[eligible]
    fold_seed = task.seed + task.test_positions[0]
    model = build_wave1_model(task.base_model_id, fold_seed=fold_seed)
    features = tuple(definition.feature_columns)
    x_train = train.loc[:, list(features)].copy()
    x_test = test.loc[:, list(features)].copy()
    fit_context = FitContext(
        experiment_id="structural-wave-v3-exact-nested-oof",
        fold_id=task.task_id,
        seed=fold_seed,
        train_end=pd.Timestamp(task.train_end_iso),
        target_name="observed_pe",
        feature_metadata=feature_metadata(features),
    )
    predict_context = PredictContext(
        experiment_id="structural-wave-v3-exact-nested-oof",
        fold_id=task.task_id,
        seed=fold_seed,
        prediction_start=pd.Timestamp(task.test_start_iso),
        prediction_end=pd.Timestamp(task.test_end_iso),
        feature_metadata=feature_metadata(features),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.fit(x_train, train["observed_pe"], context=fit_context)
        prediction = model.predict(x_test, context=predict_context)
    values = prediction.to_numpy(dtype=np.float64)
    diagnostics = model.diagnostics()
    train_ends = tuple(task.train_end_iso for _ in task.test_positions)
    adapter_evidence = {
        "adapter": "pe_regime_v04.model_lab.models.wave1.adapters.build_wave1_model",
        "model_id": task.base_model_id,
        "fold_seed": fold_seed,
        "eligible_train_rows": int(eligible.sum()),
        "feature_ids": list(features),
        "active_features": list(diagnostics.active_features),
        "dropped_all_missing_features": list(diagnostics.dropped_all_missing_features),
        "dropped_constant_spline_features": list(diagnostics.dropped_constant_spline_features),
        "resolved_parameters_sha256": sha256_bytes(
            canonical_json_bytes(diagnostics.resolved_parameters)
        ),
        "canonical_csv_sha256": sha256_file(inspected["canonical_path"]),
        "output_csv_sha256": sha256_file(inspected["output_path"]),
        "within_test_target_updates": False,
        "truth_access": False,
    }
    return _finalize_replay(task, authorization, values, train_ends, adapter_evidence)


def _finalize_replay(
    task: NestedOOFTaskPayload,
    authorization: StructuralExecutionAuthorization,
    predictions: np.ndarray,
    train_ends: tuple[str, ...],
    adapter_evidence: Mapping[str, Any],
) -> ExactBaseTaskReplay:
    values = np.asarray(predictions, dtype=np.float64)
    test_dates = normalize_dates(
        [date for _, date in task.test_records], context="exact replay test dates"
    )
    normalized_train_ends = normalize_dates(train_ends, context="exact replay train ends")
    if len(values) != len(test_dates) or not (normalized_train_ends < test_dates).all():
        raise StructuralContractError("exact replay temporal provenance is invalid")
    receipt = seal_payload(
        {
            "format_version": 3,
            "execution_scope": "FORMAL_SPENT",
            "execution_mode": "EXACT_LOCKED_BASE_REPLAY_NO_CALLER_PREDICTIONS",
            "task_sha256": task.task_sha256,
            "role": task.role,
            "task_id": task.task_id,
            "candidate_id": task.candidate_id,
            "seed": task.seed,
            "outer_fold_id": task.outer_fold_id,
            "outer_cutoff_iso": task.outer_cutoff_iso,
            "train_identity_sha256": task.train_identity_sha256,
            "test_identity_sha256": task.test_identity_sha256,
            "train_end_iso": task.train_end_iso,
            "test_start_iso": task.test_start_iso,
            "test_end_iso": task.test_end_iso,
            "base_model_id": task.base_model_id,
            "base_source_sha256": task.base_source_sha256,
            "base_config_sha256": task.base_config_sha256,
            "environment_sha256": task.environment_sha256,
            "feature_registry_sha256": task.feature_registry_sha256,
            "authorization_sha256": authorization.authorization_sha256,
            "rowwise_base_train_end_isos": list(train_ends),
            "rowwise_base_train_end_sha256": sha256_bytes(canonical_json_bytes(list(train_ends))),
            "prediction_count": len(values),
            "prediction_bytes_sha256": sha256_bytes(_prediction_bytes(values)),
            "adapter_evidence": dict(adapter_evidence),
            "adapter_evidence_sha256": sha256_bytes(canonical_json_bytes(adapter_evidence)),
        },
        field="receipt_sha256",
    )
    return ExactBaseTaskReplay._mint(
        token=_REPLAY_RESULT_TOKEN,
        predictions=values,
        base_train_end_isos=train_ends,
        receipt=receipt,
    )


def execute_exact_spent_base_task(
    task: NestedOOFTaskPayload,
    authorization: StructuralExecutionAuthorization,
) -> ExactBaseTaskReplay:
    """Run exactly one plan task through its authorization-selected base adapter."""

    if not isinstance(task, NestedOOFTaskPayload):
        raise StructuralContractError("formal replay requires a factory-created nested task")
    if not isinstance(authorization, StructuralExecutionAuthorization):
        raise StructuralContractError("formal replay requires sealed authorization")
    authorization.require_formal_spent_go()
    if task.base_model_id == V04_MODEL_ID:
        authorization.require_residual_base(task.candidate_id, task.base_model_id)
        return _v04_task_replay(task, authorization)
    if task.base_model_id in EXPECTED_PAIR:
        return _wave1_task_replay(task, authorization)
    raise StructuralContractError("no exact replay adapter exists for the requested base")
