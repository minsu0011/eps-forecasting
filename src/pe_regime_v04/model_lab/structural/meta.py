"""Plan-executed, factory-only generated meta-feature artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from .authorization import StructuralExecutionAuthorization
from .contracts import (
    FULL_META_IDENTITY_COLUMNS,
    STRUCTURAL_DESIGN_SHA256,
    StructuralContractError,
    canonical_json_bytes,
    logical_frame_sha256,
    normalize_dates,
    require_positive_finite,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .nested import NestedOOFPlan, NestedOOFTaskPayload


MetaArtifactRole = Literal["INNER_OOS_FIT", "OUTER_TEST_PREDICT"]
META_ROLES = ("INNER_OOS_FIT", "OUTER_TEST_PREDICT")
OUTER_TEST_INNER_FOLD_SENTINEL = "__OUTER_TEST__"
META_FRAME_COLUMNS = (
    "seed",
    "date",
    "outer_fold_id",
    "inner_fold_id",
    "base_model_id",
    "base_source_sha256",
    "base_config_sha256",
    "base_train_end",
    "session_position",
    "base_prediction",
)
_ARTIFACT_TOKEN = object()


@dataclass(frozen=True)
class RecomputedContentBinding:
    """Legacy compatibility value; it cannot mint a verified artifact."""

    base_source_bytes: bytes
    base_config_bytes: bytes
    environment_bytes: bytes

    @property
    def base_source_sha256(self) -> str:
        return sha256_bytes(self.base_source_bytes)

    @property
    def base_config_sha256(self) -> str:
        return sha256_bytes(self.base_config_bytes)

    @property
    def environment_sha256(self) -> str:
        return sha256_bytes(self.environment_bytes)


def _normalize_meta_frame(frame: pd.DataFrame, *, role: MetaArtifactRole) -> pd.DataFrame:
    if role not in META_ROLES:
        raise StructuralContractError("generated meta role is invalid")
    if frame.columns.has_duplicates or tuple(frame.columns) != META_FRAME_COLUMNS:
        raise StructuralContractError(f"generated meta schema must be exact: {META_FRAME_COLUMNS}")
    if frame.empty:
        raise StructuralContractError("generated meta artifact cannot be empty")
    output = frame.copy()
    output["date"] = normalize_dates(output["date"], context="meta date").to_numpy(copy=True)
    output["base_train_end"] = normalize_dates(
        output["base_train_end"], context="meta train end"
    ).to_numpy(copy=True)
    if not (output["base_train_end"] < output["date"]).all():
        raise StructuralContractError("every base train end must strictly precede its row")
    if output.loc[:, list(FULL_META_IDENTITY_COLUMNS)].isna().any().any():
        raise StructuralContractError("generated meta identity contains missing values")
    if output.duplicated(list(FULL_META_IDENTITY_COLUMNS)).any():
        raise StructuralContractError("generated meta identity must be one-to-one")
    positions = pd.to_numeric(output["session_position"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    if (
        not np.isfinite(positions).all()
        or not np.equal(positions, np.floor(positions)).all()
        or (positions < 0).any()
    ):
        raise StructuralContractError("meta session positions must be non-negative integers")
    output["session_position"] = positions.astype(np.int64)
    output["base_prediction"] = require_positive_finite(
        output["base_prediction"], context="generated base prediction"
    )
    expected_inner = (
        output["inner_fold_id"].ne(OUTER_TEST_INNER_FOLD_SENTINEL).all()
        if role == "INNER_OOS_FIT"
        else output["inner_fold_id"].eq(OUTER_TEST_INNER_FOLD_SENTINEL).all()
    )
    if not expected_inner:
        raise StructuralContractError("generated meta role/fold sentinel mismatch")
    if not output["session_position"].is_monotonic_increasing:
        raise StructuralContractError("generated meta session order changed")
    return output.reset_index(drop=True)


def _prediction_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or not np.isfinite(array).all() or (array <= 0).any():
        raise StructuralContractError("prediction byte binding requires positive finite vector")
    return array.tobytes(order="C")


def _synthetic_predictions(task: NestedOOFTaskPayload) -> np.ndarray:
    """Deterministic no-target replay used only for score-free provenance tests."""

    values: list[float] = []
    for position, date in task.test_records:
        digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "mode": "SYNTHETIC_NO_SCORE",
                    "task_sha256": task.task_sha256,
                    "position": position,
                    "date": date,
                    "base_model_id": task.base_model_id,
                }
            )
        )
        values.append(8.0 + int(digest[:12], 16) / float(16**12) * 8.0)
    return np.asarray(values, dtype=np.float64)


def _synthetic_receipt(task: NestedOOFTaskPayload, values: np.ndarray) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "execution_scope": "SYNTHETIC_NO_SCORE",
        "task_sha256": task.task_sha256,
        "role": task.role,
        "task_id": task.task_id,
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
        "prediction_count": len(values),
        "prediction_bytes_sha256": sha256_bytes(_prediction_bytes(values)),
    }
    return seal_payload(payload, field="receipt_sha256")


def _task_rows(
    task: NestedOOFTaskPayload,
    values: np.ndarray,
    *,
    base_train_end_isos: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    inner_fold_id = task.task_id if task.role == "INNER_OOS_FIT" else OUTER_TEST_INNER_FOLD_SENTINEL
    train_ends = base_train_end_isos or tuple(task.train_end_iso for _ in task.test_records)
    if len(train_ends) != len(task.test_records):
        raise StructuralContractError("base train-end provenance row count differs")
    return pd.DataFrame(
        {
            "seed": task.seed,
            "date": [date for _, date in task.test_records],
            "outer_fold_id": task.outer_fold_id,
            "inner_fold_id": inner_fold_id,
            "base_model_id": task.base_model_id,
            "base_source_sha256": task.base_source_sha256,
            "base_config_sha256": task.base_config_sha256,
            "base_train_end": list(train_ends),
            "session_position": [position for position, _ in task.test_records],
            "base_prediction": values,
        }
    ).loc[:, list(META_FRAME_COLUMNS)]


@dataclass(frozen=True, init=False)
class GeneratedMetaFeatureArtifact:
    """Opaque artifact whose rows are reconstructed from a verified plan."""

    role: MetaArtifactRole
    execution_scope: str
    plan: NestedOOFPlan
    authorization_sha256: str
    csv_text: str
    row_count: int
    task_receipts: tuple[dict[str, Any], ...]
    task_receipt_manifest_sha256: str
    identity_sha256: str
    fold_sha256: str
    base_model_id: str
    base_source_sha256: str
    base_config_sha256: str
    environment_sha256: str
    prediction_bytes_sha256: str
    prediction_content_sha256: str
    artifact_sha256: str
    design_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError(
            "GeneratedMetaFeatureArtifact is factory-only; execute a NestedOOFPlan"
        )

    @classmethod
    def _mint(cls, *, token: object, **values: Any) -> "GeneratedMetaFeatureArtifact":
        if token is not _ARTIFACT_TOKEN:
            raise StructuralContractError("invalid generated-meta mint token")
        output = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(output, key, value)
        return output

    def to_frame(self) -> pd.DataFrame:
        try:
            frame = pd.read_csv(StringIO(self.csv_text), float_precision="round_trip")
        except Exception as exc:
            raise StructuralContractError("generated meta CSV is unreadable") from exc
        return _normalize_meta_frame(frame, role=self.role)

    def _unsigned(self) -> dict[str, Any]:
        return {
            "format_version": 2,
            "mode": "plan_executed_generated_meta_feature",
            "role": self.role,
            "execution_scope": self.execution_scope,
            "plan_sha256": self.plan.plan_sha256,
            "authorization_sha256": self.authorization_sha256,
            "row_count": self.row_count,
            "task_receipts": list(self.task_receipts),
            "task_receipt_manifest_sha256": self.task_receipt_manifest_sha256,
            "identity_sha256": self.identity_sha256,
            "fold_sha256": self.fold_sha256,
            "base_model_id": self.base_model_id,
            "base_source_sha256": self.base_source_sha256,
            "base_config_sha256": self.base_config_sha256,
            "environment_sha256": self.environment_sha256,
            "prediction_bytes_sha256": self.prediction_bytes_sha256,
            "prediction_content_sha256": self.prediction_content_sha256,
            "csv_sha256": sha256_bytes(self.csv_text.encode("utf-8")),
            "design_sha256": self.design_sha256,
        }

    def to_process_payload(self) -> dict[str, Any]:
        value = self._unsigned()
        value["plan"] = self.plan.to_sealed_payload()
        value["csv_text"] = self.csv_text
        value["artifact_sha256"] = self.artifact_sha256
        canonical_json_bytes(value)
        return value

    def verify(
        self,
        authorization: StructuralExecutionAuthorization,
        *,
        expected_role: MetaArtifactRole,
        expected_candidate_id: str,
    ) -> pd.DataFrame:
        if not isinstance(authorization, StructuralExecutionAuthorization):
            raise StructuralContractError("meta artifact requires sealed authorization")
        authorization.require_candidate(expected_candidate_id)
        self.plan.verify(authorization)
        if self.execution_scope != authorization.scope:
            raise StructuralContractError("meta execution scope is not authorized")
        if self.authorization_sha256 != authorization.authorization_sha256:
            raise StructuralContractError("meta artifact authorization binding changed")
        if self.role != expected_role or self.plan.candidate_id != expected_candidate_id:
            raise StructuralContractError("meta artifact role/candidate mismatch")
        if self.design_sha256 != STRUCTURAL_DESIGN_SHA256:
            raise StructuralContractError("meta artifact design changed")
        tasks = self.plan.tasks if self.role == "INNER_OOS_FIT" else (self.plan.outer_task,)
        expected_frames: list[pd.DataFrame] = []
        expected_receipts: list[dict[str, Any]] = []
        if self.execution_scope == "SYNTHETIC_NO_SCORE":
            for task in tasks:
                predictions = _synthetic_predictions(task)
                expected_frames.append(_task_rows(task, predictions))
                expected_receipts.append(_synthetic_receipt(task, predictions))
        elif self.execution_scope == "FORMAL_SPENT":
            authorization.require_formal_spent_go()
            from .replay import execute_exact_spent_base_task

            for task in tasks:
                replay = execute_exact_spent_base_task(task, authorization)
                expected_frames.append(
                    _task_rows(
                        task,
                        np.asarray(replay.predictions, dtype=np.float64),
                        base_train_end_isos=replay.base_train_end_isos,
                    )
                )
                expected_receipts.append(replay.receipt)
        else:
            raise StructuralContractError("unknown generated-meta execution scope")
        expected = _normalize_meta_frame(
            pd.concat(expected_frames, ignore_index=True), role=self.role
        )
        actual = self.to_frame()
        if not expected.equals(actual):
            raise StructuralContractError(
                "meta rows cannot be reconstructed from the executable nested plan"
            )
        if tuple(expected_receipts) != self.task_receipts:
            raise StructuralContractError("meta task execution receipts changed")
        receipt_manifest = sha256_bytes(canonical_json_bytes(expected_receipts))
        if receipt_manifest != self.task_receipt_manifest_sha256:
            raise StructuralContractError("meta task receipt manifest changed")
        identity = actual.loc[:, list(FULL_META_IDENTITY_COLUMNS)]
        if logical_frame_sha256(identity) != self.identity_sha256:
            raise StructuralContractError("meta identity hash changed")
        if self.fold_sha256 != self.plan.fold_sha256:
            raise StructuralContractError("meta fold hash is not the plan fold hash")
        prediction = actual["base_prediction"].to_numpy(dtype=np.float64)
        if sha256_bytes(_prediction_bytes(prediction)) != self.prediction_bytes_sha256:
            raise StructuralContractError("meta prediction bytes changed")
        content = actual.loc[:, [*FULL_META_IDENTITY_COLUMNS, "base_prediction"]]
        if logical_frame_sha256(content) != self.prediction_content_sha256:
            raise StructuralContractError("meta prediction content changed")
        if (
            self.base_model_id,
            self.base_source_sha256,
            self.base_config_sha256,
            self.environment_sha256,
        ) != (
            self.plan.base_model_id,
            self.plan.base_source_sha256,
            self.plan.base_config_sha256,
            self.plan.environment_sha256,
        ):
            raise StructuralContractError("meta base replay binding changed")
        if sha256_bytes(canonical_json_bytes(self._unsigned())) != self.artifact_sha256:
            raise StructuralContractError("meta artifact seal changed")
        return actual


def _mint_executed_artifact(
    *,
    plan: NestedOOFPlan,
    authorization: StructuralExecutionAuthorization,
    role: MetaArtifactRole,
    frame: pd.DataFrame,
    receipts: list[dict[str, Any]],
) -> GeneratedMetaFeatureArtifact:
    frame = _normalize_meta_frame(frame, role=role)
    csv_text = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )
    round_trip = _normalize_meta_frame(
        pd.read_csv(StringIO(csv_text), float_precision="round_trip"), role=role
    )
    values: dict[str, Any] = {
        "role": role,
        "execution_scope": authorization.scope,
        "plan": plan,
        "authorization_sha256": authorization.authorization_sha256,
        "csv_text": csv_text,
        "row_count": len(round_trip),
        "task_receipts": tuple(receipts),
        "task_receipt_manifest_sha256": sha256_bytes(canonical_json_bytes(receipts)),
        "identity_sha256": logical_frame_sha256(
            round_trip.loc[:, list(FULL_META_IDENTITY_COLUMNS)]
        ),
        "fold_sha256": plan.fold_sha256,
        "base_model_id": plan.base_model_id,
        "base_source_sha256": plan.base_source_sha256,
        "base_config_sha256": plan.base_config_sha256,
        "environment_sha256": plan.environment_sha256,
        "prediction_bytes_sha256": sha256_bytes(
            _prediction_bytes(round_trip["base_prediction"].to_numpy(dtype=np.float64))
        ),
        "prediction_content_sha256": logical_frame_sha256(
            round_trip.loc[:, [*FULL_META_IDENTITY_COLUMNS, "base_prediction"]]
        ),
        "design_sha256": STRUCTURAL_DESIGN_SHA256,
    }
    provisional = GeneratedMetaFeatureArtifact._mint(
        token=_ARTIFACT_TOKEN, **values, artifact_sha256="0" * 64
    )
    values["artifact_sha256"] = sha256_bytes(canonical_json_bytes(provisional._unsigned()))
    output = GeneratedMetaFeatureArtifact._mint(token=_ARTIFACT_TOKEN, **values)
    output.verify(authorization, expected_role=role, expected_candidate_id=plan.candidate_id)
    return output


def execute_nested_oof_plan_score_free(
    plan: NestedOOFPlan,
    authorization: StructuralExecutionAuthorization,
    *,
    role: MetaArtifactRole,
) -> GeneratedMetaFeatureArtifact:
    """Execute the fixed no-target replay; arbitrary prediction input is impossible."""

    if not isinstance(plan, NestedOOFPlan):
        raise StructuralContractError("nested execution requires a factory-created plan")
    if authorization.scope != "SYNTHETIC_NO_SCORE":
        raise StructuralContractError("score-free executor requires SYNTHETIC_NO_SCORE scope")
    plan.verify(authorization)
    if role not in META_ROLES:
        raise StructuralContractError("nested execution role is invalid")
    tasks = plan.tasks if role == "INNER_OOS_FIT" else (plan.outer_task,)
    if not tasks:
        raise StructuralContractError("nested plan has no tasks for the requested role")
    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    for task in tasks:
        predictions = _synthetic_predictions(task)
        frames.append(_task_rows(task, predictions))
        receipts.append(_synthetic_receipt(task, predictions))
    return _mint_executed_artifact(
        plan=plan,
        authorization=authorization,
        role=role,
        frame=pd.concat(frames, ignore_index=True),
        receipts=receipts,
    )


def execute_nested_oof_plan_formal_spent(
    plan: NestedOOFPlan,
    authorization: StructuralExecutionAuthorization,
    *,
    role: MetaArtifactRole,
) -> GeneratedMetaFeatureArtifact:
    """Execute only authorization-selected locked bases; no prediction input exists."""

    if not isinstance(plan, NestedOOFPlan):
        raise StructuralContractError("nested execution requires a factory-created plan")
    if not isinstance(authorization, StructuralExecutionAuthorization):
        raise StructuralContractError("formal nested execution requires sealed authorization")
    authorization.require_formal_spent_go()
    plan.verify(authorization)
    if role not in META_ROLES:
        raise StructuralContractError("nested execution role is invalid")
    tasks = plan.tasks if role == "INNER_OOS_FIT" else (plan.outer_task,)
    if not tasks:
        raise StructuralContractError("nested plan has no tasks for the requested role")

    from .replay import execute_exact_spent_base_task

    frames: list[pd.DataFrame] = []
    receipts: list[dict[str, Any]] = []
    for task in tasks:
        replay = execute_exact_spent_base_task(task, authorization)
        frames.append(
            _task_rows(
                task,
                np.asarray(replay.predictions, dtype=np.float64),
                base_train_end_isos=replay.base_train_end_isos,
            )
        )
        receipts.append(replay.receipt)
    frame = _normalize_meta_frame(pd.concat(frames, ignore_index=True), role=role)
    return _mint_executed_artifact(
        plan=plan,
        authorization=authorization,
        role=role,
        frame=frame,
        receipts=receipts,
    )


def create_generated_meta_artifact(*_args: object, **_kwargs: object) -> None:
    """Removed unsafe API: a caller-provided frame can never assert OOF provenance."""

    raise StructuralContractError(
        "caller-minted generated meta artifacts are disabled; execute a NestedOOFPlan"
    )


@dataclass(frozen=True, init=False)
class NestedOOFCacheKey:
    key_sha256: str
    plan_sha256: str
    authorization_sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise StructuralContractError("NestedOOFCacheKey is factory-only")

    @classmethod
    def from_plan(
        cls,
        plan: NestedOOFPlan,
        authorization: StructuralExecutionAuthorization,
    ) -> "NestedOOFCacheKey":
        plan.verify(authorization)
        payload = {
            "seed": plan.seed,
            "outer_cutoff": plan.outer_cutoff_iso,
            "base_source_sha256": plan.base_source_sha256,
            "base_config_sha256": plan.base_config_sha256,
            "fold_sha256": plan.fold_sha256,
            "plan_sha256": plan.plan_sha256,
            "authorization_sha256": authorization.authorization_sha256,
        }
        output = object.__new__(cls)
        object.__setattr__(output, "key_sha256", sha256_bytes(canonical_json_bytes(payload)))
        object.__setattr__(output, "plan_sha256", plan.plan_sha256)
        object.__setattr__(output, "authorization_sha256", authorization.authorization_sha256)
        return output


class NestedOOFArtifactCache:
    """Immutable cache whose reads require the original plan and authorization."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, key: NestedOOFCacheKey) -> Path:
        if not isinstance(key, NestedOOFCacheKey):
            raise StructuralContractError("cache key must be factory-created")
        return self.root / f"{key.key_sha256}.json"

    def put(
        self,
        plan: NestedOOFPlan,
        authorization: StructuralExecutionAuthorization,
        artifact: GeneratedMetaFeatureArtifact,
    ) -> Path:
        key = NestedOOFCacheKey.from_plan(plan, authorization)
        artifact.verify(
            authorization,
            expected_role="INNER_OOS_FIT",
            expected_candidate_id=plan.candidate_id,
        )
        path = self.path_for(key)
        self.root.mkdir(parents=True, exist_ok=True)
        payload = seal_payload(
            {
                "format_version": 2,
                "mode": "immutable_plan_bound_nested_oof_cache",
                "cache_key_sha256": key.key_sha256,
                "plan_sha256": plan.plan_sha256,
                "authorization_sha256": authorization.authorization_sha256,
                "artifact": artifact.to_process_payload(),
            }
        )
        encoded = canonical_json_bytes(payload) + b"\n"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            raise
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                path.unlink(missing_ok=True)
            finally:
                raise
        return path

    def get(
        self,
        plan: NestedOOFPlan,
        authorization: StructuralExecutionAuthorization,
    ) -> GeneratedMetaFeatureArtifact:
        key = NestedOOFCacheKey.from_plan(plan, authorization)
        path = self.path_for(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StructuralContractError("nested cache artifact is unreadable") from exc
        if not isinstance(payload, Mapping):
            raise StructuralContractError("nested cache payload must be an object")
        verify_payload_seal(payload)
        if (
            payload.get("cache_key_sha256") != key.key_sha256
            or payload.get("plan_sha256") != plan.plan_sha256
            or payload.get("authorization_sha256") != authorization.authorization_sha256
        ):
            raise StructuralContractError("nested cache provenance binding changed")
        raw = payload.get("artifact")
        if not isinstance(raw, Mapping):
            raise StructuralContractError("nested cache artifact payload is missing")
        # Never trust serialized row claims: reconstruct the exact execution and
        # compare its complete sealed payload byte-for-byte.
        if authorization.scope == "SYNTHETIC_NO_SCORE":
            reconstructed = execute_nested_oof_plan_score_free(
                plan, authorization, role="INNER_OOS_FIT"
            )
        elif authorization.scope == "FORMAL_SPENT":
            reconstructed = execute_nested_oof_plan_formal_spent(
                plan, authorization, role="INNER_OOS_FIT"
            )
        else:
            raise StructuralContractError("unknown nested cache authorization scope")
        if dict(raw) != reconstructed.to_process_payload():
            raise StructuralContractError("cached artifact differs from plan replay")
        return reconstructed
