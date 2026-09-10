"""Disk-only handoff and single-use physically separated evaluator boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .artifacts import immutable_write_json
from .authorization import (
    ExecutionAuthorization,
    FORMAL_SCOPE,
    common_mask_manifest,
    load_execution_authorization,
)
from .contracts import (
    ProbabilisticContractError,
    VerifiedPredictionBatch,
    canonical_json_bytes,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .custody import (
    VerifiedPointComparatorArtifact,
    load_verified_point_comparator_with_receipt,
    load_verified_prediction_artifact,
    load_verified_reference_prediction_artifact,
    write_content_addressed_json,
)
from .formal_pins import (
    FormalLaunchAuthority,
    require_formal_launch_authority,
    verify_externally_pinned_pointer,
)
from .references import (
    REFERENCE_IDS,
    VerifiedReferenceExecutionReceipt,
    load_verified_reference_execution_receipt,
)
from .resources import VerifiedRuntimeReceipt, load_verified_runtime_receipt
from .spec import CANDIDATE_IDS


HANDOFF_SPEC_SCHEMA = "expected_pe_model_zoo.physical_evaluator_bundle_spec.v4"
HANDOFF_SCHEMA = "expected_pe_model_zoo.physical_evaluator_handoff.v4"
RESULT_SCHEMA = "expected_pe_model_zoo.physical_evaluator_result.v4"
COMPARATOR_IDS = ("v04_expected_pe", "ml_expected_pe")
_PREFLIGHT_TOKEN = object()


def _resolve(path: str | Path, *, repo_root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def _file_record(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": sha256_bytes(raw), "bytes": len(raw)}


def _verify_record(record: Mapping[str, Any]) -> Path:
    if set(record) != {"path", "sha256", "bytes"}:
        raise ProbabilisticContractError("physical handoff file record schema differs")
    path = Path(str(record["path"]))
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError("physical handoff artifact is unavailable") from exc
    if sha256_bytes(raw) != record["sha256"] or len(raw) != record["bytes"]:
        raise ProbabilisticContractError("physical handoff artifact bytes changed")
    return path


def _load_pointer(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        pointer = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("authorization pointer is unavailable or invalid") from exc
    if sha256_bytes(raw) not in path.name.split("."):
        raise ProbabilisticContractError("authorization pointer is not content-addressed")
    verify_payload_seal(pointer)
    return pointer


def _load_authorization_from_pointer(
    pointer_path: Path,
    *,
    repo_root: Path,
    launch_authority: FormalLaunchAuthority | None = None,
) -> tuple[dict[str, Any], ExecutionAuthorization]:
    pointer = _load_pointer(pointer_path)
    is_formal = pointer.get("authorization_scope") == FORMAL_SCOPE
    if is_formal:
        if launch_authority is None:
            raise ProbabilisticContractError(
                "formal pointer requires externally pinned launch authority"
            )
        launch_authority = require_formal_launch_authority(launch_authority)
        pointer = verify_externally_pinned_pointer(
            pointer_path,
            request=launch_authority.request,
        )
    elif launch_authority is not None:
        raise ProbabilisticContractError("formal launch authority cannot authorize score-free data")
    activation_path = pointer.get("formal_activation_path")
    activation_sha256 = pointer.get("formal_activation_raw_sha256")
    if is_formal and (
        not isinstance(activation_path, str) or not isinstance(activation_sha256, str)
    ):
        raise ProbabilisticContractError("formal pointer lacks its activation/policy address")
    authorization = load_execution_authorization(
        _resolve(pointer["authorization_path"], repo_root=repo_root),
        _resolve(pointer["identity_path"], repo_root=repo_root),
        _resolve(pointer["source_snapshot_path"], repo_root=repo_root),
        expected_precommit_sha256=pointer["authorization_raw_sha256"],
        require_formal=is_formal,
        formal_activation_path=(
            _resolve(activation_path, repo_root=repo_root) if is_formal else None
        ),
        expected_formal_activation_sha256=activation_sha256 if is_formal else None,
    )
    return pointer, authorization


def _verify_common_mask(path: Path, authorization: ExecutionAuthorization) -> str:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("common-mask artifact is unavailable or invalid") from exc
    digest = sha256_bytes(raw)
    if digest not in path.name.split("."):
        raise ProbabilisticContractError("common-mask raw bytes are not content-addressed")
    verify_payload_seal(payload)
    if payload != common_mask_manifest(authorization.identity_frame()):
        raise ProbabilisticContractError("common-mask payload differs from executable identity")
    if (
        sha256_bytes(canonical_json_bytes(payload))
        != authorization.bindings["common_mask_manifest_sha256"]
    ):
        raise ProbabilisticContractError("common-mask canonical bytes differ from authorization")
    return digest


def _require_exact_mapping(
    value: object, expected: tuple[str, ...], *, context: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ProbabilisticContractError(f"{context} must contain exact IDs {expected}")
    return value


def build_physical_evaluator_handoff(
    spec_path: Path,
    directory: Path,
    *,
    repo_root: Path,
    launch_authority: FormalLaunchAuthority | None = None,
) -> Path:
    """Parent-side disk preflight; never reads truth and returns only a handoff path."""

    try:
        spec_raw = Path(spec_path).read_bytes()
        spec = json.loads(spec_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("physical evaluator bundle spec is invalid") from exc
    verify_payload_seal(spec)
    if spec.get("schema_version") != HANDOFF_SPEC_SCHEMA:
        raise ProbabilisticContractError("physical evaluator bundle spec schema differs")
    pointer_path = _resolve(spec["authorization_pointer_path"], repo_root=repo_root)
    pointer, authorization = _load_authorization_from_pointer(
        pointer_path,
        repo_root=repo_root,
        launch_authority=launch_authority,
    )
    common_mask_path = _resolve(spec["common_mask_path"], repo_root=repo_root)
    _verify_common_mask(common_mask_path, authorization)

    candidate_spec = _require_exact_mapping(
        spec.get("candidates"), CANDIDATE_IDS, context="candidate artifact set"
    )
    reference_spec = _require_exact_mapping(
        spec.get("references"), REFERENCE_IDS, context="reference artifact set"
    )
    comparator_spec = _require_exact_mapping(
        spec.get("comparators"), COMPARATOR_IDS, context="point-comparator artifact set"
    )
    candidate_records: dict[str, Any] = {}
    for model_id in CANDIDATE_IDS:
        item = candidate_spec[model_id]
        prediction_path = _resolve(item["prediction_path"], repo_root=repo_root)
        receipt_path = _resolve(item["receipt_path"], repo_root=repo_root)
        runtime_path = _resolve(item["runtime_receipt_path"], repo_root=repo_root)
        predictions = load_verified_prediction_artifact(
            prediction_path, receipt_path, authorization=authorization
        )
        runtime = load_verified_runtime_receipt(
            runtime_path, authorization=authorization, predictions=predictions
        )
        if runtime.participant_id != model_id:
            raise ProbabilisticContractError("candidate runtime receipt is relabeled")
        candidate_records[model_id] = {
            "prediction": _file_record(prediction_path),
            "receipt": _file_record(receipt_path),
            "runtime_receipt": _file_record(runtime_path),
        }

    reference_records: dict[str, Any] = {}
    for reference_id in REFERENCE_IDS:
        item = reference_spec[reference_id]
        prediction_path = _resolve(item["prediction_path"], repo_root=repo_root)
        receipt_path = _resolve(item["receipt_path"], repo_root=repo_root)
        execution_path = _resolve(item["execution_receipt_path"], repo_root=repo_root)
        predictions = load_verified_reference_prediction_artifact(
            prediction_path, receipt_path, authorization=authorization
        )
        load_verified_reference_execution_receipt(
            execution_path,
            authorization=authorization,
            predictions=predictions,
            reference_id=reference_id,
        )
        reference_records[reference_id] = {
            "prediction": _file_record(prediction_path),
            "receipt": _file_record(receipt_path),
            "execution_receipt": _file_record(execution_path),
        }

    comparator_records: dict[str, Any] = {}
    comparator_hashes: set[str] = set()
    for comparator_id in COMPARATOR_IDS:
        item = comparator_spec[comparator_id]
        prediction_path = _resolve(item["prediction_path"], repo_root=repo_root)
        receipt_path = _resolve(item["receipt_path"], repo_root=repo_root)
        artifact = load_verified_point_comparator_with_receipt(
            prediction_path,
            receipt_path,
            comparator_id=comparator_id,
            authorization=authorization,
        )
        comparator_hashes.add(artifact.raw_sha256)
        comparator_records[comparator_id] = {
            "prediction": _file_record(prediction_path),
            "receipt": _file_record(receipt_path),
        }
    if len(comparator_hashes) != 2:
        raise ProbabilisticContractError("point comparator prediction bytes must be distinct")

    truth_path = _resolve(spec["truth_path"], repo_root=repo_root)
    if authorization.scope == FORMAL_SCOPE:
        from .spent import UPSTREAM_INPUTS

        expected_truth_path = _resolve(UPSTREAM_INPUTS["evaluate"]["path"], repo_root=repo_root)
        if (
            truth_path != expected_truth_path
            or authorization.bindings["detached_truth_manifest_sha256"]
            != UPSTREAM_INPUTS["evaluate"]["raw_sha256"]
        ):
            raise ProbabilisticContractError("formal truth-manifest address differs")
    elif str(truth_path) != str(_resolve(pointer["truth_path"], repo_root=repo_root)):
        raise ProbabilisticContractError("truth path differs from authorization pointer")
    payload_body = {
        "schema_version": HANDOFF_SCHEMA,
        "parent_pid": os.getpid(),
        "authorization_pointer": _file_record(pointer_path),
        "authorization_raw_sha256": authorization.raw_sha256,
        "source_closure_sha256": authorization.source_closure_sha256,
        "common_mask": _file_record(common_mask_path),
        "candidates": candidate_records,
        "references": reference_records,
        "comparators": comparator_records,
        # The parent copies the precommitted address and path without reading truth bytes.
        "truth": {
            "path": str(truth_path),
            "expected_sha256": authorization.bindings["detached_truth_manifest_sha256"],
            "parent_read": False,
        },
        "required_child_process": True,
        "single_use_no_retry": True,
    }
    if authorization.scope == FORMAL_SCOPE:
        if launch_authority is None:
            raise ProbabilisticContractError("formal handoff lacks external launch authority")
        if (
            spec.get("external_execution_request_raw_sha256") != launch_authority.request.raw_sha256
            or _resolve(spec["external_execution_request_path"], repo_root=repo_root)
            != launch_authority.request.path
            or spec.get("independent_go_raw_sha256") != launch_authority.independent_go.raw_sha256
            or _resolve(spec["independent_go_path"], repo_root=repo_root)
            != launch_authority.independent_go.path
        ):
            raise ProbabilisticContractError("formal bundle differs from external launch pins")
        payload_body["external_execution_request"] = _file_record(launch_authority.request.path)
        payload_body["independent_go"] = _file_record(launch_authority.independent_go.path)
    payload = seal_payload(payload_body)
    return write_content_addressed_json(Path(directory), "physical_evaluator_handoff", payload)


class VerifiedPhysicalPreflight:
    """Child-private proof that the complete immutable surface was independently reloaded."""

    __slots__ = (
        "_authorization_sha256",
        "_candidates",
        "_child_pid",
        "_claim_path",
        "_comparators",
        "_handoff_sha256",
        "_parent_pid",
        "_references",
        "_runtime",
        "_truth_opened",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _PREFLIGHT_TOKEN:
            raise ProbabilisticContractError(
                "physical preflight must come from the child disk verifier"
            )
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        authorization: ExecutionAuthorization,
        candidates: Mapping[str, VerifiedPredictionBatch],
        references: Mapping[str, tuple[VerifiedPredictionBatch, VerifiedReferenceExecutionReceipt]],
        comparators: Mapping[str, VerifiedPointComparatorArtifact],
        runtime: Mapping[str, VerifiedRuntimeReceipt],
        handoff_sha256: str,
        parent_pid: int,
        claim_path: Path,
    ) -> None:
        if token is not _PREFLIGHT_TOKEN:
            raise ProbabilisticContractError("invalid physical preflight token")
        object.__setattr__(self, "_authorization_sha256", authorization.raw_sha256)
        object.__setattr__(self, "_candidates", dict(candidates))
        object.__setattr__(self, "_references", dict(references))
        object.__setattr__(self, "_comparators", dict(comparators))
        object.__setattr__(self, "_runtime", dict(runtime))
        object.__setattr__(self, "_handoff_sha256", handoff_sha256)
        object.__setattr__(self, "_parent_pid", int(parent_pid))
        object.__setattr__(self, "_child_pid", os.getpid())
        object.__setattr__(self, "_claim_path", Path(claim_path))
        object.__setattr__(self, "_truth_opened", False)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedPhysicalPreflight is immutable")

    def verify_for_truth_open(self, *, authorization: ExecutionAuthorization) -> None:
        if self._truth_opened:
            raise ProbabilisticContractError("physical truth preflight is single-use")
        authorization.verify_integrity()
        if (
            os.getpid() != self._child_pid
            or self._child_pid == self._parent_pid
            or authorization.raw_sha256 != self._authorization_sha256
            or set(self._candidates) != set(CANDIDATE_IDS)
            or set(self._references) != set(REFERENCE_IDS)
            or set(self._comparators) != set(COMPARATOR_IDS)
            or set(self._runtime) != set(CANDIDATE_IDS)
        ):
            raise ProbabilisticContractError("physical evaluator process/custody set differs")
        try:
            claim = json.loads(self._claim_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProbabilisticContractError("single-use physical claim is unavailable") from exc
        verify_payload_seal(claim)
        if (
            claim.get("handoff_sha256") != self._handoff_sha256
            or claim.get("child_pid") != self._child_pid
            or claim.get("parent_pid") != self._parent_pid
        ):
            raise ProbabilisticContractError("single-use physical claim differs")
        for model_id, predictions in self._candidates.items():
            predictions.verify_integrity()
            self._runtime[model_id].verify_integrity()
        for predictions, receipt in self._references.values():
            predictions.verify_integrity()
            receipt.verify_integrity()
        for comparator in self._comparators.values():
            comparator.verify_integrity()
        object.__setattr__(self, "_truth_opened", True)


def _claim_handoff(handoff_path: Path, *, handoff_sha256: str, parent_pid: int) -> Path:
    claim_path = handoff_path.with_suffix(handoff_path.suffix + ".claimed")
    payload = seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.physical_evaluator_claim.v4",
            "handoff_sha256": handoff_sha256,
            "parent_pid": parent_pid,
            "child_pid": os.getpid(),
            "retry_permitted": False,
        }
    )
    raw = canonical_json_bytes(payload)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise ProbabilisticContractError("physical evaluator handoff was already consumed") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return claim_path


def _load_child_preflight(
    handoff_path: Path,
    *,
    repo_root: Path,
    launch_authority: FormalLaunchAuthority | None = None,
) -> tuple[ExecutionAuthorization, VerifiedPhysicalPreflight, Path]:
    raw = handoff_path.read_bytes()
    handoff_sha = sha256_bytes(raw)
    if handoff_sha not in handoff_path.name.split("."):
        raise ProbabilisticContractError("physical handoff is not content-addressed")
    try:
        handoff = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("physical handoff cannot be decoded") from exc
    verify_payload_seal(handoff)
    if handoff.get("schema_version") != HANDOFF_SCHEMA:
        raise ProbabilisticContractError("physical handoff schema differs")
    parent_pid = int(handoff["parent_pid"])
    if parent_pid == os.getpid():
        raise ProbabilisticContractError("physical evaluator must run in a distinct process")
    pointer_path = _verify_record(handoff["authorization_pointer"])
    _, authorization = _load_authorization_from_pointer(
        pointer_path,
        repo_root=repo_root,
        launch_authority=launch_authority,
    )
    if authorization.scope == FORMAL_SCOPE:
        if launch_authority is None:
            raise ProbabilisticContractError("formal child lacks external launch authority")
        request_path = _verify_record(handoff.get("external_execution_request", {}))
        go_path = _verify_record(handoff.get("independent_go", {}))
        if (
            request_path.resolve() != launch_authority.request.path
            or sha256_bytes(request_path.read_bytes()) != launch_authority.request.raw_sha256
            or go_path.resolve() != launch_authority.independent_go.path
            or sha256_bytes(go_path.read_bytes()) != launch_authority.independent_go.raw_sha256
        ):
            raise ProbabilisticContractError("child handoff differs from external launch pins")
    if (
        handoff.get("authorization_raw_sha256") != authorization.raw_sha256
        or handoff.get("source_closure_sha256") != authorization.source_closure_sha256
    ):
        raise ProbabilisticContractError("physical handoff authorization/source differs")
    common_mask_path = _verify_record(handoff["common_mask"])
    _verify_common_mask(common_mask_path, authorization)
    candidate_map = _require_exact_mapping(
        handoff.get("candidates"), CANDIDATE_IDS, context="child candidate artifact set"
    )
    candidates: dict[str, VerifiedPredictionBatch] = {}
    runtime: dict[str, VerifiedRuntimeReceipt] = {}
    for model_id in CANDIDATE_IDS:
        item = candidate_map[model_id]
        prediction = _verify_record(item["prediction"])
        receipt = _verify_record(item["receipt"])
        runtime_path = _verify_record(item["runtime_receipt"])
        batch = load_verified_prediction_artifact(prediction, receipt, authorization=authorization)
        candidates[model_id] = batch
        runtime[model_id] = load_verified_runtime_receipt(
            runtime_path, authorization=authorization, predictions=batch
        )
    reference_map = _require_exact_mapping(
        handoff.get("references"), REFERENCE_IDS, context="child reference artifact set"
    )
    references: dict[str, tuple[VerifiedPredictionBatch, VerifiedReferenceExecutionReceipt]] = {}
    for reference_id in REFERENCE_IDS:
        item = reference_map[reference_id]
        prediction = _verify_record(item["prediction"])
        receipt = _verify_record(item["receipt"])
        execution = _verify_record(item["execution_receipt"])
        batch = load_verified_reference_prediction_artifact(
            prediction, receipt, authorization=authorization
        )
        references[reference_id] = (
            batch,
            load_verified_reference_execution_receipt(
                execution,
                authorization=authorization,
                predictions=batch,
                reference_id=reference_id,
            ),
        )
    comparator_map = _require_exact_mapping(
        handoff.get("comparators"), COMPARATOR_IDS, context="child comparator artifact set"
    )
    comparators: dict[str, VerifiedPointComparatorArtifact] = {}
    for comparator_id in COMPARATOR_IDS:
        item = comparator_map[comparator_id]
        comparators[comparator_id] = load_verified_point_comparator_with_receipt(
            _verify_record(item["prediction"]),
            _verify_record(item["receipt"]),
            comparator_id=comparator_id,
            authorization=authorization,
        )
    truth = handoff.get("truth")
    if (
        not isinstance(truth, dict)
        or set(truth) != {"path", "expected_sha256", "parent_read"}
        or truth.get("parent_read") is not False
        or truth.get("expected_sha256") != authorization.bindings["detached_truth_manifest_sha256"]
    ):
        raise ProbabilisticContractError("physical truth handoff differs")
    claim_path = _claim_handoff(handoff_path, handoff_sha256=handoff_sha, parent_pid=parent_pid)
    preflight = VerifiedPhysicalPreflight(
        _PREFLIGHT_TOKEN,
        authorization=authorization,
        candidates=candidates,
        references=references,
        comparators=comparators,
        runtime=runtime,
        handoff_sha256=handoff_sha,
        parent_pid=parent_pid,
        claim_path=claim_path,
    )
    return authorization, preflight, Path(str(truth["path"]))


def execute_physical_evaluator(
    handoff_path: Path,
    output_path: Path,
    *,
    repo_root: Path,
    launch_authority: FormalLaunchAuthority | None = None,
) -> Path:
    """Child entry point. Reload all disk custody, then and only then open truth once."""

    authorization, preflight, truth_path = _load_child_preflight(
        Path(handoff_path),
        repo_root=repo_root,
        launch_authority=launch_authority,
    )
    from .evaluation import (
        _load_detached_truth_after_physical_preflight,
        evaluate_candidate_receipt,
        evaluate_point_comparator_receipt,
        evaluate_reference_receipt,
    )

    truth = _load_detached_truth_after_physical_preflight(
        truth_path, authorization=authorization, preflight=preflight
    )
    candidate_receipts = {
        model_id: evaluate_candidate_receipt(
            predictions,
            truth,
            authorization=authorization,
            candidate_id=model_id,
        )
        for model_id, predictions in preflight._candidates.items()
    }
    reference_receipts = {
        reference_id: evaluate_reference_receipt(
            predictions,
            truth,
            execution_receipt=execution_receipt,
            authorization=authorization,
            reference_id=reference_id,
        )
        for reference_id, (predictions, execution_receipt) in preflight._references.items()
    }
    comparator_receipts = {
        comparator_id: evaluate_point_comparator_receipt(
            artifact,
            truth,
            authorization=authorization,
            comparator_id=comparator_id,
        )
        for comparator_id, artifact in preflight._comparators.items()
    }
    formal_selection = None
    if authorization.scope == FORMAL_SCOPE:
        from .governance import FormalScreenOrchestrator

        formal_selection = FormalScreenOrchestrator(authorization).execute(
            candidates=candidate_receipts,
            probabilistic_references=reference_receipts,
            point_comparators=comparator_receipts,
            runtime_by_candidate=preflight._runtime,
        )

    def receipt_payload(receipt: object) -> dict[str, Any]:
        raw = receipt.raw_bytes  # type: ignore[attr-defined]
        return json.loads(raw)

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "PASS_PHYSICAL_EVALUATOR_SINGLE_USE",
        "handoff_sha256": preflight._handoff_sha256,
        "authorization_raw_sha256": authorization.raw_sha256,
        "source_closure_sha256": authorization.source_closure_sha256,
        "external_execution_request_raw_sha256": (
            launch_authority.request.raw_sha256 if launch_authority is not None else None
        ),
        "independent_go_raw_sha256": (
            launch_authority.independent_go.raw_sha256 if launch_authority is not None else None
        ),
        "parent_pid": preflight._parent_pid,
        "evaluator_pid": os.getpid(),
        "candidate_metric_receipts": {
            model_id: receipt_payload(receipt) for model_id, receipt in candidate_receipts.items()
        },
        "reference_metric_receipts": {
            reference_id: receipt_payload(receipt)
            for reference_id, receipt in reference_receipts.items()
        },
        "point_comparator_metric_receipts": {
            comparator_id: receipt_payload(receipt)
            for comparator_id, receipt in comparator_receipts.items()
        },
        "runtime_receipt_sha256_by_candidate": {
            model_id: receipt.raw_sha256 for model_id, receipt in preflight._runtime.items()
        },
        "formal_selection": formal_selection,
        "truth_bytes_returned": False,
        "prediction_bytes_returned": False,
        "metric_receipts_only": True,
        "retry_permitted": False,
        "formal_execution_authorized": authorization.scope == FORMAL_SCOPE,
    }
    immutable_write_json(Path(output_path), result)
    return Path(output_path)
