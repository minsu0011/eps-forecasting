"""Opaque metric receipts, fixed ranking, and single-use one-winner governance."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .authorization import ExecutionAuthorization, FORMAL_SCOPE
from .contracts import (
    ProbabilisticContractError,
    seal_payload,
)
from .evaluation import (
    VerifiedEvaluationReceipt,
    _cheap_screen_decision,
    score_free_evaluation_receipts,
)
from .resources import VerifiedRuntimeReceipt, score_free_runtime_receipts
from .spec import CANDIDATE_IDS


RANK_ORDER = (
    "pooled_wis",
    "pooled_mean_pinball_loss",
    "p50_fair_log_mae",
    "p50_fair_log_rmse",
    "maximum_calibration_error",
    "worst_seed_wis",
    "runtime",
    "model_id",
)


def _finite(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProbabilisticContractError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ProbabilisticContractError(f"{field} must be finite")
    return number


def _stronger_probabilistic_reference(
    references: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(references) != {"rolling_log_quantiles_252", "point_residual_quantiles_252"}:
        raise ProbabilisticContractError("both exact probabilistic references are required")
    values = list(references.values())
    seed_sets = {tuple(sorted(item["per_seed"], key=str)) for item in values}
    if len(seed_sets) != 1:
        raise ProbabilisticContractError("probabilistic reference seed supports differ")
    seeds = next(iter(seed_sets))
    return {
        "wis": min(_finite(item["wis"], field="reference.wis") for item in values),
        "mean_pinball_loss": min(
            _finite(item["mean_pinball_loss"], field="reference.mean_pinball_loss")
            for item in values
        ),
        "per_seed": {
            seed: {
                "wis": min(
                    _finite(item["per_seed"][seed]["wis"], field="reference.seed.wis")
                    for item in values
                )
            }
            for seed in seeds
        },
    }


def _stronger_point_comparator(
    comparators: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(comparators) != {"v04_expected_pe", "ml_expected_pe"}:
        raise ProbabilisticContractError("both exact point comparators are required")
    values = list(comparators.values())
    seed_sets = {tuple(sorted(item["per_seed"], key=str)) for item in values}
    if len(seed_sets) != 1:
        raise ProbabilisticContractError("point comparator seed supports differ")
    seeds = next(iter(seed_sets))
    return {
        "fair_log_mae": min(
            _finite(item["fair_log_mae"], field="point.fair_log_mae") for item in values
        ),
        "fair_log_rmse": min(
            _finite(item["fair_log_rmse"], field="point.fair_log_rmse") for item in values
        ),
        "per_seed": {
            seed: {
                "fair_log_mae": min(
                    _finite(
                        item["per_seed"][seed]["fair_log_mae"],
                        field="point.seed.fair_log_mae",
                    )
                    for item in values
                )
            }
            for seed in seeds
        },
    }


def _rank_locked_metrics(
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    probabilistic_references: Mapping[str, Mapping[str, Any]],
    point_comparators: Mapping[str, Mapping[str, Any]],
    runtime_by_candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(candidates) != set(CANDIDATE_IDS):
        raise ProbabilisticContractError("selection requires all three locked candidates")
    if set(runtime_by_candidate) != set(candidates):
        raise ProbabilisticContractError("runtime evidence differs from candidate set")
    reference = _stronger_probabilistic_reference(probabilistic_references)
    point = _stronger_point_comparator(point_comparators)
    decisions: dict[str, dict[str, Any]] = {}
    for model_id, metrics in candidates.items():
        runtime = runtime_by_candidate[model_id]
        decisions[model_id] = _cheap_screen_decision(
            metrics,
            stronger_probabilistic_reference=reference,
            stronger_point_comparator=point,
            crossing_row_rate=_finite(runtime["crossing_row_rate"], field="crossing rate"),
            crossing_p95_magnitude=_finite(
                runtime["crossing_p95_magnitude"], field="crossing magnitude"
            ),
            post_repair_crossing_rate=_finite(
                runtime["post_repair_crossing_rate"], field="post crossing rate"
            ),
            common_mask_coverage=_finite(
                runtime["common_mask_coverage"], field="common mask coverage"
            ),
            runtime_minutes=_finite(runtime["runtime_minutes"], field="runtime"),
            aggregate_rss_gib=_finite(runtime["aggregate_rss_gib"], field="rss"),
        )
    eligible = [
        model_id
        for model_id, decision in decisions.items()
        if decision["all_guards_pass"] and decision["advancement_path"] != "NO_ADVANCEMENT"
    ]

    def rank_key(model_id: str) -> tuple[Any, ...]:
        metrics = candidates[model_id]
        runtime = runtime_by_candidate[model_id]
        return (
            _finite(metrics["wis"], field="candidate.wis"),
            _finite(metrics["mean_pinball_loss"], field="candidate.pinball"),
            _finite(metrics["fair_median"]["fair_log_mae"], field="candidate.mae"),
            _finite(metrics["fair_median"]["fair_log_rmse"], field="candidate.rmse"),
            _finite(
                metrics["maximum_absolute_quantile_calibration_error"],
                field="candidate.max_calibration",
            ),
            max(
                _finite(seed_metrics["wis"], field="candidate.seed.wis")
                for seed_metrics in metrics["per_seed"].values()
            ),
            _finite(runtime["runtime_minutes"], field="candidate.runtime"),
            model_id,
        )

    ranked = sorted(eligible, key=rank_key)
    return seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.probabilistic_selection.v2",
            "rank_order": list(RANK_ORDER),
            "candidate_order": list(candidates),
            "decisions": decisions,
            "eligible_ranked": ranked,
            "selected": ranked[:1],
            "maximum_candidates_advanced": 1,
            "gate_application_count_per_candidate": 1,
        }
    )


def select_from_verified_receipts(
    *,
    authorization: ExecutionAuthorization,
    candidates: Mapping[str, VerifiedEvaluationReceipt],
    probabilistic_references: Mapping[str, VerifiedEvaluationReceipt],
    point_comparators: Mapping[str, VerifiedEvaluationReceipt],
    runtime_by_candidate: Mapping[str, VerifiedRuntimeReceipt],
) -> dict[str, Any]:
    """Only opaque, same-authorization receipts can reach ranking/selection."""

    authorization.verify_integrity()
    evaluation_groups = (
        (candidates, "candidate_distribution"),
        (probabilistic_references, "reference_distribution"),
        (point_comparators, "point_comparator"),
    )
    for receipts, expected_kind in evaluation_groups:
        for participant_id, receipt in receipts.items():
            if not isinstance(receipt, VerifiedEvaluationReceipt):
                raise ProbabilisticContractError("selection rejects arbitrary metric mappings")
            receipt.verify_integrity()
            if receipt.participant_id != participant_id or receipt.receipt_kind != expected_kind:
                raise ProbabilisticContractError("screen receipt identity/kind differs")
            if receipt.authorization_sha256 != authorization.raw_sha256:
                raise ProbabilisticContractError("screen receipt authorization differs")
    for participant_id, receipt in runtime_by_candidate.items():
        if not isinstance(receipt, VerifiedRuntimeReceipt):
            raise ProbabilisticContractError("selection rejects arbitrary runtime mappings")
        receipt.verify_integrity()
        if receipt.participant_id != participant_id or receipt.receipt_kind != "candidate_runtime":
            raise ProbabilisticContractError("runtime receipt identity/kind differs")
        if receipt.authorization_sha256 != authorization.raw_sha256:
            raise ProbabilisticContractError("runtime receipt authorization differs")
        candidate_provenance = candidates[participant_id].provenance
        runtime_provenance = receipt.provenance
        for field in (
            "prediction_raw_sha256",
            "prediction_logical_sha256",
            "prediction_receipt_sha256",
            "common_identity_sha256",
            "authorization_raw_sha256",
        ):
            if candidate_provenance[field] != runtime_provenance[field]:
                raise ProbabilisticContractError(
                    f"candidate/runtime custody provenance differs: {field}"
                )
    return _rank_locked_metrics(
        candidates={key: value._metrics_copy() for key, value in candidates.items()},
        probabilistic_references={
            key: value._metrics_copy() for key, value in probabilistic_references.items()
        },
        point_comparators={key: value._metrics_copy() for key, value in point_comparators.items()},
        runtime_by_candidate={
            key: value._metrics_copy() for key, value in runtime_by_candidate.items()
        },
    )


def run_score_free_governance_evidence(
    authorization: ExecutionAuthorization,
) -> dict[str, Any]:
    """Execute all-three receipt/ranking wiring with fixed score-free factories."""

    candidates, references, point_comparators = score_free_evaluation_receipts(authorization)
    runtime = score_free_runtime_receipts(authorization)
    result = select_from_verified_receipts(
        authorization=authorization,
        candidates=candidates,
        probabilistic_references=references,
        point_comparators=point_comparators,
        runtime_by_candidate=runtime,
    )
    result["synthetic_only"] = True
    result["formal_execution_authorized"] = False
    result["candidate_receipt_count"] = len(candidates)
    result["candidate_receipt_ids"] = list(candidates)
    result["receipt_evidence_mode"] = "SCORE_FREE_FIXED_INTERNAL_FIXTURE"
    return seal_payload(result)


class FormalScreenOrchestrator:
    """Single-use formal gate; unavailable until an external trust root is pinned."""

    __slots__ = ("_authorization", "_used")

    def __init__(self, authorization: ExecutionAuthorization) -> None:
        if not isinstance(authorization, ExecutionAuthorization):
            raise ProbabilisticContractError("formal orchestrator requires factory authorization")
        authorization.verify_integrity()
        if authorization.scope != FORMAL_SCOPE:
            raise ProbabilisticContractError("score-free authorization cannot create orchestrator")
        object.__setattr__(self, "_authorization", authorization)
        object.__setattr__(self, "_used", False)

    def execute(
        self,
        *,
        candidates: Mapping[str, VerifiedEvaluationReceipt],
        probabilistic_references: Mapping[str, VerifiedEvaluationReceipt],
        point_comparators: Mapping[str, VerifiedEvaluationReceipt],
        runtime_by_candidate: Mapping[str, VerifiedRuntimeReceipt],
    ) -> dict[str, Any]:
        if self._used:
            raise ProbabilisticContractError("formal gate is single-use; retry is forbidden")
        object.__setattr__(self, "_used", True)
        return select_from_verified_receipts(
            authorization=self._authorization,
            candidates=candidates,
            probabilistic_references=probabilistic_references,
            point_comparators=point_comparators,
            runtime_by_candidate=runtime_by_candidate,
        )
