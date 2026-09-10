"""Deterministic Structural-Wave trigger resolver over final sealed evidence.

The module has no writer for ``TRIGGER_DECISION.json``.  A verified evidence
object can only be constructed from a final terminal summary plus an
independent final GO audit that binds the exact terminal file bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from .contracts import (
    STRUCTURAL_DESIGN_SHA256,
    StructuralContractError,
    canonical_json_bytes,
    finite_float,
    require_sha256,
    sha256_bytes,
    sha256_file,
    verify_payload_seal,
)


WAVE1_DESIGN_LOCK_FILE_SHA256 = "b3a190cbf80045b460551af4b9a53476d5c369e47f2cbf460a02d8779672c82d"
FINAL_TERMINAL_FILE_SHA256 = "d4355270771aa830d74e7bcfc04fa5a2e103fcacf88c75e3e48308616bc1bae7"
FINAL_TERMINAL_LOGICAL_SHA256 = "1104dddb8dbeb93347a2f1837ff1834225c5631990987f9ef1759dd2bf50ccaf"
FINAL_AUDIT_MANIFEST_FILE_SHA256 = (
    "ea40d901a7a5d889e217f38ee13b3260191f2c24b38238256a66e3fd74540d2d"
)
FINAL_AUDIT_MANIFEST_LOGICAL_SHA256 = (
    "f2488546811ae4178c160c1cf803f9ba62480ed62ba5bf0db38425c48661e7b3"
)
FINAL_AUDIT_REPORT_FILE_SHA256 = "444fe108ed2fa28dfbae4cd731ec6847bca6b0662d6fc0136109379f49ea289e"
FINAL_TRIGGER_INPUT_FILE_SHA256 = "697a68616868f9b19b5a440ed0c7b7c1c51c31d0418c63ea415670229973cf28"
FINAL_TRIGGER_INPUT_LOGICAL_SHA256 = (
    "9625a81e73f6827eb9cb8ec082ecf192d74fef03227140fe56cacb2aa7a67440"
)
PRIMARY_GAIN_THRESHOLD = 0.005
OTHER_PRIMARY_MAX_DEGRADATION = 0.005
P95_MAX_DEGRADATION = 0.005
WORST_SEED_MAX_DEGRADATION = 0.02
COMPLEMENTARITY_MIN_ORACLE_GAIN = 0.01
COMPLEMENTARITY_MAX_ABS_ERROR_CORRELATION = 0.95
STABLE_BIAS_MIN_ABS_POOLED = 0.005
STABLE_BIAS_REQUIRED_SEEDS = 4
SERIAL_MEDIAN_MIN = 0.20
SERIAL_PER_SEED_MIN = 0.15
SERIAL_REQUIRED_SEEDS = 4
SPENT_SEED_COUNT = 5

TRIGGER_IDS = (
    "T_A_TRACK_GAP",
    "T_DIRECT_PLATEAU_OR_TAIL",
    "T_COMPLEMENTARITY",
    "T_STABLE_BIAS",
    "T_SERIAL_RESIDUAL",
)
CANDIDATE_TRIGGER_MAP = {
    "decomp_block_ridge_ar1_lag1": ("T_A_TRACK_GAP",),
    "decomp_block_ridge_ar1_current": (
        "T_DIRECT_PLATEAU_OR_TAIL",
        "T_COMPLEMENTARITY",
    ),
    "residual_huber_nested_oof": ("T_STABLE_BIAS",),
    "residual_ar1_nested_oof": ("T_SERIAL_RESIDUAL",),
    "stack_geometric_equal_pair": ("T_COMPLEMENTARITY",),
    "stack_simplex_pair_frozen": ("T_COMPLEMENTARITY",),
}


@dataclass(frozen=True)
class TriggerMetricRow:
    model_id: str
    family: str
    fair_log_mae: float
    fair_log_rmse: float
    fair_abs_log_error_p95: float
    worst_seed_fair_log_mae: float
    runtime_seconds: float
    natural_coverage: float

    def __post_init__(self) -> None:
        if not self.model_id or not self.family:
            raise StructuralContractError("trigger metric identity must be non-empty")
        for field_name in (
            "fair_log_mae",
            "fair_log_rmse",
            "fair_abs_log_error_p95",
            "worst_seed_fair_log_mae",
            "runtime_seconds",
            "natural_coverage",
        ):
            value = finite_float(getattr(self, field_name), field=field_name)
            if value < 0.0:
                raise StructuralContractError(f"{field_name} must be non-negative")
        if not 0.0 <= self.natural_coverage <= 1.0:
            raise StructuralContractError("natural_coverage must be in [0,1]")

    @property
    def rank_tuple(self) -> tuple[float, float, float, float, float, str]:
        return (
            self.fair_log_mae,
            self.fair_log_rmse,
            self.fair_abs_log_error_p95,
            self.worst_seed_fair_log_mae,
            self.runtime_seconds,
            self.model_id,
        )


@dataclass(frozen=True)
class ComplementarityPairInput:
    base0_model_id: str
    base1_model_id: str
    distinct_families: bool
    same_track_c_cutoff: bool
    oracle_relative_gain_vs_best_single: float
    absolute_error_correlation: float
    prediction_disagreement_frequency: float
    base0_natural_coverage: float
    base1_natural_coverage: float

    def __post_init__(self) -> None:
        if not self.base0_model_id or not self.base1_model_id:
            raise StructuralContractError("complementarity base ids must be non-empty")
        if self.base0_model_id == self.base1_model_id:
            raise StructuralContractError("complementarity bases must differ")
        for field_name in (
            "oracle_relative_gain_vs_best_single",
            "absolute_error_correlation",
            "prediction_disagreement_frequency",
            "base0_natural_coverage",
            "base1_natural_coverage",
        ):
            finite_float(getattr(self, field_name), field=field_name)
        if not -1.0 <= self.absolute_error_correlation <= 1.0:
            raise StructuralContractError("absolute-error correlation must be in [-1,1]")
        if not 0.0 <= self.prediction_disagreement_frequency <= 1.0:
            raise StructuralContractError("prediction disagreement frequency must be in [0,1]")
        if (
            not 0.0 <= self.base0_natural_coverage <= 1.0
            or not 0.0 <= self.base1_natural_coverage <= 1.0
        ):
            raise StructuralContractError("pair natural coverage must be in [0,1]")


@dataclass(frozen=True)
class Wave1TriggerInputs:
    track_a_eligible_supervised_baseline_count: int
    primary_comparator: TriggerMetricRow
    direct_learners: tuple[TriggerMetricRow, ...]
    complementarity_pair: ComplementarityPairInput | None
    selected_residual_base_model_id: str
    pooled_fair_log_bias: float
    per_seed_fair_log_bias: tuple[float, ...]
    per_seed_contiguous_lag1_residual_correlation: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.track_a_eligible_supervised_baseline_count, int)
            or isinstance(self.track_a_eligible_supervised_baseline_count, bool)
            or self.track_a_eligible_supervised_baseline_count < 0
        ):
            raise StructuralContractError("Track-A eligible baseline count is invalid")
        if not self.direct_learners:
            raise StructuralContractError("trigger inputs require at least one direct learner")
        if len({row.model_id for row in self.direct_learners}) != len(self.direct_learners):
            raise StructuralContractError("direct learner trigger rows must be unique")
        if not self.selected_residual_base_model_id:
            raise StructuralContractError("selected residual base id must be non-empty")
        finite_float(self.pooled_fair_log_bias, field="pooled_fair_log_bias")
        if len(self.per_seed_fair_log_bias) != SPENT_SEED_COUNT:
            raise StructuralContractError("stable-bias trigger requires exactly five spent seeds")
        if len(self.per_seed_contiguous_lag1_residual_correlation) != SPENT_SEED_COUNT:
            raise StructuralContractError("serial trigger requires exactly five spent seeds")
        if not all(math.isfinite(value) for value in self.per_seed_fair_log_bias):
            raise StructuralContractError("per-seed biases must be finite")
        if not all(
            math.isfinite(value) and -1.0 <= value <= 1.0
            for value in self.per_seed_contiguous_lag1_residual_correlation
        ):
            raise StructuralContractError("per-seed serial correlations must be finite in [-1,1]")


_VERIFICATION_TOKEN = object()


class VerifiedWave1TerminalEvidence:
    """Opaque evidence object; direct construction is intentionally rejected."""

    __slots__ = (
        "inputs",
        "terminal_file_sha256",
        "terminal_manifest_sha256",
        "audit_file_sha256",
        "audit_manifest_sha256",
        "trigger_inputs_sha256",
        "bound_artifact_sha256s",
    )

    def __init__(
        self,
        *,
        token: object,
        inputs: Wave1TriggerInputs,
        terminal_file_sha256: str,
        terminal_manifest_sha256: str,
        audit_file_sha256: str,
        audit_manifest_sha256: str,
        trigger_inputs_sha256: str,
        bound_artifact_sha256s: tuple[tuple[str, str], ...],
    ) -> None:
        if token is not _VERIFICATION_TOKEN:
            raise StructuralContractError(
                "verified Wave1 evidence can only be created by sealed terminal/audit loading"
            )
        self.inputs = inputs
        self.terminal_file_sha256 = terminal_file_sha256
        self.terminal_manifest_sha256 = terminal_manifest_sha256
        self.audit_file_sha256 = audit_file_sha256
        self.audit_manifest_sha256 = audit_manifest_sha256
        self.trigger_inputs_sha256 = trigger_inputs_sha256
        self.bound_artifact_sha256s = bound_artifact_sha256s


def _metric_from_mapping(value: object, *, context: str) -> TriggerMetricRow:
    if not isinstance(value, Mapping):
        raise StructuralContractError(f"{context} must be an object")
    required = {
        "model_id",
        "family",
        "fair_log_mae",
        "fair_log_rmse",
        "fair_abs_log_error_p95",
        "worst_seed_fair_log_mae",
        "runtime_seconds",
        "natural_coverage",
    }
    if set(value) != required:
        raise StructuralContractError(f"{context} metric schema changed")
    return TriggerMetricRow(
        model_id=str(value["model_id"]),
        family=str(value["family"]),
        fair_log_mae=float(value["fair_log_mae"]),
        fair_log_rmse=float(value["fair_log_rmse"]),
        fair_abs_log_error_p95=float(value["fair_abs_log_error_p95"]),
        worst_seed_fair_log_mae=float(value["worst_seed_fair_log_mae"]),
        runtime_seconds=float(value["runtime_seconds"]),
        natural_coverage=float(value["natural_coverage"]),
    )


def _inputs_from_mapping(value: object) -> Wave1TriggerInputs:
    if not isinstance(value, Mapping):
        raise StructuralContractError("terminal trigger_inputs must be an object")
    required = {
        "track_a_eligible_supervised_baseline_count",
        "primary_comparator",
        "direct_learners",
        "complementarity_pair",
        "selected_residual_base_model_id",
        "pooled_fair_log_bias",
        "per_seed_fair_log_bias",
        "per_seed_contiguous_lag1_residual_correlation",
    }
    if set(value) != required:
        raise StructuralContractError("terminal trigger_inputs schema changed")
    direct_raw = value["direct_learners"]
    if not isinstance(direct_raw, list):
        raise StructuralContractError("direct_learners must be an array")
    pair_raw = value["complementarity_pair"]
    pair: ComplementarityPairInput | None
    if pair_raw is None:
        pair = None
    elif isinstance(pair_raw, Mapping):
        pair_fields = {
            "base0_model_id",
            "base1_model_id",
            "distinct_families",
            "same_track_c_cutoff",
            "oracle_relative_gain_vs_best_single",
            "absolute_error_correlation",
            "prediction_disagreement_frequency",
            "base0_natural_coverage",
            "base1_natural_coverage",
        }
        if set(pair_raw) != pair_fields:
            raise StructuralContractError("complementarity pair schema changed")
        pair = ComplementarityPairInput(
            base0_model_id=str(pair_raw["base0_model_id"]),
            base1_model_id=str(pair_raw["base1_model_id"]),
            distinct_families=pair_raw["distinct_families"] is True,
            same_track_c_cutoff=pair_raw["same_track_c_cutoff"] is True,
            oracle_relative_gain_vs_best_single=float(
                pair_raw["oracle_relative_gain_vs_best_single"]
            ),
            absolute_error_correlation=float(pair_raw["absolute_error_correlation"]),
            prediction_disagreement_frequency=float(pair_raw["prediction_disagreement_frequency"]),
            base0_natural_coverage=float(pair_raw["base0_natural_coverage"]),
            base1_natural_coverage=float(pair_raw["base1_natural_coverage"]),
        )
    else:
        raise StructuralContractError("complementarity_pair must be null or an object")
    biases = value["per_seed_fair_log_bias"]
    correlations = value["per_seed_contiguous_lag1_residual_correlation"]
    if not isinstance(biases, list) or not isinstance(correlations, list):
        raise StructuralContractError("per-seed trigger inputs must be arrays")
    return Wave1TriggerInputs(
        track_a_eligible_supervised_baseline_count=int(
            value["track_a_eligible_supervised_baseline_count"]
        ),
        primary_comparator=_metric_from_mapping(
            value["primary_comparator"], context="primary_comparator"
        ),
        direct_learners=tuple(
            _metric_from_mapping(item, context="direct_learner") for item in direct_raw
        ),
        complementarity_pair=pair,
        selected_residual_base_model_id=str(value["selected_residual_base_model_id"]),
        pooled_fair_log_bias=float(value["pooled_fair_log_bias"]),
        per_seed_fair_log_bias=tuple(float(item) for item in biases),
        per_seed_contiguous_lag1_residual_correlation=tuple(float(item) for item in correlations),
    )


def load_verified_wave1_terminal_evidence(
    terminal_path: Path,
    audit_path: Path,
    *,
    expected_terminal_file_sha256: str,
    expected_audit_file_sha256: str,
) -> VerifiedWave1TerminalEvidence:
    """Load only final, doubly sealed inputs; interim evidence is rejected."""

    raise StructuralContractError(
        "generic trigger envelopes are disabled; use the exact four-file final authority loader"
    )

    require_sha256(expected_terminal_file_sha256, field="expected_terminal_file_sha256")
    require_sha256(expected_audit_file_sha256, field="expected_audit_file_sha256")
    if sha256_file(terminal_path) != expected_terminal_file_sha256:
        raise StructuralContractError("Wave1 terminal file differs from the expected final SHA")
    if sha256_file(audit_path) != expected_audit_file_sha256:
        raise StructuralContractError("Wave1 audit file differs from the expected final SHA")
    try:
        terminal = json.loads(Path(terminal_path).read_text(encoding="utf-8"))
        audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralContractError("Wave1 terminal/audit JSON is unreadable") from exc
    if not isinstance(terminal, dict) or not isinstance(audit, dict):
        raise StructuralContractError("Wave1 terminal/audit roots must be objects")
    verify_payload_seal(terminal)
    verify_payload_seal(audit)
    terminal_fields = {
        "format_version",
        "mode",
        "state",
        "wave1_design_lock_file_sha256",
        "structural_design_sha256",
        "trigger_inputs",
        "trigger_inputs_sha256",
        "manifest_sha256",
    }
    audit_fields = {
        "format_version",
        "mode",
        "decision",
        "audit_final",
        "terminal_file_sha256",
        "terminal_manifest_sha256",
        "trigger_inputs_sha256",
        "checks",
        "manifest_sha256",
    }
    if set(terminal) != terminal_fields or set(audit) != audit_fields:
        raise StructuralContractError("Wave1 terminal/audit trigger schema changed")
    if (
        terminal.get("format_version") != 1
        or terminal.get("mode") != "wave1_terminal_trigger_inputs"
        or terminal.get("state") != "FINAL"
    ):
        raise StructuralContractError("Wave1 trigger terminal is not final")
    if terminal.get("wave1_design_lock_file_sha256") != WAVE1_DESIGN_LOCK_FILE_SHA256:
        raise StructuralContractError("Wave1 trigger terminal uses a different Wave1 design")
    if terminal.get("structural_design_sha256") != STRUCTURAL_DESIGN_SHA256:
        raise StructuralContractError("Wave1 trigger terminal uses a different structural design")
    inputs_hash = sha256_bytes(canonical_json_bytes(terminal["trigger_inputs"]))
    if terminal.get("trigger_inputs_sha256") != inputs_hash:
        raise StructuralContractError("Wave1 terminal trigger-input hash is invalid")
    if (
        audit.get("format_version") != 1
        or audit.get("mode") != "wave1_terminal_trigger_independent_audit"
        or audit.get("decision") != "GO"
        or audit.get("audit_final") is not True
    ):
        raise StructuralContractError("Wave1 independent terminal audit is not final GO")
    if audit.get("terminal_file_sha256") != expected_terminal_file_sha256:
        raise StructuralContractError("Wave1 audit does not bind the terminal file bytes")
    if audit.get("terminal_manifest_sha256") != terminal.get("manifest_sha256"):
        raise StructuralContractError("Wave1 audit does not bind the terminal manifest seal")
    if audit.get("trigger_inputs_sha256") != inputs_hash:
        raise StructuralContractError("Wave1 audit does not bind the trigger inputs")
    checks = audit.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise StructuralContractError("Wave1 final audit checks are incomplete")
    inputs = _inputs_from_mapping(terminal["trigger_inputs"])
    return VerifiedWave1TerminalEvidence(
        token=_VERIFICATION_TOKEN,
        inputs=inputs,
        terminal_file_sha256=expected_terminal_file_sha256,
        terminal_manifest_sha256=str(terminal["manifest_sha256"]),
        audit_file_sha256=expected_audit_file_sha256,
        audit_manifest_sha256=str(audit["manifest_sha256"]),
        trigger_inputs_sha256=inputs_hash,
        bound_artifact_sha256s=(
            ("terminal_manifest", expected_terminal_file_sha256),
            ("audit_manifest", expected_audit_file_sha256),
        ),
    )


def _load_exact_json(path: Path, *, expected_sha256: str, context: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise StructuralContractError(f"{context} file SHA-256 differs from final authority")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralContractError(f"{context} is unreadable") from exc
    if not isinstance(payload, dict):
        raise StructuralContractError(f"{context} root must be an object")
    return payload


def _require_record(
    record: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    context: str,
    relative: bool = False,
) -> None:
    if not isinstance(record, Mapping):
        raise StructuralContractError(f"{context} record must be an object")
    path_key = "relative_path" if relative else "path"
    required = {path_key, "bytes", "sha256"}
    allowed = required.union({"logical_sha256", "rows"})
    if not required.issubset(record) or not set(record).issubset(allowed):
        raise StructuralContractError(f"{context} record schema changed")
    expected_path = Path(expected_path).resolve(strict=True)
    if relative:
        if Path(str(record[path_key])).name != expected_path.name:
            raise StructuralContractError(f"{context} relative path differs")
    elif Path(str(record[path_key])).resolve(strict=True) != expected_path:
        raise StructuralContractError(f"{context} absolute path differs")
    if int(record["bytes"]) != expected_path.stat().st_size:
        raise StructuralContractError(f"{context} byte count differs")
    if record["sha256"] != expected_sha256 or sha256_file(expected_path) != expected_sha256:
        raise StructuralContractError(f"{context} file hash differs")


def load_authorized_final_wave1_evidence(
    *,
    terminal_path: Path,
    audit_manifest_path: Path,
    audit_report_path: Path,
    trigger_inputs_path: Path,
) -> VerifiedWave1TerminalEvidence:
    """Bind the exact 2026-08-19 terminal authority named by the final audit.

    Only these four JSON files are opened.  Candidate prediction/score files,
    pair CSVs, fresh seeds, and heldout data are outside this loader's surface.
    """

    terminal_path = Path(terminal_path).resolve(strict=True)
    audit_manifest_path = Path(audit_manifest_path).resolve(strict=True)
    audit_report_path = Path(audit_report_path).resolve(strict=True)
    trigger_inputs_path = Path(trigger_inputs_path).resolve(strict=True)
    terminal = _load_exact_json(
        terminal_path,
        expected_sha256=FINAL_TERMINAL_FILE_SHA256,
        context="final Wave1 terminal manifest",
    )
    audit = _load_exact_json(
        audit_manifest_path,
        expected_sha256=FINAL_AUDIT_MANIFEST_FILE_SHA256,
        context="final independent audit manifest",
    )
    report = _load_exact_json(
        audit_report_path,
        expected_sha256=FINAL_AUDIT_REPORT_FILE_SHA256,
        context="final independent audit report",
    )
    trigger = _load_exact_json(
        trigger_inputs_path,
        expected_sha256=FINAL_TRIGGER_INPUT_FILE_SHA256,
        context="final structural trigger inputs",
    )

    terminal_fields = {
        "format_version",
        "mode",
        "evidence_stage",
        "design_lock_sha256",
        "seeds",
        "fresh_seeds_consumed",
        "heldout_opened",
        "stage2_tuning_authorized",
        "lock_or_promotion_authorized",
        "advanced_model_ids",
        "audit_candidate",
        "execution_binding",
        "execution_precommit",
        "predict_inputs",
        "evaluate_inputs",
        "prediction_manifest",
        "evaluation_manifest",
        "registry_definition_append",
        "registry_result_append",
        "external_terminal_records",
        "prediction_rows",
        "fold_diagnostic_rows",
        "fold_diagnostic_status_counts",
        "fold_failures",
        "prediction_end_to_end_wall_seconds_diagnostic_only",
        "runtime_seconds_by_model",
        "resource_by_seed",
        "expected_common_identity_rows",
        "actual_common_identity_rows",
        "common_model_columns",
        "baseline_rank_by_fair_log_mae",
        "candidate_rank_by_fair_log_mae",
        "stage1_gates",
        "registry_terminal_state",
        "output_inventory_excludes",
        "output_inventory_file_count",
        "output_inventory_sha256",
        "output_inventory",
        "terminal_checksums",
        "manifest_sha256",
    }
    audit_fields = {
        "format_version",
        "mode",
        "decision",
        "severity_counts",
        "terminal_manifest",
        "terminal_output_inventory_sha256",
        "structural_design_observed",
        "recomputation_evidence",
        "structural_trigger_inputs",
        "attestations",
        "recommendation",
        "audit_inventory_file_count",
        "audit_inventory_sha256",
        "audit_inventory",
        "manifest_sha256",
    }
    report_fields = {
        "format_version",
        "mode",
        "audited_at",
        "decision",
        "severity_counts",
        "terminal_binding",
        "integrity",
        "evidence_access",
        "registry",
        "folds_and_failures",
        "formal_recomputation",
        "global_common_mask_assessment",
        "complete_coverage_diagnostic",
        "structural_trigger_inputs",
        "runtime_and_resources",
        "P2_findings",
        "recommendation",
    }
    trigger_fields = {
        "format_version",
        "mode",
        "structural_design_observed",
        "wave1_terminal_manifest",
        "surface",
        "current_v04_comparator",
        "pair_selection",
        "trigger_decisions_read_only",
        "direct_trigger_inputs",
        "manifest_sha256",
    }
    if set(terminal) != terminal_fields:
        raise StructuralContractError("final Wave1 terminal manifest schema changed")
    if set(audit) != audit_fields:
        raise StructuralContractError("final independent audit manifest schema changed")
    if set(report) != report_fields:
        raise StructuralContractError("final independent audit report schema changed")
    if set(trigger) != trigger_fields:
        raise StructuralContractError("final structural trigger-input schema changed")
    verify_payload_seal(terminal)
    verify_payload_seal(audit)
    verify_payload_seal(trigger)
    if terminal["manifest_sha256"] != FINAL_TERMINAL_LOGICAL_SHA256:
        raise StructuralContractError("final terminal logical seal differs")
    if audit["manifest_sha256"] != FINAL_AUDIT_MANIFEST_LOGICAL_SHA256:
        raise StructuralContractError("final audit logical seal differs")
    if trigger["manifest_sha256"] != FINAL_TRIGGER_INPUT_LOGICAL_SHA256:
        raise StructuralContractError("final trigger-input logical seal differs")

    expected_seeds = [6301, 6421, 6521, 6607, 6701]
    if (
        terminal["format_version"] != 1
        or terminal["mode"] != "wave1_stage1_terminal"
        or terminal["evidence_stage"] != "SPENT_SEED_STAGE1_RESEARCH_ONLY"
        or terminal["design_lock_sha256"] != WAVE1_DESIGN_LOCK_FILE_SHA256
        or terminal["seeds"] != expected_seeds
        or terminal["fresh_seeds_consumed"] is not False
        or terminal["heldout_opened"] is not False
        or terminal["stage2_tuning_authorized"] is not False
        or terminal["lock_or_promotion_authorized"] is not False
        or terminal["advanced_model_ids"] != []
    ):
        raise StructuralContractError(
            "final Wave1 terminal state is not reject-all spent-seed only"
        )

    if (
        audit["format_version"] != 1
        or audit["mode"] != "wave1_stage1_terminal_independent_audit_manifest"
        or audit["decision"] != "PASS_TERMINAL_REJECT_ALL"
    ):
        raise StructuralContractError("final independent audit decision differs")
    severity = audit["severity_counts"]
    if not isinstance(severity, Mapping) or set(severity) != {"P0", "P1", "P2"}:
        raise StructuralContractError("final audit severity schema changed")
    if severity["P0"] != 0 or severity["P1"] != 0:
        raise StructuralContractError("final audit contains a blocking P0/P1 finding")
    _require_record(
        audit["terminal_manifest"],
        expected_path=terminal_path,
        expected_sha256=FINAL_TERMINAL_FILE_SHA256,
        context="audit terminal binding",
    )
    if audit["terminal_manifest"].get("logical_sha256") != FINAL_TERMINAL_LOGICAL_SHA256:
        raise StructuralContractError("audit terminal logical binding differs")
    _require_record(
        audit["structural_trigger_inputs"],
        expected_path=trigger_inputs_path,
        expected_sha256=FINAL_TRIGGER_INPUT_FILE_SHA256,
        context="audit trigger-input binding",
        relative=True,
    )
    if (
        audit["structural_trigger_inputs"].get("logical_sha256")
        != FINAL_TRIGGER_INPUT_LOGICAL_SHA256
    ):
        raise StructuralContractError("audit trigger-input logical binding differs")
    design_record = audit["structural_design_observed"]
    if (
        not isinstance(design_record, Mapping)
        or design_record.get("sha256") != STRUCTURAL_DESIGN_SHA256
    ):
        raise StructuralContractError("final audit observed a different structural design")
    attestations = audit["attestations"]
    expected_attestations = {
        "source_modified",
        "registry_modified",
        "wave1_terminal_modified",
        "models_run",
        "fresh_seeds_consumed",
        "heldout_opened",
        "formal_decision_changed",
    }
    if (
        not isinstance(attestations, Mapping)
        or set(attestations) != expected_attestations
        or any(value is not False for value in attestations.values())
    ):
        raise StructuralContractError("final audit mutation/access attestations differ")
    inventory = audit["audit_inventory"]
    if not isinstance(inventory, list):
        raise StructuralContractError("final audit inventory must be an array")
    report_records = [item for item in inventory if item.get("relative_path") == "REPORT.json"]
    if len(report_records) != 1:
        raise StructuralContractError("final audit inventory does not uniquely bind REPORT.json")
    _require_record(
        report_records[0],
        expected_path=audit_report_path,
        expected_sha256=FINAL_AUDIT_REPORT_FILE_SHA256,
        context="audit report binding",
        relative=True,
    )

    if (
        report["format_version"] != 1
        or report["mode"] != "wave1_stage1_terminal_independent_audit"
        or report["decision"] != "PASS_TERMINAL_REJECT_ALL"
        or report["severity_counts"] != severity
    ):
        raise StructuralContractError("final audit report identity/decision differs")
    terminal_binding = report["terminal_binding"]
    if (
        not isinstance(terminal_binding, Mapping)
        or terminal_binding.get("terminal_manifest_sha256") != FINAL_TERMINAL_FILE_SHA256
        or terminal_binding.get("terminal_manifest_logical_sha256") != FINAL_TERMINAL_LOGICAL_SHA256
    ):
        raise StructuralContractError("final audit report terminal binding differs")
    evidence_access = report["evidence_access"]
    if (
        not isinstance(evidence_access, Mapping)
        or evidence_access.get("spent_seeds") != expected_seeds
        or evidence_access.get("fresh_seeds_consumed") is not False
        or evidence_access.get("heldout_opened") is not False
    ):
        raise StructuralContractError("final audit report seed/heldout boundary differs")
    registry = report["registry"]
    if (
        not isinstance(registry, Mapping)
        or registry.get("stage2_tuning_authorized") is not False
        or registry.get("lock_or_promotion_authorized") is not False
        or registry.get("advanced_model_count") != 0
    ):
        raise StructuralContractError("final audit report registry terminal state differs")

    if (
        trigger["format_version"] != 1
        or trigger["mode"] != "wave1_spent_surface_structural_trigger_inputs"
    ):
        raise StructuralContractError("final trigger-input identity differs")
    trigger_design = trigger["structural_design_observed"]
    if (
        not isinstance(trigger_design, Mapping)
        or trigger_design.get("sha256") != STRUCTURAL_DESIGN_SHA256
    ):
        raise StructuralContractError("trigger inputs use a different structural design")
    trigger_terminal = trigger["wave1_terminal_manifest"]
    if not isinstance(trigger_terminal, Mapping):
        raise StructuralContractError("trigger terminal binding is invalid")
    _require_record(
        trigger_terminal,
        expected_path=terminal_path,
        expected_sha256=FINAL_TERMINAL_FILE_SHA256,
        context="trigger terminal binding",
    )
    surface = trigger["surface"]
    if (
        not isinstance(surface, Mapping)
        or set(surface)
        != {"seeds", "rows_per_seed", "rows", "fresh_seeds_consumed", "heldout_opened"}
        or surface["seeds"] != expected_seeds
        or surface["rows_per_seed"] != 1296
        or surface["rows"] != 6480
        or surface["fresh_seeds_consumed"] is not False
        or surface["heldout_opened"] is not False
    ):
        raise StructuralContractError("trigger-input spent-seed surface differs")

    comparator = trigger["current_v04_comparator"]
    if not isinstance(comparator, Mapping):
        raise StructuralContractError("current-v04 trigger evidence is invalid")
    comparator_fields = {
        "model_id",
        "signed_residual_definition",
        "pooled_fair_log_bias",
        "pooled_bias_sign",
        "same_nonzero_sign_seed_count",
        "per_seed",
        "lag1_pairing",
        "total_lag1_pairs",
        "median_per_seed_lag1_autocorrelation",
        "seed_count_lag1_autocorrelation_at_least_0_15",
    }
    if set(comparator) != comparator_fields:
        raise StructuralContractError("current-v04 trigger evidence schema changed")
    if (
        comparator["model_id"] != "v04_expected_pe"
        or comparator["signed_residual_definition"] != "log(prediction/true_fair_pe)"
    ):
        raise StructuralContractError("current-v04 residual definition/base differs")
    per_seed = comparator["per_seed"]
    if not isinstance(per_seed, list) or len(per_seed) != SPENT_SEED_COUNT:
        raise StructuralContractError("current-v04 trigger evidence needs five seed rows")
    per_seed_fields = {
        "seed",
        "rows",
        "lag1_pair_count",
        "fair_log_bias",
        "bias_sign",
        "lag1_signed_fair_log_residual_autocorrelation",
    }
    if any(not isinstance(row, Mapping) or set(row) != per_seed_fields for row in per_seed):
        raise StructuralContractError("current-v04 per-seed trigger schema changed")
    if [row["seed"] for row in per_seed] != expected_seeds:
        raise StructuralContractError("current-v04 per-seed order/universe differs")
    if any(row["rows"] != 1296 or row["lag1_pair_count"] != 1295 for row in per_seed):
        raise StructuralContractError("current-v04 per-seed row/pair counts differ")
    biases = tuple(float(row["fair_log_bias"]) for row in per_seed)
    correlations = tuple(
        float(row["lag1_signed_fair_log_residual_autocorrelation"]) for row in per_seed
    )
    pooled_bias = float(comparator["pooled_fair_log_bias"])
    pooled_sign = 1 if pooled_bias > 0.0 else -1 if pooled_bias < 0.0 else 0
    matching_sign = sum(
        value != 0.0 and (1 if value > 0.0 else -1) == pooled_sign for value in biases
    )
    if not math.isclose(pooled_bias, sum(biases) / len(biases), rel_tol=0.0, abs_tol=1e-15):
        raise StructuralContractError("pooled bias does not recompute from equal seed surfaces")
    if (
        comparator["pooled_bias_sign"] != pooled_sign
        or comparator["same_nonzero_sign_seed_count"] != matching_sign
        or comparator["total_lag1_pairs"] != sum(int(row["lag1_pair_count"]) for row in per_seed)
        or not math.isclose(
            float(comparator["median_per_seed_lag1_autocorrelation"]),
            median(correlations),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or comparator["seed_count_lag1_autocorrelation_at_least_0_15"]
        != sum(value >= SERIAL_PER_SEED_MIN for value in correlations)
    ):
        raise StructuralContractError("bias/serial trigger summaries do not recompute")

    pair_selection = trigger["pair_selection"]
    if not isinstance(pair_selection, Mapping):
        raise StructuralContractError("pair-selection trigger evidence is invalid")
    family_pool = pair_selection.get("family_pool")
    selected_pair = pair_selection.get("selected_pair")
    qualifier = pair_selection.get("qualifier")
    if (
        not isinstance(family_pool, list)
        or not isinstance(selected_pair, Mapping)
        or not isinstance(qualifier, Mapping)
    ):
        raise StructuralContractError("pair-selection trigger evidence schema is invalid")
    pool_fields = {
        "family",
        "model_id",
        "fair_log_mae",
        "fair_log_rmse",
        "fair_abs_log_error_p95",
        "worst_seed_fair_log_mae",
        "runtime_seconds",
    }
    if any(not isinstance(row, Mapping) or set(row) != pool_fields for row in family_pool):
        raise StructuralContractError("pair family-pool schema changed")
    if len({row["family"] for row in family_pool}) != len(family_pool):
        raise StructuralContractError("pair pool contains more than one model per family")
    primary = min(
        family_pool,
        key=lambda row: (
            float(row["fair_log_mae"]),
            float(row["fair_log_rmse"]),
            float(row["fair_abs_log_error_p95"]),
            float(row["worst_seed_fair_log_mae"]),
            float(row["runtime_seconds"]),
            str(row["model_id"]),
        ),
    )
    if pair_selection.get("primary_model_id") != primary["model_id"]:
        raise StructuralContractError("pair primary base does not follow the frozen rank tuple")
    expected_qualifier = {
        "min_oracle_relative_gain_vs_best_single": COMPLEMENTARITY_MIN_ORACLE_GAIN,
        "max_absolute_error_correlation": COMPLEMENTARITY_MAX_ABS_ERROR_CORRELATION,
        "prediction_disagreement_frequency_strictly_greater_than": 0.0,
        "required_natural_coverage": 1.0,
    }
    if dict(qualifier) != expected_qualifier:
        raise StructuralContractError("pair qualifier thresholds differ from the design")
    primary_id = str(pair_selection["primary_model_id"])
    secondary_id = str(pair_selection["selected_secondary_model_id"])
    pool_by_id = {str(row["model_id"]): row for row in family_pool}
    if primary_id not in pool_by_id or secondary_id not in pool_by_id:
        raise StructuralContractError("selected pair is not contained in the eligible family pool")
    if pool_by_id[primary_id]["family"] == pool_by_id[secondary_id]["family"]:
        raise StructuralContractError("selected pair does not use distinct families")
    selected_ids = {str(selected_pair.get("model_a")), str(selected_pair.get("model_b"))}
    if selected_ids != {primary_id, secondary_id}:
        raise StructuralContractError("selected pair ids differ from primary/secondary binding")
    pair_qualifies = bool(
        float(selected_pair["oracle_relative_gain_vs_best_single"])
        >= COMPLEMENTARITY_MIN_ORACLE_GAIN
        and abs(float(selected_pair["absolute_error_correlation"]))
        <= COMPLEMENTARITY_MAX_ABS_ERROR_CORRELATION
        and float(selected_pair["prediction_disagreement_frequency"]) > 0.0
        and int(selected_pair["common_rows"]) == int(surface["rows"])
    )
    if (
        selected_pair.get("qualifies") is not True
        or not pair_qualifies
        or pair_selection.get("status") != "QUALIFIED_COMPLEMENTARY_PAIR"
        or int(pair_selection.get("qualifying_pair_count", 0)) < 1
    ):
        raise StructuralContractError("selected complementarity pair does not pass frozen gates")

    direct = trigger["direct_trigger_inputs"]
    direct_fields = {
        "best_direct_model_id",
        "mae_relative_gain_vs_stronger_primary_comparator",
        "rmse_relative_gain_vs_stronger_primary_comparator",
        "p95_relative_worsening_vs_stronger_primary_comparator",
        "worst_seed_relative_degradation",
        "no_direct_primary_path_pass",
    }
    if not isinstance(direct, Mapping) or set(direct) != direct_fields:
        raise StructuralContractError("direct trigger evidence schema changed")
    mae_gain = float(direct["mae_relative_gain_vs_stronger_primary_comparator"])
    rmse_gain = float(direct["rmse_relative_gain_vs_stronger_primary_comparator"])
    p95_worsening = float(direct["p95_relative_worsening_vs_stronger_primary_comparator"])
    worst_degradation = float(direct["worst_seed_relative_degradation"])
    if not all(
        math.isfinite(value) for value in (mae_gain, rmse_gain, p95_worsening, worst_degradation)
    ):
        raise StructuralContractError("direct trigger metrics must be finite")
    if direct["best_direct_model_id"] != primary_id:
        raise StructuralContractError("direct best model differs from the frozen rank primary")
    direct_trigger = bool(
        direct["no_direct_primary_path_pass"] is True
        or p95_worsening > P95_MAX_DEGRADATION
        or worst_degradation > WORST_SEED_MAX_DEGRADATION
    )
    stable_bias = bool(
        abs(pooled_bias) >= STABLE_BIAS_MIN_ABS_POOLED
        and pooled_sign != 0
        and matching_sign >= STABLE_BIAS_REQUIRED_SEEDS
    )
    serial = bool(
        median(correlations) >= SERIAL_MEDIAN_MIN
        and sum(value >= SERIAL_PER_SEED_MIN for value in correlations) >= SERIAL_REQUIRED_SEEDS
    )
    recomputed_tuple = {
        "T_A_TRACK_GAP": True,
        "T_DIRECT_PLATEAU_OR_TAIL": direct_trigger,
        "T_COMPLEMENTARITY": pair_qualifies,
        "T_STABLE_BIAS": stable_bias,
        "T_SERIAL_RESIDUAL": serial,
    }
    declared_tuple = trigger["trigger_decisions_read_only"]
    if not isinstance(declared_tuple, Mapping) or dict(declared_tuple) != recomputed_tuple:
        raise StructuralContractError("declared structural triggers differ from recomputation")
    report_trigger = report["structural_trigger_inputs"]
    if (
        not isinstance(report_trigger, Mapping)
        or report_trigger.get("structural_design_sha256") != STRUCTURAL_DESIGN_SHA256
        or report_trigger.get("trigger_inputs_manifest_sha256")
        != FINAL_TRIGGER_INPUT_LOGICAL_SHA256
        or report_trigger.get("trigger_tuple") != recomputed_tuple
    ):
        raise StructuralContractError("independent report trigger binding differs")

    # Normalize the audited relative direct metrics to unit reference losses so
    # the shared resolver recomputes the same frozen formulas without inventing
    # or opening any row-level score artifact.
    metric_comparator = TriggerMetricRow(
        model_id="stronger_primary_comparator",
        family="audited_comparator",
        fair_log_mae=1.0,
        fair_log_rmse=1.0,
        fair_abs_log_error_p95=1.0,
        worst_seed_fair_log_mae=1.0,
        runtime_seconds=0.0,
        natural_coverage=1.0,
    )
    best_direct = TriggerMetricRow(
        model_id=primary_id,
        family=str(pool_by_id[primary_id]["family"]),
        fair_log_mae=1.0 - mae_gain,
        fair_log_rmse=1.0 - rmse_gain,
        fair_abs_log_error_p95=1.0 + p95_worsening,
        worst_seed_fair_log_mae=1.0 + worst_degradation,
        runtime_seconds=0.0,
        natural_coverage=1.0,
    )
    inputs = Wave1TriggerInputs(
        track_a_eligible_supervised_baseline_count=0,
        primary_comparator=metric_comparator,
        direct_learners=(best_direct,),
        complementarity_pair=ComplementarityPairInput(
            base0_model_id=primary_id,
            base1_model_id=secondary_id,
            distinct_families=True,
            same_track_c_cutoff=True,
            oracle_relative_gain_vs_best_single=float(
                selected_pair["oracle_relative_gain_vs_best_single"]
            ),
            absolute_error_correlation=float(selected_pair["absolute_error_correlation"]),
            prediction_disagreement_frequency=float(
                selected_pair["prediction_disagreement_frequency"]
            ),
            base0_natural_coverage=1.0,
            base1_natural_coverage=1.0,
        ),
        selected_residual_base_model_id=str(comparator["model_id"]),
        pooled_fair_log_bias=pooled_bias,
        per_seed_fair_log_bias=biases,
        per_seed_contiguous_lag1_residual_correlation=correlations,
    )
    return VerifiedWave1TerminalEvidence(
        token=_VERIFICATION_TOKEN,
        inputs=inputs,
        terminal_file_sha256=FINAL_TERMINAL_FILE_SHA256,
        terminal_manifest_sha256=FINAL_TERMINAL_LOGICAL_SHA256,
        audit_file_sha256=FINAL_AUDIT_MANIFEST_FILE_SHA256,
        audit_manifest_sha256=FINAL_AUDIT_MANIFEST_LOGICAL_SHA256,
        trigger_inputs_sha256=FINAL_TRIGGER_INPUT_LOGICAL_SHA256,
        bound_artifact_sha256s=(
            ("wave1_terminal_manifest", FINAL_TERMINAL_FILE_SHA256),
            ("independent_audit_manifest", FINAL_AUDIT_MANIFEST_FILE_SHA256),
            ("independent_audit_report", FINAL_AUDIT_REPORT_FILE_SHA256),
            ("structural_trigger_inputs", FINAL_TRIGGER_INPUT_FILE_SHA256),
        ),
    )


def _relative_gain(reference: float, candidate: float) -> float:
    if reference <= 0.0:
        raise StructuralContractError("trigger reference loss must be positive")
    return (reference - candidate) / reference


@dataclass(frozen=True)
class TriggerResolution:
    triggers: tuple[tuple[str, bool], ...]
    enabled_candidates: tuple[str, ...]
    disabled_candidates: tuple[str, ...]
    selected_pair: tuple[str, str] | None
    selected_residual_base_model_id: str
    evidence_terminal_file_sha256: str
    evidence_audit_file_sha256: str
    trigger_inputs_sha256: str
    bound_artifact_sha256s: tuple[tuple[str, str], ...]
    design_sha256: str = STRUCTURAL_DESIGN_SHA256

    def __post_init__(self) -> None:
        if tuple(key for key, _ in self.triggers) != TRIGGER_IDS:
            raise StructuralContractError("trigger resolution order/schema changed")
        if set(self.enabled_candidates).intersection(self.disabled_candidates):
            raise StructuralContractError("candidate cannot be both enabled and disabled")
        if set(self.enabled_candidates).union(self.disabled_candidates) != set(
            CANDIDATE_TRIGGER_MAP
        ):
            raise StructuralContractError("trigger resolution candidate universe changed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "mode": "structural_trigger_resolution_in_memory",
            "design_sha256": self.design_sha256,
            "triggers": {key: value for key, value in self.triggers},
            "enabled_candidates": list(self.enabled_candidates),
            "disabled_candidates": list(self.disabled_candidates),
            "selected_pair": list(self.selected_pair) if self.selected_pair else None,
            "selected_residual_base_model_id": self.selected_residual_base_model_id,
            "evidence_terminal_file_sha256": self.evidence_terminal_file_sha256,
            "evidence_audit_file_sha256": self.evidence_audit_file_sha256,
            "trigger_inputs_sha256": self.trigger_inputs_sha256,
            "bound_artifact_sha256s": {key: value for key, value in self.bound_artifact_sha256s},
        }


def resolve_structural_triggers(
    evidence: VerifiedWave1TerminalEvidence,
) -> TriggerResolution:
    if not isinstance(evidence, VerifiedWave1TerminalEvidence):
        raise StructuralContractError("trigger resolver accepts verified terminal evidence only")
    inputs = evidence.inputs
    comparator = inputs.primary_comparator
    direct = tuple(sorted(inputs.direct_learners, key=lambda row: row.rank_tuple))
    qualifying_primary_path = []
    for row in direct:
        mae_gain = _relative_gain(comparator.fair_log_mae, row.fair_log_mae)
        rmse_gain = _relative_gain(comparator.fair_log_rmse, row.fair_log_rmse)
        qualifies = (
            mae_gain >= PRIMARY_GAIN_THRESHOLD and rmse_gain >= -OTHER_PRIMARY_MAX_DEGRADATION
        ) or (rmse_gain >= PRIMARY_GAIN_THRESHOLD and mae_gain >= -OTHER_PRIMARY_MAX_DEGRADATION)
        qualifying_primary_path.append(qualifies)
    best = direct[0]
    p95_degradation = -_relative_gain(
        comparator.fair_abs_log_error_p95, best.fair_abs_log_error_p95
    )
    worst_degradation = -_relative_gain(
        comparator.worst_seed_fair_log_mae, best.worst_seed_fair_log_mae
    )
    direct_trigger = (
        not any(qualifying_primary_path)
        or p95_degradation > P95_MAX_DEGRADATION
        or worst_degradation > WORST_SEED_MAX_DEGRADATION
    )

    pair = inputs.complementarity_pair
    complementarity = bool(
        pair is not None
        and pair.distinct_families
        and pair.same_track_c_cutoff
        and pair.oracle_relative_gain_vs_best_single >= COMPLEMENTARITY_MIN_ORACLE_GAIN
        and abs(pair.absolute_error_correlation) <= COMPLEMENTARITY_MAX_ABS_ERROR_CORRELATION
        and pair.prediction_disagreement_frequency > 0.0
        and pair.base0_natural_coverage == 1.0
        and pair.base1_natural_coverage == 1.0
    )

    pooled_bias = inputs.pooled_fair_log_bias
    pooled_sign = 1 if pooled_bias > 0.0 else -1 if pooled_bias < 0.0 else 0
    matching_bias_seeds = sum(
        1
        for value in inputs.per_seed_fair_log_bias
        if value != 0.0 and (1 if value > 0.0 else -1) == pooled_sign
    )
    stable_bias = (
        abs(pooled_bias) >= STABLE_BIAS_MIN_ABS_POOLED
        and pooled_sign != 0
        and matching_bias_seeds >= STABLE_BIAS_REQUIRED_SEEDS
    )

    correlations = inputs.per_seed_contiguous_lag1_residual_correlation
    serial_residual = (
        median(correlations) >= SERIAL_MEDIAN_MIN
        and sum(value >= SERIAL_PER_SEED_MIN for value in correlations) >= SERIAL_REQUIRED_SEEDS
    )
    trigger_values = {
        "T_A_TRACK_GAP": inputs.track_a_eligible_supervised_baseline_count == 0,
        "T_DIRECT_PLATEAU_OR_TAIL": direct_trigger,
        "T_COMPLEMENTARITY": complementarity,
        "T_STABLE_BIAS": stable_bias,
        "T_SERIAL_RESIDUAL": serial_residual,
    }
    enabled: list[str] = []
    disabled: list[str] = []
    for candidate_id, trigger_ids in CANDIDATE_TRIGGER_MAP.items():
        # Track-C decomposition is the only OR-trigger candidate; all others have one trigger.
        if any(trigger_values[trigger_id] for trigger_id in trigger_ids):
            enabled.append(candidate_id)
        else:
            disabled.append(candidate_id)
    selected_pair = (
        (pair.base0_model_id, pair.base1_model_id) if complementarity and pair is not None else None
    )
    return TriggerResolution(
        triggers=tuple(
            (trigger_id, bool(trigger_values[trigger_id])) for trigger_id in TRIGGER_IDS
        ),
        enabled_candidates=tuple(enabled),
        disabled_candidates=tuple(disabled),
        selected_pair=selected_pair,
        selected_residual_base_model_id=inputs.selected_residual_base_model_id,
        evidence_terminal_file_sha256=evidence.terminal_file_sha256,
        evidence_audit_file_sha256=evidence.audit_file_sha256,
        trigger_inputs_sha256=evidence.trigger_inputs_sha256,
        bound_artifact_sha256s=evidence.bound_artifact_sha256s,
    )
