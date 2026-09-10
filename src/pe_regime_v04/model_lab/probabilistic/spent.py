"""Fixed authority and policy for the five-seed probabilistic spent screen."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    PROBABILISTIC_DESIGN_SHA256,
    ProbabilisticContractError,
    require_sha256,
    sha256_bytes,
    verify_payload_seal,
)
from .spec import CANDIDATE_IDS


FORMAL_ACTIVATION_SCHEMA = "expected_pe_model_zoo.probabilistic_spent_activation.v1"
SPENT_SEEDS = (6301, 6421, 6521, 6607, 6701)
ENTITY_ID = "DEMO"
WORKER_COUNT = 32
INNER_THREADS = 1
FOLDS_PER_SEED = 62
ROWS_PER_SEED = 1296

INDEPENDENT_GO = {
    "path": "outputs/model_zoo_probabilistic_wave_fifth_independent_audit_20260819/AUDIT.json",
    "raw_sha256": "2693aa03807aa26f821f17263bc1d52c540297247b6d407cef4e3753ea44c391",
    "logical_sha256": "65a5d23e07b9c7b9b04e2502dee9b2eae0ffa4605a85e2827b0d62cc1abc700d",
}
V5_ROOTS = {
    "audit_candidate": {
        "path": "outputs/model_zoo_probabilistic_wave_screen_20260819/AUDIT_CANDIDATE_V5.json",
        "raw_sha256": "aa2aca662364b1d6a425c28f0cc578707c0b77e7b8ad35ce8825cfdc0f0f1551",
    },
    "checksums": {
        "path": "outputs/model_zoo_probabilistic_wave_screen_20260819/CHECKSUMS_V5.sha256",
        "raw_sha256": "d6048bc2fded274cddece60dd5e43a9416303581bce296e44f7af57659d342d1",
    },
    "score_free_authorization": {
        "path": "outputs/model_zoo_probabilistic_wave_screen_20260819/"
        "SCORE_FREE_EXECUTION_AUTHORIZATION."
        "be85fbebfbc56ac6c98a7a7f2b609f7fe0e879b6926d6439d66c30772eec4cea.json",
        "raw_sha256": "be85fbebfbc56ac6c98a7a7f2b609f7fe0e879b6926d6439d66c30772eec4cea",
    },
    "score_free_source_snapshot": {
        "path": "outputs/model_zoo_probabilistic_wave_screen_20260819/"
        "SCORE_FREE_SOURCE_SNAPSHOT."
        "45bfb5d76a1ad784381d3d51847d0e24d5be1c0cca8b9aaeb9f6722c941918cc.json",
        "raw_sha256": "45bfb5d76a1ad784381d3d51847d0e24d5be1c0cca8b9aaeb9f6722c941918cc",
    },
    "score_free_pointer": {
        "path": "outputs/model_zoo_probabilistic_wave_screen_20260819/"
        "SCORE_FREE_AUTHORIZATION_POINTER."
        "4b585139299eac038d890e653db5bda2d6d7ebb43883baab88e156f7d87f53ea.json",
        "raw_sha256": "4b585139299eac038d890e653db5bda2d6d7ebb43883baab88e156f7d87f53ea",
    },
}
UPSTREAM_INPUTS = {
    "predict": {
        "path": "outputs/model_zoo_wave1_screen_20260819/PREDICT_INPUTS.json",
        "raw_sha256": "f8f30ab74d812909b644984ba9ee3c4a8f509dfabb2ab566c7048313245d500c",
        "logical_sha256": "b08dd0a63c9b02ef14f06b3fd48764ac44c954100be5f360d03df7390aa494e8",
    },
    "evaluate": {
        "path": "outputs/model_zoo_wave1_screen_20260819/EVALUATE_INPUTS.json",
        "raw_sha256": "6cc6c750905784d615dc5ba200843b296b6f29fe1a1209d859da5fd266eda98d",
        "logical_sha256": "5f39e7495f81485695cb38443e5a8771bf76406e10d735e2d221745861f2f70a",
    },
}


def _read_exact_json(path: Path, *, expected_raw_sha256: str, context: str) -> dict[str, Any]:
    require_sha256(expected_raw_sha256, field=f"{context}.raw_sha256")
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError(f"{context} is unavailable or invalid") from exc
    if sha256_bytes(raw) != expected_raw_sha256:
        raise ProbabilisticContractError(f"{context} raw bytes differ from the pin")
    if not isinstance(payload, dict):
        raise ProbabilisticContractError(f"{context} must be an object")
    return payload


def _verify_file_root(root: Path, record: Mapping[str, str], *, context: str) -> None:
    relative = record.get("path")
    expected = record.get("raw_sha256")
    if not isinstance(relative, str):
        raise ProbabilisticContractError(f"{context} path is invalid")
    require_sha256(expected, field=f"{context}.raw_sha256")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
        raw = path.read_bytes()
    except (ValueError, OSError) as exc:
        raise ProbabilisticContractError(f"{context} root is unavailable") from exc
    if sha256_bytes(raw) != expected:
        raise ProbabilisticContractError(f"{context} root bytes drifted")


def verify_fixed_external_authorities(
    repo_root: Path, *, include_evaluation_input: bool = False
) -> None:
    """Reverify the independent GO, V5 roots, and exact spent input manifests."""

    root = Path(repo_root).resolve()
    audit_path = root / INDEPENDENT_GO["path"]
    audit = _read_exact_json(
        audit_path,
        expected_raw_sha256=INDEPENDENT_GO["raw_sha256"],
        context="independent V5 GO",
    )
    if (
        audit.get("manifest_sha256") != INDEPENDENT_GO["logical_sha256"]
        or audit.get("status") != "FINAL_GO"
        or audit.get("decision", {}).get("spent_screen_activation") != "GO"
        or audit.get("decision", {}).get("p0_count") != 0
        or audit.get("decision", {}).get("p1_count") != 0
    ):
        raise ProbabilisticContractError("independent V5 GO authority differs")
    for name, record in V5_ROOTS.items():
        _verify_file_root(root, record, context=f"V5.{name}")
    selected_inputs = ("predict", "evaluate") if include_evaluation_input else ("predict",)
    for name in selected_inputs:
        record = UPSTREAM_INPUTS[name]
        _verify_file_root(root, record, context=f"upstream.{name}")
        payload = json.loads((root / record["path"]).read_bytes())
        if payload.get("manifest_sha256") != record["logical_sha256"]:
            raise ProbabilisticContractError(f"upstream.{name} logical seal differs")
    predict = json.loads((root / UPSTREAM_INPUTS["predict"]["path"]).read_bytes())
    if (
        predict.get("mode") != "predict_inputs"
        or predict.get("evaluation_data_excluded") is not True
        or tuple(item.get("seed") for item in predict.get("seeds", ())) != SPENT_SEEDS
    ):
        raise ProbabilisticContractError("spent input role/seed authority differs")
    if include_evaluation_input:
        evaluate = json.loads((root / UPSTREAM_INPUTS["evaluate"]["path"]).read_bytes())
        if (
            evaluate.get("mode") != "evaluate_inputs"
            or evaluate.get("predict_process_must_not_receive_this_manifest") is not True
            or tuple(item.get("seed") for item in evaluate.get("seeds", ())) != SPENT_SEEDS
        ):
            raise ProbabilisticContractError("spent evaluation input role/seed authority differs")


def exact_policy() -> dict[str, Any]:
    return {
        "candidates": list(CANDIDATE_IDS),
        "spent_seeds": list(SPENT_SEEDS),
        "entity_id": ENTITY_ID,
        "folds_per_seed": FOLDS_PER_SEED,
        "rows_per_seed": ROWS_PER_SEED,
        "outer_workers": WORKER_COUNT,
        "inner_threads": INNER_THREADS,
        "process_start_method": "spawn",
        "gpu": "OFF",
        "fit_attempts_per_fold": 1,
        "retry_allowed": False,
        "candidate_substitution_allowed": False,
        "fresh_seed_reserved_or_opened": False,
        "heldout_opened": False,
        "registry_scope": "RESEARCH_ONLY_RESULT_REVISIONS",
        "promotion_authorized": False,
        "post_score_tuning_authorized": False,
    }


def verify_formal_activation(
    path: Path,
    *,
    expected_raw_sha256: str,
    repo_root: Path,
    authorization_raw_sha256: str,
    source_snapshot_raw_sha256: str,
    identity_raw_sha256: str,
    authorization_bindings: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Verify the atomically published external policy before formal capability use."""

    from .formal_pins import PRIOR_EXTERNAL_ROOTS, PRIOR_NO_GO

    payload = _read_exact_json(
        path,
        expected_raw_sha256=expected_raw_sha256,
        context="formal spent activation",
    )
    if expected_raw_sha256 not in Path(path).name.split("."):
        raise ProbabilisticContractError("formal activation is not content-addressed")
    verify_payload_seal(payload)
    verify_fixed_external_authorities(repo_root)
    if (
        payload.get("schema_version") != FORMAL_ACTIVATION_SCHEMA
        or payload.get("design_sha256") != PROBABILISTIC_DESIGN_SHA256
        or payload.get("status") != "ACTIVE_BEFORE_ANY_PROBABILISTIC_PREDICTION"
        or payload.get("independent_go") != INDEPENDENT_GO
        or payload.get("v5_roots") != V5_ROOTS
        or payload.get("upstream_inputs") != UPSTREAM_INPUTS
        or payload.get("policy") != exact_policy()
        or payload.get("formal_authorization_raw_sha256") != authorization_raw_sha256
        or payload.get("formal_source_snapshot_raw_sha256") != source_snapshot_raw_sha256
        or payload.get("formal_identity_raw_sha256") != identity_raw_sha256
        or payload.get("formal_required_bindings") != json.loads(json.dumps(authorization_bindings))
        or payload.get("predecessor_external_roots") != PRIOR_EXTERNAL_ROOTS
        or payload.get("prior_no_go") != PRIOR_NO_GO
        or payload.get("probabilistic_predictions_existed_before_activation") is not False
    ):
        raise ProbabilisticContractError("formal activation/policy differs from fixed authority")
    return payload
