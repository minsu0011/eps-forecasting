"""Fail-closed authority for the V3 two-survivor spent-screen continuation.

This module is deliberately separate from the historical three-candidate execution
authorization.  It imports two completed V2 prediction custodies by exact raw hash;
it never converts the terminally broken NGBoost candidate into a synthetic receipt and
it never authorizes a refit or retry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contracts import (
    PROBABILISTIC_DESIGN_SHA256,
    ProbabilisticContractError,
    require_sha256,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .source_closure import VerifiedSourceClosure, load_verified_source_closure


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


CONTINUATION_ACTIVATION_SCHEMA = (
    "expected_pe_model_zoo.probabilistic_two_survivor_continuation_activation.v1"
)
CONTINUATION_REQUEST_SCHEMA = (
    "expected_pe_model_zoo.probabilistic_two_survivor_external_request.v1"
)
CONTINUATION_GO_SCHEMA = "independent_probabilistic_two_survivor_continuation_audit.v1"
CONTINUATION_POINTER_SCHEMA = (
    "expected_pe_model_zoo.probabilistic_two_survivor_continuation_pointer.v1"
)
REFERENCE_SURFACE_SCHEMA = (
    "expected_pe_model_zoo.probabilistic_two_survivor_reference_surface.v1"
)
REFERENCE_ATTESTATION_SCHEMA = (
    "expected_pe_model_zoo.probabilistic_two_survivor_reference_attestation.v1"
)
GLOBAL_RUNTIME_SCHEMA = (
    "expected_pe_model_zoo.probabilistic_two_survivor_global_runtime_receipt.v1"
)
BUNDLE_SCHEMA = "expected_pe_model_zoo.probabilistic_two_survivor_bundle.v1"

SURVIVOR_IDS = (
    "qlinear_l1_with_regime_v1",
    "qhistgb_with_regime_v1",
)
BROKEN_CANDIDATE_ID = "ngboost_normal_crps_with_regime_v1"
REFERENCE_IDS = (
    "rolling_log_quantiles_252",
    "point_residual_quantiles_252",
)
COMPARATOR_IDS = ("v04_expected_pe", "ml_expected_pe")
SPENT_SEEDS = (6301, 6421, 6521, 6607, 6701)
ROWS_PER_SEED = 1296
PREDICTION_ROWS = 6480
FOLDS_PER_SEED = 62


V2_ROOTS: Mapping[str, Mapping[str, Any]] = _deep_freeze(
    {
        "publication": {
            "path": (
                "outputs/model_zoo_probabilistic_wave_spent_screen_20260819/"
                "FORMAL_ACTIVATION_PUBLICATION_V2.json"
            ),
            "raw_sha256": (
                "6017fc678df460235f76d01ad3160465dd3a79a0ae1f34bc86ccff91768a538f"
            ),
        },
        "external_request": {
            "path": (
                "outputs/p5s/a2/FORMAL_EXTERNAL_EXECUTION_REQUEST."
                "f51737ac5f271cb0eb83574d34f12543ea6b795d4c0f7d10d09646547c1e1d30.json"
            ),
            "raw_sha256": (
                "f51737ac5f271cb0eb83574d34f12543ea6b795d4c0f7d10d09646547c1e1d30"
            ),
        },
        "pointer": {
            "path": (
                "outputs/p5s/a2/SPENT_FORMAL_AUTHORIZATION_POINTER."
                "cb89f45f15463e6de061b2d16cfe86cc35e2820822704d8c18eb24a4ea16c739.json"
            ),
            "raw_sha256": (
                "cb89f45f15463e6de061b2d16cfe86cc35e2820822704d8c18eb24a4ea16c739"
            ),
        },
        "activation": {
            "path": (
                "outputs/p5s/a2/FORMAL_ACTIVATION_POLICY_PIN."
                "a5e86b25788162ddb31dd3595c4b8ea75c14538a4ff81d95bcda95c035411868.json"
            ),
            "raw_sha256": (
                "a5e86b25788162ddb31dd3595c4b8ea75c14538a4ff81d95bcda95c035411868"
            ),
        },
        "authorization": {
            "path": (
                "outputs/p5s/a2/SPENT_FORMAL_EXECUTION_AUTHORIZATION."
                "1118e6eb77ee2b1817a3b7fd564a77cef93b249cb0eaa3c4a0f7d5e36928febf.json"
            ),
            "raw_sha256": (
                "1118e6eb77ee2b1817a3b7fd564a77cef93b249cb0eaa3c4a0f7d5e36928febf"
            ),
        },
        "source_snapshot": {
            "path": (
                "outputs/p5s/a2/SPENT_FORMAL_SOURCE_SNAPSHOT."
                "c7611fff08da61b4dba4e809d2c585f185de488fd07fb08721dd21f5518692c6.json"
            ),
            "raw_sha256": (
                "c7611fff08da61b4dba4e809d2c585f185de488fd07fb08721dd21f5518692c6"
            ),
            "logical_sha256": (
                "d3e86a27be9a662f3d2ceb70c0b139680e80daa4921f1d0599fde01e5c104452"
            ),
            "record_count": 58,
        },
        "predict_input_manifest": {
            "path": "outputs/p5s/i/SPENT_PREDICT_INPUTS_MANIFEST.json",
            "raw_sha256": (
                "8fd72abfc592a7df0c2cb50fbfb9bf6b9e3cd1e47c1200e1d7aa095022d133cd"
            ),
            "logical_sha256": (
                "a28f5fc8d11793cfb203cdb6f8cb6abecf07af23573a0b3fbe596c4e5bf5bb52"
            ),
        },
        "v2_independent_go": {
            "path": (
                "outputs/model_zoo_probabilistic_wave_formal_activation_v2_"
                "independent_reaudit_20260819/AUDIT.json"
            ),
            "raw_sha256": (
                "c4565ab03b949bec23bfd95fba50aef59ed50c29b315f98139ae39b14df970b5"
            ),
            "logical_sha256": (
                "9edb88658cea06c53f624edef57009f1220a8ed2e52d6f1c71cb3282a741c461"
            ),
        },
        "terminal_failure_receipt": {
            "path": (
                "outputs/model_zoo_probabilistic_wave_formal_execution_fail_closed_"
                "20260819/FAILURE.json"
            ),
            "raw_sha256": (
                "f99260f3bc36fd3a440d926305401c4b55fe472dc9d2c27e80252b52807dcc27"
            ),
            "logical_sha256": (
                "f7303b23a420b619946ab79c7d1d354382248cf43d81cb12c0d2fcc56c63b396"
            ),
        },
        "terminal_failure_audit": {
            "path": (
                "outputs/model_zoo_probabilistic_wave_spent_terminal_failure_"
                "independent_audit_20260819/AUDIT.json"
            ),
            "raw_sha256": (
                "cf08ec76b6928ac58d3e47d259a316eaf79a59ce2259bccb5be2f7f4c5379287"
            ),
            "logical_sha256": (
                "2166e9666b9f309bdd0354fe83808c1b2aa82e391e762caeaa6ff526bb88ac8f"
            ),
        },
    }
)


SURVIVOR_CUSTODY: Mapping[str, Mapping[str, Any]] = _deep_freeze(
    {
        "qlinear_l1_with_regime_v1": {
            "surface": {
                "path": "outputs/p5s/c/0/CANDIDATE_SURFACE.json",
                "raw_sha256": (
                    "e7fa93c5fbf3833e019e1bec92364ae415d53fee076ab7601594ef95722940e4"
                ),
                "logical_sha256": (
                    "146ac5eb075f387e6feb2af47984cb44b755097182a16bd2d115126364ba3c09"
                ),
            },
            "prediction": {
                "path": (
                    "outputs/p5s/c/0/predictions."
                    "7d6c7934330e7547118d4d64676389f6102025edb474c559fae5f7c0f05e2fc8.csv"
                ),
                "raw_sha256": (
                    "7d6c7934330e7547118d4d64676389f6102025edb474c559fae5f7c0f05e2fc8"
                ),
            },
            "prediction_receipt": {
                "path": (
                    "outputs/p5s/c/0/prediction_receipt."
                    "7950de7186f07144c2455d32e7b49562a42ff5b732283c8a94cb2d68bccaca81.json"
                ),
                "raw_sha256": (
                    "7950de7186f07144c2455d32e7b49562a42ff5b732283c8a94cb2d68bccaca81"
                ),
                "logical_sha256": (
                    "9acab08c50f22f8fbceefe28912f639b0fe13cc99c60dff00266cb1c62715dd4"
                ),
            },
            "runtime_receipt": {
                "path": (
                    "outputs/p5s/c/0/runtime."
                    "5a8d709c8e5998eac07e37dd49497de468885539e71498c3fc57339916949543.json"
                ),
                "raw_sha256": (
                    "5a8d709c8e5998eac07e37dd49497de468885539e71498c3fc57339916949543"
                ),
                "logical_sha256": (
                    "32590622a613b6dd88a64781076ac1488034961494d7860bb3730d0302451944"
                ),
            },
            "diagnostics": {
                "path": (
                    "outputs/p5s/c/0/raw_crossing_diagnostics."
                    "ca6953c93c2c530e990f71523264b5422cc1ef02a3fe57f039fc630aa6a2165e.csv"
                ),
                "raw_sha256": (
                    "ca6953c93c2c530e990f71523264b5422cc1ef02a3fe57f039fc630aa6a2165e"
                ),
            },
        },
        "qhistgb_with_regime_v1": {
            "surface": {
                "path": "outputs/p5s/c/1/CANDIDATE_SURFACE.json",
                "raw_sha256": (
                    "ed8769377917c05c135e3678f05c5b43d6eecd72d2cc9670b571a909ac9cb479"
                ),
                "logical_sha256": (
                    "a46a8a7280f93823a6361f3767cc1b958b4da48de532c6193f2d95ecd894d797"
                ),
            },
            "prediction": {
                "path": (
                    "outputs/p5s/c/1/predictions."
                    "786d648ee4145392ab58373947b580c4cc6a83f254ba42d810d5633a7398b10e.csv"
                ),
                "raw_sha256": (
                    "786d648ee4145392ab58373947b580c4cc6a83f254ba42d810d5633a7398b10e"
                ),
            },
            "prediction_receipt": {
                "path": (
                    "outputs/p5s/c/1/prediction_receipt."
                    "60b27bb0c15f30bd0f8da2f15376dc154ecb5d316d4c718e1268c69cc40ebbee.json"
                ),
                "raw_sha256": (
                    "60b27bb0c15f30bd0f8da2f15376dc154ecb5d316d4c718e1268c69cc40ebbee"
                ),
                "logical_sha256": (
                    "22c55b817d9f7c949efb98b68ba142b0204c250cd2e5adc3eff3ce59be11bbc1"
                ),
            },
            "runtime_receipt": {
                "path": (
                    "outputs/p5s/c/1/runtime."
                    "97008171c3f4af9c67f06a89b1c458782cff6b6949ae3d8b866a60ee1b64550f.json"
                ),
                "raw_sha256": (
                    "97008171c3f4af9c67f06a89b1c458782cff6b6949ae3d8b866a60ee1b64550f"
                ),
                "logical_sha256": (
                    "891338d6357bac1492bf633f976caa5e90f6bad7cb4d49acaec686dae397f1df"
                ),
            },
            "diagnostics": {
                "path": (
                    "outputs/p5s/c/1/raw_crossing_diagnostics."
                    "223b612233d5aac64ffac85f99c7e0dd474435b71167a470cf465570cc39ba94.csv"
                ),
                "raw_sha256": (
                    "223b612233d5aac64ffac85f99c7e0dd474435b71167a470cf465570cc39ba94"
                ),
            },
        },
    }
)


COMPARATOR_CUSTODY: Mapping[str, Mapping[str, Mapping[str, str]]] = _deep_freeze(
    {
        "v04_expected_pe": {
            "prediction": {
                "path": (
                    "outputs/p5s/i/SPENT_V04_COMPARATOR."
                    "e43e62af7f2f8140132043a3d0dbadd233fff20a0637f8465d7384f5899b0900.csv"
                ),
                "raw_sha256": (
                    "e43e62af7f2f8140132043a3d0dbadd233fff20a0637f8465d7384f5899b0900"
                ),
            },
            "receipt": {
                "path": (
                    "outputs/p5s/a2/point_comparator_receipt_v04_expected_pe."
                    "37d2cb712a3cee4f3dccd49f403f4cc3a4c5bf844b97bf741cf02e60a83ac008.json"
                ),
                "raw_sha256": (
                    "37d2cb712a3cee4f3dccd49f403f4cc3a4c5bf844b97bf741cf02e60a83ac008"
                ),
            },
        },
        "ml_expected_pe": {
            "prediction": {
                "path": (
                    "outputs/p5s/i/SPENT_ML_COMPARATOR."
                    "346853392b834e2fd8d9295463814fee40d7e3ec31381b63bf87836590baabff.csv"
                ),
                "raw_sha256": (
                    "346853392b834e2fd8d9295463814fee40d7e3ec31381b63bf87836590baabff"
                ),
            },
            "receipt": {
                "path": (
                    "outputs/p5s/a2/point_comparator_receipt_ml_expected_pe."
                    "90d02a616396bf62503843056b0ae878425ae65e15034a87d78bba6da17fbdfa.json"
                ),
                "raw_sha256": (
                    "90d02a616396bf62503843056b0ae878425ae65e15034a87d78bba6da17fbdfa"
                ),
            },
        },
    }
)


REGISTRY_TERMINAL_STATE: Mapping[str, Any] = _deep_freeze(
    {
        "append_receipt": {
            "path": (
                "outputs/model_zoo_probabilistic_wave_terminal_registry_append_20260819/"
                "REGISTRY_APPEND.json"
            ),
            "raw_sha256": (
                "0b4ba6d5fb1a1a535db48d68eb619e89424bddc0d390c0d86146c0de1a28dc44"
            ),
            "logical_sha256": (
                "9e68e3724a048799bb3e5452b5ae0adf2818b4a19d5f6c201e6bb1d063b502d3"
            ),
        },
        "model_registry_csv": {
            "path": "research/model_zoo/model_registry.csv",
            "raw_sha256": (
                "a4c827519f0eed1e1dbc2389ab1035df68803509f13ca87f05f71dc428997d6a"
            ),
        },
        "model_registry_json": {
            "path": "research/model_zoo/model_registry.json",
            "raw_sha256": (
                "752577e3dc37ad3be73a605b739450c4a79e721f8498288f0f5d12b65856142f"
            ),
            "logical_sha256": (
                "8b8503990c129845d646f3c2661f98cb3a64f8532fbdb197f9e2d32b2dfe3b77"
            ),
            "record_count": 41,
        },
        "recoverable_predecessor_csv": {
            "path": (
                "outputs/model_zoo_probabilistic_wave_terminal_registry_append_20260819/"
                "model_registry.before."
                "0cf9b9e568265419f372593d72900df3a3dbbea88087f243d66d62f4d0fca3f1.csv"
            ),
            "raw_sha256": (
                "0cf9b9e568265419f372593d72900df3a3dbbea88087f243d66d62f4d0fca3f1"
            ),
        },
        "recoverable_predecessor_json": {
            "path": (
                "outputs/model_zoo_probabilistic_wave_terminal_registry_append_20260819/"
                "model_registry.before."
                "2f7628a254fe711d3b004ee85e425cd02fd2bc1604405f83b5f600303b9e2c25.json"
            ),
            "raw_sha256": (
                "2f7628a254fe711d3b004ee85e425cd02fd2bc1604405f83b5f600303b9e2c25"
            ),
        },
    }
)


WORKFLOW_POLICY: Mapping[str, Any] = _deep_freeze(
    {
        "survivor_ids": list(SURVIVOR_IDS),
        "excluded_candidate": {
            "model_id": BROKEN_CANDIDATE_ID,
            "status": "BROKEN_PERMANENT_NO_RETRY_NO_FALLBACK",
        },
        "candidate_refit_or_retry_allowed": False,
        "candidate_prediction_receipts_required": 2,
        "probabilistic_reference_participants_required": 2,
        "reference_prediction_receipts_required": 2,
        "reference_execution_receipts_required": 2,
        "point_comparator_receipts_required": 2,
        "candidate_runtime_receipts_required": 2,
        "global_runtime_receipts_required": 1,
        "same_continuation_authority_required_everywhere": True,
        "reference_before_bundle": True,
        "bundle_before_truth": True,
        "truth_child_process_required": True,
        "truth_child_single_use": True,
        "parent_truth_read_allowed": False,
        "generic_three_candidate_bypass_allowed": False,
        "fresh_seed_reserved_or_opened": False,
        "heldout_opened": False,
        "registry_write_allowed": False,
    }
)


_TERMINAL_TOKEN = object()
_ACTIVATION_TOKEN = object()
_REQUEST_TOKEN = object()
_GO_TOKEN = object()
_AUTHORITY_TOKEN = object()


def _json_copy(value: Any) -> Any:
    return _thaw(value)


def _resolve(root: Path, relative: object, *, context: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ProbabilisticContractError(f"{context} path must be repository-relative")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProbabilisticContractError(f"{context} path escapes repository") from exc
    return path


def _read_exact(
    root: Path,
    record: Mapping[str, Any],
    *,
    context: str,
    require_addressed_name: bool = False,
) -> bytes:
    expected = require_sha256(record.get("raw_sha256"), field=f"{context}.raw_sha256")
    path = _resolve(root, record.get("path"), context=context)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError(f"{context} is unavailable") from exc
    if sha256_bytes(raw) != expected:
        raise ProbabilisticContractError(f"{context} raw bytes changed")
    if require_addressed_name and expected not in path.name.split("."):
        raise ProbabilisticContractError(f"{context} is not content-addressed")
    return raw


def verify_file_against_record(path: Path, record: Mapping[str, Any], *, context: str) -> None:
    """Small public tamper-test helper; it grants no execution capability."""

    expected = require_sha256(record.get("raw_sha256"), field=f"{context}.raw_sha256")
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise ProbabilisticContractError(f"{context} is unavailable") from exc
    if sha256_bytes(raw) != expected:
        raise ProbabilisticContractError(f"{context} raw bytes changed")


def _read_exact_json(
    root: Path,
    record: Mapping[str, Any],
    *,
    context: str,
    sealed: bool = True,
    require_addressed_name: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    raw = _read_exact(
        root,
        record,
        context=context,
        require_addressed_name=require_addressed_name,
    )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError(f"{context} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ProbabilisticContractError(f"{context} must be a JSON object")
    if sealed:
        verify_payload_seal(payload)
    logical = record.get("logical_sha256")
    if logical is not None and payload.get("manifest_sha256") != logical:
        raise ProbabilisticContractError(f"{context} logical seal changed")
    return raw, payload


def _verify_v2_roots(root: Path) -> None:
    payloads: dict[str, dict[str, Any]] = {}
    for name, record in V2_ROOTS.items():
        _, payload = _read_exact_json(
            root,
            record,
            context=f"V2.{name}",
            require_addressed_name=name
            in {"external_request", "pointer", "activation", "authorization", "source_snapshot"},
        )
        payloads[name] = payload
    if payloads["v2_independent_go"].get("status") != "FINAL_GO":
        raise ProbabilisticContractError("V2 independent GO is no longer FINAL_GO")
    source = payloads["source_snapshot"]
    if (
        source.get("manifest_sha256")
        != V2_ROOTS["source_snapshot"]["logical_sha256"]
        or len(source.get("records", ())) != V2_ROOTS["source_snapshot"]["record_count"]
    ):
        raise ProbabilisticContractError("V2 source snapshot record closure differs")
    failure = payloads["terminal_failure_receipt"]
    if (
        failure.get("status") != "TERMINAL_FAIL_CLOSED"
        or failure.get("failure", {}).get("candidate_id") != BROKEN_CANDIDATE_ID
        or failure.get("decision", {}).get("retry_authorized") is not False
        or failure.get("decision", {}).get("fallback_authorized") is not False
        or failure.get("failure", {}).get("partial_custody_exists", False) is not False
    ):
        raise ProbabilisticContractError("terminal NGBoost failure disposition changed")
    audit = payloads["terminal_failure_audit"]
    decision = audit.get("decision", {})
    if (
        decision.get("probabilistic_spent_screen") != "NO_GO_TERMINAL_INCOMPLETE"
        or decision.get("completed_candidate_custody_accepted") is not True
        or decision.get("ngboost_candidate_status") != "BROKEN"
        or decision.get("evaluation_or_gate_authorized") is not False
        or audit.get("ngboost_terminal_failure", {}).get("retry_attempted") is not False
        or audit.get("ngboost_terminal_failure", {}).get("fallback_attempted") is not False
    ):
        raise ProbabilisticContractError("terminal independent audit disposition changed")


def _verify_input_manifest(root: Path) -> dict[str, Any]:
    _, manifest = _read_exact_json(
        root,
        V2_ROOTS["predict_input_manifest"],
        context="spent predict-input manifest",
    )
    artifacts = manifest.get("artifacts")
    expected_names = {
        "common_mask",
        "feature",
        "feature_provenance",
        "identity",
        "label",
        "ml_comparator",
        "point_history",
        "role",
        "v04_comparator",
    }
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != expected_names
        or tuple(manifest.get("spent_seeds", ())) != SPENT_SEEDS
        or manifest.get("rows") != {"features": 9000, "identity": 6480, "labels": 9000}
        or manifest.get("truth_bytes_read") is not False
        or manifest.get("scores_generated_or_read") is not False
        or manifest.get("evaluation_manifest_read") is not False
        or manifest.get("fresh_seed_reserved_or_opened") is not False
        or manifest.get("heldout_opened") is not False
    ):
        raise ProbabilisticContractError("spent predict-input manifest policy changed")
    for name, item in artifacts.items():
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise ProbabilisticContractError(f"spent input record differs: {name}")
        raw = _read_exact(
            root,
            {"path": item["path"], "raw_sha256": item["sha256"]},
            context=f"spent input {name}",
            require_addressed_name=name != "predict_input_manifest",
        )
        if len(raw) != item["bytes"]:
            raise ProbabilisticContractError(f"spent input byte count changed: {name}")
    return manifest


def _verify_survivor(root: Path, model_id: str) -> None:
    if model_id not in SURVIVOR_IDS:
        raise ProbabilisticContractError("continuation candidate is not an exact survivor")
    records = SURVIVOR_CUSTODY[model_id]
    _, surface = _read_exact_json(root, records["surface"], context=f"{model_id}.surface")
    _read_exact(
        root,
        records["prediction"],
        context=f"{model_id}.prediction",
        require_addressed_name=True,
    )
    _, receipt = _read_exact_json(
        root,
        records["prediction_receipt"],
        context=f"{model_id}.prediction_receipt",
        require_addressed_name=True,
    )
    _, runtime = _read_exact_json(
        root,
        records["runtime_receipt"],
        context=f"{model_id}.runtime_receipt",
        require_addressed_name=True,
    )
    _read_exact(
        root,
        records["diagnostics"],
        context=f"{model_id}.diagnostics",
        require_addressed_name=True,
    )
    old_auth = V2_ROOTS["authorization"]["raw_sha256"]
    old_source = V2_ROOTS["source_snapshot"]["raw_sha256"]
    if (
        surface.get("model_id") != model_id
        or surface.get("status") != "PASS_FULL_MASK_FORMAL_PREDICTION_CUSTODY"
        or surface.get("prediction_raw_sha256") != records["prediction"]["raw_sha256"]
        or surface.get("receipt_raw_sha256") != records["prediction_receipt"]["raw_sha256"]
        or surface.get("runtime_receipt_raw_sha256")
        != records["runtime_receipt"]["raw_sha256"]
        or surface.get("source_closure_sha256") != old_source
        or surface.get("authorization_raw_sha256") != old_auth
        or surface.get("independent_go_raw_sha256")
        != V2_ROOTS["v2_independent_go"]["raw_sha256"]
        or surface.get("rows") != PREDICTION_ROWS
        or surface.get("folds_per_seed") != FOLDS_PER_SEED
        or tuple(surface.get("spent_seeds", ())) != SPENT_SEEDS
        or surface.get("process_pool_workers") != 32
        or surface.get("inner_threads") != 1
        or surface.get("gpu") != "OFF"
        or surface.get("fit_attempts_per_fold") != 1
        or surface.get("retry_permitted") is not False
        or surface.get("truth_bytes_read") is not False
        or surface.get("scores_generated_or_read") is not False
    ):
        raise ProbabilisticContractError(f"{model_id} surface custody changed")
    if (
        receipt.get("model_id") != model_id
        or receipt.get("authorization_raw_sha256") != old_auth
        or receipt.get("candidate_source_snapshot_sha256") != old_source
        or receipt.get("prediction_raw_sha256") != records["prediction"]["raw_sha256"]
        or receipt.get("manifest_sha256")
        != records["prediction_receipt"]["logical_sha256"]
    ):
        raise ProbabilisticContractError(f"{model_id} prediction receipt binding changed")
    provenance = runtime.get("provenance", {})
    metrics = runtime.get("metrics", {})
    if (
        runtime.get("participant_id") != model_id
        or runtime.get("authorization_raw_sha256") != old_auth
        or runtime.get("manifest_sha256") != records["runtime_receipt"]["logical_sha256"]
        or provenance.get("authorization_raw_sha256") != old_auth
        or provenance.get("source_closure_sha256") != old_source
        or provenance.get("prediction_raw_sha256") != records["prediction"]["raw_sha256"]
        or provenance.get("prediction_receipt_sha256")
        != records["prediction_receipt"]["raw_sha256"]
        or metrics.get("worker_count") != 32
        or metrics.get("common_mask_coverage") != 1.0
        or metrics.get("post_repair_crossing_rate") != 0.0
    ):
        raise ProbabilisticContractError(f"{model_id} runtime receipt binding changed")


def _verify_comparators(root: Path) -> None:
    old_auth = V2_ROOTS["authorization"]["raw_sha256"]
    old_source = V2_ROOTS["source_snapshot"]["raw_sha256"]
    for comparator_id in COMPARATOR_IDS:
        records = COMPARATOR_CUSTODY[comparator_id]
        _read_exact(
            root,
            records["prediction"],
            context=f"{comparator_id}.prediction",
            require_addressed_name=True,
        )
        _, receipt = _read_exact_json(
            root,
            records["receipt"],
            context=f"{comparator_id}.receipt",
            require_addressed_name=True,
        )
        if (
            receipt.get("comparator_id") != comparator_id
            or receipt.get("authorization_raw_sha256") != old_auth
            or receipt.get("source_closure_sha256") != old_source
            or receipt.get("comparator_raw_sha256")
            != records["prediction"]["raw_sha256"]
        ):
            raise ProbabilisticContractError(f"{comparator_id} receipt binding changed")


def _verify_registry_terminal_state(root: Path) -> None:
    for name, record in REGISTRY_TERMINAL_STATE.items():
        if name == "append_receipt":
            continue
        _read_exact(root, record, context=f"terminal registry.{name}")
    _, receipt = _read_exact_json(
        root,
        REGISTRY_TERMINAL_STATE["append_receipt"],
        context="terminal registry append receipt",
    )
    expected_after = {
        "csv_sha256": REGISTRY_TERMINAL_STATE["model_registry_csv"]["raw_sha256"],
        "json_sha256": REGISTRY_TERMINAL_STATE["model_registry_json"]["raw_sha256"],
        "logical_sha256": REGISTRY_TERMINAL_STATE["model_registry_json"][
            "logical_sha256"
        ],
        "records": REGISTRY_TERMINAL_STATE["model_registry_json"]["record_count"],
    }
    expected_registration_ids = [
        "qlinear_l1_with_regime_v1@sparse_linear_quantiles@prob-v5-ae2442-v1#0",
        "qlinear_l1_with_regime_v1@sparse_linear_quantiles@prob-v5-ae2442-v1#1",
        (
            "qhistgb_with_regime_v1@nonlinear_histogram_tree_quantiles@"
            "prob-v5-ae2442-v1#0"
        ),
        (
            "qhistgb_with_regime_v1@nonlinear_histogram_tree_quantiles@"
            "prob-v5-ae2442-v1#1"
        ),
        (
            "ngboost_normal_crps_with_regime_v1@conditional_normal_log_pe_crps@"
            "prob-v5-ae2442-v1#0"
        ),
        (
            "ngboost_normal_crps_with_regime_v1@conditional_normal_log_pe_crps@"
            "prob-v5-ae2442-v1#1"
        ),
    ]
    if (
        receipt.get("status") != "PASS_APPEND_ONLY_TERMINAL_STATE"
        or receipt.get("after") != expected_after
        or receipt.get("appended_registration_ids") != expected_registration_ids
        or receipt.get("terminal_failure_receipt")
        != _thaw(V2_ROOTS["terminal_failure_receipt"])
        or receipt.get("terminal_failure_audit")
        != _thaw(V2_ROOTS["terminal_failure_audit"])
        or receipt.get("csv_json_parity_verified") is not True
        or receipt.get("history_prefix_preserved") is not True
        or receipt.get("latest_projection_verified") is not True
        or receipt.get("metrics_generated_or_read") is not False
        or receipt.get("truth_opened") is not False
    ):
        raise ProbabilisticContractError("terminal registry append receipt differs")

    from ..registry import RegistryPaths, load_model_registry, load_model_registry_history

    paths = RegistryPaths(
        _resolve(
            root,
            REGISTRY_TERMINAL_STATE["model_registry_csv"]["path"],
            context="model registry CSV",
        ),
        _resolve(
            root,
            REGISTRY_TERMINAL_STATE["model_registry_json"]["path"],
            context="model registry JSON",
        ),
    )
    history = load_model_registry_history(paths)
    latest = {record.model_id: record for record in load_model_registry(paths)}
    if (
        len(history) != 41
        or [record.registration_id for record in history[-6:]]
        != expected_registration_ids
        or set(SURVIVOR_IDS).difference(latest)
        or BROKEN_CANDIDATE_ID not in latest
    ):
        raise ProbabilisticContractError("terminal registry history/latest semantics differ")
    for model_id in SURVIVOR_IDS:
        record = latest[model_id]
        required_hashes = SURVIVOR_CUSTODY[model_id]
        if (
            record.registry_revision != 1
            or record.status != "RESEARCH_ONLY"
            or record.tuning_status != "UNTESTED"
            or record.locked_status != "UNTESTED"
            or record.heldout_status != "NOT_OPENED"
            or any(
                value is not None
                for value in (
                    record.fair_log_mae,
                    record.fair_log_rmse,
                    record.worst_seed,
                    record.compute_time,
                )
            )
            or "result_status=UNSCORED_PREDICTION_CUSTODY_ONLY" not in record.notes
            or any(
                nested["raw_sha256"] not in record.notes
                for nested in (
                    required_hashes["surface"],
                    required_hashes["prediction"],
                    required_hashes["prediction_receipt"],
                    required_hashes["runtime_receipt"],
                )
            )
        ):
            raise ProbabilisticContractError(f"{model_id} registry state is not unscored")
    broken = latest[BROKEN_CANDIDATE_ID]
    if (
        broken.registry_revision != 1
        or broken.status != "BROKEN"
        or broken.tuning_status != "UNTESTED"
        or broken.locked_status != "UNTESTED"
        or broken.heldout_status != "NOT_OPENED"
        or any(
            value is not None
            for value in (
                broken.fair_log_mae,
                broken.fair_log_rmse,
                broken.worst_seed,
                broken.compute_time,
            )
        )
        or "result_status=BROKEN_TERMINAL_NO_RETRY_NO_FALLBACK" not in broken.notes
        or V2_ROOTS["terminal_failure_receipt"]["raw_sha256"] not in broken.notes
        or V2_ROOTS["terminal_failure_audit"]["raw_sha256"] not in broken.notes
        or V2_ROOTS["terminal_failure_audit"]["logical_sha256"] not in broken.notes
    ):
        raise ProbabilisticContractError("NGBoost registry state is not terminal BROKEN")


class VerifiedTerminalCustody:
    """Opaque proof that the terminal V2 audit and both survivor bytes still match."""

    __slots__ = ("_repo_root",)

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _TERMINAL_TOKEN:
            raise ProbabilisticContractError("terminal custody requires the exact-root loader")
        return super().__new__(cls)

    def __init__(self, token: object, *, repo_root: Path) -> None:
        if token is not _TERMINAL_TOKEN:
            raise ProbabilisticContractError("invalid terminal-custody factory token")
        object.__setattr__(self, "_repo_root", Path(repo_root).resolve())

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedTerminalCustody is immutable")

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    def verify_integrity(self) -> None:
        _verify_v2_roots(self._repo_root)
        _verify_input_manifest(self._repo_root)
        if tuple(SURVIVOR_CUSTODY) != SURVIVOR_IDS:
            raise ProbabilisticContractError("survivor custody set/order is not exactly two")
        for model_id in SURVIVOR_IDS:
            _verify_survivor(self._repo_root, model_id)
        _verify_comparators(self._repo_root)
        _verify_registry_terminal_state(self._repo_root)
        broken_directory = self._repo_root / "outputs/p5s/c/2"
        if broken_directory.exists():
            raise ProbabilisticContractError("NGBoost partial or replacement custody is forbidden")


def load_verified_terminal_custody(repo_root: Path) -> VerifiedTerminalCustody:
    custody = VerifiedTerminalCustody(_TERMINAL_TOKEN, repo_root=repo_root)
    custody.verify_integrity()
    return custody


def continuation_policy() -> dict[str, Any]:
    return _json_copy(WORKFLOW_POLICY)


def frozen_v2_roots() -> dict[str, Any]:
    return _json_copy(V2_ROOTS)


def frozen_survivor_custody() -> dict[str, Any]:
    return _json_copy(SURVIVOR_CUSTODY)


def frozen_comparator_custody() -> dict[str, Any]:
    return _json_copy(COMPARATOR_CUSTODY)


def frozen_registry_terminal_state() -> dict[str, Any]:
    return _json_copy(REGISTRY_TERMINAL_STATE)


def build_continuation_activation_payload(
    *, source_snapshot_path: str, source_snapshot_raw_sha256: str
) -> dict[str, Any]:
    require_sha256(source_snapshot_raw_sha256, field="continuation source snapshot")
    if not isinstance(source_snapshot_path, str) or source_snapshot_raw_sha256 not in Path(
        source_snapshot_path
    ).name.split("."):
        raise ProbabilisticContractError("continuation source snapshot is not content-addressed")
    return seal_payload(
        {
            "schema_version": CONTINUATION_ACTIVATION_SCHEMA,
            "design_sha256": PROBABILISTIC_DESIGN_SHA256,
            "status": "INDEPENDENT_AUDIT_REQUIRED_INACTIVE_SCORE_FREE",
            "formal": False,
            "heavy_compute_authorized": False,
            "reference_generation_authorized": False,
            "truth_open_authorized": False,
            "score_computation_authorized": False,
            "candidate_refit_or_retry_authorized": False,
            "v2_roots": frozen_v2_roots(),
            "survivor_custody": frozen_survivor_custody(),
            "comparator_custody": frozen_comparator_custody(),
            "terminal_registry_state": frozen_registry_terminal_state(),
            "workflow_policy": continuation_policy(),
            "continuation_source_snapshot": {
                "path": source_snapshot_path,
                "raw_sha256": source_snapshot_raw_sha256,
            },
            "output_namespaces": {
                "references": "outputs/p5s/v3/r",
                "bundle": "outputs/p5s/v3/b",
                "evaluator": "outputs/p5s/v3/e",
                "handoff": "outputs/p5s/v3/h",
            },
            "independent_go_required_before_reference_or_evaluation": True,
            "old_v2_authority_is_custody_only": True,
            "fresh_seed_reserved_or_opened": False,
            "heldout_opened": False,
            "registry_modified": False,
        }
    )


def build_continuation_request_payload(
    *,
    activation_path: str,
    activation_raw_sha256: str,
    source_snapshot_path: str,
    source_snapshot_raw_sha256: str,
) -> dict[str, Any]:
    require_sha256(activation_raw_sha256, field="continuation activation")
    require_sha256(source_snapshot_raw_sha256, field="continuation source snapshot")
    if activation_raw_sha256 not in Path(activation_path).name.split("."):
        raise ProbabilisticContractError("continuation activation is not content-addressed")
    return seal_payload(
        {
            "schema_version": CONTINUATION_REQUEST_SCHEMA,
            "design_sha256": PROBABILISTIC_DESIGN_SHA256,
            "status": "EXTERNAL_TWO_SURVIVOR_CONTINUATION_AUDIT_REQUEST",
            "activation": {
                "path": activation_path,
                "raw_sha256": activation_raw_sha256,
            },
            "source_snapshot": {
                "path": source_snapshot_path,
                "raw_sha256": source_snapshot_raw_sha256,
            },
            "v2_go_raw_sha256": V2_ROOTS["v2_independent_go"]["raw_sha256"],
            "terminal_failure_receipt_raw_sha256": V2_ROOTS[
                "terminal_failure_receipt"
            ]["raw_sha256"],
            "terminal_failure_audit_raw_sha256": V2_ROOTS["terminal_failure_audit"][
                "raw_sha256"
            ],
            "terminal_failure_audit_logical_sha256": V2_ROOTS[
                "terminal_failure_audit"
            ]["logical_sha256"],
            "terminal_registry_append_raw_sha256": REGISTRY_TERMINAL_STATE[
                "append_receipt"
            ]["raw_sha256"],
            "model_registry_csv_raw_sha256": REGISTRY_TERMINAL_STATE[
                "model_registry_csv"
            ]["raw_sha256"],
            "model_registry_json_raw_sha256": REGISTRY_TERMINAL_STATE[
                "model_registry_json"
            ]["raw_sha256"],
            "model_registry_logical_sha256": REGISTRY_TERMINAL_STATE[
                "model_registry_json"
            ]["logical_sha256"],
            "survivor_ids": list(SURVIVOR_IDS),
            "independently_authored_go_required": True,
            "same_request_raw_required_by_reference_bundle_parent_child": True,
            "heavy_compute_authorized_by_request_alone": False,
            "formal": False,
            "truth_open_authorized": False,
            "score_computation_authorized": False,
            "fresh_seed_reserved_or_opened": False,
            "heldout_opened": False,
        }
    )


class VerifiedContinuationActivation:
    __slots__ = ("_path", "_raw", "_payload", "_source", "_terminal")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _ACTIVATION_TOKEN:
            raise ProbabilisticContractError("continuation activation requires its exact loader")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        path: Path,
        raw: bytes,
        payload: Mapping[str, Any],
        source: VerifiedSourceClosure,
        terminal: VerifiedTerminalCustody,
    ) -> None:
        if token is not _ACTIVATION_TOKEN:
            raise ProbabilisticContractError("invalid continuation-activation token")
        object.__setattr__(self, "_path", Path(path).resolve())
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_payload", _json_copy(payload))
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_terminal", terminal)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedContinuationActivation is immutable")

    @property
    def raw_sha256(self) -> str:
        return sha256_bytes(self._raw)

    @property
    def source_snapshot_raw_sha256(self) -> str:
        return self._source.raw_sha256

    @property
    def payload(self) -> Mapping[str, Any]:
        self.verify_integrity()
        return MappingProxyType(_json_copy(self._payload))

    def verify_integrity(self) -> None:
        self._terminal.verify_integrity()
        self._source.verify_integrity()
        try:
            decoded = json.loads(self._raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProbabilisticContractError("continuation activation bytes are invalid") from exc
        verify_payload_seal(decoded)
        if decoded != self._payload:
            raise ProbabilisticContractError("continuation activation payload changed")


def load_continuation_activation(
    path: Path, *, expected_raw_sha256: str, repo_root: Path
) -> VerifiedContinuationActivation:
    expected = require_sha256(expected_raw_sha256, field="continuation activation raw")
    path = Path(path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("continuation activation is unavailable") from exc
    if sha256_bytes(raw) != expected or expected not in path.name.split("."):
        raise ProbabilisticContractError("continuation activation differs from external raw pin")
    if not isinstance(payload, dict):
        raise ProbabilisticContractError("continuation activation must be an object")
    verify_payload_seal(payload)
    source_record = payload.get("continuation_source_snapshot")
    if not isinstance(source_record, dict) or set(source_record) != {"path", "raw_sha256"}:
        raise ProbabilisticContractError("continuation source record differs")
    expected_payload = build_continuation_activation_payload(
        source_snapshot_path=source_record["path"],
        source_snapshot_raw_sha256=source_record["raw_sha256"],
    )
    if payload != expected_payload:
        raise ProbabilisticContractError("continuation activation policy differs")
    terminal = load_verified_terminal_custody(repo_root)
    source_path = _resolve(repo_root, source_record["path"], context="continuation source")
    source = load_verified_source_closure(
        source_path,
        expected_raw_sha256=source_record["raw_sha256"],
        repo_root=repo_root,
    )
    return VerifiedContinuationActivation(
        _ACTIVATION_TOKEN,
        path=path,
        raw=raw,
        payload=payload,
        source=source,
        terminal=terminal,
    )


class VerifiedContinuationRequest:
    __slots__ = ("_path", "_raw", "_payload", "_activation")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _REQUEST_TOKEN:
            raise ProbabilisticContractError("continuation request requires its exact loader")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        path: Path,
        raw: bytes,
        payload: Mapping[str, Any],
        activation: VerifiedContinuationActivation,
    ) -> None:
        if token is not _REQUEST_TOKEN:
            raise ProbabilisticContractError("invalid continuation-request token")
        object.__setattr__(self, "_path", Path(path).resolve())
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_payload", _json_copy(payload))
        object.__setattr__(self, "_activation", activation)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedContinuationRequest is immutable")

    @property
    def raw_sha256(self) -> str:
        return sha256_bytes(self._raw)

    @property
    def activation(self) -> VerifiedContinuationActivation:
        self.verify_integrity()
        return self._activation

    def verify_integrity(self) -> None:
        self._activation.verify_integrity()
        decoded = json.loads(self._raw)
        verify_payload_seal(decoded)
        if decoded != self._payload:
            raise ProbabilisticContractError("continuation request payload changed")


def load_continuation_request(
    path: Path,
    *,
    expected_raw_sha256: str,
    activation: VerifiedContinuationActivation,
) -> VerifiedContinuationRequest:
    if not isinstance(activation, VerifiedContinuationActivation):
        raise ProbabilisticContractError("request requires V3 continuation activation")
    activation.verify_integrity()
    expected = require_sha256(expected_raw_sha256, field="continuation request raw")
    path = Path(path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("continuation request is unavailable") from exc
    if sha256_bytes(raw) != expected or expected not in path.name.split("."):
        raise ProbabilisticContractError("continuation request differs from external raw pin")
    verify_payload_seal(payload)
    expected_payload = build_continuation_request_payload(
        activation_path=activation._path.relative_to(activation._terminal.repo_root).as_posix(),
        activation_raw_sha256=activation.raw_sha256,
        source_snapshot_path=activation._payload["continuation_source_snapshot"]["path"],
        source_snapshot_raw_sha256=activation.source_snapshot_raw_sha256,
    )
    if payload != expected_payload:
        raise ProbabilisticContractError("continuation request policy differs")
    return VerifiedContinuationRequest(
        _REQUEST_TOKEN,
        path=path,
        raw=raw,
        payload=payload,
        activation=activation,
    )


class VerifiedContinuationGo:
    __slots__ = ("_path", "_raw", "_payload", "_request")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _GO_TOKEN:
            raise ProbabilisticContractError("continuation GO requires its exact loader")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        path: Path,
        raw: bytes,
        payload: Mapping[str, Any],
        request: VerifiedContinuationRequest,
    ) -> None:
        if token is not _GO_TOKEN:
            raise ProbabilisticContractError("invalid continuation-GO token")
        object.__setattr__(self, "_path", Path(path).resolve())
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_payload", _json_copy(payload))
        object.__setattr__(self, "_request", request)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedContinuationGo is immutable")

    @property
    def raw_sha256(self) -> str:
        return sha256_bytes(self._raw)

    def verify_integrity(self) -> None:
        self._request.verify_integrity()
        decoded = json.loads(self._raw)
        verify_payload_seal(decoded)
        if decoded != self._payload:
            raise ProbabilisticContractError("continuation GO payload changed")


class TwoSurvivorLaunchAuthority:
    """Opaque V3 authority; historical V2 authority objects are never accepted."""

    __slots__ = ("_request", "_go")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _AUTHORITY_TOKEN:
            raise ProbabilisticContractError("two-survivor authority requires request plus GO")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        request: VerifiedContinuationRequest,
        go: VerifiedContinuationGo,
    ) -> None:
        if token is not _AUTHORITY_TOKEN:
            raise ProbabilisticContractError("invalid two-survivor authority token")
        object.__setattr__(self, "_request", request)
        object.__setattr__(self, "_go", go)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("TwoSurvivorLaunchAuthority is immutable")

    @property
    def request_raw_sha256(self) -> str:
        self.verify_integrity()
        return self._request.raw_sha256

    @property
    def activation_raw_sha256(self) -> str:
        self.verify_integrity()
        return self._request._activation.raw_sha256

    @property
    def go_raw_sha256(self) -> str:
        self.verify_integrity()
        return self._go.raw_sha256

    @property
    def source_snapshot_raw_sha256(self) -> str:
        self.verify_integrity()
        return self._request._activation.source_snapshot_raw_sha256

    @property
    def repo_root(self) -> Path:
        return self._request._activation._terminal.repo_root

    def verify_integrity(self) -> None:
        self._request.verify_integrity()
        self._go.verify_integrity()


def require_two_survivor_authority(value: object) -> TwoSurvivorLaunchAuthority:
    if not isinstance(value, TwoSurvivorLaunchAuthority):
        raise ProbabilisticContractError(
            "isolated continuation path rejects old or generic execution authority"
        )
    value.verify_integrity()
    return value


def load_two_survivor_launch_authority(
    *,
    repo_root: Path,
    activation_path: Path,
    expected_activation_raw_sha256: str,
    request_path: Path,
    expected_request_raw_sha256: str,
    independent_go_path: Path,
    expected_independent_go_raw_sha256: str,
) -> TwoSurvivorLaunchAuthority:
    activation = load_continuation_activation(
        activation_path,
        expected_raw_sha256=expected_activation_raw_sha256,
        repo_root=repo_root,
    )
    request = load_continuation_request(
        request_path,
        expected_raw_sha256=expected_request_raw_sha256,
        activation=activation,
    )
    expected = require_sha256(
        expected_independent_go_raw_sha256, field="continuation independent GO raw"
    )
    try:
        raw = Path(independent_go_path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("continuation independent GO is unavailable") from exc
    if sha256_bytes(raw) != expected:
        raise ProbabilisticContractError("continuation GO differs from external raw pin")
    if not isinstance(payload, dict):
        raise ProbabilisticContractError("continuation GO must be an object")
    verify_payload_seal(payload)
    bindings = payload.get("bindings")
    expected_bindings = {
        "activation_raw_sha256": activation.raw_sha256,
        "request_raw_sha256": request.raw_sha256,
        "source_snapshot_raw_sha256": activation.source_snapshot_raw_sha256,
        "v2_go_raw_sha256": V2_ROOTS["v2_independent_go"]["raw_sha256"],
        "terminal_failure_receipt_raw_sha256": V2_ROOTS[
            "terminal_failure_receipt"
        ]["raw_sha256"],
        "terminal_failure_audit_raw_sha256": V2_ROOTS["terminal_failure_audit"][
            "raw_sha256"
        ],
        "terminal_failure_audit_logical_sha256": V2_ROOTS[
            "terminal_failure_audit"
        ]["logical_sha256"],
        "terminal_registry_append_raw_sha256": REGISTRY_TERMINAL_STATE[
            "append_receipt"
        ]["raw_sha256"],
        "model_registry_csv_raw_sha256": REGISTRY_TERMINAL_STATE[
            "model_registry_csv"
        ]["raw_sha256"],
        "model_registry_json_raw_sha256": REGISTRY_TERMINAL_STATE[
            "model_registry_json"
        ]["raw_sha256"],
        "model_registry_logical_sha256": REGISTRY_TERMINAL_STATE[
            "model_registry_json"
        ]["logical_sha256"],
    }
    decision = payload.get("decision", {})
    if (
        payload.get("schema_version") != CONTINUATION_GO_SCHEMA
        or payload.get("status") != "FINAL_GO"
        or bindings != expected_bindings
        or decision.get("two_survivor_continuation") != "GO"
        or decision.get("p0_count") != 0
        or decision.get("p1_count") != 0
        or decision.get("candidate_refit_or_retry_authorized") is not False
        or decision.get("reference_generation_authorized") is not True
        or decision.get("separate_truth_child_authorized_after_complete_bundle") is not True
        or decision.get("score_before_complete_bundle_authorized") is not False
        or payload.get("survivor_ids") != list(SURVIVOR_IDS)
        or payload.get("broken_candidate_id") != BROKEN_CANDIDATE_ID
        or payload.get("fresh_seed_reserved_or_opened") is not False
        or payload.get("heldout_opened") is not False
    ):
        raise ProbabilisticContractError("continuation independent GO authority differs")
    go = VerifiedContinuationGo(
        _GO_TOKEN,
        path=independent_go_path,
        raw=raw,
        payload=payload,
        request=request,
    )
    return TwoSurvivorLaunchAuthority(_AUTHORITY_TOKEN, request=request, go=go)


def authority_binding(authority: TwoSurvivorLaunchAuthority) -> dict[str, str]:
    authority = require_two_survivor_authority(authority)
    return {
        "activation_raw_sha256": authority.activation_raw_sha256,
        "request_raw_sha256": authority.request_raw_sha256,
        "independent_go_raw_sha256": authority.go_raw_sha256,
        "source_snapshot_raw_sha256": authority.source_snapshot_raw_sha256,
    }


def build_reference_attestation_payload(
    *,
    authority: TwoSurvivorLaunchAuthority,
    reference_id: str,
    prediction: Mapping[str, str],
    prediction_receipt: Mapping[str, str],
    execution_receipt: Mapping[str, str],
) -> dict[str, Any]:
    """Build the V3 wrapper around one newly generated causal reference custody."""

    authority = require_two_survivor_authority(authority)
    if reference_id not in REFERENCE_IDS:
        raise ProbabilisticContractError("continuation reference ID is not locked")
    records = {
        "prediction": dict(prediction),
        "prediction_receipt": dict(prediction_receipt),
        "execution_receipt": dict(execution_receipt),
    }
    for name, record in records.items():
        if set(record) != {"path", "raw_sha256"}:
            raise ProbabilisticContractError(f"reference {name} record differs")
        require_sha256(record["raw_sha256"], field=f"reference.{name}.raw_sha256")
    return seal_payload(
        {
            "schema_version": REFERENCE_ATTESTATION_SCHEMA,
            "reference_id": reference_id,
            "authority": authority_binding(authority),
            "historic_v2_authorization_raw_sha256": V2_ROOTS["authorization"][
                "raw_sha256"
            ],
            "terminal_failure_audit_raw_sha256": V2_ROOTS["terminal_failure_audit"][
                "raw_sha256"
            ],
            "records": records,
            "observed_fold_count": FOLDS_PER_SEED * len(SPENT_SEEDS),
            "rows": PREDICTION_ROWS,
            "candidate_prediction_or_refit_performed": False,
            "truth_bytes_read": False,
            "scores_generated_or_read": False,
            "fresh_seed_reserved_or_opened": False,
            "heldout_opened": False,
        }
    )


def validate_reference_surface_payload(
    payload: Mapping[str, Any], *, authority: TwoSurvivorLaunchAuthority
) -> None:
    authority = require_two_survivor_authority(authority)
    if (
        payload.get("schema_version") != REFERENCE_SURFACE_SCHEMA
        or payload.get("authority") != authority_binding(authority)
        or payload.get("status") != "PASS_EXACT_TWO_CAUSAL_REFERENCE_CUSTODY"
        or payload.get("candidate_prediction_or_refit_performed") is not False
        or payload.get("truth_bytes_read") is not False
        or payload.get("scores_generated_or_read") is not False
        or payload.get("fresh_seed_reserved_or_opened") is not False
        or payload.get("heldout_opened") is not False
    ):
        raise ProbabilisticContractError("two-survivor reference surface policy differs")
    records = _require_exact_keys(
        payload.get("records"), REFERENCE_IDS, context="reference surface"
    )
    for reference_id, record in records.items():
        if not isinstance(record, dict) or set(record) != {
            "prediction",
            "prediction_receipt",
            "execution_receipt",
            "continuation_attestation",
        }:
            raise ProbabilisticContractError(f"{reference_id} reference receipt set differs")
        attestation = _verify_bound_record(
            record["continuation_attestation"],
            repo_root=authority.repo_root,
            authority=authority,
            context=f"{reference_id}.continuation_attestation",
            expected_schema=REFERENCE_ATTESTATION_SCHEMA,
        )
        if (
            attestation is None
            or attestation.get("reference_id") != reference_id
            or attestation.get("historic_v2_authorization_raw_sha256")
            != V2_ROOTS["authorization"]["raw_sha256"]
            or attestation.get("terminal_failure_audit_raw_sha256")
            != V2_ROOTS["terminal_failure_audit"]["raw_sha256"]
            or attestation.get("records")
            != {
                name: record[name]
                for name in ("prediction", "prediction_receipt", "execution_receipt")
            }
            or attestation.get("observed_fold_count")
            != FOLDS_PER_SEED * len(SPENT_SEEDS)
            or attestation.get("rows") != PREDICTION_ROWS
        ):
            raise ProbabilisticContractError(f"{reference_id} attestation binding differs")
        for name in ("prediction", "prediction_receipt", "execution_receipt"):
            _verify_bound_record(
                record[name],
                repo_root=authority.repo_root,
                authority=authority,
                context=f"{reference_id}.{name}",
            )


def build_global_runtime_receipt_payload(
    *,
    authority: TwoSurvivorLaunchAuthority,
    reference_surface_raw_sha256: str,
    elapsed_seconds: float,
    peak_process_tree_rss_gib: float,
    minimum_free_ram_gib: float,
) -> dict[str, Any]:
    authority = require_two_survivor_authority(authority)
    require_sha256(reference_surface_raw_sha256, field="reference surface raw")
    numeric = (elapsed_seconds, peak_process_tree_rss_gib, minimum_free_ram_gib)
    if any(not isinstance(value, (int, float)) or value < 0 for value in numeric):
        raise ProbabilisticContractError("global runtime metrics must be non-negative")
    return seal_payload(
        {
            "schema_version": GLOBAL_RUNTIME_SCHEMA,
            "authority": authority_binding(authority),
            "survivor_ids": list(SURVIVOR_IDS),
            "candidate_runtime_raw_sha256_by_id": {
                model_id: SURVIVOR_CUSTODY[model_id]["runtime_receipt"]["raw_sha256"]
                for model_id in SURVIVOR_IDS
            },
            "reference_ids": list(REFERENCE_IDS),
            "reference_surface_raw_sha256": reference_surface_raw_sha256,
            "metrics": {
                "elapsed_seconds": float(elapsed_seconds),
                "peak_process_tree_rss_gib": float(peak_process_tree_rss_gib),
                "minimum_free_ram_gib": float(minimum_free_ram_gib),
            },
            "candidate_refit_or_retry_performed": False,
            "gpu": "OFF",
            "inner_threads": 1,
            "truth_bytes_read": False,
            "scores_generated_or_read": False,
            "fresh_seed_reserved_or_opened": False,
            "heldout_opened": False,
        }
    )


def build_two_survivor_bundle_payload(
    *,
    authority: TwoSurvivorLaunchAuthority,
    references: Mapping[str, Any],
    global_runtime: Mapping[str, str],
) -> dict[str, Any]:
    authority = require_two_survivor_authority(authority)
    _require_exact_keys(references, REFERENCE_IDS, context="reference receipts")
    payload = seal_payload(
        {
            "schema_version": BUNDLE_SCHEMA,
            "authority": authority_binding(authority),
            "candidates": {
                model_id: {
                    "prediction": _json_copy(SURVIVOR_CUSTODY[model_id]["prediction"]),
                    "prediction_receipt": _json_copy(
                        SURVIVOR_CUSTODY[model_id]["prediction_receipt"]
                    ),
                }
                for model_id in SURVIVOR_IDS
            },
            "references": _json_copy(dict(references)),
            "comparators": frozen_comparator_custody(),
            "candidate_runtimes": {
                model_id: _json_copy(SURVIVOR_CUSTODY[model_id]["runtime_receipt"])
                for model_id in SURVIVOR_IDS
            },
            "global_runtime": dict(global_runtime),
            "truth_parent_read": False,
            "truth_path_present": False,
            "fresh_seed_reserved_or_opened": False,
            "heldout_opened": False,
        }
    )
    validate_bundle_shape(payload)
    return payload


def _require_exact_keys(
    value: object, expected: Sequence[str], *, context: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ProbabilisticContractError(
            f"{context} must contain exactly {len(tuple(expected))} locked participants"
        )
    if BROKEN_CANDIDATE_ID in value:
        raise ProbabilisticContractError("NGBoost inclusion is forbidden in V3 continuation")
    return value


def validate_bundle_shape(payload: Mapping[str, Any]) -> None:
    """Validate exact receipt cardinalities before opening any referenced file."""

    _require_exact_keys(payload.get("candidates"), SURVIVOR_IDS, context="candidate receipts")
    _require_exact_keys(payload.get("references"), REFERENCE_IDS, context="reference receipts")
    _require_exact_keys(payload.get("comparators"), COMPARATOR_IDS, context="comparator receipts")
    _require_exact_keys(
        payload.get("candidate_runtimes"), SURVIVOR_IDS, context="candidate runtime receipts"
    )
    global_runtime = payload.get("global_runtime")
    if not isinstance(global_runtime, dict) or set(global_runtime) != {"path", "raw_sha256"}:
        raise ProbabilisticContractError("exactly one global runtime receipt is required")


def _verify_bound_record(
    record: object,
    *,
    repo_root: Path,
    authority: TwoSurvivorLaunchAuthority,
    context: str,
    expected_schema: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(record, dict) or set(record) != {"path", "raw_sha256"}:
        raise ProbabilisticContractError(f"{context} file record differs")
    raw = _read_exact(
        repo_root,
        record,
        context=context,
        require_addressed_name=True,
    )
    if expected_schema is None:
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError(f"{context} is not valid JSON") from exc
    verify_payload_seal(payload)
    if payload.get("schema_version") != expected_schema:
        raise ProbabilisticContractError(f"{context} schema differs")
    if payload.get("authority") != authority_binding(authority):
        raise ProbabilisticContractError(f"{context} authority binding differs")
    return payload


@dataclass(frozen=True)
class VerifiedTwoSurvivorBundle:
    path: Path
    raw_sha256: str
    payload: Mapping[str, Any]
    authority: TwoSurvivorLaunchAuthority

    def verify_integrity(self) -> None:
        self.authority.verify_integrity()
        verify_file_against_record(
            self.path,
            {"raw_sha256": self.raw_sha256},
            context="two-survivor bundle",
        )
        verify_payload_seal(self.payload)
        validate_bundle_shape(self.payload)


def load_two_survivor_bundle(
    path: Path,
    *,
    expected_raw_sha256: str,
    authority: TwoSurvivorLaunchAuthority,
) -> VerifiedTwoSurvivorBundle:
    authority = require_two_survivor_authority(authority)
    expected = require_sha256(expected_raw_sha256, field="two-survivor bundle raw")
    path = Path(path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("two-survivor bundle is unavailable") from exc
    if sha256_bytes(raw) != expected or expected not in path.name.split("."):
        raise ProbabilisticContractError("two-survivor bundle differs from external raw pin")
    verify_payload_seal(payload)
    if (
        payload.get("schema_version") != BUNDLE_SCHEMA
        or payload.get("authority") != authority_binding(authority)
        or payload.get("truth_parent_read") is not False
        or payload.get("truth_path_present") is not False
        or payload.get("fresh_seed_reserved_or_opened") is not False
        or payload.get("heldout_opened") is not False
    ):
        raise ProbabilisticContractError("two-survivor bundle policy differs")
    validate_bundle_shape(payload)
    for model_id in SURVIVOR_IDS:
        candidate = payload["candidates"][model_id]
        if candidate != {
            "prediction": SURVIVOR_CUSTODY[model_id]["prediction"],
            "prediction_receipt": SURVIVOR_CUSTODY[model_id]["prediction_receipt"],
        }:
            raise ProbabilisticContractError(f"{model_id} bundle custody differs")
        if payload["candidate_runtimes"][model_id] != SURVIVOR_CUSTODY[model_id][
            "runtime_receipt"
        ]:
            raise ProbabilisticContractError(f"{model_id} runtime bundle custody differs")
    for comparator_id in COMPARATOR_IDS:
        if payload["comparators"][comparator_id] != COMPARATOR_CUSTODY[comparator_id]:
            raise ProbabilisticContractError(f"{comparator_id} bundle custody differs")
    for reference_id in REFERENCE_IDS:
        reference = payload["references"][reference_id]
        if not isinstance(reference, dict) or set(reference) != {
            "prediction",
            "prediction_receipt",
            "execution_receipt",
            "continuation_attestation",
        }:
            raise ProbabilisticContractError(f"{reference_id} reference receipt set differs")
        _verify_bound_record(
            reference["continuation_attestation"],
            repo_root=authority.repo_root,
            authority=authority,
            context=f"{reference_id}.continuation_attestation",
            expected_schema=REFERENCE_ATTESTATION_SCHEMA,
        )
        for name in ("prediction", "prediction_receipt", "execution_receipt"):
            _verify_bound_record(
                reference[name],
                repo_root=authority.repo_root,
                authority=authority,
                context=f"{reference_id}.{name}",
            )
    global_payload = _verify_bound_record(
        payload["global_runtime"],
        repo_root=authority.repo_root,
        authority=authority,
        context="global runtime receipt",
        expected_schema=GLOBAL_RUNTIME_SCHEMA,
    )
    if (
        global_payload is None
        or global_payload.get("survivor_ids") != list(SURVIVOR_IDS)
        or global_payload.get("candidate_runtime_raw_sha256_by_id")
        != {
            model_id: SURVIVOR_CUSTODY[model_id]["runtime_receipt"]["raw_sha256"]
            for model_id in SURVIVOR_IDS
        }
        or global_payload.get("reference_ids") != list(REFERENCE_IDS)
        or global_payload.get("truth_bytes_read") is not False
        or global_payload.get("scores_generated_or_read") is not False
    ):
        raise ProbabilisticContractError("global runtime receipt binding differs")
    return VerifiedTwoSurvivorBundle(
        path=path.resolve(),
        raw_sha256=expected,
        payload=MappingProxyType(_json_copy(payload)),
        authority=authority,
    )
