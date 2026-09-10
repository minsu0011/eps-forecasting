"""Out-of-band raw pins required by every formal execution entry point."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import (
    PROBABILISTIC_DESIGN_SHA256,
    ProbabilisticContractError,
    require_sha256,
    sha256_bytes,
    verify_payload_seal,
)


EXECUTION_REQUEST_SCHEMA = "expected_pe_model_zoo.probabilistic_external_execution_request.v1"
INDEPENDENT_GO_SCHEMA = "independent_probabilistic_formal_activation_reaudit.v1"
PRIOR_EXTERNAL_ROOTS = {
    "pointer_raw_sha256": "a91c315d460c36182a531a34804c4b9ad68578f358e22ca97a67253e9e806d39",
    "activation_raw_sha256": "0f5b28c5cd5469042e3818b559f01e69eda928fb644b3814734848f66f5a9883",
    "authorization_raw_sha256": "941ca7f6ab2035fb4a1e9ce00f2a7d509148853950be4d5050ec0a6f6e631301",
    "source_snapshot_raw_sha256": "fd4b034799a2ec56fe09a27888e0953643db175a1c9adec628d4e4dfa3fd16bc",
    "predict_input_manifest_raw_sha256": (
        "8fd72abfc592a7df0c2cb50fbfb9bf6b9e3cd1e47c1200e1d7aa095022d133cd"
    ),
}
PRIOR_NO_GO = {
    "path": "outputs/model_zoo_probabilistic_wave_formal_activation_independent_audit_20260819/"
    "AUDIT.json",
    "raw_sha256": "9e350dd3bdc1d2e5f1ad8d7d8f3d31e4d5929ed2a303e8a71bb2f169e52f1139",
    "logical_sha256": "faaf7750e9e99d05ae6e532240e6c5d78143f211e29d733f17803f4d354053a6",
}
TERMINAL_V2_REVOCATION = {
    "failure_receipt": {
        "path": (
            "outputs/model_zoo_probabilistic_wave_formal_execution_fail_closed_20260819/"
            "FAILURE.json"
        ),
        "raw_sha256": "f99260f3bc36fd3a440d926305401c4b55fe472dc9d2c27e80252b52807dcc27",
        "logical_sha256": "f7303b23a420b619946ab79c7d1d354382248cf43d81cb12c0d2fcc56c63b396",
    },
    "independent_terminal_audit": {
        "path": (
            "outputs/model_zoo_probabilistic_wave_spent_terminal_failure_"
            "independent_audit_20260819/AUDIT.json"
        ),
        "raw_sha256": "cf08ec76b6928ac58d3e47d259a316eaf79a59ce2259bccb5be2f7f4c5379287",
        "logical_sha256": "2166e9666b9f309bdd0354fe83808c1b2aa82e391e762caeaa6ff526bb88ac8f",
    },
}
PREDICT_INPUT_MANIFEST_PATH = "outputs/p5s/i/SPENT_PREDICT_INPUTS_MANIFEST.json"
_REQUEST_TOKEN = object()
_GO_TOKEN = object()
_AUTHORITY_TOKEN = object()


def _read_explicitly_addressed_json(
    path: Path, *, expected_raw_sha256: str, context: str, content_addressed: bool
) -> tuple[bytes, dict[str, Any]]:
    require_sha256(expected_raw_sha256, field=f"{context}.expected_raw_sha256")
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError(f"{context} is unavailable or invalid") from exc
    if sha256_bytes(raw) != expected_raw_sha256:
        raise ProbabilisticContractError(f"{context} differs from the external raw pin")
    if content_addressed and expected_raw_sha256 not in Path(path).name.split("."):
        raise ProbabilisticContractError(f"{context} is not content-addressed")
    if not isinstance(payload, dict):
        raise ProbabilisticContractError(f"{context} must be an object")
    verify_payload_seal(payload)
    return raw, payload


def _require_exact_hash_mapping(value: object, *, context: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(PRIOR_EXTERNAL_ROOTS):
        raise ProbabilisticContractError(f"{context} root set differs")
    output = {}
    for key, digest in value.items():
        output[key] = require_sha256(digest, field=f"{context}.{key}")
    return output


class VerifiedExternalExecutionRequest:
    __slots__ = ("_path", "_payload", "_raw", "_raw_sha256")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _REQUEST_TOKEN:
            raise ProbabilisticContractError("execution request requires the external-pin loader")
        return super().__new__(cls)

    def __init__(self, token: object, *, path: Path, raw: bytes, payload: Mapping[str, Any]):
        if token is not _REQUEST_TOKEN:
            raise ProbabilisticContractError("invalid execution-request factory token")
        object.__setattr__(self, "_path", Path(path).resolve())
        object.__setattr__(self, "_payload", json.loads(json.dumps(payload)))
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedExternalExecutionRequest is immutable")

    @property
    def raw_sha256(self) -> str:
        return self._raw_sha256

    @property
    def path(self) -> Path:
        return self._path

    @property
    def current_roots(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._payload["superseding_roots"]))

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw) != self._raw_sha256:
            raise ProbabilisticContractError("external execution-request bytes changed")
        decoded = json.loads(self._raw)
        verify_payload_seal(decoded)
        if decoded != self._payload:
            raise ProbabilisticContractError("external execution-request payload changed")


class VerifiedIndependentExecutionGo:
    __slots__ = ("_path", "_payload", "_raw", "_raw_sha256")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _GO_TOKEN:
            raise ProbabilisticContractError("independent GO requires the external-pin loader")
        return super().__new__(cls)

    def __init__(self, token: object, *, path: Path, raw: bytes, payload: Mapping[str, Any]):
        if token is not _GO_TOKEN:
            raise ProbabilisticContractError("invalid independent-GO factory token")
        object.__setattr__(self, "_path", Path(path).resolve())
        object.__setattr__(self, "_payload", json.loads(json.dumps(payload)))
        object.__setattr__(self, "_raw", bytes(raw))
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedIndependentExecutionGo is immutable")

    @property
    def path(self) -> Path:
        return self._path

    @property
    def raw_sha256(self) -> str:
        return self._raw_sha256

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw) != self._raw_sha256:
            raise ProbabilisticContractError("independent-GO bytes changed")
        decoded = json.loads(self._raw)
        verify_payload_seal(decoded)
        if decoded != self._payload:
            raise ProbabilisticContractError("independent-GO payload changed")


class FormalLaunchAuthority:
    __slots__ = ("_request", "_go")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _AUTHORITY_TOKEN:
            raise ProbabilisticContractError("formal launch authority requires external pins")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        request: VerifiedExternalExecutionRequest,
        independent_go: VerifiedIndependentExecutionGo,
    ) -> None:
        if token is not _AUTHORITY_TOKEN:
            raise ProbabilisticContractError("invalid formal-launch factory token")
        object.__setattr__(self, "_request", request)
        object.__setattr__(self, "_go", independent_go)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("FormalLaunchAuthority is immutable")

    @property
    def request(self) -> VerifiedExternalExecutionRequest:
        self.verify_integrity()
        return self._request

    @property
    def independent_go(self) -> VerifiedIndependentExecutionGo:
        self.verify_integrity()
        return self._go

    def verify_integrity(self) -> None:
        self._request.verify_integrity()
        self._go.verify_integrity()


def load_external_execution_request(
    path: Path, *, expected_raw_sha256: str, repo_root: Path
) -> VerifiedExternalExecutionRequest:
    raw, payload = _read_explicitly_addressed_json(
        path,
        expected_raw_sha256=expected_raw_sha256,
        context="formal external execution request",
        content_addressed=True,
    )
    predecessor = _require_exact_hash_mapping(
        payload.get("predecessor_external_roots"), context="predecessor external"
    )
    superseding = _require_exact_hash_mapping(
        payload.get("superseding_roots"), context="superseding external"
    )
    try:
        prior_raw = (Path(repo_root) / PRIOR_NO_GO["path"]).read_bytes()
        prior_payload = json.loads(prior_raw)
        input_raw = (Path(repo_root) / PREDICT_INPUT_MANIFEST_PATH).read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError(
            "externally pinned predecessor evidence is unavailable"
        ) from exc
    prior_records = prior_payload.get("trust_roots")
    prior_key_map = {
        "pointer_raw_sha256": "pointer",
        "activation_raw_sha256": "activation",
        "authorization_raw_sha256": "authorization",
        "source_snapshot_raw_sha256": "source_snapshot",
        "predict_input_manifest_raw_sha256": "predict_input_manifest",
    }
    if not isinstance(prior_records, dict) or set(prior_records) != set(prior_key_map.values()):
        raise ProbabilisticContractError("prior audit trust-root set differs")
    for digest_key, record_key in prior_key_map.items():
        record = prior_records[record_key]
        if (
            not isinstance(record, dict)
            or record.get("raw_sha256") != PRIOR_EXTERNAL_ROOTS[digest_key]
            or record.get("raw_match") is not True
            or record.get("logical_seal_valid") is not True
        ):
            raise ProbabilisticContractError("prior audit trust-root record differs")
        try:
            predecessor_raw = (Path(repo_root) / record["path"]).read_bytes()
        except (KeyError, OSError, TypeError) as exc:
            raise ProbabilisticContractError("prior trust-root artifact is unavailable") from exc
        if sha256_bytes(predecessor_raw) != PRIOR_EXTERNAL_ROOTS[digest_key]:
            raise ProbabilisticContractError("prior trust-root artifact bytes changed")
    if (
        payload.get("schema_version") != EXECUTION_REQUEST_SCHEMA
        or payload.get("design_sha256") != PROBABILISTIC_DESIGN_SHA256
        or payload.get("status") != "SUPERSEDING_CHAIN_AUDIT_CANDIDATE"
        or predecessor != PRIOR_EXTERNAL_ROOTS
        or payload.get("prior_no_go") != PRIOR_NO_GO
        or superseding["predict_input_manifest_raw_sha256"]
        != PRIOR_EXTERNAL_ROOTS["predict_input_manifest_raw_sha256"]
        or sha256_bytes(prior_raw) != PRIOR_NO_GO["raw_sha256"]
        or prior_payload.get("manifest_sha256") != PRIOR_NO_GO["logical_sha256"]
        or sha256_bytes(input_raw) != PRIOR_EXTERNAL_ROOTS["predict_input_manifest_raw_sha256"]
        or payload.get("same_pin_required_by_candidate_reference_bundle_parent_child") is not True
        or payload.get("independent_go_required_before_compute") is not True
        or payload.get("heavy_compute_authorized_by_request_alone") is not False
    ):
        raise ProbabilisticContractError("formal execution request differs from external policy")
    return VerifiedExternalExecutionRequest(_REQUEST_TOKEN, path=path, raw=raw, payload=payload)


def load_independent_execution_go(
    path: Path,
    *,
    expected_raw_sha256: str,
    request: VerifiedExternalExecutionRequest,
) -> VerifiedIndependentExecutionGo:
    request.verify_integrity()
    raw, payload = _read_explicitly_addressed_json(
        path,
        expected_raw_sha256=expected_raw_sha256,
        context="independent formal activation GO",
        content_addressed=False,
    )
    decision = payload.get("decision")
    roots = payload.get("superseding_roots")
    if (
        payload.get("schema_version") != INDEPENDENT_GO_SCHEMA
        or payload.get("status") != "FINAL_GO"
        or not isinstance(decision, dict)
        or decision.get("formal_spent_screen_execution") != "GO"
        or decision.get("p0_count") != 0
        or decision.get("p1_count") != 0
        or decision.get("heavy_compute_authorized") is not True
        or payload.get("execution_request_raw_sha256") != request.raw_sha256
        or roots != dict(request.current_roots)
        or payload.get("supersedes_no_go_raw_sha256") != PRIOR_NO_GO["raw_sha256"]
    ):
        raise ProbabilisticContractError("independent GO does not authorize this exact request")
    return VerifiedIndependentExecutionGo(_GO_TOKEN, path=path, raw=raw, payload=payload)


def load_formal_launch_authority(
    *,
    execution_request_path: Path,
    expected_execution_request_sha256: str,
    independent_go_path: Path,
    expected_independent_go_sha256: str,
    repo_root: Path,
) -> FormalLaunchAuthority:
    # This exact V2 request/GO was consumed once and terminated fail-closed by
    # NGBoost.  Verify both terminal records before rejecting so deletion or
    # coordinated resealing cannot revive any old candidate/reference/evaluator
    # entry point.  Completed c/0 and c/1 bytes remain historical custody only.
    terminal_payloads: dict[str, dict[str, Any]] = {}
    for name, record in TERMINAL_V2_REVOCATION.items():
        _, payload = _read_explicitly_addressed_json(
            Path(repo_root) / record["path"],
            expected_raw_sha256=record["raw_sha256"],
            context=f"terminal V2 revocation {name}",
            content_addressed=False,
        )
        if payload.get("manifest_sha256") != record["logical_sha256"]:
            raise ProbabilisticContractError(
                f"terminal V2 revocation {name} logical seal differs"
            )
        terminal_payloads[name] = payload
    failure = terminal_payloads["failure_receipt"]
    audit = terminal_payloads["independent_terminal_audit"]
    if (
        failure.get("status") != "TERMINAL_FAIL_CLOSED"
        or failure.get("audited_launch_roots", {}).get("execution_request_raw_sha256")
        != expected_execution_request_sha256
        or failure.get("audited_launch_roots", {}).get("independent_go_raw_sha256")
        != expected_independent_go_sha256
        or failure.get("decision", {}).get("retry_authorized") is not False
        or failure.get("decision", {}).get("fallback_authorized") is not False
        or audit.get("decision", {}).get("probabilistic_spent_screen")
        != "NO_GO_TERMINAL_INCOMPLETE"
        or audit.get("decision", {}).get("retry_or_relaunch_authorized_by_this_audit")
        is not False
    ):
        raise ProbabilisticContractError("terminal V2 revocation evidence differs")
    raise ProbabilisticContractError(
        "V2 formal launch authority is terminally revoked; use an independently audited "
        "two-survivor continuation authority"
    )

    # Unreachable historical construction is retained below as an auditable record
    # of the consumed authorization contract.
    request = load_external_execution_request(
        execution_request_path,
        expected_raw_sha256=expected_execution_request_sha256,
        repo_root=repo_root,
    )
    independent_go = load_independent_execution_go(
        independent_go_path,
        expected_raw_sha256=expected_independent_go_sha256,
        request=request,
    )
    return FormalLaunchAuthority(_AUTHORITY_TOKEN, request=request, independent_go=independent_go)


def require_formal_launch_authority(value: object) -> FormalLaunchAuthority:
    """Accept only the opaque capability minted by both external-pin loaders."""

    if not isinstance(value, FormalLaunchAuthority):
        raise ProbabilisticContractError(
            "formal entry point requires externally pinned launch authority"
        )
    value.verify_integrity()
    # Access both capabilities so a request-only lookalike cannot reach execution.
    value.request.verify_integrity()
    value.independent_go.verify_integrity()
    return value


def verify_externally_pinned_pointer(
    pointer_path: Path,
    *,
    request: VerifiedExternalExecutionRequest,
) -> dict[str, Any]:
    """Reject a coordinated self-reseal unless its raw pointer equals the external pin."""

    request.verify_integrity()
    expected = request.current_roots
    raw, pointer = _read_explicitly_addressed_json(
        pointer_path,
        expected_raw_sha256=expected["pointer_raw_sha256"],
        context="formal authorization pointer",
        content_addressed=True,
    )
    del raw
    if (
        pointer.get("authorization_scope") != "SPENT_SCREEN_EXECUTION"
        or pointer.get("authorization_raw_sha256") != expected["authorization_raw_sha256"]
        or pointer.get("formal_activation_raw_sha256") != expected["activation_raw_sha256"]
        or pointer.get("source_snapshot_raw_sha256") != expected["source_snapshot_raw_sha256"]
        or pointer.get("truth_path_present") is not False
    ):
        raise ProbabilisticContractError("formal pointer differs from external root set")
    return pointer
