"""Live CPU-only resource enforcement and score-free spawn-parity benchmark."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
import base64
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from .authorization import ExecutionAuthorization, SCORE_FREE_SCOPE, canonical_csv_bytes
from .contracts import (
    IDENTITY_COLUMNS,
    ProbabilisticContractError,
    QuantilePredictionBatch,
    VerifiedPredictionBatch,
    canonical_json_bytes,
    logical_frame_sha256,
    seal_payload,
    sha256_bytes,
    verify_payload_seal,
)
from .spec import CANDIDATE_IDS


@dataclass(frozen=True)
class ResourcePolicy:
    outer_workers: int = 32
    inner_threads: int = 1
    cpu_affinity_start: int = 0
    cpu_affinity_end: int = 31
    aggregate_rss_max_gib: int = 64
    minimum_free_ram_gib: int = 16
    full_screen_wall_minutes: int = 90
    gpu: str = "OFF"
    quantile_execution: str = "five_quantiles_sequential_inside_outer_worker"
    paging_allowed: bool = False

    def __post_init__(self) -> None:
        if self.outer_workers not in {8, 16, 24, 32}:
            raise ProbabilisticContractError("worker count is outside the locked benchmark set")
        if self.inner_threads != 1 or self.gpu != "OFF" or self.paging_allowed:
            raise ProbabilisticContractError("nested threads/GPU/paging violate resource policy")
        if self.aggregate_rss_max_gib != 64 or self.minimum_free_ram_gib != 16:
            raise ProbabilisticContractError("RAM guard differs from design")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
_RESOURCE_GUARD_TOKEN = object()
_RUNTIME_RECEIPT_TOKEN = object()


class VerifiedRuntimeReceipt(Mapping[str, Any]):
    """Opaque resource-guard receipt accepted by selection."""

    __slots__ = ("_payload", "_raw", "_raw_sha256")

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _RUNTIME_RECEIPT_TOKEN:
            raise ProbabilisticContractError(
                "runtime receipts must come from the live resource guard"
            )
        return super().__new__(cls)

    def __init__(self, token: object, *, payload: Mapping[str, Any]) -> None:
        if token is not _RUNTIME_RECEIPT_TOKEN:
            raise ProbabilisticContractError("invalid runtime receipt factory token")
        sealed = seal_payload(payload)
        raw = canonical_json_bytes(sealed)
        object.__setattr__(self, "_payload", sealed)
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_raw_sha256", sha256_bytes(raw))

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("VerifiedRuntimeReceipt is immutable")

    def __getitem__(self, key: str) -> Any:
        self.verify_integrity()
        if key not in self._payload["metrics"]:
            raise KeyError(key)
        return json.loads(json.dumps(self._payload["metrics"][key]))

    def __iter__(self) -> Iterator[str]:
        self.verify_integrity()
        return iter(tuple(self._payload["metrics"]))

    def __len__(self) -> int:
        return len(self._payload["metrics"])

    @property
    def authorization_sha256(self) -> str:
        return str(self._payload["authorization_raw_sha256"])

    @property
    def participant_id(self) -> str:
        return str(self._payload["participant_id"])

    @property
    def receipt_kind(self) -> str:
        return "candidate_runtime"

    @property
    def provenance(self) -> Mapping[str, Any]:
        self.verify_integrity()
        return json.loads(json.dumps(self._payload["provenance"]))

    @property
    def raw_bytes(self) -> bytes:
        self.verify_integrity()
        return bytes(self._raw)

    @property
    def raw_sha256(self) -> str:
        self.verify_integrity()
        return self._raw_sha256

    def verify_integrity(self) -> None:
        if sha256_bytes(self._raw) != self._raw_sha256:
            raise ProbabilisticContractError("runtime receipt raw bytes changed")
        decoded = json.loads(self._raw)
        verify_payload_seal(decoded)
        if decoded != self._payload:
            raise ProbabilisticContractError("runtime receipt payload changed")

    def _metrics_copy(self) -> dict[str, Any]:
        self.verify_integrity()
        return json.loads(json.dumps(self._payload["metrics"]))


def write_runtime_receipt(receipt: VerifiedRuntimeReceipt, directory: Path) -> Path:
    """Persist one immutable content-addressed runtime receipt."""

    if not isinstance(receipt, VerifiedRuntimeReceipt):
        raise ProbabilisticContractError("runtime receipt writer requires verified custody")
    raw = receipt.raw_bytes
    path = Path(directory) / f"runtime.{sha256_bytes(raw)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    from .artifacts import immutable_write_bytes

    immutable_write_bytes(path, raw)
    return path


def load_verified_runtime_receipt(
    path: Path,
    *,
    authorization: ExecutionAuthorization,
    predictions: VerifiedPredictionBatch,
) -> VerifiedRuntimeReceipt:
    """Load a live receipt and bind it to actual sealed prediction/receipt bytes."""

    authorization.verify_integrity()
    predictions.verify_integrity()
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("runtime receipt is unavailable or invalid") from exc
    digest = sha256_bytes(raw)
    if digest not in Path(path).name.split("."):
        raise ProbabilisticContractError("runtime receipt is not content-addressed")
    verify_payload_seal(payload)
    receipt = VerifiedRuntimeReceipt(_RUNTIME_RECEIPT_TOKEN, payload=payload)
    if receipt.raw_bytes != raw:
        raise ProbabilisticContractError("runtime receipt canonical bytes changed")
    model_ids = tuple(pd.unique(predictions.frame["model_id"]))
    if len(model_ids) != 1:
        raise ProbabilisticContractError("runtime prediction participant is ambiguous")
    provenance = receipt.provenance
    if (
        receipt.authorization_sha256 != authorization.raw_sha256
        or receipt.participant_id != model_ids[0]
        or provenance.get("evidence_mode") != "LIVE_RESOURCE_GUARD"
        or provenance.get("prediction_raw_sha256") != predictions.raw_sha256
        or provenance.get("prediction_logical_sha256") != predictions.prediction_sha256
        or provenance.get("prediction_receipt_sha256") != predictions.receipt_sha256
        or provenance.get("common_identity_sha256") != predictions.identity_sha256
        or provenance.get("source_closure_sha256") != authorization.source_closure_sha256
    ):
        raise ProbabilisticContractError(
            "live runtime receipt differs from prediction/source custody"
        )
    assert_resource_snapshot(
        aggregate_rss_gib=float(receipt["aggregate_rss_gib"]),
        free_ram_gib=float(receipt["minimum_free_ram_gib"]),
        wall_minutes=float(receipt["runtime_minutes"]),
    )
    if int(receipt["live_snapshot_count"]) < 3:
        raise ProbabilisticContractError("runtime receipt lacks live guard checkpoints")
    return receipt


def install_thread_guards(environment: dict[str, str] | None = None) -> dict[str, str]:
    target = os.environ if environment is None else environment
    for key, value in THREAD_ENVIRONMENT.items():
        existing = target.get(key)
        if existing not in {None, value}:
            raise ProbabilisticContractError(f"{key}={existing!r} violates one-thread policy")
        target[key] = value
    return dict(THREAD_ENVIRONMENT)


def _windows_live_metrics() -> tuple[float, float, list[int]]:
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ProbabilisticContractError("GlobalMemoryStatusEx failed")
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        raise ProbabilisticContractError("process snapshot failed")
    parents: dict[int, int] = {}
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        present = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while present:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            present = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    root_pid = os.getpid()
    tree = {root_pid}
    changed = True
    while changed:
        before = len(tree)
        tree.update(pid for pid, parent in parents.items() if parent in tree)
        changed = len(tree) != before
    rss_bytes = 0
    for pid in tree:
        process = kernel32.OpenProcess(0x1000 | 0x0010, False, pid)
        if not process:
            continue
        try:
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), ctypes.sizeof(counters)):
                rss_bytes += int(counters.WorkingSetSize)
        finally:
            kernel32.CloseHandle(process)
    process_mask = ctypes.c_size_t()
    system_mask = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        kernel32.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
    ):
        raise ProbabilisticContractError("GetProcessAffinityMask failed")
    affinity = [index for index in range(64) if process_mask.value & (1 << index)]
    return rss_bytes / 1024**3, status.ullAvailPhys / 1024**3, affinity


def _live_metrics() -> tuple[float, float, list[int]]:
    try:
        import psutil
    except ImportError:
        if os.name == "nt":
            return _windows_live_metrics()
        raise ProbabilisticContractError("live resource enforcement backend is unavailable")
    process = psutil.Process()
    processes = [process, *process.children(recursive=True)]
    rss = sum(item.memory_info().rss for item in processes if item.is_running()) / 1024**3
    free = psutil.virtual_memory().available / 1024**3
    try:
        affinity = list(process.cpu_affinity())
    except (AttributeError, NotImplementedError):
        affinity = list(range(os.cpu_count() or 1))
    return float(rss), float(free), affinity


def capture_resource_snapshot(
    *,
    worker_count: int,
    started_monotonic: float,
    started_cpu: float,
) -> dict[str, Any]:
    """Measure and enforce live process-tree RSS, free floor, affinity, threads, and GPU."""

    if worker_count not in {8, 16, 24, 32}:
        raise ProbabilisticContractError("resource snapshot worker count is not predeclared")
    installed = install_thread_guards()
    rss, free, affinity = _live_metrics()
    elapsed = time.monotonic() - started_monotonic
    cpu_time = time.process_time() - started_cpu
    if not set(range(32)).issubset(set(affinity)):
        raise ProbabilisticContractError("process affinity does not include locked CPUs 0-31")
    assert_resource_snapshot(
        aggregate_rss_gib=rss,
        free_ram_gib=free,
        wall_minutes=elapsed / 60.0,
    )
    gpu_modules = sorted(
        name for name in sys.modules if name.split(".", 1)[0] in {"cupy", "tensorflow"}
    )
    if gpu_modules or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProbabilisticContractError("GPU-off contract is not satisfied")
    return seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.probabilistic_resource_snapshot.v2",
            "worker_count": worker_count,
            "process_start_method": "spawn",
            "wall_seconds": elapsed,
            "cpu_seconds_parent": cpu_time,
            "aggregate_process_tree_rss_gib": rss,
            "free_ram_gib": free,
            "affinity": affinity,
            "thread_environment": installed,
            "gpu": "OFF",
            "gpu_modules_loaded": gpu_modules,
            "host": platform.node(),
            "python": platform.python_version(),
        }
    )


class CandidateResourceGuard:
    """Factory-only live guard spanning every fold of exactly one candidate run."""

    __slots__ = (
        "_authorization",
        "_model_id",
        "_worker_count",
        "_started_wall",
        "_started_cpu",
        "_snapshots",
        "_fold_keys",
        "_negative_gaps",
        "_observed_outputs",
        "_crossing_rows",
        "_rows",
        "_finalized",
    )

    def __new__(cls, token: object | None = None, *_: object, **__: object):
        if token is not _RESOURCE_GUARD_TOKEN:
            raise ProbabilisticContractError("resource guards must come from the live factory")
        return super().__new__(cls)

    def __init__(
        self,
        token: object,
        *,
        authorization: ExecutionAuthorization,
        model_id: str,
        worker_count: int,
    ) -> None:
        if token is not _RESOURCE_GUARD_TOKEN:
            raise ProbabilisticContractError("invalid resource guard factory token")
        object.__setattr__(self, "_authorization", authorization)
        object.__setattr__(self, "_model_id", model_id)
        object.__setattr__(self, "_worker_count", worker_count)
        object.__setattr__(self, "_started_wall", time.monotonic())
        object.__setattr__(self, "_started_cpu", time.process_time())
        object.__setattr__(self, "_snapshots", [])
        object.__setattr__(self, "_fold_keys", set())
        object.__setattr__(self, "_negative_gaps", [])
        object.__setattr__(self, "_observed_outputs", [])
        object.__setattr__(self, "_crossing_rows", 0)
        object.__setattr__(self, "_rows", 0)
        object.__setattr__(self, "_finalized", False)
        self._checkpoint("candidate_start")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("CandidateResourceGuard is immutable")

    def _checkpoint(self, phase: str) -> None:
        if self._finalized:
            raise ProbabilisticContractError("resource guard is finalized; retry is forbidden")
        snapshot = capture_resource_snapshot(
            worker_count=self._worker_count,
            started_monotonic=self._started_wall,
            started_cpu=self._started_cpu,
        )
        snapshot["phase"] = phase
        self._snapshots.append(seal_payload(snapshot))

    def before_fold(
        self,
        *,
        authorization: ExecutionAuthorization,
        model_id: str,
        seed: int,
        entity_id: str,
        fold_id: str,
    ) -> None:
        if authorization.raw_sha256 != self._authorization.raw_sha256 or model_id != self._model_id:
            raise ProbabilisticContractError("resource guard execution binding differs")
        key = (seed, entity_id, fold_id)
        if key in self._fold_keys:
            raise ProbabilisticContractError("candidate fold retry is forbidden")
        self._fold_keys.add(key)
        self._checkpoint(f"before:{seed}:{entity_id}:{fold_id}")

    def after_fold(
        self, *, batch: QuantilePredictionBatch, output: pd.DataFrame, fold_id: str
    ) -> None:
        if batch.model_id != self._model_id:
            raise ProbabilisticContractError("resource guard batch candidate differs")
        if tuple(pd.unique(output["model_id"])) != (self._model_id,):
            raise ProbabilisticContractError("resource guard output candidate differs")
        seeds = tuple(pd.unique(output["seed"]))
        entities = tuple(pd.unique(output["entity_id"]))
        if len(seeds) != 1 or len(entities) != 1:
            raise ProbabilisticContractError("resource guard fold group differs")
        expected_identity = self._authorization.identity_frame()
        expected_identity = expected_identity.loc[
            (expected_identity["fold_id"] == fold_id)
            & (expected_identity["seed"] == seeds[0])
            & (expected_identity["entity_id"] == entities[0])
        ].reset_index(drop=True)
        actual_identity = output.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
        if not actual_identity.equals(expected_identity):
            raise ProbabilisticContractError("resource guard fold identity differs")
        expected_payload = batch.output_frame().reset_index(drop=True)
        actual_payload = output.loc[:, list(expected_payload.columns)].reset_index(drop=True)
        if not actual_payload.equals(expected_payload):
            raise ProbabilisticContractError("resource guard output differs from adapter batch")
        self._observed_outputs.append(output.copy(deep=True))
        adjacent = np.diff(batch.raw_log_quantiles, axis=1)
        gaps = np.maximum(-adjacent, 0.0).sum(axis=1)
        self._negative_gaps.extend(gaps.tolist())
        object.__setattr__(
            self,
            "_crossing_rows",
            self._crossing_rows + int((adjacent < 0.0).any(axis=1).sum()),
        )
        object.__setattr__(self, "_rows", self._rows + batch.rows)
        self._checkpoint(f"after:{fold_id}")

    def finalize(self, predictions: VerifiedPredictionBatch):
        """Produce the only runtime receipt accepted by formal selection."""

        from .coverage import verify_prediction_coverage

        if self._finalized:
            raise ProbabilisticContractError("resource guard is single-use; retry is forbidden")
        try:
            self._checkpoint("candidate_finalize")
        finally:
            object.__setattr__(self, "_finalized", True)
        verify_prediction_coverage(predictions, authorization=self._authorization)
        if str(predictions.frame["model_id"].iloc[0]) != self._model_id:
            raise ProbabilisticContractError("resource receipt prediction candidate differs")
        identity = self._authorization.identity_frame()
        expected_fold_keys = set(
            identity.loc[:, ["seed", "entity_id", "fold_id"]].itertuples(index=False, name=None)
        )
        if self._fold_keys != expected_fold_keys or self._rows != len(identity):
            raise ProbabilisticContractError(
                "resource guard did not observe every authorized fold row"
            )
        observed = pd.concat(self._observed_outputs, ignore_index=True)
        identity_columns = list(IDENTITY_COLUMNS)
        if observed.duplicated(identity_columns).any():
            raise ProbabilisticContractError("resource guard observed duplicate prediction rows")
        observed_indexed = observed.set_index(identity_columns, drop=False)
        expected_index = pd.MultiIndex.from_frame(identity.loc[:, identity_columns])
        try:
            observed = observed_indexed.loc[expected_index].reset_index(drop=True)
        except KeyError as exc:
            raise ProbabilisticContractError(
                "resource guard observed prediction identity differs"
            ) from exc
        if (
            sha256_bytes(canonical_csv_bytes(observed)) != predictions.raw_sha256
            or logical_frame_sha256(observed) != predictions.prediction_sha256
        ):
            raise ProbabilisticContractError(
                "sealed candidate predictions differ from observed fold outputs"
            )
        peak_rss = max(float(item["aggregate_process_tree_rss_gib"]) for item in self._snapshots)
        minimum_free = min(float(item["free_ram_gib"]) for item in self._snapshots)
        elapsed_minutes = (time.monotonic() - self._started_wall) / 60.0
        gaps = np.asarray(self._negative_gaps, dtype=np.float64)
        snapshot_sha256 = [
            sha256_bytes(canonical_json_bytes(snapshot)) for snapshot in self._snapshots
        ]
        return VerifiedRuntimeReceipt(
            _RUNTIME_RECEIPT_TOKEN,
            payload={
                "schema_version": "expected_pe_model_zoo.probabilistic_runtime_receipt.v2",
                "authorization_raw_sha256": self._authorization.raw_sha256,
                "participant_id": self._model_id,
                "provenance": {
                    "evidence_mode": "LIVE_RESOURCE_GUARD",
                    "prediction_raw_sha256": predictions.raw_sha256,
                    "prediction_logical_sha256": predictions.prediction_sha256,
                    "prediction_receipt_sha256": predictions.receipt_sha256,
                    "common_identity_sha256": predictions.identity_sha256,
                    "resource_snapshot_sha256": snapshot_sha256,
                    "authorization_raw_sha256": self._authorization.raw_sha256,
                    "source_closure_sha256": self._authorization.source_closure_sha256,
                },
                "metrics": {
                    "crossing_row_rate": self._crossing_rows / self._rows,
                    "crossing_p95_magnitude": float(np.quantile(gaps, 0.95)),
                    "post_repair_crossing_rate": 0.0,
                    "common_mask_coverage": 1.0,
                    "runtime_minutes": elapsed_minutes,
                    "aggregate_rss_gib": peak_rss,
                    "minimum_free_ram_gib": minimum_free,
                    "worker_count": self._worker_count,
                    "live_snapshot_count": len(self._snapshots),
                },
            },
        )


def create_candidate_resource_guard(
    *,
    authorization: ExecutionAuthorization,
    model_id: str,
    worker_count: int,
) -> CandidateResourceGuard:
    if not isinstance(authorization, ExecutionAuthorization):
        raise ProbabilisticContractError("resource guard requires factory authorization")
    authorization.verify_integrity()
    if model_id not in CANDIDATE_IDS:
        raise ProbabilisticContractError("resource guard candidate is not locked")
    ResourcePolicy(outer_workers=worker_count)
    return CandidateResourceGuard(
        _RESOURCE_GUARD_TOKEN,
        authorization=authorization,
        model_id=model_id,
        worker_count=worker_count,
    )


def score_free_runtime_receipts(
    authorization: ExecutionAuthorization,
) -> dict[str, VerifiedRuntimeReceipt]:
    """Create fixed zero-input runtime receipts for score-free governance wiring."""

    authorization.verify_integrity()
    if authorization.scope != SCORE_FREE_SCOPE:
        raise ProbabilisticContractError("fixed runtime receipts require score-free scope")
    common_identity = authorization.bindings["formal_identity_logical_sha256"]
    output: dict[str, VerifiedRuntimeReceipt] = {}
    for model_id in CANDIDATE_IDS:
        fixture_hash = sha256_bytes(f"SCORE_FREE_FIXED_PREDICTION:{model_id}".encode())
        output[model_id] = VerifiedRuntimeReceipt(
            _RUNTIME_RECEIPT_TOKEN,
            payload={
                "schema_version": "expected_pe_model_zoo.probabilistic_runtime_receipt.v2",
                "authorization_raw_sha256": authorization.raw_sha256,
                "participant_id": model_id,
                "provenance": {
                    "evidence_mode": "SCORE_FREE_FIXED_INTERNAL_FIXTURE",
                    "prediction_raw_sha256": fixture_hash,
                    "prediction_logical_sha256": fixture_hash,
                    "prediction_receipt_sha256": fixture_hash,
                    "common_identity_sha256": common_identity,
                    "resource_snapshot_sha256": [fixture_hash],
                    "authorization_raw_sha256": authorization.raw_sha256,
                    "source_closure_sha256": authorization.source_closure_sha256,
                },
                "metrics": {
                    "crossing_row_rate": 0.0,
                    "crossing_p95_magnitude": 0.0,
                    "post_repair_crossing_rate": 0.0,
                    "common_mask_coverage": 1.0,
                    "runtime_minutes": 1.0,
                    "aggregate_rss_gib": 1.0,
                    "minimum_free_ram_gib": 95.0,
                    "worker_count": 8,
                    "live_snapshot_count": 1,
                },
            },
        )
    return output


def assert_resource_snapshot(
    *, aggregate_rss_gib: float, free_ram_gib: float, wall_minutes: float
) -> None:
    values = (aggregate_rss_gib, free_ram_gib, wall_minutes)
    if any(not isinstance(value, (int, float)) or value < 0 for value in values):
        raise ProbabilisticContractError("resource snapshot is invalid")
    if aggregate_rss_gib > 64:
        raise ProbabilisticContractError("aggregate RSS exceeded 64 GiB; no retry allowed")
    if free_ram_gib < 16:
        raise ProbabilisticContractError("free RAM fell below 16 GiB; no retry allowed")
    if wall_minutes > 90:
        raise ProbabilisticContractError("candidate runtime exceeded 90 minutes; no retry allowed")


def _outer_fit_worker(task: tuple[int, str]) -> dict[str, Any]:
    """Fit one real adapter on the deterministic first formal synthetic outer fold."""

    os.environ.update(THREAD_ENVIRONMENT)
    task_index, model_id = task
    from ..contracts import FitContext, PredictContext
    from .adapters import create_adapter
    from .spec import FEATURE_COLUMNS, feature_metadata

    rng = np.random.default_rng(20260819)
    matrix = rng.normal(size=(525, len(FEATURE_COLUMNS)))
    train = pd.DataFrame(matrix[:504], columns=FEATURE_COLUMNS)
    test = pd.DataFrame(matrix[504:], columns=FEATURE_COLUMNS)
    target = pd.Series(
        np.exp(
            2.7 + 0.05 * train[FEATURE_COLUMNS[0]].to_numpy() + rng.normal(0.0, 0.12, len(train))
        ),
        name="observed_pe",
    )
    fit_context = FitContext(
        experiment_id="score-free-actual-outer-fit-v3",
        fold_id="fold_012",
        seed=0,
        train_end=pd.Timestamp("2020-05-18"),
        target_name="observed_pe",
        feature_metadata=feature_metadata(),
    )
    predict_context = PredictContext(
        experiment_id="score-free-actual-outer-fit-v3",
        fold_id="fold_012",
        seed=0,
        prediction_start=pd.Timestamp("2020-05-19"),
        prediction_end=pd.Timestamp("2020-06-08"),
        feature_metadata=feature_metadata(),
    )
    started = time.monotonic()
    started_cpu = time.process_time()
    model = create_adapter(model_id)
    model.fit(train, target, context=fit_context)
    batch = model.predict(test, context=predict_context)
    fit_predict_seconds = time.monotonic() - started
    cpu_seconds = time.process_time() - started_cpu
    if batch.raw_log_quantiles.shape != (21, 5):
        raise ProbabilisticContractError("outer-fit benchmark did not publish five quantiles")
    estimator_count = len(getattr(model, "_estimators", ()))
    if model_id in CANDIDATE_IDS[:2] and estimator_count != 5:
        raise ProbabilisticContractError(
            "quantile candidate did not fit five estimators sequentially"
        )
    if model_id == CANDIDATE_IDS[2] and getattr(model, "_estimator", None) is None:
        raise ProbabilisticContractError("NGBoost distribution was not fitted")
    output_bytes = canonical_csv_bytes(batch.output_frame())
    model_bytes = pickle.dumps(model, protocol=5)
    replay = pickle.loads(model_bytes).predict(test, context=predict_context)
    replay_bytes = canonical_csv_bytes(replay.output_frame())
    if output_bytes != replay_bytes:
        raise ProbabilisticContractError("actual outer-fit serialization replay changed output")
    return {
        "task_index": task_index,
        "pid": os.getpid(),
        "model_id": model_id,
        "train_rows": len(train),
        "test_rows": len(test),
        "outer_fold_id": "fold_012",
        "fit_attempts": model.fit_attempts,
        "sequential_quantile_estimator_count": estimator_count,
        "published_quantile_count": batch.raw_log_quantiles.shape[1],
        "fit_predict_seconds": fit_predict_seconds,
        "cpu_seconds": cpu_seconds,
        "serialized_model_bytes": len(model_bytes),
        "serialized_model_sha256": sha256_bytes(model_bytes),
        "serialized_output": output_bytes,
        "serialized_output_sha256": sha256_bytes(output_bytes),
        "serialization_replay_sha256": sha256_bytes(replay_bytes),
    }


def _run_parity_lane(worker_count: int) -> dict[str, Any]:
    tasks = [(index, CANDIDATE_IDS[index % len(CANDIDATE_IDS)]) for index in range(worker_count)]
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    context = mp.get_context("spawn")
    peak_rss = 0.0
    minimum_free = float("inf")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        futures = [pool.submit(_outer_fit_worker, task) for task in tasks]
        rss, free, _ = _live_metrics()
        peak_rss = rss
        minimum_free = free
        while not all(future.done() for future in futures):
            rss, free, _ = _live_metrics()
            peak_rss = max(peak_rss, rss)
            minimum_free = min(minimum_free, free)
            assert_resource_snapshot(
                aggregate_rss_gib=rss,
                free_ram_gib=free,
                wall_minutes=(time.monotonic() - started_wall) / 60.0,
            )
            time.sleep(0.005)
        results = [future.result() for future in futures]
    observed_pids = sorted({int(result["pid"]) for result in results})
    if len(observed_pids) != worker_count:
        raise ProbabilisticContractError("actual outer-fit lane did not use every spawn worker")
    canonical_outputs: dict[str, dict[str, Any]] = {}
    duration_by_candidate: dict[str, list[float]] = {}
    for model_id in CANDIDATE_IDS:
        records = [result for result in results if result["model_id"] == model_id]
        if not records:
            raise ProbabilisticContractError("outer-fit lane omitted a locked candidate")
        output_hashes = {str(record["serialized_output_sha256"]) for record in records}
        if len(output_hashes) != 1:
            raise ProbabilisticContractError("parallel outer fits changed candidate output bytes")
        first = min(records, key=lambda value: int(value["task_index"]))
        raw = bytes(first.pop("serialized_output"))
        canonical_outputs[model_id] = {
            "serialized_output_sha256": sha256_bytes(raw),
            "serialized_output_bytes": len(raw),
            "serialized_output_csv_base64": base64.b64encode(raw).decode("ascii"),
            "published_quantile_count": first["published_quantile_count"],
            "sequential_quantile_estimator_count": first["sequential_quantile_estimator_count"],
        }
        duration_by_candidate[model_id] = [
            float(record["fit_predict_seconds"]) for record in records
        ]
    for record in results:
        record.pop("serialized_output", None)
    common_hash = sha256_bytes(
        canonical_json_bytes(
            {
                model_id: canonical_outputs[model_id]["serialized_output_sha256"]
                for model_id in CANDIDATE_IDS
            }
        )
    )
    snapshot = capture_resource_snapshot(
        worker_count=worker_count,
        started_monotonic=started_wall,
        started_cpu=started_cpu,
    )
    return {
        "worker_count": worker_count,
        "requested_processes": worker_count,
        "observed_worker_pids": observed_pids,
        "actual_outer_fit_task_count": len(results),
        "candidate_ids": list(CANDIDATE_IDS),
        "canonical_outputs": canonical_outputs,
        "candidate_fit_seconds": duration_by_candidate,
        "prediction_sha256": common_hash,
        "task_receipts": sorted(results, key=lambda value: int(value["task_index"])),
        "live_peak_process_tree_rss_gib": peak_rss,
        "live_minimum_free_ram_gib": minimum_free,
        "resource_snapshot": snapshot,
    }


def run_score_free_parity_benchmark(*, rows: int = 1296) -> dict[str, Any]:
    """Run real synthetic outer-fold fits for all candidates at every spawn lane."""

    if rows != 1296:
        raise ProbabilisticContractError("runtime projection is fixed to the 1,296-row screen")
    lanes = [_run_parity_lane(workers) for workers in (8, 16, 24, 32)]
    prediction_hash = verify_worker_count_parity(
        {lane["worker_count"]: lane["prediction_sha256"] for lane in lanes}
    )
    projections: list[dict[str, Any]] = []
    for lane in lanes:
        worker_count = int(lane["worker_count"])
        candidate_minutes: dict[str, float] = {}
        for model_id in CANDIDATE_IDS:
            durations = np.asarray(lane["candidate_fit_seconds"][model_id], dtype=np.float64)
            conservative_seconds = float(np.quantile(durations, 0.95)) * math.ceil(
                62 / worker_count
            ) * 2.0 + float(lane["resource_snapshot"]["wall_seconds"])
            candidate_minutes[model_id] = conservative_seconds / 60.0
        projections.append(
            {
                "worker_count": worker_count,
                "conservative_candidate_minutes": candidate_minutes,
                "maximum_candidate_minutes": max(candidate_minutes.values()),
                "within_90_minutes": max(candidate_minutes.values()) <= 90.0,
            }
        )
    eligible = [item for item in projections if item["within_90_minutes"]]
    if not eligible:
        raise ProbabilisticContractError("actual outer-fit projection exceeds 90 minutes")
    selected = min(
        eligible,
        key=lambda value: (float(value["maximum_candidate_minutes"]), int(value["worker_count"])),
    )
    return seal_payload(
        {
            "schema_version": "expected_pe_model_zoo.probabilistic_spawn_parity.v3",
            "score_free": True,
            "project_predictions_generated": False,
            "rows": rows,
            "workload": "actual_synthetic_outer_fold_012_fit_predict_serialize_replay",
            "candidate_ids": list(CANDIDATE_IDS),
            "formal_outer_fold_count_per_candidate": 62,
            "quantile_execution": "five_sequential_for_qlinear_qhistgb;five_from_one_normal_for_ngboost",
            "lanes": lanes,
            "common_prediction_sha256": prediction_hash,
            "formal_runtime_projections": projections,
            "selected_worker_count": selected["worker_count"],
            "selected_projection_maximum_candidate_minutes": selected["maximum_candidate_minutes"],
            "full_screen_wall_limit_minutes": 90,
            "projection_guard_passed": True,
        }
    )


def verify_worker_count_parity(prediction_hashes: dict[int, str]) -> str:
    if tuple(sorted(prediction_hashes)) != (8, 16, 24, 32):
        raise ProbabilisticContractError("worker parity requires 8/16/24/32 evidence")
    hashes = set(prediction_hashes.values())
    if len(hashes) != 1:
        raise ProbabilisticContractError("worker count changed prediction bytes")
    return next(iter(hashes))
