"""Spawn-only no-score runner with live host resource guards."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import platform
from statistics import median
import time
from typing import Any, Iterable, Iterator, Mapping

from .contracts import (
    STRUCTURAL_DESIGN_SHA256,
    StructuralContractError,
    canonical_json_bytes,
    seal_payload,
    sha256_bytes,
)
from .resources import (
    CPU_AFFINITY,
    MAX_TOTAL_RSS_BYTES,
    MINIMUM_FREE_RAM_BYTES,
    THREAD_ENVIRONMENT,
    freeze_benchmarked_process_backend_policy,
    require_worker_hash_parity,
)


BENCHMARK_WORKER_GRID = (8, 16, 24, 32)
GPU_DISABLED_VALUE = "-1"


def _psutil():
    try:
        import psutil
    except ImportError as exc:
        raise StructuralContractError("psutil is required for live resource guards") from exc
    return psutil


def _worker_initializer() -> None:
    for key, value in THREAD_ENVIRONMENT.items():
        os.environ[key] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DISABLED_VALUE
    psutil = _psutil()
    process = psutil.Process()
    try:
        available = tuple(int(value) for value in process.cpu_affinity())
        desired = tuple(value for value in CPU_AFFINITY if value in available)
        if desired:
            process.cpu_affinity(list(desired))
    except (AttributeError, OSError, ValueError) as exc:
        raise RuntimeError(f"cannot apply worker CPU affinity: {exc}") from exc


def _synthetic_no_score_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Hash-only workload.  It has no model, target, prediction, seed, or score."""

    required = {"task_index", "rounds", "payload_sha256", "design_sha256"}
    if set(payload) != required or payload.get("design_sha256") != STRUCTURAL_DESIGN_SHA256:
        raise RuntimeError("synthetic worker payload schema changed")
    task_index = int(payload["task_index"])
    rounds = int(payload["rounds"])
    expected_payload_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "task_index": task_index,
                "rounds": rounds,
                "design_sha256": STRUCTURAL_DESIGN_SHA256,
            }
        )
    )
    if payload["payload_sha256"] != expected_payload_hash or rounds < 1:
        raise RuntimeError("synthetic worker payload seal changed")
    state = bytes.fromhex(expected_payload_hash)
    started_cpu = time.process_time()
    for round_index in range(rounds):
        state = hashlib.sha256(
            state + task_index.to_bytes(4, "little") + round_index.to_bytes(4, "little")
        ).digest()
    psutil = _psutil()
    process = psutil.Process()
    try:
        affinity = tuple(int(value) for value in process.cpu_affinity())
    except (AttributeError, OSError, ValueError):
        affinity = ()
    native_pool_threads: list[int] = []
    try:
        from threadpoolctl import threadpool_info

        native_pool_threads = sorted(
            {
                int(row["num_threads"])
                for row in threadpool_info()
                if row.get("num_threads") is not None
            }
        )
    except Exception:
        native_pool_threads = []
    return {
        "task_index": task_index,
        "result_sha256": state.hex(),
        "pid": os.getpid(),
        "affinity": list(affinity),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENVIRONMENT},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "native_pool_threads": native_pool_threads,
        "rss_bytes": int(process.memory_info().rss),
        "cpu_seconds": time.process_time() - started_cpu,
        "warnings": [],
    }


@contextmanager
def _spawn_environment() -> Iterator[None]:
    keys = (*THREAD_ENVIRONMENT.keys(), "CUDA_VISIBLE_DEVICES")
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key, value in THREAD_ENVIRONMENT.items():
            os.environ[key] = value
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_DISABLED_VALUE
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _aggregate_process_rss_bytes() -> tuple[int, list[int]]:
    psutil = _psutil()
    parent = psutil.Process()
    processes = [parent, *parent.children(recursive=True)]
    total = 0
    pids: list[int] = []
    for process in processes:
        try:
            total += int(process.memory_info().rss)
            pids.append(int(process.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total, sorted(set(pids))


def _payloads(*, task_count: int, rounds: int) -> list[dict[str, Any]]:
    if task_count < 1 or rounds < 1:
        raise StructuralContractError("synthetic benchmark dimensions must be positive")
    output: list[dict[str, Any]] = []
    for task_index in range(task_count):
        core = {
            "task_index": task_index,
            "rounds": rounds,
            "design_sha256": STRUCTURAL_DESIGN_SHA256,
        }
        output.append({**core, "payload_sha256": sha256_bytes(canonical_json_bytes(core))})
    return output


def _terminate_executor_processes(executor: ProcessPoolExecutor) -> None:
    """Best-effort immediate stop after a hard live resource breach."""

    processes = getattr(executor, "_processes", None)
    if not isinstance(processes, dict):
        return
    for process in tuple(processes.values()):
        try:
            process.terminate()
        except (AttributeError, OSError):
            continue


def run_guarded_no_score_tasks(
    payloads: Iterable[Mapping[str, Any]],
    *,
    worker_count: int,
) -> dict[str, Any]:
    """Run primitive payloads under spawn while continuously enforcing RAM guards."""

    if worker_count not in BENCHMARK_WORKER_GRID:
        raise StructuralContractError("worker count is outside the frozen benchmark grid")
    materialized = [dict(payload) for payload in payloads]
    if not materialized:
        raise StructuralContractError("guarded runner requires at least one task")
    psutil = _psutil()
    initial_free = int(psutil.virtual_memory().available)
    if initial_free < MINIMUM_FREE_RAM_BYTES:
        raise StructuralContractError("free RAM is below the 16-GiB admission floor")
    started = time.perf_counter()
    max_rss = 0
    min_free = initial_free
    observed_pids: set[int] = set()
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    breached: str | None = None
    context = mp.get_context("spawn")
    with _spawn_environment():
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_worker_initializer,
        ) as executor:
            future_to_index = {
                executor.submit(_synthetic_no_score_worker, payload): int(payload["task_index"])
                for payload in materialized
            }
            pending = set(future_to_index)
            while pending:
                rss, pids = _aggregate_process_rss_bytes()
                free = int(psutil.virtual_memory().available)
                max_rss = max(max_rss, rss)
                min_free = min(min_free, free)
                observed_pids.update(pids)
                if rss > MAX_TOTAL_RSS_BYTES:
                    breached = "aggregate_rss_above_64_gib"
                elif free < MINIMUM_FREE_RAM_BYTES:
                    breached = "free_ram_below_16_gib"
                if breached is not None:
                    for future in pending:
                        future.cancel()
                    _terminate_executor_processes(executor)
                    failures.append({"kind": "resource_guard", "message": breached})
                    break
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    task_index = future_to_index[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        failures.append(
                            {
                                "kind": type(exc).__name__,
                                "task_index": task_index,
                                "message": str(exc),
                            }
                        )
                    else:
                        results.append(row)
                        for message in row.get("warnings", []):
                            warnings.append({"task_index": task_index, "message": str(message)})
    wall = time.perf_counter() - started
    results.sort(key=lambda row: int(row["task_index"]))
    if breached is not None or failures:
        raise StructuralContractError(
            f"guarded spawn runner failed: breached={breached}, failures={failures}"
        )
    if len(results) != len(materialized):
        raise StructuralContractError("guarded spawn runner lost task results")
    for row in results:
        if row["thread_environment"] != THREAD_ENVIRONMENT:
            raise StructuralContractError("worker native thread environment changed")
        if row["cuda_visible_devices"] != GPU_DISABLED_VALUE:
            raise StructuralContractError("worker GPU guard changed")
        if row["native_pool_threads"] and any(value != 1 for value in row["native_pool_threads"]):
            raise StructuralContractError("worker native thread pool exceeds one thread")
        if tuple(row["affinity"]) != CPU_AFFINITY:
            raise StructuralContractError("worker CPU affinity differs from logical CPUs 0..31")
    result_hashes = [str(row["result_sha256"]) for row in results]
    return {
        "worker_count": worker_count,
        "task_count": len(materialized),
        "wall_seconds": wall,
        "worker_cpu_seconds_sum": sum(float(row["cpu_seconds"]) for row in results),
        "peak_live_aggregate_rss_bytes": max_rss,
        "minimum_live_free_ram_bytes": min_free,
        "observed_process_ids": sorted(observed_pids),
        "result_hashes": result_hashes,
        "result_manifest_sha256": sha256_bytes(canonical_json_bytes(result_hashes)),
        "worker_max_reported_rss_bytes": max(int(row["rss_bytes"]) for row in results),
        "affinity_all_exact_0_31": True,
        "thread_environment_all_one": True,
        "gpu_guard_all_off": True,
        "warnings": warnings,
        "failures": failures,
        "resource_guard_breached": False,
    }


def benchmark_no_score_backend(
    *,
    task_count: int = 64,
    rounds: int = 12000,
    repeats: int = 2,
) -> dict[str, Any]:
    """Benchmark 8/16/24/32 using identical deterministic hash-only payloads."""

    if repeats < 2:
        raise StructuralContractError("benchmark requires at least two repeats")
    psutil = _psutil()
    logical_cpus = os.cpu_count()
    if logical_cpus is None or logical_cpus < max(BENCHMARK_WORKER_GRID):
        raise StructuralContractError("host lacks the frozen 32 logical CPU surface")
    payloads = _payloads(task_count=task_count, rounds=rounds)
    records: list[dict[str, Any]] = []
    reference_hashes: list[str] | None = None
    reference_worker_count: int | None = None
    for worker_count in BENCHMARK_WORKER_GRID:
        runs: list[dict[str, Any]] = []
        for repeat in range(repeats):
            row = run_guarded_no_score_tasks(payloads, worker_count=worker_count)
            row["repeat"] = repeat
            runs.append(row)
            if reference_hashes is None:
                reference_hashes = list(row["result_hashes"])
                reference_worker_count = worker_count
            else:
                if worker_count == reference_worker_count:
                    if tuple(reference_hashes) != tuple(row["result_hashes"]):
                        raise StructuralContractError("same-worker replay hash parity failed")
                else:
                    require_worker_hash_parity(
                        reference_hashes,
                        row["result_hashes"],
                        reference_worker_count=int(reference_worker_count),
                        candidate_worker_count=worker_count,
                    )
        wall_values = sorted(float(row["wall_seconds"]) for row in runs)
        median_wall = float(median(wall_values))
        records.append(
            {
                "worker_count": worker_count,
                "median_wall_seconds": median_wall,
                "runs": runs,
                "all_result_hash_parity": True,
                "all_resource_guards_pass": True,
            }
        )
    selected = min(records, key=lambda row: (row["median_wall_seconds"], row["worker_count"]))
    available = int(psutil.virtual_memory().available)
    policy = freeze_benchmarked_process_backend_policy(
        selected_worker_count=int(selected["worker_count"]),
        logical_cpu_count=int(logical_cpus),
        available_physical_bytes=available,
    )
    root = Path(__file__).resolve().parents[4]
    generator_paths = (
        root / "src/pe_regime_v04/model_lab/structural/runner.py",
        root / "src/pe_regime_v04/model_lab/structural/resources.py",
        root / "src/pe_regime_v04/model_lab/structural/contracts.py",
    )
    payload = {
        "format_version": 2,
        "mode": "structural_no_score_spawn_backend_benchmark",
        "design_sha256": STRUCTURAL_DESIGN_SHA256,
        "generator_source_inventory": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_bytes(path.read_bytes()),
            }
            for path in generator_paths
        ],
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": logical_cpus,
            "physical_memory_bytes": int(psutil.virtual_memory().total),
            "available_memory_at_policy_freeze_bytes": available,
        },
        "workload": {
            "kind": "deterministic_hash_only_no_model_no_target_no_prediction_no_score",
            "task_count": task_count,
            "rounds_per_task": rounds,
            "repeats": repeats,
            "payload_manifest_sha256": sha256_bytes(canonical_json_bytes(payloads)),
        },
        "worker_grid": list(BENCHMARK_WORKER_GRID),
        "records": records,
        "selection_rule": "minimum_median_wall_seconds_then_lower_worker_count",
        "selected_worker_count": policy.outer_workers,
        "frozen_policy": {
            "backend": policy.backend,
            "start_method": policy.start_method,
            "outer_workers": policy.outer_workers,
            "task_chunksize": policy.task_chunksize,
            "estimator_inner_threads": policy.estimator_inner_threads,
            "blas_openmp_threads": policy.blas_openmp_threads,
            "max_total_rss_bytes": policy.max_total_rss_bytes,
            "minimum_free_ram_bytes": policy.minimum_free_ram_bytes,
            "cpu_affinity": list(policy.cpu_affinity),
            "thread_environment": dict(policy.thread_environment),
            "gpu_enabled": policy.gpu_enabled,
            "benchmark_performed": policy.benchmark_performed,
        },
        "parity": {
            "result_manifest_sha256": sha256_bytes(canonical_json_bytes(reference_hashes)),
            "all_8_16_24_32_repeats_exact": True,
        },
        "warning_failure_ledger": {
            "warnings": [
                item for record in records for run in record["runs"] for item in run["warnings"]
            ],
            "failures": [
                item for record in records for run in record["runs"] for item in run["failures"]
            ],
        },
        "attestations": {
            "spent_input_surface_opened": False,
            "model_fit_or_predict_called": False,
            "target_or_truth_read": False,
            "candidate_prediction_generated": False,
            "candidate_score_generated": False,
            "fresh_seed_selected_or_reserved": False,
            "heldout_opened": False,
        },
    }
    return seal_payload(payload)
