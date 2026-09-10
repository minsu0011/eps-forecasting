"""Precommitted process backend policy and frozen benchmark selection."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping, Sequence

from .contracts import StructuralContractError, require_sha256


DEFAULT_OUTER_WORKERS = 32
MAX_TOTAL_RSS_BYTES = 64 * 1024**3
MINIMUM_FREE_RAM_BYTES = 16 * 1024**3
ESTIMATED_MAX_WORKER_RSS_BYTES = 1536 * 1024**2
CPU_AFFINITY = tuple(range(32))
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class StructuralProcessBackendPolicy:
    backend: str
    start_method: str
    outer_workers: int
    task_chunksize: int
    estimator_inner_threads: int
    blas_openmp_threads: int
    max_total_rss_bytes: int
    minimum_free_ram_bytes: int
    cpu_affinity: tuple[int, ...]
    thread_environment: tuple[tuple[str, str], ...]
    gpu_enabled: bool
    benchmark_performed: bool

    def __post_init__(self) -> None:
        if self.backend != "process" or self.start_method != "spawn":
            raise StructuralContractError(
                "structural execution must use spawn-based process workers"
            )
        if not 1 <= self.outer_workers <= DEFAULT_OUTER_WORKERS:
            raise StructuralContractError("structural outer worker count is invalid")
        if self.task_chunksize != 1:
            raise StructuralContractError("structural task chunksize must preserve fold isolation")
        if self.estimator_inner_threads != 1 or self.blas_openmp_threads != 1:
            raise StructuralContractError("nested/BLAS thread counts must be one")
        if self.max_total_rss_bytes != MAX_TOTAL_RSS_BYTES:
            raise StructuralContractError("structural RSS cap differs from the design")
        if self.minimum_free_ram_bytes != MINIMUM_FREE_RAM_BYTES:
            raise StructuralContractError("structural free-RAM floor differs from the design")
        if self.cpu_affinity != CPU_AFFINITY:
            raise StructuralContractError("structural CPU affinity differs from logical CPUs 0..31")
        if dict(self.thread_environment) != THREAD_ENVIRONMENT:
            raise StructuralContractError("structural native thread environment differs")
        if self.gpu_enabled:
            raise StructuralContractError("GPU must remain off for Structural Wave kernels")


def resolve_process_backend_policy(
    *,
    logical_cpu_count: int | None = None,
    available_physical_bytes: int,
) -> StructuralProcessBackendPolicy:
    """Resolve worker count from frozen caps without timing candidate kernels."""

    cpus = logical_cpu_count if logical_cpu_count is not None else os.cpu_count()
    if not isinstance(cpus, int) or isinstance(cpus, bool) or cpus < 1:
        raise StructuralContractError("logical CPU count must be positive")
    if (
        not isinstance(available_physical_bytes, int)
        or isinstance(available_physical_bytes, bool)
        or available_physical_bytes <= MINIMUM_FREE_RAM_BYTES
    ):
        raise StructuralContractError("insufficient available RAM for structural process backend")
    usable_by_free_floor = available_physical_bytes - MINIMUM_FREE_RAM_BYTES
    memory_budget = min(MAX_TOTAL_RSS_BYTES, usable_by_free_floor)
    memory_workers = max(1, memory_budget // ESTIMATED_MAX_WORKER_RSS_BYTES)
    workers = min(DEFAULT_OUTER_WORKERS, cpus, int(memory_workers))
    return StructuralProcessBackendPolicy(
        backend="process",
        start_method="spawn",
        outer_workers=workers,
        task_chunksize=1,
        estimator_inner_threads=1,
        blas_openmp_threads=1,
        max_total_rss_bytes=MAX_TOTAL_RSS_BYTES,
        minimum_free_ram_bytes=MINIMUM_FREE_RAM_BYTES,
        cpu_affinity=CPU_AFFINITY,
        thread_environment=tuple(sorted(THREAD_ENVIRONMENT.items())),
        gpu_enabled=False,
        benchmark_performed=False,
    )


def freeze_benchmarked_process_backend_policy(
    *,
    selected_worker_count: int,
    logical_cpu_count: int,
    available_physical_bytes: int,
) -> StructuralProcessBackendPolicy:
    """Construct the policy only after a passing 8/16/24/32 no-score benchmark."""

    eligible = resolve_process_backend_policy(
        logical_cpu_count=logical_cpu_count,
        available_physical_bytes=available_physical_bytes,
    )
    if selected_worker_count not in (8, 16, 24, 32):
        raise StructuralContractError("selected workers are outside the frozen benchmark grid")
    if selected_worker_count > eligible.outer_workers:
        raise StructuralContractError("selected workers exceed CPU/RAM admission")
    return StructuralProcessBackendPolicy(
        backend="process",
        start_method="spawn",
        outer_workers=selected_worker_count,
        task_chunksize=1,
        estimator_inner_threads=1,
        blas_openmp_threads=1,
        max_total_rss_bytes=MAX_TOTAL_RSS_BYTES,
        minimum_free_ram_bytes=MINIMUM_FREE_RAM_BYTES,
        cpu_affinity=CPU_AFFINITY,
        thread_environment=tuple(sorted(THREAD_ENVIRONMENT.items())),
        gpu_enabled=False,
        benchmark_performed=True,
    )


def require_worker_hash_parity(
    reference_hashes: Sequence[str],
    candidate_hashes: Sequence[str],
    *,
    reference_worker_count: int,
    candidate_worker_count: int,
) -> None:
    if reference_worker_count == candidate_worker_count:
        raise StructuralContractError("worker parity check requires distinct worker counts")
    for index, value in enumerate(reference_hashes):
        require_sha256(value, field=f"reference_hashes[{index}]")
    for index, value in enumerate(candidate_hashes):
        require_sha256(value, field=f"candidate_hashes[{index}]")
    if tuple(reference_hashes) != tuple(candidate_hashes):
        raise StructuralContractError("worker-count prediction hash parity failed")


def apply_native_thread_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return an explicit child-process environment; the parent is not mutated."""

    output = dict(os.environ if environment is None else environment)
    output.update(THREAD_ENVIRONMENT)
    return output
