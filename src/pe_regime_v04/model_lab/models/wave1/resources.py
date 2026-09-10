"""Fail-closed CPU-only and memory policy for Wave-1 execution."""

from __future__ import annotations

from contextlib import AbstractContextManager
import ctypes
from ctypes import wintypes
import os
import platform
import threading
from typing import Any

from ...contracts import ContractError
from .spec import RESOURCE_POLICY


CPU_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
GPU_SEAL_ENV = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "NVIDIA_VISIBLE_DEVICES": "void",
}


class ResourcePolicyError(ContractError):
    """Raised whenever the sealed resource contract is not satisfied."""


def seal_cpu_environment() -> None:
    """Overwrite thread/GPU variables; callers must invoke before numerical imports."""

    for name, value in {**CPU_THREAD_ENV, **GPU_SEAL_ENV}.items():
        os.environ[name] = value


def assert_cpu_environment() -> None:
    expected = {**CPU_THREAD_ENV, **GPU_SEAL_ENV}
    wrong = {name: os.environ.get(name) for name, value in expected.items() if os.environ.get(name) != value}
    if wrong:
        raise ResourcePolicyError(f"CPU-only environment seal mismatch: {wrong}")


def set_full_process_affinity_0_31() -> tuple[int, ...]:
    """Set and verify full logical affinity for processors 0..31."""

    requested = tuple(RESOURCE_POLICY["affinity_logical_processors"])
    if os.cpu_count() is None or os.cpu_count() < len(requested):
        raise ResourcePolicyError("host exposes fewer than 32 logical processors")
    if platform.system() == "Windows":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_process = kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        set_mask = kernel32.SetProcessAffinityMask
        set_mask.argtypes = (wintypes.HANDLE, ctypes.c_size_t)
        set_mask.restype = wintypes.BOOL
        get_mask = kernel32.GetProcessAffinityMask
        get_mask.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        )
        get_mask.restype = wintypes.BOOL
        handle = get_process()
        requested_mask = (1 << len(requested)) - 1
        if not set_mask(handle, requested_mask):
            raise ResourcePolicyError(
                f"SetProcessAffinityMask failed with WinError {ctypes.get_last_error()}"
            )
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not get_mask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise ResourcePolicyError(
                f"GetProcessAffinityMask failed with WinError {ctypes.get_last_error()}"
            )
        actual = tuple(index for index in requested if process_mask.value & (1 << index))
    elif hasattr(os, "sched_setaffinity") and hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, set(requested))
        actual = tuple(sorted(os.sched_getaffinity(0)))
    else:
        raise ResourcePolicyError("process affinity control is unsupported on this platform")
    if actual != requested:
        raise ResourcePolicyError(f"logical affinity mismatch: expected={requested}, actual={actual}")
    return actual


class _MemoryStatusEx(ctypes.Structure):
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


class _ProcessMemoryCountersEx(ctypes.Structure):
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
        ("PrivateUsage", ctypes.c_size_t),
    ]


def memory_snapshot() -> dict[str, int]:
    if platform.system() != "Windows":
        try:
            import resource

            rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return {"process_rss_bytes": rss, "available_physical_bytes": pages * page_size}
        except (ImportError, OSError, ValueError) as exc:
            raise ResourcePolicyError("memory accounting is unavailable") from exc

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    global_memory_status = kernel32.GlobalMemoryStatusEx
    global_memory_status.argtypes = (ctypes.POINTER(_MemoryStatusEx),)
    global_memory_status.restype = wintypes.BOOL
    get_process = kernel32.GetCurrentProcess
    get_process.argtypes = ()
    get_process.restype = wintypes.HANDLE
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not global_memory_status(ctypes.byref(status)):
        raise ResourcePolicyError(
            f"GlobalMemoryStatusEx failed with WinError {ctypes.get_last_error()}"
        )
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    get_process_memory = psapi.GetProcessMemoryInfo
    get_process_memory.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCountersEx),
        wintypes.DWORD,
    )
    get_process_memory.restype = wintypes.BOOL
    counters = _ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    handle = get_process()
    if not get_process_memory(handle, ctypes.byref(counters), counters.cb):
        raise ResourcePolicyError(
            f"GetProcessMemoryInfo failed with WinError {ctypes.get_last_error()}"
        )
    return {
        "process_rss_bytes": int(counters.WorkingSetSize),
        "available_physical_bytes": int(status.ullAvailPhys),
    }


def assert_memory_budget() -> dict[str, int]:
    snapshot = memory_snapshot()
    if snapshot["process_rss_bytes"] > int(RESOURCE_POLICY["max_total_rss_bytes"]):
        raise ResourcePolicyError("Wave1 process RSS exceeded the sealed 64 GiB cap")
    if snapshot["available_physical_bytes"] < int(RESOURCE_POLICY["minimum_free_ram_bytes"]):
        raise ResourcePolicyError("host free physical memory fell below the sealed 16 GiB floor")
    return snapshot


def assert_threadpools_one() -> list[dict[str, Any]]:
    from threadpoolctl import threadpool_info

    inventory = threadpool_info()
    excessive = [
        item
        for item in inventory
        if isinstance(item.get("num_threads"), int) and int(item["num_threads"]) != 1
    ]
    if excessive:
        raise ResourcePolicyError(f"native thread pools are not capped at one: {excessive}")
    return inventory


class ResourceGuard(AbstractContextManager["ResourceGuard"]):
    """Sample memory during a ThreadPool run and fail the whole run on breach."""

    def __init__(self, *, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample_lock = threading.Lock()
        self._failure: BaseException | None = None
        self.peak_rss_bytes = 0
        self.minimum_available_bytes: int | None = None

    def _sample(self) -> None:
        with self._sample_lock:
            try:
                snapshot = assert_memory_budget()
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes, snapshot["process_rss_bytes"]
                )
                available = snapshot["available_physical_bytes"]
                self.minimum_available_bytes = (
                    available
                    if self.minimum_available_bytes is None
                    else min(self.minimum_available_bytes, available)
                )
            except BaseException as exc:  # retain exact asynchronous policy failure
                self._failure = exc
                self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> ResourceGuard:
        assert_cpu_environment()
        self._sample()
        if self._failure is not None:
            raise self._failure
        self._thread = threading.Thread(target=self._loop, name="wave1-resource-guard", daemon=True)
        self._thread.start()
        return self

    def check(self) -> None:
        if self._failure is not None:
            raise ResourcePolicyError("resource monitor recorded a policy breach") from self._failure
        self._sample()
        if self._failure is not None:
            raise ResourcePolicyError("resource monitor recorded a policy breach") from self._failure

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample()
        if exc is None and self._failure is not None:
            raise ResourcePolicyError("resource monitor recorded a policy breach") from self._failure
        return False
