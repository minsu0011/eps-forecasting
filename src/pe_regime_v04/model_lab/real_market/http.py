"""Bounded official-source HTTP transport with a process-global SEC throttle."""

from __future__ import annotations

from datetime import datetime, timezone
import re
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .contracts import RawArtifact, RealMarketContractError, SEC_SOURCE_ID


SEC_MAX_REQUESTS_PER_SECOND = 8.0
DEFAULT_MAX_BYTES = 32 * 1024 * 1024
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


class MinimumIntervalRateLimiter:
    """Thread-safe start-time limiter.

    One request every 1/rate seconds is stricter than an eight-request rolling
    burst.  The module-global instance makes the SEC limit process-wide.
    """

    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_second <= 0 or requests_per_second > SEC_MAX_REQUESTS_PER_SECOND:
            raise RealMarketContractError("SEC limiter must be in (0, 8] requests/second")
        self.requests_per_second = float(requests_per_second)
        self._interval = 1.0 / self.requests_per_second
        self._clock = clock
        self._sleeper = sleeper
        self._next_start = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            delay = self._next_start - now
            if delay > 0:
                self._sleeper(delay)
                now = self._clock()
            self._next_start = max(now, self._next_start) + self._interval


SEC_GLOBAL_LIMITER = MinimumIntervalRateLimiter(SEC_MAX_REQUESTS_PER_SECOND)


def validate_sec_user_agent(user_agent: str) -> str:
    if not isinstance(user_agent, str):
        raise RealMarketContractError("SEC User-Agent must be a string")
    normalized = " ".join(user_agent.split())
    if len(normalized) < 12 or len(normalized) > 240 or _EMAIL.search(normalized) is None:
        raise RealMarketContractError(
            "SEC User-Agent must identify the project/organization and contain a contact email"
        )
    lowered = normalized.lower()
    if lowered.startswith(("python-urllib", "python-requests", "mozilla/5.0")):
        raise RealMarketContractError("generic browser/library SEC User-Agent is forbidden")
    return normalized


class OfficialSourceHttpClient:
    """Fetch exact bytes only after an adapter validates the requested URL."""

    def __init__(
        self,
        *,
        user_agent: str,
        now: Callable[[], datetime] | None = None,
        sec_limiter: MinimumIntervalRateLimiter = SEC_GLOBAL_LIMITER,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.user_agent = " ".join(str(user_agent).split())
        if not self.user_agent:
            raise RealMarketContractError("a descriptive User-Agent is required")
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._sec_limiter = sec_limiter
        self._opener = opener or urlopen

    def fetch(
        self,
        *,
        source_id: str,
        url: str,
        validate_url: Callable[[str], None],
        timeout_seconds: float = 30.0,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> RawArtifact:
        if timeout_seconds <= 0 or timeout_seconds > 120:
            raise RealMarketContractError("HTTP timeout must be in (0, 120] seconds")
        if max_bytes <= 0 or max_bytes > 256 * 1024 * 1024:
            raise RealMarketContractError("HTTP max_bytes must be in (0, 256 MiB]")
        validate_url(url)
        if source_id == SEC_SOURCE_ID:
            validate_sec_user_agent(self.user_agent)
            self._sec_limiter.acquire()
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "identity",
                "Accept": "application/json,text/csv,application/xml,text/xml;q=0.9",
            },
            method="GET",
        )
        try:
            with self._opener(request, timeout=timeout_seconds) as response:
                status = int(response.getcode())
                final_url = str(response.geturl())
                validate_url(final_url)
                if status != 200:
                    raise RealMarketContractError(f"official source returned HTTP {status}")
                body = response.read(max_bytes + 1)
                if len(body) > max_bytes:
                    raise RealMarketContractError(
                        f"official source object exceeds bounded maximum of {max_bytes} bytes"
                    )
                retained_headers: dict[str, str] = {}
                for name in ("Content-Type", "ETag", "Last-Modified", "Date", "Retry-After"):
                    value = response.headers.get(name)
                    if value is not None:
                        retained_headers[name] = value
        except HTTPError as exc:
            raise RealMarketContractError(
                f"official source request failed with HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise RealMarketContractError("official source request failed") from exc
        retrieved_at = self._now()
        return RawArtifact.from_bytes(
            source_id=source_id,
            requested_url=url,
            final_url=final_url,
            retrieved_at=retrieved_at,
            http_status=status,
            headers=retained_headers,
            body=body,
        )
