"""Fail-closed contracts for the score-free public point-in-time ledger.

This package is deliberately separate from the synthetic Model Lab evaluator.  It
contains source observations and their availability only; it has no price, target,
prediction, or candidate-score vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


SCHEMA_VERSION = "real_market_pit_event_v1"
IMPLEMENTATION_STATUS = "CONDITIONAL"
SCOREABILITY = "NOT_SCOREABLE"
EXCHANGE_ID = "XNYS"
DECISION_CUTOFF_LOCAL = time(16, 15)

SEC_SOURCE_ID = "SEC_EDGAR"
TREASURY_SOURCE_ID = "US_TREASURY_DAILY_YIELD_CURVE"
NYFED_SOURCE_ID = "NYFED_REFERENCE_RATES"


class RealMarketContractError(ValueError):
    """Raised before an uncertain source observation can enter the ledger."""


@dataclass(frozen=True)
class SourcePolicy:
    source_id: str
    source_name: str
    license_id: str
    terms_url: str
    documentation_url: str
    key_required: bool


SOURCE_POLICIES: Mapping[str, SourcePolicy] = MappingProxyType(
    {
        SEC_SOURCE_ID: SourcePolicy(
            source_id=SEC_SOURCE_ID,
            source_name="SEC EDGAR",
            license_id="SEC_PUBLIC_FILINGS_FAIR_ACCESS",
            terms_url="https://www.sec.gov/about/privacy-information",
            documentation_url=(
                "https://www.sec.gov/search-filings/"
                "edgar-application-programming-interfaces"
            ),
            key_required=False,
        ),
        TREASURY_SOURCE_ID: SourcePolicy(
            source_id=TREASURY_SOURCE_ID,
            source_name="U.S. Treasury Daily Treasury Par Yield Curve Rates",
            license_id="US_TREASURY_PUBLIC_DATA_ATTRIBUTION",
            terms_url=(
                "https://home.treasury.gov/utility/"
                "website-policies-and-notices"
            ),
            documentation_url=(
                "https://home.treasury.gov/resource-center/data-chart-center/"
                "interest-rates/TextView?type=daily_treasury_yield_curve"
            ),
            key_required=False,
        ),
        NYFED_SOURCE_ID: SourcePolicy(
            source_id=NYFED_SOURCE_ID,
            source_name="Federal Reserve Bank of New York Reference Rates",
            license_id="NYFED_REFERENCE_RATE_TERMS_AND_ATTRIBUTION",
            terms_url="https://www.newyorkfed.org/privacy/termsofuse.html",
            documentation_url="https://markets.newyorkfed.org/static/docs/markets-api.html",
            key_required=False,
        ),
    }
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "candidate_score",
        "candidate_scores",
        "evaluation_score",
        "expected_pe",
        "future_return",
        "label",
        "model_id",
        "observed_pe",
        "prediction",
        "price",
        "target_value",
        "true_fair_pe",
    }
)


def _new_york_zone() -> ZoneInfo:
    try:
        return ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError as exc:  # pragma: no cover - environment failure
        raise RealMarketContractError(
            "America/New_York timezone data is required for PIT availability"
        ) from exc


NEW_YORK = _new_york_zone()


def utc_datetime(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RealMarketContractError(f"{field_name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def isoformat_utc(value: datetime) -> str:
    normalized = utc_datetime(value, field_name="datetime")
    return normalized.isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso_datetime(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise RealMarketContractError(f"{field_name} must be a non-empty ISO datetime")
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RealMarketContractError(f"{field_name} is not an ISO datetime") from exc
    return utc_datetime(parsed, field_name=field_name)


def parse_iso_date(value: Any, *, field_name: str, allow_empty: bool = False) -> date | None:
    if allow_empty and (value is None or value == ""):
        return None
    if not isinstance(value, str):
        raise RealMarketContractError(f"{field_name} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise RealMarketContractError(f"{field_name} is not an ISO date") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise RealMarketContractError("non-finite values are forbidden")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise RealMarketContractError("non-finite decimal values are forbidden")
        return str(value)
    if isinstance(value, datetime):
        return isoformat_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise RealMarketContractError("JSON object keys must be non-empty strings")
            if raw_key in result:
                raise RealMarketContractError(f"duplicate JSON key: {raw_key!r}")
            result[raw_key] = json_ready(item)
        return result
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    raise RealMarketContractError(f"unsupported JSON value type: {type(value).__name__}")


def strict_json_loads(body: bytes) -> Any:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RealMarketContractError("official JSON is not valid UTF-8") from exc

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise RealMarketContractError(f"official JSON contains duplicate key {key!r}")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise RealMarketContractError(f"official JSON contains non-finite constant {value!r}")

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_float=Decimal,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise RealMarketContractError("official JSON is malformed") from exc


@dataclass(frozen=True)
class RawArtifact:
    """The exact downloaded bytes plus transport provenance.

    Ledger serialization intentionally records only the byte count and hash.  The
    raw bytes belong in a separate immutable bronze object.
    """

    source_id: str
    requested_url: str
    final_url: str
    retrieved_at: datetime
    http_status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes = field(repr=False)
    sha256: str

    def __post_init__(self) -> None:
        if self.source_id not in SOURCE_POLICIES:
            raise RealMarketContractError(f"unsupported source_id: {self.source_id!r}")
        for field_name, url in (
            ("requested_url", self.requested_url),
            ("final_url", self.final_url),
        ):
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.fragment:
                raise RealMarketContractError(f"{field_name} must be an HTTPS URL without fragment")
        object.__setattr__(
            self,
            "retrieved_at",
            utc_datetime(self.retrieved_at, field_name="retrieved_at"),
        )
        if self.http_status != 200:
            raise RealMarketContractError("only HTTP 200 artifacts may enter the ledger")
        if not isinstance(self.body, bytes):
            raise RealMarketContractError("raw artifact body must be bytes")
        expected = hashlib.sha256(self.body).hexdigest()
        if self.sha256 != expected:
            raise RealMarketContractError("raw artifact SHA-256 does not match its bytes")
        normalized_headers: list[tuple[str, str]] = []
        seen: set[str] = set()
        for name, value in self.headers:
            lowered = str(name).strip().lower()
            if not lowered or lowered in seen:
                raise RealMarketContractError("raw artifact headers must be unique and non-empty")
            seen.add(lowered)
            normalized_headers.append((lowered, str(value).strip()))
        object.__setattr__(self, "headers", tuple(sorted(normalized_headers)))

    @classmethod
    def from_bytes(
        cls,
        *,
        source_id: str,
        requested_url: str,
        body: bytes,
        retrieved_at: datetime,
        final_url: str | None = None,
        http_status: int = 200,
        headers: Mapping[str, str] | Iterable[tuple[str, str]] = (),
    ) -> "RawArtifact":
        header_items = tuple(headers.items()) if isinstance(headers, Mapping) else tuple(headers)
        return cls(
            source_id=source_id,
            requested_url=requested_url,
            final_url=final_url or requested_url,
            retrieved_at=retrieved_at,
            http_status=http_status,
            headers=header_items,
            body=body,
            sha256=hashlib.sha256(body).hexdigest(),
        )

    def record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "requested_url": self.requested_url,
            "final_url": self.final_url,
            "retrieved_at": isoformat_utc(self.retrieved_at),
            "http_status": self.http_status,
            "headers": dict(self.headers),
            "bytes": len(self.body),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ExplicitSessionCalendar:
    """An explicit exchange-session allow-list; it never guesses weekdays."""

    sessions: tuple[date, ...]
    exchange_id: str = EXCHANGE_ID
    decision_cutoff_local: time = DECISION_CUTOFF_LOCAL

    def __post_init__(self) -> None:
        if self.exchange_id != EXCHANGE_ID:
            raise RealMarketContractError("the pilot is locked to an explicit XNYS calendar")
        if not self.sessions:
            raise RealMarketContractError("at least one explicit exchange session is required")
        if any(not isinstance(item, date) or isinstance(item, datetime) for item in self.sessions):
            raise RealMarketContractError("sessions must contain date objects")
        if tuple(sorted(set(self.sessions))) != self.sessions:
            raise RealMarketContractError("sessions must be strictly increasing and unique")
        if self.decision_cutoff_local != DECISION_CUTOFF_LOCAL:
            raise RealMarketContractError("pilot decision cutoff is locked to 16:15 America/New_York")

    @classmethod
    def from_iso_dates(cls, values: Iterable[str]) -> "ExplicitSessionCalendar":
        parsed: list[date] = []
        for index, value in enumerate(values):
            item = parse_iso_date(value, field_name=f"sessions[{index}]")
            assert item is not None
            parsed.append(item)
        return cls(tuple(parsed))

    def next_session_after(self, local_date: date) -> date:
        for session in self.sessions:
            if session > local_date:
                return session
        raise RealMarketContractError(
            f"explicit XNYS calendar has no session after {local_date.isoformat()}"
        )

    def session_on_or_after(self, local_date: date) -> date:
        for session in self.sessions:
            if session >= local_date:
                return session
        raise RealMarketContractError(
            f"explicit XNYS calendar has no session on or after {local_date.isoformat()}"
        )

    def first_observable_session(self, available_at: datetime) -> date:
        local = utc_datetime(available_at, field_name="available_at").astimezone(NEW_YORK)
        candidate = self.session_on_or_after(local.date())
        if candidate == local.date() and local.timetz().replace(tzinfo=None) > self.decision_cutoff_local:
            return self.next_session_after(candidate)
        return candidate


def _walk_payload_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", str(key)).replace("-", "_").lower()
            yield normalized
            yield from _walk_payload_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_payload_keys(item)


def make_event_id(*parts: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(list(parts))).hexdigest()


@dataclass(frozen=True)
class LedgerEvent:
    schema_version: str
    event_id: str
    event_type: str
    source_id: str
    source_license_id: str
    source_terms_url: str
    entity_id: str
    security_id: None
    cik: str | None
    effective_at: datetime
    availability_at: datetime
    availability_basis: str
    effective_session: date
    revision_id: str
    revision_kind: str
    revision_of: str | None
    source_url: str
    source_document_sha256: str
    retrieved_at: datetime
    payload: Mapping[str, Any]
    implementation_status: str = IMPLEMENTATION_STATUS
    scoreability: str = SCOREABILITY

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RealMarketContractError("event schema version mismatch")
        if _SHA256.fullmatch(self.event_id) is None:
            raise RealMarketContractError("event_id must be a lowercase SHA-256")
        policy = SOURCE_POLICIES.get(self.source_id)
        if policy is None:
            raise RealMarketContractError("event source is not an approved public source")
        if (
            self.source_license_id != policy.license_id
            or self.source_terms_url != policy.terms_url
        ):
            raise RealMarketContractError("event source/license fields do not match source policy")
        if not self.event_type or not self.entity_id or not self.availability_basis:
            raise RealMarketContractError("event identity and availability basis are required")
        if self.security_id is not None:
            raise RealMarketContractError("this public PIT pilot must not contain price security IDs")
        if self.cik is not None and re.fullmatch(r"[0-9]{10}", self.cik) is None:
            raise RealMarketContractError("CIK must be exactly ten digits")
        effective = utc_datetime(self.effective_at, field_name="effective_at")
        available = utc_datetime(self.availability_at, field_name="availability_at")
        retrieved = utc_datetime(self.retrieved_at, field_name="retrieved_at")
        object.__setattr__(self, "effective_at", effective)
        object.__setattr__(self, "availability_at", available)
        object.__setattr__(self, "retrieved_at", retrieved)
        if retrieved < available:
            raise RealMarketContractError(
                "retrieved_at precedes source availability; historical reconstruction must not "
                "pretend to be a contemporaneous first-seen snapshot"
            )
        if not isinstance(self.effective_session, date) or isinstance(
            self.effective_session, datetime
        ):
            raise RealMarketContractError("effective_session must be a date")
        if not self.revision_id or self.revision_kind not in {
            "ORIGINAL",
            "AMENDMENT",
            "CORRECTION",
            "SNAPSHOT",
        }:
            raise RealMarketContractError("revision metadata is incomplete or unsupported")
        if self.implementation_status != IMPLEMENTATION_STATUS:
            raise RealMarketContractError("public PIT events must remain CONDITIONAL")
        if self.scoreability != SCOREABILITY:
            raise RealMarketContractError("public PIT events must remain NOT_SCOREABLE")
        parsed_url = urlsplit(self.source_url)
        if parsed_url.scheme != "https" or not parsed_url.hostname or parsed_url.fragment:
            raise RealMarketContractError("source_url must be an HTTPS URL without fragment")
        if _SHA256.fullmatch(self.source_document_sha256) is None:
            raise RealMarketContractError("source_document_sha256 must be lowercase SHA-256")
        normalized_payload = json_ready(self.payload)
        forbidden = _FORBIDDEN_PAYLOAD_KEYS.intersection(_walk_payload_keys(normalized_payload))
        if forbidden:
            raise RealMarketContractError(
                f"score/price vocabulary is forbidden in public PIT payload: {sorted(forbidden)}"
            )
        object.__setattr__(self, "payload", MappingProxyType(normalized_payload))

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_id": self.source_id,
            "source_license_id": self.source_license_id,
            "source_terms_url": self.source_terms_url,
            "entity_id": self.entity_id,
            "security_id": self.security_id,
            "cik": self.cik,
            "effective_at": isoformat_utc(self.effective_at),
            "availability_at": isoformat_utc(self.availability_at),
            "availability_basis": self.availability_basis,
            "effective_session": self.effective_session.isoformat(),
            "revision_id": self.revision_id,
            "revision_kind": self.revision_kind,
            "revision_of": self.revision_of,
            "source_url": self.source_url,
            "source_document_sha256": self.source_document_sha256,
            "retrieved_at": isoformat_utc(self.retrieved_at),
            "payload": dict(self.payload),
            "implementation_status": self.implementation_status,
            "scoreability": self.scoreability,
        }


def build_event(
    *,
    event_type: str,
    source_id: str,
    entity_id: str,
    cik: str | None,
    effective_at: datetime,
    availability_at: datetime,
    availability_basis: str,
    effective_session: date,
    revision_id: str,
    revision_kind: str,
    revision_of: str | None,
    artifact: RawArtifact,
    payload: Mapping[str, Any],
    identity_parts: Sequence[Any],
) -> LedgerEvent:
    policy = SOURCE_POLICIES[source_id]
    if artifact.source_id != source_id:
        raise RealMarketContractError("artifact source does not match event source")
    return LedgerEvent(
        schema_version=SCHEMA_VERSION,
        event_id=make_event_id(source_id, event_type, *identity_parts, artifact.sha256),
        event_type=event_type,
        source_id=source_id,
        source_license_id=policy.license_id,
        source_terms_url=policy.terms_url,
        entity_id=entity_id,
        security_id=None,
        cik=cik,
        effective_at=effective_at,
        availability_at=availability_at,
        availability_basis=availability_basis,
        effective_session=effective_session,
        revision_id=revision_id,
        revision_kind=revision_kind,
        revision_of=revision_of,
        source_url=artifact.final_url,
        source_document_sha256=artifact.sha256,
        retrieved_at=artifact.retrieved_at,
        payload=payload,
    )


def local_datetime(local_date: date, local_time: time) -> datetime:
    return datetime.combine(local_date, local_time, tzinfo=NEW_YORK).astimezone(timezone.utc)
