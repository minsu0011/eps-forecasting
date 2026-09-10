"""Immutable, score-free event-ledger serialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    IMPLEMENTATION_STATUS,
    SCOREABILITY,
    LedgerEvent,
    RawArtifact,
    RealMarketContractError,
    canonical_json_bytes,
)


@dataclass(frozen=True)
class EventLedger:
    events: tuple[LedgerEvent, ...]

    @classmethod
    def from_events(cls, events: Iterable[LedgerEvent]) -> "EventLedger":
        materialized = tuple(events)
        ids = [event.event_id for event in materialized]
        if len(ids) != len(set(ids)):
            raise RealMarketContractError("ledger event_id values must be unique")
        ordered = tuple(
            sorted(
                materialized,
                key=lambda event: (
                    event.availability_at,
                    event.source_id,
                    event.entity_id,
                    event.revision_id,
                    event.event_id,
                ),
            )
        )
        return cls(ordered)

    def records(self) -> list[dict[str, Any]]:
        return [event.to_record() for event in self.events]

    def canonical_jsonl_bytes(self) -> bytes:
        if not self.events:
            raise RealMarketContractError("an empty ledger cannot be sealed")
        return b"".join(canonical_json_bytes(record) + b"\n" for record in self.records())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_jsonl_bytes()).hexdigest()

    def manifest(self, raw_artifacts: Sequence[RawArtifact]) -> dict[str, Any]:
        artifact_records = sorted(
            (artifact.record() for artifact in raw_artifacts),
            key=lambda record: (record["source_id"], record["sha256"], record["final_url"]),
        )
        source_ids = sorted({event.source_id for event in self.events})
        return seal_payload(
            {
                "format_version": 1,
                "mode": "public_real_market_pit_ledger",
                "implementation_status": IMPLEMENTATION_STATUS,
                "scoreability": SCOREABILITY,
                "price_data_present": False,
                "fred_data_present": False,
                "candidate_scores_computed": False,
                "source_ids": source_ids,
                "event_count": len(self.events),
                "ledger_sha256": self.sha256(),
                "raw_artifacts": artifact_records,
            }
        )


def seal_payload(payload: Mapping[str, Any], *, field: str = "manifest_sha256") -> dict[str, Any]:
    output = dict(payload)
    output.pop(field, None)
    output[field] = hashlib.sha256(canonical_json_bytes(output)).hexdigest()
    return output


def verify_payload_seal(payload: Mapping[str, Any], *, field: str = "manifest_sha256") -> None:
    recorded = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if recorded != expected:
        raise RealMarketContractError(f"{field} is missing or invalid")


def write_immutable_bytes(path: Path, content: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"immutable output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    write_immutable_bytes(path, content)
