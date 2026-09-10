"""Hash-chained experiment logging without implicit clocks or machine state."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Iterator, Mapping

from .contracts import ContractError


LOGGER_FORMAT_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_DIRECTORY_LOCKS_GUARD = threading.Lock()
_DIRECTORY_LOCKS: dict[str, threading.RLock] = {}


class ExperimentLogError(ContractError):
    """Raised for conflicting or tampered deterministic experiment logs."""


@dataclass(frozen=True)
class ExperimentEvent:
    experiment_id: str
    sequence: int
    event_key: str
    event_type: str
    payload: dict[str, Any]
    previous_entry_sha256: str
    entry_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "sequence": self.sequence,
            "event_key": self.event_key,
            "event_type": self.event_type,
            "payload": self.payload,
            "previous_entry_sha256": self.previous_entry_sha256,
            "entry_sha256": self.entry_sha256,
        }


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentLogError("experiment payload must contain canonical JSON values") from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _genesis(experiment_id: str) -> str:
    return _sha256(f"model-lab-experiment-v1:{experiment_id}".encode("utf-8"))


def _seal_record(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(payload)
    sealed[field] = _sha256(_canonical_json(payload))
    return sealed


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    if temporary.exists():
        raise FileExistsError(f"experiment temporary path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _directory_thread_lock(directory: Path) -> threading.RLock:
    key = str(directory.resolve())
    with _DIRECTORY_LOCKS_GUARD:
        return _DIRECTORY_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize independent logger instances and worker processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class DeterministicExperimentLogger:
    """Append-only logical event log with idempotent event keys.

    The logger deliberately does not add timestamps, hostnames, process IDs,
    or random UUIDs. Identical ordered calls therefore produce identical bytes.
    """

    def __init__(self, directory: Path, *, experiment_id: str) -> None:
        if _SAFE_ID.fullmatch(experiment_id) is None:
            raise ExperimentLogError("experiment_id is not path-safe")
        self.directory = Path(directory)
        self.experiment_id = experiment_id
        self.events_path = self.directory / "events.jsonl"
        self.manifest_path = self.directory / "manifest.json"
        self.lock_path = self.directory / ".single-writer.lock"
        self._lock = threading.Lock()
        self.verify()

    @contextmanager
    def _operation_lock(self) -> Iterator[None]:
        with _directory_thread_lock(self.directory):
            with _exclusive_file_lock(self.lock_path):
                yield

    def _read_events(self) -> tuple[ExperimentEvent, ...]:
        if not self.events_path.exists() and not self.manifest_path.exists():
            return ()
        if not self.events_path.exists():
            raise ExperimentLogError("manifest exists without its source event ledger")
        raw = self.events_path.read_bytes()
        if not raw or not raw.endswith(b"\n") or b"\r" in raw:
            raise ExperimentLogError("event ledger must be non-empty canonical LF-delimited JSONL")
        events: list[ExperimentEvent] = []
        for line_number, raw_line in enumerate(raw[:-1].split(b"\n"), start=1):
            try:
                line = raw_line.decode("utf-8")
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ExperimentLogError(f"invalid JSONL event at line {line_number}") from exc
            required = {
                "experiment_id",
                "sequence",
                "event_key",
                "event_type",
                "payload",
                "previous_entry_sha256",
                "entry_sha256",
            }
            if not isinstance(record, dict) or set(record) != required:
                raise ExperimentLogError(f"event fields are invalid at line {line_number}")
            if not isinstance(record["payload"], dict):
                raise ExperimentLogError("event payload must be an object")
            if _canonical_json(record) != raw_line:
                raise ExperimentLogError(
                    f"event bytes are not canonical JSON at line {line_number}"
                )
            events.append(ExperimentEvent(**record))
        return tuple(events)

    def _manifest_core(
        self,
        events: tuple[ExperimentEvent, ...] | list[ExperimentEvent],
        events_bytes: bytes,
    ) -> dict[str, Any]:
        if not events:
            raise ExperimentLogError("cannot build a manifest for an empty event ledger")
        return {
            "format_version": LOGGER_FORMAT_VERSION,
            "experiment_id": self.experiment_id,
            "event_count": len(events),
            "genesis_sha256": _genesis(self.experiment_id),
            "head_entry_sha256": events[-1].entry_sha256,
            "events_file_sha256": _sha256(events_bytes),
        }

    @staticmethod
    def _events_bytes(events: tuple[ExperimentEvent, ...] | list[ExperimentEvent]) -> bytes:
        return b"".join(_canonical_json(event.to_record()) + b"\n" for event in events)

    def _write_manifest_for(
        self, events: tuple[ExperimentEvent, ...] | list[ExperimentEvent]
    ) -> None:
        events_bytes = self._events_bytes(events)
        manifest = _seal_record(self._manifest_core(events, events_bytes), "manifest_sha256")
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        _atomic_write(self.manifest_path, manifest_bytes)

    def _verify_unlocked(self) -> tuple[ExperimentEvent, ...]:
        events = self._read_events()
        if not events:
            return ()
        previous = _genesis(self.experiment_id)
        keys: set[str] = set()
        for expected_sequence, event in enumerate(events, start=1):
            if event.experiment_id != self.experiment_id:
                raise ExperimentLogError("event experiment_id mismatch")
            if event.sequence != expected_sequence:
                raise ExperimentLogError("event sequences must be contiguous")
            if (
                _SAFE_ID.fullmatch(event.event_key) is None
                or _SAFE_ID.fullmatch(event.event_type) is None
            ):
                raise ExperimentLogError("event key and type must be safe identifiers")
            if event.event_key in keys:
                raise ExperimentLogError("event keys must be unique")
            keys.add(event.event_key)
            record = event.to_record()
            recorded_sha256 = record.pop("entry_sha256")
            if event.previous_entry_sha256 != previous:
                raise ExperimentLogError("event hash chain is broken")
            if _sha256(_canonical_json(record)) != recorded_sha256:
                raise ExperimentLogError("event SHA-256 mismatch")
            previous = event.entry_sha256

        current_events_bytes = self._events_bytes(events)
        expected_manifest = self._manifest_core(events, current_events_bytes)
        if not self.manifest_path.exists():
            self._write_manifest_for(events)
            return events

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExperimentLogError("experiment manifest is invalid JSON") from exc
        if not isinstance(manifest, dict):
            raise ExperimentLogError("experiment manifest must be an object")
        logical = dict(manifest)
        recorded_manifest_sha256 = logical.pop("manifest_sha256", None)
        if _sha256(_canonical_json(logical)) != recorded_manifest_sha256:
            raise ExperimentLogError("experiment manifest SHA-256 mismatch")
        if logical == expected_manifest:
            return events

        # ``events.jsonl`` is replaced first.  If a process stops before the
        # derived manifest replace, a valid old manifest must bind an exact
        # prefix of the now-longer canonical event chain.  That sole stale
        # direction is deterministically recoverable. A newer manifest, a
        # same-length mismatch, or invalid event bytes remains fail-closed.
        stale_count = logical.get("event_count")
        if (
            isinstance(stale_count, int)
            and not isinstance(stale_count, bool)
            and 0 < stale_count < len(events)
        ):
            prefix = events[:stale_count]
            prefix_bytes = self._events_bytes(prefix)
            if logical == self._manifest_core(prefix, prefix_bytes):
                self._write_manifest_for(events)
                return events
        raise ExperimentLogError("experiment manifest content mismatch")

    def verify(self) -> tuple[ExperimentEvent, ...]:
        with self._operation_lock():
            return self._verify_unlocked()

    def log(
        self,
        *,
        event_key: str,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> ExperimentEvent:
        if _SAFE_ID.fullmatch(event_key) is None or _SAFE_ID.fullmatch(event_type) is None:
            raise ExperimentLogError("event_key and event_type must be safe identifiers")
        normalized_payload = json.loads(_canonical_json(dict(payload)).decode("utf-8"))
        with self._lock, self._operation_lock():
            events = list(self._verify_unlocked())
            for event in events:
                if event.event_key == event_key:
                    if event.event_type != event_type or _canonical_json(
                        event.payload
                    ) != _canonical_json(normalized_payload):
                        raise ExperimentLogError(f"conflicting event_key: {event_key}")
                    return event
            previous = events[-1].entry_sha256 if events else _genesis(self.experiment_id)
            core = {
                "experiment_id": self.experiment_id,
                "sequence": len(events) + 1,
                "event_key": event_key,
                "event_type": event_type,
                "payload": normalized_payload,
                "previous_entry_sha256": previous,
            }
            record = _seal_record(core, "entry_sha256")
            event = ExperimentEvent(**record)
            events.append(event)
            events_bytes = self._events_bytes(events)
            _atomic_write(self.events_path, events_bytes)
            self._write_manifest_for(events)
            self._verify_unlocked()
            return event
