"""SEC EDGAR submissions, companyfacts, and accession-index PIT adapters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .contracts import (
    NEW_YORK,
    SEC_SOURCE_ID,
    ExplicitSessionCalendar,
    LedgerEvent,
    RawArtifact,
    RealMarketContractError,
    build_event,
    parse_iso_date,
    parse_iso_datetime,
    strict_json_loads,
)


_ACCESSION = re.compile(r"^[0-9]{10}-[0-9]{2}-[0-9]{6}$")
_HISTORY_NAME = re.compile(r"^CIK[0-9]{10}-submissions-[0-9]{3}\.json$")
_SAFE_ARCHIVE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def normalize_cik(value: Any) -> str:
    text = str(value).strip()
    if not text.isdigit() or len(text) > 10 or int(text) <= 0:
        raise RealMarketContractError("SEC CIK must be one to ten positive digits")
    return text.zfill(10)


def sec_submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{normalize_cik(cik)}.json"


def sec_companyfacts_url(cik: str) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalize_cik(cik)}.json"


def sec_accession_index_url(cik: str, accession_number: str) -> str:
    normalized_cik = normalize_cik(cik)
    if _ACCESSION.fullmatch(accession_number) is None:
        raise RealMarketContractError("SEC accession number format is invalid")
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(normalized_cik)}/{accession_number.replace('-', '')}/index.json"
    )


def _validate_https_url(url: str) -> tuple[str, str, str]:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
    ):
        raise RealMarketContractError("SEC endpoint must be plain HTTPS without credentials")
    return (parsed.hostname or "").lower(), parsed.path, parsed.query


def validate_sec_submissions_url(url: str) -> None:
    host, path, query = _validate_https_url(url)
    if host != "data.sec.gov" or query or re.fullmatch(
        r"/submissions/CIK[0-9]{10}(?:-submissions-[0-9]{3})?\.json", path
    ) is None:
        raise RealMarketContractError("URL is not an official SEC submissions endpoint")


def validate_sec_companyfacts_url(url: str) -> None:
    host, path, query = _validate_https_url(url)
    if host != "data.sec.gov" or query or re.fullmatch(
        r"/api/xbrl/companyfacts/CIK[0-9]{10}\.json", path
    ) is None:
        raise RealMarketContractError("URL is not an official SEC companyfacts endpoint")


def validate_sec_accession_index_url(url: str) -> None:
    host, path, query = _validate_https_url(url)
    if host != "www.sec.gov" or query or re.fullmatch(
        r"/Archives/edgar/data/[1-9][0-9]*/[0-9]{18}/index\.json", path
    ) is None:
        raise RealMarketContractError("URL is not an official SEC accession index endpoint")


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RealMarketContractError(f"{field_name} must be an object")
    return value


def _list(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise RealMarketContractError(f"{field_name} must be an array")
    return value


def _nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealMarketContractError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class SubmissionRecord:
    cik: str
    entity_id: str
    accession_number: str
    filing_date: date
    report_date: date | None
    acceptance_at: datetime
    effective_session: date
    form: str
    primary_document: str
    is_xbrl: bool | None
    is_inline_xbrl: bool | None
    artifact: RawArtifact

    @property
    def revision_kind(self) -> str:
        return "AMENDMENT" if self.form.upper().endswith("/A") else "ORIGINAL"


@dataclass(frozen=True)
class SubmissionBundle:
    cik: str
    entity_id: str
    entity_name: str
    tickers: tuple[str, ...]
    exchanges: tuple[str, ...]
    submissions: tuple[SubmissionRecord, ...]
    events: tuple[LedgerEvent, ...]
    artifacts: tuple[RawArtifact, ...]
    history_urls: tuple[str, ...]

    def by_accession(self) -> Mapping[str, SubmissionRecord]:
        return MappingProxyType(
            {record.accession_number: record for record in self.submissions}
        )


def _parse_main_submissions(artifact: RawArtifact) -> tuple[Mapping[str, Any], str]:
    if artifact.source_id != SEC_SOURCE_ID:
        raise RealMarketContractError("SEC submissions artifact has the wrong source_id")
    validate_sec_submissions_url(artifact.requested_url)
    validate_sec_submissions_url(artifact.final_url)
    if artifact.requested_url != artifact.final_url:
        raise RealMarketContractError("SEC submissions redirect is not accepted")
    payload = _mapping(strict_json_loads(artifact.body), field_name="SEC submissions root")
    cik = normalize_cik(payload.get("cik"))
    if artifact.final_url != sec_submissions_url(cik):
        raise RealMarketContractError("SEC submissions URL CIK differs from response CIK")
    return payload, cik


def discover_submission_history_urls(artifact: RawArtifact) -> tuple[str, ...]:
    payload, _ = _parse_main_submissions(artifact)
    filings = _mapping(payload.get("filings"), field_name="filings")
    files = _list(filings.get("files"), field_name="filings.files")
    urls: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(files):
        record = _mapping(item, field_name=f"filings.files[{index}]")
        name = _nonempty_string(record.get("name"), field_name=f"filings.files[{index}].name")
        if _HISTORY_NAME.fullmatch(name) is None:
            raise RealMarketContractError("SEC supplemental submission filename is unsafe")
        url = f"https://data.sec.gov/submissions/{name}"
        if url in seen:
            raise RealMarketContractError("duplicate SEC supplemental submission file")
        seen.add(url)
        urls.append(url)
    return tuple(urls)


_SUBMISSION_REQUIRED_ARRAYS = (
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "primaryDocument",
)
_SUBMISSION_OPTIONAL_ARRAYS = ("isXBRL", "isInlineXBRL")


def _submission_rows(
    payload: Mapping[str, Any],
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    arrays: dict[str, list[Any]] = {}
    for name in _SUBMISSION_REQUIRED_ARRAYS:
        arrays[name] = _list(payload.get(name), field_name=f"{field_name}.{name}")
    row_count = len(arrays["accessionNumber"])
    if row_count == 0:
        return []
    if any(len(values) != row_count for values in arrays.values()):
        raise RealMarketContractError(f"{field_name} arrays have inconsistent lengths")
    for name in _SUBMISSION_OPTIONAL_ARRAYS:
        if name in payload:
            values = _list(payload[name], field_name=f"{field_name}.{name}")
            if len(values) != row_count:
                raise RealMarketContractError(f"{field_name}.{name} length mismatch")
            arrays[name] = values
        else:
            arrays[name] = [None] * row_count
    return [{name: values[index] for name, values in arrays.items()} for index in range(row_count)]


def _optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    raise RealMarketContractError(f"{field_name} must be boolean, 0/1, or null")


def parse_sec_submissions(
    main_artifact: RawArtifact,
    *,
    calendar: ExplicitSessionCalendar,
    history_artifacts: Sequence[RawArtifact] = (),
    require_complete_history: bool = True,
) -> SubmissionBundle:
    main_payload, cik = _parse_main_submissions(main_artifact)
    entity_name = _nonempty_string(main_payload.get("name"), field_name="name")
    entity_id = f"SEC_CIK:{cik}"
    tickers_raw = _list(main_payload.get("tickers", []), field_name="tickers")
    exchanges_raw = _list(main_payload.get("exchanges", []), field_name="exchanges")
    tickers = tuple(_nonempty_string(item, field_name="ticker") for item in tickers_raw)
    exchanges = tuple(_nonempty_string(item, field_name="exchange") for item in exchanges_raw)
    if len(tickers) != len(exchanges):
        raise RealMarketContractError("SEC tickers and exchanges arrays must have equal length")

    expected_history_urls = discover_submission_history_urls(main_artifact)
    supplied_history: dict[str, RawArtifact] = {}
    for artifact in history_artifacts:
        if artifact.source_id != SEC_SOURCE_ID:
            raise RealMarketContractError("SEC history artifact has the wrong source_id")
        validate_sec_submissions_url(artifact.requested_url)
        validate_sec_submissions_url(artifact.final_url)
        if artifact.requested_url != artifact.final_url:
            raise RealMarketContractError("SEC history redirect is not accepted")
        if artifact.final_url in supplied_history:
            raise RealMarketContractError("duplicate SEC history artifact URL")
        supplied_history[artifact.final_url] = artifact
    extra = set(supplied_history).difference(expected_history_urls)
    if extra:
        raise RealMarketContractError(f"unreferenced SEC history artifact: {sorted(extra)[0]}")
    missing = set(expected_history_urls).difference(supplied_history)
    if require_complete_history and missing:
        raise RealMarketContractError(
            f"SEC submission history is incomplete; missing {len(missing)} supplemental files"
        )

    filings = _mapping(main_payload.get("filings"), field_name="filings")
    recent = _mapping(filings.get("recent"), field_name="filings.recent")
    row_sources: list[tuple[dict[str, Any], RawArtifact]] = [
        (row, main_artifact) for row in _submission_rows(recent, field_name="filings.recent")
    ]
    for url in expected_history_urls:
        artifact = supplied_history.get(url)
        if artifact is None:
            continue
        history_payload = _mapping(
            strict_json_loads(artifact.body), field_name=f"history {url}"
        )
        row_sources.extend(
            (row, artifact)
            for row in _submission_rows(history_payload, field_name=f"history {url}")
        )

    records: list[SubmissionRecord] = []
    events: list[LedgerEvent] = []
    seen_accessions: set[str] = set()
    for row_index, (row, artifact) in enumerate(row_sources):
        accession = _nonempty_string(
            row["accessionNumber"], field_name=f"submission[{row_index}].accessionNumber"
        )
        if _ACCESSION.fullmatch(accession) is None:
            raise RealMarketContractError("SEC accession number format is invalid")
        if accession in seen_accessions:
            raise RealMarketContractError(f"duplicate SEC accession: {accession}")
        seen_accessions.add(accession)
        if accession[:10] != cik:
            raise RealMarketContractError("SEC accession prefix differs from issuer CIK")
        filing_date = parse_iso_date(
            row["filingDate"], field_name=f"submission[{row_index}].filingDate"
        )
        report_date = parse_iso_date(
            row["reportDate"],
            field_name=f"submission[{row_index}].reportDate",
            allow_empty=True,
        )
        assert filing_date is not None
        acceptance_at = parse_iso_datetime(
            row["acceptanceDateTime"],
            field_name=f"submission[{row_index}].acceptanceDateTime",
        )
        if acceptance_at.date() < filing_date:
            raise RealMarketContractError("SEC acceptance datetime precedes filing date")
        form = _nonempty_string(row["form"], field_name=f"submission[{row_index}].form")
        primary_document = str(row["primaryDocument"] or "").strip()
        if primary_document and _SAFE_ARCHIVE_NAME.fullmatch(primary_document) is None:
            raise RealMarketContractError("SEC primaryDocument is unsafe")
        local_acceptance_date = acceptance_at.astimezone(NEW_YORK).date()
        effective_session = calendar.next_session_after(local_acceptance_date)
        record = SubmissionRecord(
            cik=cik,
            entity_id=entity_id,
            accession_number=accession,
            filing_date=filing_date,
            report_date=report_date,
            acceptance_at=acceptance_at,
            effective_session=effective_session,
            form=form,
            primary_document=primary_document,
            is_xbrl=_optional_bool(
                row["isXBRL"], field_name=f"submission[{row_index}].isXBRL"
            ),
            is_inline_xbrl=_optional_bool(
                row["isInlineXBRL"],
                field_name=f"submission[{row_index}].isInlineXBRL",
            ),
            artifact=artifact,
        )
        records.append(record)
        valid_date = report_date or filing_date
        events.append(
            build_event(
                event_type="SEC_FILING_ACCEPTED",
                source_id=SEC_SOURCE_ID,
                entity_id=entity_id,
                cik=cik,
                effective_at=datetime.combine(valid_date, time.min, tzinfo=timezone.utc),
                availability_at=acceptance_at,
                availability_basis=(
                    "SEC_SUBMISSIONS_ACCEPTANCE_DATETIME; CONSERVATIVE_NEXT_XNYS_SESSION"
                ),
                effective_session=effective_session,
                revision_id=accession,
                revision_kind=record.revision_kind,
                revision_of=None,
                artifact=artifact,
                payload={
                    "accession_number": accession,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "form": form,
                    "primary_document": primary_document,
                    "is_xbrl": record.is_xbrl,
                    "is_inline_xbrl": record.is_inline_xbrl,
                    "entity_name": entity_name,
                    "tickers": tickers,
                    "exchanges": exchanges,
                },
                identity_parts=(cik, accession, "submission"),
            )
        )
    if not records:
        raise RealMarketContractError("SEC submissions response contains no filing rows")
    artifacts = (main_artifact,) + tuple(
        supplied_history[url] for url in expected_history_urls if url in supplied_history
    )
    return SubmissionBundle(
        cik=cik,
        entity_id=entity_id,
        entity_name=entity_name,
        tickers=tickers,
        exchanges=exchanges,
        submissions=tuple(records),
        events=tuple(events),
        artifacts=artifacts,
        history_urls=expected_history_urls,
    )


def _safe_fact_value(value: Any) -> str | int | bool | None:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise RealMarketContractError("SEC companyfact value must be finite")
        return str(value)
    raise RealMarketContractError("SEC companyfact value must be a JSON scalar")


def parse_sec_companyfacts(
    artifact: RawArtifact,
    *,
    submissions: SubmissionBundle,
) -> tuple[LedgerEvent, ...]:
    if artifact.source_id != SEC_SOURCE_ID:
        raise RealMarketContractError("SEC companyfacts artifact has the wrong source_id")
    validate_sec_companyfacts_url(artifact.requested_url)
    validate_sec_companyfacts_url(artifact.final_url)
    if artifact.requested_url != artifact.final_url:
        raise RealMarketContractError("SEC companyfacts redirect is not accepted")
    if artifact.final_url != sec_companyfacts_url(submissions.cik):
        raise RealMarketContractError("SEC companyfacts URL differs from submissions CIK")
    root = _mapping(strict_json_loads(artifact.body), field_name="companyfacts root")
    cik = normalize_cik(root.get("cik"))
    if cik != submissions.cik:
        raise RealMarketContractError("SEC companyfacts CIK differs from submissions CIK")
    entity_name = _nonempty_string(root.get("entityName"), field_name="entityName")
    if entity_name != submissions.entity_name:
        raise RealMarketContractError("SEC companyfacts entityName differs from submissions")
    facts = _mapping(root.get("facts"), field_name="facts")
    by_accession = submissions.by_accession()
    output: list[LedgerEvent] = []
    seen_fact_keys: set[tuple[Any, ...]] = set()
    for taxonomy in sorted(facts):
        taxonomy_facts = _mapping(facts[taxonomy], field_name=f"facts.{taxonomy}")
        for concept in sorted(taxonomy_facts):
            concept_record = _mapping(
                taxonomy_facts[concept], field_name=f"facts.{taxonomy}.{concept}"
            )
            units = _mapping(
                concept_record.get("units"),
                field_name=f"facts.{taxonomy}.{concept}.units",
            )
            for unit in sorted(units):
                observations = _list(
                    units[unit], field_name=f"facts.{taxonomy}.{concept}.units.{unit}"
                )
                for index, raw_observation in enumerate(observations):
                    observation = _mapping(
                        raw_observation,
                        field_name=f"facts.{taxonomy}.{concept}.{unit}[{index}]",
                    )
                    accession = _nonempty_string(
                        observation.get("accn"), field_name="companyfact.accn"
                    )
                    submission = by_accession.get(accession)
                    if submission is None:
                        raise RealMarketContractError(
                            "companyfact accession lacks complete submissions acceptance metadata"
                        )
                    fact_form = _nonempty_string(
                        observation.get("form"), field_name="companyfact.form"
                    )
                    if fact_form != submission.form:
                        raise RealMarketContractError(
                            "companyfact form differs from accession metadata"
                        )
                    fact_filed = parse_iso_date(
                        observation.get("filed"), field_name="companyfact.filed"
                    )
                    assert fact_filed is not None
                    if fact_filed != submission.filing_date:
                        raise RealMarketContractError(
                            "companyfact filed date differs from accession metadata"
                        )
                    period_start = parse_iso_date(
                        observation.get("start"),
                        field_name="companyfact.start",
                        allow_empty=True,
                    )
                    period_end = parse_iso_date(
                        observation.get("end"), field_name="companyfact.end"
                    )
                    assert period_end is not None
                    if period_start is not None and period_start > period_end:
                        raise RealMarketContractError("companyfact period starts after it ends")
                    frame = observation.get("frame")
                    if frame is not None and not isinstance(frame, str):
                        raise RealMarketContractError("companyfact frame must be a string or null")
                    fact_key = (
                        accession,
                        taxonomy,
                        concept,
                        unit,
                        period_start,
                        period_end,
                        frame,
                    )
                    if fact_key in seen_fact_keys:
                        raise RealMarketContractError("duplicate companyfact identity")
                    seen_fact_keys.add(fact_key)
                    value = _safe_fact_value(observation.get("val"))
                    revision_id = ":".join(
                        (
                            accession,
                            taxonomy,
                            concept,
                            unit,
                            period_start.isoformat() if period_start else "instant",
                            period_end.isoformat(),
                            frame or "no-frame",
                        )
                    )
                    output.append(
                        build_event(
                            event_type="SEC_XBRL_FACT_ACCEPTED",
                            source_id=SEC_SOURCE_ID,
                            entity_id=submissions.entity_id,
                            cik=cik,
                            effective_at=datetime.combine(
                                period_end, time.min, tzinfo=timezone.utc
                            ),
                            availability_at=submission.acceptance_at,
                            availability_basis=(
                                "FACT_ACCN_JOIN_TO_SEC_SUBMISSIONS_ACCEPTANCE_DATETIME; "
                                "CONSERVATIVE_NEXT_XNYS_SESSION"
                            ),
                            effective_session=submission.effective_session,
                            revision_id=revision_id,
                            revision_kind=submission.revision_kind,
                            revision_of=None,
                            artifact=artifact,
                            payload={
                                "accession_number": accession,
                                "taxonomy": taxonomy,
                                "concept": concept,
                                "unit": unit,
                                "period_start": period_start,
                                "period_end": period_end,
                                "value": value,
                                "form": fact_form,
                                "filed_date": fact_filed,
                                "fiscal_year": observation.get("fy"),
                                "fiscal_period": observation.get("fp"),
                                "frame": frame,
                            },
                            identity_parts=fact_key,
                        )
                    )
    if not output:
        raise RealMarketContractError("SEC companyfacts response contains no fact rows")
    return tuple(output)


def parse_sec_accession_metadata(
    artifact: RawArtifact,
    *,
    submissions: SubmissionBundle,
    accession_number: str,
) -> LedgerEvent:
    if artifact.source_id != SEC_SOURCE_ID:
        raise RealMarketContractError("SEC accession artifact has the wrong source_id")
    if _ACCESSION.fullmatch(accession_number) is None:
        raise RealMarketContractError("SEC accession number format is invalid")
    validate_sec_accession_index_url(artifact.requested_url)
    validate_sec_accession_index_url(artifact.final_url)
    if artifact.requested_url != artifact.final_url:
        raise RealMarketContractError("SEC accession index redirect is not accepted")
    expected_url = sec_accession_index_url(submissions.cik, accession_number)
    if artifact.final_url != expected_url:
        raise RealMarketContractError("SEC accession index URL differs from requested accession")
    submission = submissions.by_accession().get(accession_number)
    if submission is None:
        raise RealMarketContractError("accession index has no submissions acceptance metadata")
    root = _mapping(strict_json_loads(artifact.body), field_name="accession index root")
    directory = _mapping(root.get("directory"), field_name="directory")
    directory_name = _nonempty_string(directory.get("name"), field_name="directory.name")
    expected_suffix = (
        f"/Archives/edgar/data/{int(submissions.cik)}/{accession_number.replace('-', '')}"
    )
    if directory_name.rstrip("/") != expected_suffix:
        raise RealMarketContractError("SEC accession directory identity mismatch")
    raw_items = _list(directory.get("item"), field_name="directory.item")
    files: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        item = _mapping(raw_item, field_name=f"directory.item[{index}]")
        name = _nonempty_string(item.get("name"), field_name=f"directory.item[{index}].name")
        if _SAFE_ARCHIVE_NAME.fullmatch(name) is None or name in {".", ".."}:
            raise RealMarketContractError("SEC accession item filename is unsafe")
        if name in seen_names:
            raise RealMarketContractError("SEC accession index contains duplicate filenames")
        seen_names.add(name)
        raw_size = item.get("size")
        try:
            size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise RealMarketContractError("SEC accession item size is invalid") from exc
        if size < 0:
            raise RealMarketContractError("SEC accession item size is negative")
        files.append(
            {
                "name": name,
                "type": str(item.get("type") or ""),
                "size": size,
                "last_modified": str(item.get("last-modified") or ""),
            }
        )
    if not files:
        raise RealMarketContractError("SEC accession index contains no files")
    if submission.primary_document and submission.primary_document not in seen_names:
        raise RealMarketContractError("SEC primary document is absent from accession index")
    valid_date = submission.report_date or submission.filing_date
    return build_event(
        event_type="SEC_ACCESSION_INDEX_SNAPSHOT",
        source_id=SEC_SOURCE_ID,
        entity_id=submissions.entity_id,
        cik=submissions.cik,
        effective_at=datetime.combine(valid_date, time.min, tzinfo=timezone.utc),
        availability_at=submission.acceptance_at,
        availability_basis=(
            "ACCESSION_INDEX_JOIN_TO_SEC_SUBMISSIONS_ACCEPTANCE_DATETIME; "
            "CONSERVATIVE_NEXT_XNYS_SESSION"
        ),
        effective_session=submission.effective_session,
        revision_id=f"{accession_number}:accession-index:{artifact.sha256[:16]}",
        revision_kind=submission.revision_kind,
        revision_of=None,
        artifact=artifact,
        payload={
            "accession_number": accession_number,
            "form": submission.form,
            "primary_document": submission.primary_document,
            "directory_name": directory_name,
            "files": files,
        },
        identity_parts=(submissions.cik, accession_number, "accession-index"),
    )
