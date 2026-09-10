"""Treasury curve and New York Fed reference-rate PIT adapters."""

from __future__ import annotations

import csv
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
import io
import re
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlencode, urlsplit
import xml.etree.ElementTree as ET

from .contracts import (
    NYFED_SOURCE_ID,
    TREASURY_SOURCE_ID,
    ExplicitSessionCalendar,
    LedgerEvent,
    RawArtifact,
    RealMarketContractError,
    build_event,
    local_datetime,
    parse_iso_date,
    strict_json_loads,
)


TREASURY_TENORS = (
    "1 Mo",
    "1.5 Month",
    "2 Mo",
    "3 Mo",
    "4 Mo",
    "6 Mo",
    "1 Yr",
    "2 Yr",
    "3 Yr",
    "5 Yr",
    "7 Yr",
    "10 Yr",
    "20 Yr",
    "30 Yr",
)
_TREASURY_XML_FIELDS = {
    "BC_1MONTH": "1 Mo",
    "BC_1_5MONTH": "1.5 Month",
    "BC_2MONTH": "2 Mo",
    "BC_3MONTH": "3 Mo",
    "BC_4MONTH": "4 Mo",
    "BC_6MONTH": "6 Mo",
    "BC_1YEAR": "1 Yr",
    "BC_2YEAR": "2 Yr",
    "BC_3YEAR": "3 Yr",
    "BC_5YEAR": "5 Yr",
    "BC_7YEAR": "7 Yr",
    "BC_10YEAR": "10 Yr",
    "BC_20YEAR": "20 Yr",
    "BC_30YEAR": "30 Yr",
    "BC_30YEARDISPLAY": "30 Yr",
}


def treasury_csv_url(year: int) -> str:
    _validate_year(year)
    return (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        f"daily-treasury-rates.csv/{year}/all?_format=csv&"
        f"field_tdr_date_value={year}&type=daily_treasury_yield_curve"
    )


def treasury_xml_url(year: int) -> str:
    _validate_year(year)
    return (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        "pages/xml?data=daily_treasury_yield_curve&"
        f"field_tdr_date_value={year}"
    )


def _validate_year(year: int) -> None:
    if not isinstance(year, int) or isinstance(year, bool) or year < 1990 or year > 2100:
        raise RealMarketContractError("Treasury annual year must be an integer in [1990, 2100]")


def _official_https_parts(url: str, *, expected_host: str) -> tuple[str, Mapping[str, list[str]]]:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != expected_host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.fragment
    ):
        raise RealMarketContractError("URL is not an approved official HTTPS endpoint")
    try:
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise RealMarketContractError("official endpoint query is malformed") from exc
    return parsed.path, query


def validate_treasury_csv_url(url: str) -> None:
    path, query = _official_https_parts(url, expected_host="home.treasury.gov")
    match = re.fullmatch(
        r"/resource-center/data-chart-center/interest-rates/"
        r"daily-treasury-rates\.csv/([0-9]{4})/all",
        path,
    )
    if match is None:
        raise RealMarketContractError("URL is not the official Treasury annual CSV endpoint")
    year = int(match.group(1))
    expected = {
        "_format": ["csv"],
        "field_tdr_date_value": [str(year)],
        "type": ["daily_treasury_yield_curve"],
    }
    if query != expected:
        raise RealMarketContractError("Treasury annual CSV query contract mismatch")


def validate_treasury_xml_url(url: str) -> None:
    path, query = _official_https_parts(url, expected_host="home.treasury.gov")
    if path != "/resource-center/data-chart-center/interest-rates/pages/xml":
        raise RealMarketContractError("URL is not the official Treasury XML endpoint")
    if set(query) != {"data", "field_tdr_date_value"}:
        raise RealMarketContractError("Treasury XML query contract mismatch")
    if query["data"] != ["daily_treasury_yield_curve"]:
        raise RealMarketContractError("Treasury XML data selector is invalid")
    years = query["field_tdr_date_value"]
    if len(years) != 1 or not years[0].isdigit():
        raise RealMarketContractError("Treasury XML year selector is invalid")
    _validate_year(int(years[0]))


def _year_from_treasury_url(url: str) -> int:
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    if "daily-treasury-rates.csv" in parsed.path:
        return int(parsed.path.split("/")[-2])
    return int(query["field_tdr_date_value"][0])


def _decimal_rate(value: Any, *, field_name: str) -> str:
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError) as exc:
        raise RealMarketContractError(f"{field_name} is not a decimal rate") from exc
    if not parsed.is_finite() or parsed < Decimal("-25") or parsed > Decimal("100"):
        raise RealMarketContractError(f"{field_name} is outside the safe percent range")
    return str(parsed)


def _treasury_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact: RawArtifact,
    calendar: ExplicitSessionCalendar,
    expected_year: int,
    format_name: str,
) -> tuple[LedgerEvent, ...]:
    output: list[LedgerEvent] = []
    seen_dates: set[date] = set()
    for index, row in enumerate(rows):
        raw_date = row.get("Date")
        if not isinstance(raw_date, str):
            raise RealMarketContractError(f"Treasury row {index} Date is missing")
        try:
            observation_date = datetime.strptime(raw_date[:10], "%m/%d/%Y").date()
        except ValueError:
            try:
                observation_date = date.fromisoformat(raw_date[:10])
            except ValueError as exc:
                raise RealMarketContractError(
                    f"Treasury row {index} Date is malformed"
                ) from exc
        if observation_date.year != expected_year:
            raise RealMarketContractError("Treasury row year differs from endpoint year")
        if observation_date in seen_dates:
            raise RealMarketContractError("Treasury response has duplicate observation dates")
        seen_dates.add(observation_date)
        rates: dict[str, str] = {}
        for tenor in TREASURY_TENORS:
            raw_value = row.get(tenor)
            if raw_value is None or str(raw_value).strip() == "":
                continue
            rates[tenor] = _decimal_rate(
                raw_value, field_name=f"Treasury {observation_date} {tenor}"
            )
        if not rates:
            raise RealMarketContractError("Treasury row contains no recognized tenor values")
        effective_at = local_datetime(observation_date, time(15, 30))
        availability_at = local_datetime(observation_date, time(18, 0))
        effective_session = calendar.next_session_after(observation_date)
        output.append(
            build_event(
                event_type="TREASURY_PAR_YIELD_CURVE_SNAPSHOT",
                source_id=TREASURY_SOURCE_ID,
                entity_id="US_TREASURY:PAR_YIELD_CURVE",
                cik=None,
                effective_at=effective_at,
                availability_at=availability_at,
                availability_basis=(
                    "TREASURY_METHOD_QUOTE_NEAR_15:30_ET; CONSERVATIVE_PUBLICATION_18:00_ET; "
                    "T_MINUS_1_AT_16:15_ET_DECISION_CUTOFF"
                ),
                effective_session=effective_session,
                revision_id=(
                    f"{observation_date.isoformat()}:{format_name}:{artifact.sha256[:16]}"
                ),
                revision_kind="SNAPSHOT",
                revision_of=None,
                artifact=artifact,
                payload={
                    "observation_date": observation_date,
                    "curve_type": "daily_treasury_par_yield_curve",
                    "rates_percent": rates,
                    "source_format": format_name,
                    "publication_timestamp_observed": False,
                },
                identity_parts=(observation_date, format_name),
            )
        )
    if not output:
        raise RealMarketContractError("Treasury response contains no curve rows")
    return tuple(output)


def parse_treasury_csv(
    artifact: RawArtifact,
    *,
    calendar: ExplicitSessionCalendar,
) -> tuple[LedgerEvent, ...]:
    if artifact.source_id != TREASURY_SOURCE_ID:
        raise RealMarketContractError("Treasury CSV artifact has the wrong source_id")
    validate_treasury_csv_url(artifact.requested_url)
    validate_treasury_csv_url(artifact.final_url)
    if artifact.requested_url != artifact.final_url:
        raise RealMarketContractError("Treasury CSV redirect is not accepted")
    try:
        text = artifact.body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RealMarketContractError("Treasury CSV is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise RealMarketContractError("Treasury CSV header is missing or duplicated")
    allowed_fields = {"Date", *TREASURY_TENORS}
    unknown = set(reader.fieldnames).difference(allowed_fields)
    if "Date" not in reader.fieldnames or unknown:
        raise RealMarketContractError(
            f"Treasury CSV schema changed; unknown fields={sorted(unknown)}"
        )
    rows = list(reader)
    if any(None in row for row in rows):
        raise RealMarketContractError("Treasury CSV row has more fields than its header")
    return _treasury_events(
        rows,
        artifact=artifact,
        calendar=calendar,
        expected_year=_year_from_treasury_url(artifact.final_url),
        format_name="CSV",
    )


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_treasury_xml(
    artifact: RawArtifact,
    *,
    calendar: ExplicitSessionCalendar,
) -> tuple[LedgerEvent, ...]:
    if artifact.source_id != TREASURY_SOURCE_ID:
        raise RealMarketContractError("Treasury XML artifact has the wrong source_id")
    validate_treasury_xml_url(artifact.requested_url)
    validate_treasury_xml_url(artifact.final_url)
    if artifact.requested_url != artifact.final_url:
        raise RealMarketContractError("Treasury XML redirect is not accepted")
    upper_prefix = artifact.body[:4096].upper()
    if b"<!DOCTYPE" in upper_prefix or b"<!ENTITY" in upper_prefix:
        raise RealMarketContractError("Treasury XML DTD/entity declarations are forbidden")
    try:
        root = ET.fromstring(artifact.body)
    except ET.ParseError as exc:
        raise RealMarketContractError("Treasury XML is malformed") from exc
    rows: list[dict[str, Any]] = []
    for entry in root.iter():
        if _local_name(entry.tag) != "entry":
            continue
        properties = next(
            (item for item in entry.iter() if _local_name(item.tag) == "properties"),
            None,
        )
        if properties is None:
            raise RealMarketContractError("Treasury XML entry lacks properties")
        values: dict[str, str] = {}
        for child in list(properties):
            name = _local_name(child.tag)
            if name in values:
                raise RealMarketContractError("Treasury XML property is duplicated")
            values[name] = (child.text or "").strip()
        if "NEW_DATE" not in values:
            raise RealMarketContractError("Treasury XML entry lacks NEW_DATE")
        row: dict[str, Any] = {"Date": values["NEW_DATE"][:10]}
        for xml_name, tenor in _TREASURY_XML_FIELDS.items():
            value = values.get(xml_name)
            if value:
                if tenor in row and xml_name == "BC_30YEARDISPLAY":
                    continue
                if tenor in row:
                    raise RealMarketContractError("Treasury XML has duplicate tenor aliases")
                row[tenor] = value
        rows.append(row)
    return _treasury_events(
        rows,
        artifact=artifact,
        calendar=calendar,
        expected_year=_year_from_treasury_url(artifact.final_url),
        format_name="XML",
    )


NYFED_SERIES = frozenset({"EFFR", "SOFR"})
_NYFED_ALLOWED_FIELDS = frozenset(
    {
        "effectiveDate",
        "type",
        "percentRate",
        "percentPercentile1",
        "percentPercentile25",
        "percentPercentile75",
        "percentPercentile99",
        "volumeInBillions",
        "targetRateFrom",
        "targetRateTo",
        "revisionIndicator",
    }
)
_NYFED_NUMERIC_FIELDS = _NYFED_ALLOWED_FIELDS.difference(
    {"effectiveDate", "type", "revisionIndicator"}
)


def nyfed_search_url(series: str, start_date: date, end_date: date) -> str:
    normalized = str(series).upper()
    if normalized not in NYFED_SERIES:
        raise RealMarketContractError("NY Fed pilot series must be EFFR or SOFR")
    if not isinstance(start_date, date) or not isinstance(end_date, date) or start_date > end_date:
        raise RealMarketContractError("NY Fed search date range is invalid")
    family = "unsecured" if normalized == "EFFR" else "secured"
    query = urlencode(
        {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "type": "rate",
        }
    )
    return (
        f"https://markets.newyorkfed.org/api/rates/{family}/{normalized.lower()}/"
        f"search.json?{query}"
    )


def validate_nyfed_search_url(url: str) -> None:
    path, query = _official_https_parts(url, expected_host="markets.newyorkfed.org")
    match_path = {
        "/api/rates/unsecured/effr/search.json": "EFFR",
        "/api/rates/secured/sofr/search.json": "SOFR",
    }
    if path not in match_path:
        raise RealMarketContractError("URL is not an approved NY Fed EFFR/SOFR search endpoint")
    if set(query) != {"startDate", "endDate", "type"} or query["type"] != ["rate"]:
        raise RealMarketContractError("NY Fed search query contract mismatch")
    try:
        start = date.fromisoformat(query["startDate"][0])
        end = date.fromisoformat(query["endDate"][0])
    except (KeyError, IndexError, ValueError) as exc:
        raise RealMarketContractError("NY Fed search dates are invalid") from exc
    if len(query["startDate"]) != 1 or len(query["endDate"]) != 1 or start > end:
        raise RealMarketContractError("NY Fed search date cardinality/range is invalid")


def _nyfed_query_contract(url: str) -> tuple[str, date, date]:
    validate_nyfed_search_url(url)
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    series = "EFFR" if "/effr/" in parsed.path else "SOFR"
    return (
        series,
        date.fromisoformat(query["startDate"][0]),
        date.fromisoformat(query["endDate"][0]),
    )


def _optional_nyfed_number(
    value: Any,
    *,
    field_name: str,
    is_volume: bool,
) -> str | None:
    if value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, AttributeError) as exc:
        raise RealMarketContractError(f"{field_name} is not decimal") from exc
    minimum = Decimal("0") if is_volume else Decimal("-25")
    maximum = Decimal("1000000") if is_volume else Decimal("100")
    if not parsed.is_finite() or parsed < minimum or parsed > maximum:
        raise RealMarketContractError(f"{field_name} is outside the safe numeric range")
    return str(parsed)


def parse_nyfed_reference_rates(
    artifact: RawArtifact,
    *,
    calendar: ExplicitSessionCalendar,
) -> tuple[LedgerEvent, ...]:
    if artifact.source_id != NYFED_SOURCE_ID:
        raise RealMarketContractError("NY Fed artifact has the wrong source_id")
    validate_nyfed_search_url(artifact.requested_url)
    validate_nyfed_search_url(artifact.final_url)
    if artifact.requested_url != artifact.final_url:
        raise RealMarketContractError("NY Fed redirect is not accepted")
    requested_series, start_date, end_date = _nyfed_query_contract(artifact.final_url)
    root = strict_json_loads(artifact.body)
    if not isinstance(root, Mapping) or set(root) != {"refRates"}:
        raise RealMarketContractError("NY Fed response root schema changed")
    observations = root["refRates"]
    if not isinstance(observations, list) or not observations:
        raise RealMarketContractError("NY Fed response contains no refRates")
    output: list[LedgerEvent] = []
    seen_dates: set[date] = set()
    for index, raw_observation in enumerate(observations):
        if not isinstance(raw_observation, Mapping):
            raise RealMarketContractError(f"NY Fed refRates[{index}] must be an object")
        unknown = set(raw_observation).difference(_NYFED_ALLOWED_FIELDS)
        missing = {"effectiveDate", "type", "percentRate"}.difference(raw_observation)
        if unknown or missing:
            raise RealMarketContractError(
                f"NY Fed rate schema changed; unknown={sorted(unknown)}, missing={sorted(missing)}"
            )
        observation_date = parse_iso_date(
            raw_observation["effectiveDate"], field_name="NY Fed effectiveDate"
        )
        assert observation_date is not None
        if not start_date <= observation_date <= end_date:
            raise RealMarketContractError("NY Fed observation lies outside requested range")
        if observation_date in seen_dates:
            raise RealMarketContractError("NY Fed response has duplicate effectiveDate")
        seen_dates.add(observation_date)
        observed_series = str(raw_observation["type"]).upper()
        if observed_series != requested_series:
            raise RealMarketContractError("NY Fed response series differs from endpoint")
        normalized_values: dict[str, str | None] = {}
        for name in sorted(_NYFED_NUMERIC_FIELDS.intersection(raw_observation)):
            normalized_values[name] = _optional_nyfed_number(
                raw_observation[name],
                field_name=f"NY Fed {name}",
                is_volume=name == "volumeInBillions",
            )
        if normalized_values.get("percentRate") is None:
            raise RealMarketContractError("NY Fed percentRate must be present")
        revision_indicator = str(raw_observation.get("revisionIndicator") or "").strip()
        publication_session = calendar.next_session_after(observation_date)
        first_publication_time = time(9, 0) if requested_series == "EFFR" else time(8, 0)
        availability_at = local_datetime(publication_session, time(14, 30))
        effective_session = calendar.first_observable_session(availability_at)
        revision_kind = (
            "CORRECTION" if revision_indicator not in {"", "0"} else "ORIGINAL"
        )
        output.append(
            build_event(
                event_type=f"NYFED_{requested_series}_PUBLISHED",
                source_id=NYFED_SOURCE_ID,
                entity_id=f"NYFED_REFERENCE_RATE:{requested_series}",
                cik=None,
                effective_at=local_datetime(observation_date, time.min),
                availability_at=availability_at,
                availability_basis=(
                    f"OFFICIAL_{requested_series}_PUBLICATION_AROUND_"
                    f"{first_publication_time.strftime('%H:%M')}_ET; "
                    "CONSERVATIVE_POST_CORRECTION_SNAPSHOT_14:30_ET"
                ),
                effective_session=effective_session,
                revision_id=(
                    f"{requested_series}:{observation_date.isoformat()}:"
                    f"{revision_indicator or 'ORIGINAL'}:{artifact.sha256[:16]}"
                ),
                revision_kind=revision_kind,
                revision_of=None,
                artifact=artifact,
                payload={
                    "series": requested_series,
                    "effective_date": observation_date,
                    "reported_values": normalized_values,
                    "revision_indicator": revision_indicator,
                    "first_publication_time_local": first_publication_time.strftime("%H:%M"),
                    "conservative_confirmation_time_local": "14:30",
                    "publication_timestamp_observed": False,
                },
                identity_parts=(requested_series, observation_date, revision_indicator),
            )
        )
    return tuple(output)
