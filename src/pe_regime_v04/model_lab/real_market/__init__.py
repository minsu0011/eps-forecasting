"""Score-free public point-in-time source ledger.

This namespace is intentionally not re-exported from ``model_lab.__init__`` and is
not connected to synthetic evaluation or candidate registration.
"""

from .contracts import (
    IMPLEMENTATION_STATUS,
    NYFED_SOURCE_ID,
    SCHEMA_VERSION,
    SCOREABILITY,
    SEC_SOURCE_ID,
    SOURCE_POLICIES,
    TREASURY_SOURCE_ID,
    ExplicitSessionCalendar,
    LedgerEvent,
    RawArtifact,
    RealMarketContractError,
)
from .http import (
    SEC_MAX_REQUESTS_PER_SECOND,
    MinimumIntervalRateLimiter,
    OfficialSourceHttpClient,
    validate_sec_user_agent,
)
from .ledger import EventLedger, seal_payload, verify_payload_seal
from .rates import (
    nyfed_search_url,
    parse_nyfed_reference_rates,
    parse_treasury_csv,
    parse_treasury_xml,
    treasury_csv_url,
    treasury_xml_url,
    validate_nyfed_search_url,
    validate_treasury_csv_url,
    validate_treasury_xml_url,
)
from .sec import (
    SubmissionBundle,
    SubmissionRecord,
    discover_submission_history_urls,
    normalize_cik,
    parse_sec_accession_metadata,
    parse_sec_companyfacts,
    parse_sec_submissions,
    sec_accession_index_url,
    sec_companyfacts_url,
    sec_submissions_url,
    validate_sec_accession_index_url,
    validate_sec_companyfacts_url,
    validate_sec_submissions_url,
)


__all__ = [
    "IMPLEMENTATION_STATUS",
    "NYFED_SOURCE_ID",
    "SCHEMA_VERSION",
    "SCOREABILITY",
    "SEC_MAX_REQUESTS_PER_SECOND",
    "SEC_SOURCE_ID",
    "SOURCE_POLICIES",
    "TREASURY_SOURCE_ID",
    "EventLedger",
    "ExplicitSessionCalendar",
    "LedgerEvent",
    "MinimumIntervalRateLimiter",
    "OfficialSourceHttpClient",
    "RawArtifact",
    "RealMarketContractError",
    "SubmissionBundle",
    "SubmissionRecord",
    "discover_submission_history_urls",
    "normalize_cik",
    "nyfed_search_url",
    "parse_nyfed_reference_rates",
    "parse_sec_accession_metadata",
    "parse_sec_companyfacts",
    "parse_sec_submissions",
    "parse_treasury_csv",
    "parse_treasury_xml",
    "seal_payload",
    "sec_accession_index_url",
    "sec_companyfacts_url",
    "sec_submissions_url",
    "treasury_csv_url",
    "treasury_xml_url",
    "validate_nyfed_search_url",
    "validate_sec_accession_index_url",
    "validate_sec_companyfacts_url",
    "validate_sec_submissions_url",
    "validate_sec_user_agent",
    "validate_treasury_csv_url",
    "validate_treasury_xml_url",
    "verify_payload_seal",
]
