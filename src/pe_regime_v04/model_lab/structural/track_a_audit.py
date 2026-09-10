"""Data-bound six-part audit for the five frozen Track-A input surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .contracts import (
    STRUCTURAL_DESIGN_SHA256,
    StructuralContractError,
    canonical_json_bytes,
    logical_frame_sha256,
    seal_payload,
    sha256_bytes,
    sha256_file,
)
from .features import (
    MARKET_CURRENT,
    REGIME_CURRENT,
    REQUIRED_UPSTREAM_AUDITS,
    TRACK_A_LAGGED_COLUMNS,
    TRACK_A_LAG_SOURCE_COLUMNS,
    TrackASourceAuditBinding,
    build_track_a_lag1_artifact,
)


PREDICT_INPUTS_RELATIVE = "outputs/model_zoo_wave1_screen_20260819/PREDICT_INPUTS.json"
FEATURE_REGISTRY_SNAPSHOT_RELATIVE = (
    "outputs/model_zoo_wave1_screen_20260819/execution_snapshots/"
    "2ea4c225ed3bcd94d4bb13536a6f00735cdfadf32d1534d4b671ea75e6524d7d.snapshot"
)
FEATURE_REGISTRY_SNAPSHOT_SHA256 = (
    "2ea4c225ed3bcd94d4bb13536a6f00735cdfadf32d1534d4b671ea75e6524d7d"
)
EXPECTED_SEEDS = (6301, 6421, 6521, 6607, 6701)
EXPECTED_ROWS = 1800
AUDIT_METHOD_VERSION = "track_a_data_bound_six_audit_v2"


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StructuralContractError(f"cannot read {context}") from exc
    if not isinstance(value, dict):
        raise StructuralContractError(f"{context} must be an object")
    return value


def _validate_file_record(root: Path, raw: Mapping[str, Any], *, context: str) -> Path:
    if set(raw) != {"path", "bytes", "sha256"}:
        raise StructuralContractError(f"{context} file record schema changed")
    path = Path(str(raw["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StructuralContractError(f"{context} path escapes project root") from exc
    if path.stat().st_size != int(raw["bytes"]) or sha256_file(path) != raw["sha256"]:
        raise StructuralContractError(f"{context} bytes changed")
    return path


def _load_surface(
    root: Path, entry: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    seed = int(entry["seed"])
    canonical_path = _validate_file_record(
        root, entry["canonical_csv"], context=f"seed {seed} canonical"
    )
    current_path = _validate_file_record(root, entry["output_csv"], context=f"seed {seed} current")
    canonical_columns = ("date", "symbol", "observed_pe", *MARKET_CURRENT)
    current_columns = ("date", "symbol", *REGIME_CURRENT)
    try:
        canonical = pd.read_csv(canonical_path, usecols=list(canonical_columns))
        current = pd.read_csv(current_path, usecols=list(current_columns))
    except (OSError, ValueError) as exc:
        raise StructuralContractError(f"seed {seed} Track-A source cannot be read") from exc
    if set(canonical.columns) != set(canonical_columns) or set(current.columns) != set(
        current_columns
    ):
        raise StructuralContractError(f"seed {seed} selected source schema changed")
    canonical = canonical.loc[:, list(canonical_columns)]
    current = current.loc[:, list(current_columns)]
    if len(canonical) != EXPECTED_ROWS or len(current) != EXPECTED_ROWS:
        raise StructuralContractError(f"seed {seed} source row count changed")
    canonical_dates = pd.to_datetime(canonical["date"], utc=True).dt.tz_convert(None)
    current_dates = pd.to_datetime(current["date"], utc=True).dt.tz_convert(None)
    if canonical_dates.isna().any() or current_dates.isna().any():
        raise StructuralContractError(f"seed {seed} contains invalid dates")
    if not canonical_dates.equals(current_dates):
        raise StructuralContractError(f"seed {seed} canonical/current dates differ")
    if not canonical["symbol"].astype(str).equals(current["symbol"].astype(str)):
        raise StructuralContractError(f"seed {seed} canonical/current entities differ")
    if canonical["symbol"].nunique(dropna=False) != 1:
        raise StructuralContractError(f"seed {seed} must contain exactly one entity")
    source = pd.DataFrame(
        {
            "seed": np.repeat(seed, EXPECTED_ROWS),
            "entity_id": canonical["symbol"].astype(str),
            "date": canonical_dates,
        }
    )
    for column in MARKET_CURRENT:
        source[column] = canonical[column]
    for column in REGIME_CURRENT:
        source[column] = current[column]
    source = source.loc[:, ["seed", "entity_id", "date", *TRACK_A_LAG_SOURCE_COLUMNS]]
    selected_canonical = canonical.loc[:, list(canonical_columns)].copy()
    selected_canonical["date"] = canonical_dates
    selected_current = current.loc[:, list(current_columns)].copy()
    selected_current["date"] = current_dates
    evidence = {
        "seed": seed,
        "rows": len(source),
        "entity_count": int(source["entity_id"].nunique(dropna=False)),
        "entity_ids": sorted(source["entity_id"].unique().tolist()),
        "canonical_path": canonical_path.relative_to(root).as_posix(),
        "canonical_bytes": canonical_path.stat().st_size,
        "canonical_sha256": sha256_file(canonical_path),
        "canonical_selected_columns_sha256": logical_frame_sha256(selected_canonical),
        "current_path": current_path.relative_to(root).as_posix(),
        "current_bytes": current_path.stat().st_size,
        "current_sha256": sha256_file(current_path),
        "current_selected_columns_sha256": logical_frame_sha256(selected_current),
        "source_identity_sha256": logical_frame_sha256(
            source.loc[:, ["seed", "entity_id", "date"]]
        ),
        "source_content_sha256": logical_frame_sha256(source),
        "observed_pe_read_for_same_row_intervention_only": True,
        "true_fair_pe_read": False,
        "candidate_prediction_columns_read": False,
        "score_columns_read": False,
    }
    return source, canonical["observed_pe"].copy(), evidence


def _registry_audits(registry: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    records = registry.get("records")
    if not isinstance(records, list):
        raise StructuralContractError("feature registry records are missing")
    by_id = {str(row.get("feature_id")): row for row in records if isinstance(row, Mapping)}
    rows: list[dict[str, Any]] = []
    for feature_id in TRACK_A_LAG_SOURCE_COLUMNS:
        row = by_id.get(feature_id)
        if row is None or row.get("column_name") != feature_id:
            raise StructuralContractError(f"source feature is not registered: {feature_id}")
        prefix = row.get("prefix_invariance_status") == "PASS"
        future = row.get("future_intervention_status") == "PASS"
        same_target = row.get("same_row_target_leakage_status") == "PASS"
        pit = bool(
            row.get("pit_availability_status") == "PASS"
            and row.get("point_in_time_safe") is True
            and int(row.get("lookahead_sessions", -1)) == 0
            and row.get("evaluation_only") is False
            and row.get("allowed_for_fit") is True
            and row.get("allowed_for_predict") is True
        )
        publication_status = str(row.get("publication_date_status"))
        restatement_status = str(row.get("restatement_availability_status"))
        publication = publication_status in {"PASS", "NOT_APPLICABLE"}
        restatement = bool(
            restatement_status in {"PASS", "NOT_APPLICABLE"}
            and row.get("uses_revised_data") is False
        )
        checks = {
            "prefix_invariance": prefix,
            "future_intervention": future,
            "same_row_target_leakage": same_target,
            "pit_availability": pit,
            "publication_date": publication,
            "restatement_availability": restatement,
        }
        if not all(checks.values()):
            raise StructuralContractError(f"source feature six-audit gate failed: {feature_id}")
        rows.append(
            {
                "feature_id": feature_id,
                "feature_family": row.get("feature_family"),
                "availability": row.get("availability"),
                "availability_lag_sessions": row.get("availability_lag_sessions"),
                "lookahead_sessions": row.get("lookahead_sessions"),
                "publication_date_status": publication_status,
                "restatement_availability_status": restatement_status,
                "checks": checks,
                "provenance_sha256": row.get("provenance_sha256"),
            }
        )
    payload = {
        "registry_file_sha256": FEATURE_REGISTRY_SNAPSHOT_SHA256,
        "source_feature_count": len(rows),
        "features": rows,
    }
    return payload, sha256_bytes(canonical_json_bytes(payload))


def _algorithmic_audits(
    source: pd.DataFrame,
    *,
    observed_pe: pd.Series,
    audit_binding: TrackASourceAuditBinding,
) -> tuple[dict[str, bool], dict[str, Any]]:
    full_artifact = build_track_a_lag1_artifact(source, source_audit=audit_binding)
    full = full_artifact.to_frame()
    checkpoints = sorted({1, 2, 21, 252, 504, 900, len(source)})
    prefix_pass = True
    for count in checkpoints:
        prefix_source = source.iloc[:count].copy()
        prefix = build_track_a_lag1_artifact(prefix_source, source_audit=audit_binding).to_frame()
        if not full.iloc[:count].reset_index(drop=True).equals(prefix.reset_index(drop=True)):
            prefix_pass = False
            break

    pivot = len(source) // 2
    future_changed = source.copy()
    for index, column in enumerate(TRACK_A_LAG_SOURCE_COLUMNS):
        numeric = pd.to_numeric(future_changed.loc[pivot:, column], errors="coerce")
        future_changed.loc[pivot:, column] = numeric.fillna(0.0) + 1000.0 + index
    future = build_track_a_lag1_artifact(future_changed, source_audit=audit_binding).to_frame()
    future_pass = (
        full.iloc[:pivot].reset_index(drop=True).equals(future.iloc[:pivot].reset_index(drop=True))
    )

    with_target_a = source.copy()
    with_target_b = source.copy()
    with_target_a["observed_pe"] = pd.to_numeric(observed_pe, errors="coerce")
    with_target_b["observed_pe"] = (
        pd.to_numeric(observed_pe, errors="coerce").fillna(1.0) * 1000.0 + 7.0
    )
    target_a = build_track_a_lag1_artifact(with_target_a, source_audit=audit_binding)
    target_b = build_track_a_lag1_artifact(with_target_b, source_audit=audit_binding)
    same_target_pass = target_a == target_b

    group_starts = full.groupby(["seed", "entity_id"], sort=False).head(1)
    boundary_pass = bool(group_starts.loc[:, list(TRACK_A_LAGGED_COLUMNS)].isna().all().all())
    exact_shift_pass = True
    for source_column, derived_column in zip(TRACK_A_LAG_SOURCE_COLUMNS, TRACK_A_LAGGED_COLUMNS):
        expected = pd.to_numeric(source[source_column], errors="coerce").shift(1)
        actual = pd.to_numeric(full[derived_column], errors="coerce")
        if not np.allclose(
            expected.to_numpy(dtype=np.float64, na_value=np.nan),
            actual.to_numpy(dtype=np.float64, na_value=np.nan),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            exact_shift_pass = False
            break
    checks = {
        "prefix_invariance": prefix_pass,
        "future_intervention": future_pass,
        "same_row_target_leakage": same_target_pass,
        "pit_availability": True,
        "publication_date": True,
        "restatement_availability": True,
    }
    evidence = {
        "prefix_checkpoints_rows": checkpoints,
        "future_intervention_start_row": pivot,
        "future_intervention_all_27_source_columns": True,
        "same_row_target_intervention_uses_data_bound_observed_pe": True,
        "exact_groupwise_shift_all_rows_all_27_columns": exact_shift_pass,
        "group_boundary_first_row_all_27_missing": boundary_pass,
        "derived_rows": len(full),
        "derived_content_sha256": full_artifact.content_sha256,
        "derived_source_identity_sha256": full_artifact.source_identity_sha256,
        "derived_source_content_sha256": full_artifact.source_content_sha256,
    }
    if not all(checks.values()) or not exact_shift_pass or not boundary_pass:
        raise StructuralContractError("data-bound Track-A algorithmic audit failed")
    return checks, evidence


def audit_five_track_a_surfaces(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    predict_path = root / PREDICT_INPUTS_RELATIVE
    registry_path = root / FEATURE_REGISTRY_SNAPSHOT_RELATIVE
    if sha256_file(registry_path) != FEATURE_REGISTRY_SNAPSHOT_SHA256:
        raise StructuralContractError("frozen feature-registry snapshot changed")
    predict = _read_json(predict_path, context="PREDICT_INPUTS")
    registry = _read_json(registry_path, context="feature registry snapshot")
    generator_paths = (
        root / "src/pe_regime_v04/model_lab/structural/track_a_audit.py",
        root / "src/pe_regime_v04/model_lab/structural/features.py",
        root / "src/pe_regime_v04/model_lab/structural/contracts.py",
    )
    seeds = predict.get("seeds")
    if not isinstance(seeds, list) or tuple(int(row["seed"]) for row in seeds) != EXPECTED_SEEDS:
        raise StructuralContractError("five spent input surfaces changed")
    registry_evidence, registry_logical_sha = _registry_audits(registry)
    audit_contract = {
        "method": AUDIT_METHOD_VERSION,
        "design_sha256": STRUCTURAL_DESIGN_SHA256,
        "predict_inputs_sha256": sha256_file(predict_path),
        "feature_registry_sha256": FEATURE_REGISTRY_SNAPSHOT_SHA256,
        "feature_registry_audit_logical_sha256": registry_logical_sha,
        "required_audits": list(REQUIRED_UPSTREAM_AUDITS),
    }
    binding = TrackASourceAuditBinding(
        audit_sha256=sha256_bytes(canonical_json_bytes(audit_contract)),
        passed_audits=REQUIRED_UPSTREAM_AUDITS,
    )
    surface_results: list[dict[str, Any]] = []
    all_sources: list[pd.DataFrame] = []
    for entry in seeds:
        source, observed_pe, evidence = _load_surface(root, entry)
        checks, derived = _algorithmic_audits(
            source,
            observed_pe=observed_pe,
            audit_binding=binding,
        )
        all_sources.append(source)
        surface_results.append(
            {
                **evidence,
                "six_audits": checks,
                "all_six_pass": all(checks.values()),
                "group_boundary": derived,
            }
        )
    combined = pd.concat(all_sources, ignore_index=True)
    combined_artifact = build_track_a_lag1_artifact(combined, source_audit=binding)
    combined_frame = combined_artifact.to_frame()
    starts = combined_frame.groupby(["seed", "entity_id"], sort=False).head(1)
    combined_boundary_pass = bool(
        len(starts) == len(EXPECTED_SEEDS)
        and starts.loc[:, list(TRACK_A_LAGGED_COLUMNS)].isna().all().all()
    )
    if not combined_boundary_pass:
        raise StructuralContractError("Track-A lag crosses a seed/entity boundary")
    return seal_payload(
        {
            "format_version": 2,
            "mode": "structural_track_a_data_bound_six_audit",
            "design_sha256": STRUCTURAL_DESIGN_SHA256,
            "audit_contract": audit_contract,
            "audit_contract_sha256": binding.audit_sha256,
            "generator_source_inventory": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in generator_paths
            ],
            "predict_inputs": {
                "path": PREDICT_INPUTS_RELATIVE,
                "bytes": predict_path.stat().st_size,
                "sha256": sha256_file(predict_path),
            },
            "feature_registry_snapshot": {
                "path": FEATURE_REGISTRY_SNAPSHOT_RELATIVE,
                "bytes": registry_path.stat().st_size,
                "sha256": sha256_file(registry_path),
            },
            "feature_registry_evidence": registry_evidence,
            "surface_count": len(surface_results),
            "surfaces": surface_results,
            "combined_group_count": len(starts),
            "combined_group_boundary_isolation": combined_boundary_pass,
            "combined_derived_content_sha256": combined_artifact.content_sha256,
            "all_5_surfaces_all_6_audits_pass": all(row["all_six_pass"] for row in surface_results),
            "attestations": {
                "structural_candidate_predictions_generated": False,
                "structural_candidate_scores_generated": False,
                "wave1_candidate_prediction_columns_read": False,
                "wave1_score_columns_read": False,
                "true_fair_pe_read": False,
                "fresh_seed_selected_or_reserved": False,
                "heldout_opened": False,
            },
        }
    )
