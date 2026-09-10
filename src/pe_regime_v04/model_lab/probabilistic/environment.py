"""NGBoost dedicated-environment compatibility evidence contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import PROBABILISTIC_DESIGN_SHA256, ProbabilisticContractError, sha256_file


EXPECTED_CORE = {
    "ngboost": "0.5.11",
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scikit-learn": "1.7.2",
    "scipy": "1.15.3",
    "pyarrow": "21.0.0",
}
MODEL_LAB_LOCK_SHA256 = "0181e18dada9019faebdb50577f3a5f3d266053064bff5d640c4e77b28ba29a3"
NGBOOST_WHEEL_SHA256 = "c3683334ab6ad58d79bc50aa11d8f090f4d452010c73dc2f2304affc448b08fa"
OFFICIAL_LICENSE_URL = "https://raw.githubusercontent.com/stanfordmlgroup/ngboost/v0.5.11/LICENSE"
OFFICIAL_METADATA_URL = "https://pypi.org/pypi/ngboost/0.5.11/json"


def parse_freeze(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ProbabilisticContractError(f"unresolved freeze entry: {line!r}")
        name, version = line.split("==", 1)
        normalized = name.casefold().replace("_", "-")
        if normalized in values:
            raise ProbabilisticContractError("duplicate freeze package")
        values[normalized] = version
    return values


def verify_dedicated_freeze(text: str) -> dict[str, str]:
    values = parse_freeze(text)
    for package, version in EXPECTED_CORE.items():
        if values.get(package) != version:
            raise ProbabilisticContractError(
                f"dedicated environment drift for {package}: {values.get(package)!r}"
            )
    return values


def verify_dry_run_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbabilisticContractError("pip dry-run report is unreadable") from exc
    installs = report.get("install")
    if not isinstance(installs, list):
        raise ProbabilisticContractError("pip dry-run install inventory is missing")
    packages: dict[str, str] = {}
    artifacts: list[dict[str, str]] = []
    ngboost_hash: str | None = None
    for item in installs:
        metadata = item.get("metadata", {})
        name = str(metadata.get("name", "")).casefold().replace("_", "-")
        version = str(metadata.get("version", ""))
        packages[name] = version
        download = item.get("download_info", {})
        archive_hash = str(download.get("archive_info", {}).get("hash", "")).removeprefix("sha256=")
        if len(archive_hash) != 64:
            raise ProbabilisticContractError(f"dry-run artifact hash missing for {name}")
        artifacts.append(
            {
                "name": name,
                "version": version,
                "url": str(download.get("url", "")),
                "sha256": archive_hash,
            }
        )
        if name == "ngboost":
            ngboost_hash = archive_hash
    if packages.get("ngboost") != "0.5.11" or ngboost_hash != NGBOOST_WHEEL_SHA256:
        raise ProbabilisticContractError("dry-run NGBoost wheel/version binding failed")
    return {
        "report_sha256": sha256_file(Path(path)),
        "resolved_additions": dict(sorted(packages.items())),
        "resolved_artifacts": sorted(artifacts, key=lambda value: value["name"]),
        "ngboost_wheel_sha256": ngboost_hash,
        "design_sha256": PROBABILISTIC_DESIGN_SHA256,
    }
