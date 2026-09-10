from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import run_overlay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PE Regime Engine v0.4 bottleneck overlay"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the v0.4 overlay on a v0.3 canonical CSV")
    run.add_argument("--input-csv", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--truth-csv")
    run.add_argument("--config")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = load_config(args.config)
        result = run_overlay(
            args.input_csv,
            args.output_dir,
            config,
            truth_csv=args.truth_csv,
        )
        print(f"Output CSV : {Path(result['output_csv']).resolve()}")
        print(f"Report JSON: {Path(result['report_json']).resolve()}")
        return 0
    return 2
