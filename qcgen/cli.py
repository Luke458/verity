"""Command line interface for the synthetic generator.

    qcgen generate --suite demo --scenarios 5 --profile small
    qcgen verify --suite-dir data/suites/demo
    qcgen list-faults
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from .config import ALL_FAMILIES, STAGES, load_suite_config, suite_config
from .scenarios import generate_suite
from .verify import verify_suite

FAULT_DESCRIPTIONS = {
    "missing_stores": ("source", "Stores omitted from the newest week."),
    "new_store_backfill": ("source", "New store appears with historical weeks, unregistered."),
    "expected_event": ("source", "New store backfill matched by the expected-event registry."),
    "history_truncation": ("source", "Recent history removed for selected stores."),
    "commodity_remap": ("source", "Products moved between commodities, totals conserved."),
    "coding_error": ("coded", "Value multiplier applied at the coding stage."),
    "warehouse_transform_error": ("warehouse", "Aggregation change at the warehouse stage."),
    "recalculation": ("source", "Small changes across most historical weeks."),
    "schema_failure": ("report", "Required column missing from the report."),
    "null_duplicate_storm": ("warehouse", "Nulls and duplicated rows introduced."),
    "market_movement": ("warehouse", "Genuine trend shift; negative control."),
}


def _cmd_generate(args: argparse.Namespace) -> int:
    if args.config:
        config = load_suite_config(args.config)
    else:
        config = suite_config(args.profile)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.families:
        families = tuple(f.strip() for f in args.families.split(",") if f.strip())
        config = replace(config, families=families)
    stages = STAGES if args.stages == "all" else tuple(
        s.strip() for s in args.stages.split(",") if s.strip()
    )
    suite_dir = generate_suite(config, args.out, args.scenarios, args.suite, stages)
    suite = json.loads((suite_dir / "suite.json").read_text())

    print(
        f"suite {suite['suite_id']}  profile={suite['profile']}  "
        f"seed={suite['seed']}  scenarios={suite['n_scenarios']}"
    )
    print(f"  stages: {', '.join(suite['stages'])}")
    print(f"  output: {suite_dir}")
    for scenario in suite["scenarios"]:
        rows = ", ".join(f"{k}={v}" for k, v in scenario["row_counts"].items())
        print(
            f"  {scenario['scenario_id']}  {scenario['family']:<26} "
            f"{scenario['expected_class']:<24} rows({rows})"
        )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    passed, report = verify_suite(args.suite_dir)
    print(
        f"verify {report['suite_id']}: {report['checks']} checks, "
        f"{report['failed_count']} failed"
    )
    for failure in report["failed"]:
        print(f"  FAIL {failure['name']}  {failure['detail']}")
    print("OK" if passed else "FAILED")
    return 0 if passed else 1


def _cmd_list_faults(args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(FAULT_DESCRIPTIONS, indent=2, sort_keys=True))
        return 0
    for family in ALL_FAMILIES:
        stage, description = FAULT_DESCRIPTIONS.get(family, ("?", "no description"))
        print(f"{family:<26} {stage:<10} {description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qcgen",
        description="Generate synthetic retail QC scenarios with ground-truth faults.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a scenario suite")
    generate.add_argument("--suite", default="demo", help="suite id (directory name)")
    generate.add_argument("--scenarios", type=int, default=5)
    generate.add_argument("--seed", type=int, default=None)
    generate.add_argument("--profile", choices=["tiny", "small", "full"], default="small")
    generate.add_argument("--config", default=None, help="path to a dataset YAML config")
    generate.add_argument("--families", default=None, help="comma-separated fault families")
    generate.add_argument("--stages", default="report", help="comma-separated stages, or 'all'")
    generate.add_argument("--out", default="data/suites", help="output root directory")
    generate.set_defaults(func=_cmd_generate)

    verify = subparsers.add_parser("verify", help="verify a generated suite")
    verify.add_argument("--suite-dir", required=True)
    verify.set_defaults(func=_cmd_verify)

    faults = subparsers.add_parser("list-faults", help="list available fault families")
    faults.add_argument("--json", action="store_true")
    faults.set_defaults(func=_cmd_list_faults)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
