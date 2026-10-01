"""Cause-labelling sandbox: build data, fit challengers, compare with the rules.

    python -m experiments.labeller.run build --split train --jobs 6 --data reports/labeller
    python -m experiments.labeller.run build --split dev --jobs 6 --data reports/labeller
    python -m experiments.labeller.run build --split test --jobs 6 --data reports/labeller
    python -m experiments.labeller.run evaluate --data reports/labeller --out reports/labeller/report.json

Models are fitted on ``train``, tuned on ``dev`` and scored once on ``test``.
Results are labels only: the engine's status never depends on them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import data
from .evaluate import mcnemar, score
from .models import Labeller, RuleLabeller, TreeLabeller


def evaluate(data_dir: Path) -> dict[str, Any]:
    splits = {name: data.load(data_dir / f"{name}.jsonl") for name in ("train", "dev", "test")}
    labellers: list[Labeller] = [RuleLabeller(), TreeLabeller()]
    reports: dict[str, Any] = {}
    for labeller in labellers:
        labeller.fit(splits["train"], splits["dev"])
        reports[labeller.name] = score(splits["test"], labeller.predict(splits["test"]))
    trees, rules = reports["trees"]["exact_by_case"], reports["rules"]["exact_by_case"]
    # The rules name at most one cause, so pairs favour any multi-label model by
    # construction; single faults are the like-for-like comparison.
    single = [i for i, row in enumerate(splits["test"]) if row["kind"] == "single"]
    for report in reports.values():
        report.pop("exact_by_case")
    return {
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "code_sha256": sorted({row["code_sha256"] for rows in splits.values() for row in rows}),
        "models": reports,
        "trees_vs_rules_exact": mcnemar(trees, rules),
        "trees_vs_rules_single": mcnemar([trees[i] for i in single], [rules[i] for i in single]),
        "profile_shift": _profile_shift(splits),
    }


def _profile_shift(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Single-fault exact rate when trees are fitted on one profile and tested on the other."""
    shift: dict[str, Any] = {}
    for source, target in (("small", "realistic"), ("realistic", "small")):
        model = TreeLabeller()
        model.fit(
            [row for row in splits["train"] if row["profile"] == source],
            [row for row in splits["dev"] if row["profile"] == source],
        )
        rows = [row for row in splits["test"] if row["profile"] == target and row["kind"] == "single"]
        shift[f"{source}->{target}"] = {
            name: score(rows, labeller.predict(rows))["exact"]
            for name, labeller in (("rules", RuleLabeller()), ("trees", model))
        }
    return shift


def _table(report: dict[str, Any]) -> str:
    def cell(entry: dict[str, Any]) -> str:
        return f"{entry['hits']}/{entry['n']} ({entry['rate']:.2f})"

    lines = ["| | " + " | ".join(report["models"]) + " |", "|---|" + "---|" * len(report["models"])]
    rows = [
        ("exact set, all", lambda r: cell(r["exact"])),
        ("exact, single fault", lambda r: cell(r["by_kind"]["single"]["exact"])),
        ("exact, clean", lambda r: cell(r["by_kind"]["clean"]["exact"])),
        ("pair: names both", lambda r: cell(r["by_kind"]["pair"]["names_both"])),
        ("pair: names one or more", lambda r: cell(r["by_kind"]["pair"]["names_one_or_more"])),
        ("micro F1", lambda r: f"{r['micro_f1']:.3f}"),
    ]
    for label, render in rows:
        lines.append(f"| {label} | " + " | ".join(render(m) for m in report["models"].values()) + " |")
    lines.append("")
    for key, label in (("trees_vs_rules_exact", "all"), ("trees_vs_rules_single", "single faults")):
        test = report[key]
        lines.append(
            f"Exact-set McNemar, {label} (trees vs rules): trees-only right {test['only_first']}, "
            f"rules-only right {test['only_second']}, p = {test['p_value']:.3g}"
        )
    for direction, entry in report["profile_shift"].items():
        lines.append(
            f"Profile shift {direction}, single faults: rules {cell(entry['rules'])}, trees {cell(entry['trees'])}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--split", choices=tuple(data.SPLITS), required=True)
    build.add_argument("--data", required=True)
    build.add_argument("--workdir", default=None)
    build.add_argument("--jobs", type=int, default=1)
    run = sub.add_parser("evaluate")
    run.add_argument("--data", required=True)
    run.add_argument("--out", default=None)
    args = parser.parse_args()
    data_dir = Path(args.data)
    if args.command == "build":
        path = data.build(
            args.split, data_dir / f"{args.split}.jsonl", args.workdir or data_dir / "work", args.jobs
        )
        print(f"wrote {path}")
        return
    report = evaluate(data_dir)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(_table(report))


if __name__ == "__main__":
    main()
