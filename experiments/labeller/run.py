"""Cause-labelling sandbox: build data, fit challengers, compare with the rules.

    python -m experiments.labeller.run build --split train --jobs 4 --data reports/labeller
    python -m experiments.labeller.run build --split dev --jobs 4 --data reports/labeller
    python -m experiments.labeller.run build --split test --jobs 4 --data reports/labeller
    python -m experiments.labeller.run evaluate --data reports/labeller --out reports/labeller/report.json \
        [--service lux=reports/labeller/lux-exl3.jsonl]

Models are fitted on ``train``, tuned on ``dev`` and scored once on ``test``.
Results are labels only: the engine's status never depends on them.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from . import data
from .evaluate import mcnemar, score
from .models import Labeller, RuleLabeller, TreeLabeller
from .service import ServiceLabeller, load_probabilities


def _factories(services: dict[str, Path]) -> dict[str, Callable[[], Labeller]]:
    """Every labeller under test, each built fresh so it can be refitted on a subset."""
    factories: dict[str, Callable[[], Labeller]] = {"rules": RuleLabeller, "trees": TreeLabeller}
    for name, path in services.items():
        probabilities = load_probabilities(path)
        factories[name] = partial(ServiceLabeller, name, probabilities, tuned=False)
        factories[f"{name}-tuned"] = partial(ServiceLabeller, f"{name}-tuned", probabilities, tuned=True)
    return factories


def evaluate(data_dir: Path, services: dict[str, Path] | None = None) -> dict[str, Any]:
    splits = {name: data.load(data_dir / f"{name}.jsonl") for name in ("train", "dev", "test")}
    factories = _factories(services or {})
    reports: dict[str, Any] = {}
    for name, factory in factories.items():
        labeller = factory()
        labeller.fit(splits["train"], splits["dev"])
        reports[name] = score(splits["test"], labeller.predict(splits["test"]))
    # The rules' set is built by hand, so pairs and clean refreshes are where a
    # learned set differs most; single faults are the like-for-like comparison.
    single = [i for i, row in enumerate(splits["test"]) if row["kind"] == "single"]
    rules = reports["rules"]["exact_by_case"]
    versus_rules = {
        name: {
            "all": mcnemar(report["exact_by_case"], rules),
            "single": mcnemar([report["exact_by_case"][i] for i in single], [rules[i] for i in single]),
        }
        for name, report in reports.items()
        if name != "rules"
    }
    for report in reports.values():
        report.pop("exact_by_case")
    return {
        "sizes": {name: len(rows) for name, rows in splits.items()},
        "code_sha256": sorted({row["code_sha256"] for rows in splits.values() for row in rows}),
        "models": reports,
        "versus_rules": versus_rules,
        "profile_shift": _profile_shift(splits, factories),
    }


def _profile_shift(
    splits: dict[str, list[dict[str, Any]]], factories: dict[str, Callable[[], Labeller]]
) -> dict[str, Any]:
    """Single-fault exact rate when each labeller is fitted on one profile and tested on the other."""
    shift: dict[str, Any] = {}
    for source, target in (("small", "realistic"), ("realistic", "small")):
        rows = [row for row in splits["test"] if row["profile"] == target and row["kind"] == "single"]
        entry = {}
        for name, factory in factories.items():
            labeller = factory()
            labeller.fit(
                [row for row in splits["train"] if row["profile"] == source],
                [row for row in splits["dev"] if row["profile"] == source],
            )
            entry[name] = score(rows, labeller.predict(rows))["exact"]
        shift[f"{source}->{target}"] = entry
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
    for name, tests in report["versus_rules"].items():
        for kind, test in tests.items():
            lines.append(
                f"Exact-set McNemar, {kind} ({name} vs rules): {name}-only right {test['only_first']}, "
                f"rules-only right {test['only_second']}, p = {test['p_value']:.3g}"
            )
    for direction, entry in report["profile_shift"].items():
        cells = ", ".join(f"{name} {cell(value)}" for name, value in entry.items())
        lines.append(f"Profile shift {direction}, single faults: {cells}")
    return "\n".join(lines)


def _service(item: str) -> tuple[str, Path]:
    name, _, path = item.partition("=")
    if not name or not path:
        raise SystemExit(f"--service expects NAME=PATH, got {item!r}")
    return name, Path(path)


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
    run.add_argument(
        "--service", action="append", default=[], metavar="NAME=PATH",
        help="decision-service probabilities (JSONL from lux.py); adds NAME and NAME-tuned",
    )
    args = parser.parse_args()
    data_dir = Path(args.data)
    if args.command == "build":
        path = data.build(
            args.split, data_dir / f"{args.split}.jsonl", args.workdir or data_dir / "work", args.jobs
        )
        print(f"wrote {path}")
        return
    report = evaluate(data_dir, dict(_service(item) for item in args.service))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(_table(report))


if __name__ == "__main__":
    main()
