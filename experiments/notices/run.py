"""Notice-matching sandbox: build data, query a decision model, score against the baseline.

    python -m experiments.notices.run build --split dev --jobs 4 --data reports/notices
    python -m experiments.notices.run infer --data reports/notices --splits dev test \\
        --model ~/models/decision-2.0-lux-9b --exl3 ~/models/lux-backbone-exl3-4.0 \\
        --out reports/notices/lux.jsonl          # GPU; needs torch + exllamav3
    python -m experiments.notices.run evaluate --data reports/notices --split dev \\
        [--service lux=reports/notices/lux.jsonl] [--out report.json]

A service file holds, per notice, the model's probability for every option; the
evaluation answers its argmax (zero-shot), or "none" when the best non-none
option falls below a threshold tuned on dev for accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import data
from .match import NONE, checked_choice, lux_checks, lux_request, pattern_choice, score


def notice_key(row: dict[str, Any], notice: dict[str, Any]) -> str:
    return f"{row['split']}/{row['seed']}/{row['index']}/{notice['id']}"


def infer(args: argparse.Namespace) -> None:
    from experiments.labeller.lux import Exl3Backend

    backend = Exl3Backend(Path(args.model).expanduser(), Path(args.exl3).expanduser())
    out = Path(args.out)
    done = {json.loads(line)["key"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    with out.open("a") as handle:
        for split in args.splits:
            for row in data.load(Path(args.data) / f"{split}{args.variant}.jsonl"):
                if not row["candidates"]:
                    continue  # nothing to match; every method answers "none"
                for notice in row["notices"]:
                    key = notice_key(row, notice)
                    if key in done:
                        continue
                    state, questions = lux_request(row, notice)
                    probabilities = backend.ask(state, questions)["match"]["probabilities"]
                    record: dict[str, Any] = {"key": key, "probabilities": probabilities}
                    best = max(probabilities, key=probabilities.__getitem__)
                    if args.checks and best != NONE:
                        candidate = next(c for c in row["candidates"] if c["key"] == best)
                        answers = backend.ask(*lux_checks(row, notice, candidate))
                        record["checks"] = {name: float(a["noul"]) for name, a in answers.items()}
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                print(f"{split}/{row['seed']}/{row['index']}", flush=True)


def _choice(probabilities: dict[str, float], threshold: float) -> str:
    best = max((k for k in probabilities if k != NONE), key=probabilities.__getitem__, default=NONE)
    if best == NONE or probabilities[best] < threshold or probabilities[best] < probabilities.get(NONE, 0.0):
        return NONE
    return best


def _tune(dev: list[dict[str, Any]], probs: dict[str, dict[str, float]]) -> float:
    items = [(row, n) for row in dev if row["candidates"] for n in row["notices"]]
    best, best_hits = 0.0, -1
    for step in range(0, 20):
        threshold = step / 20
        hits = sum(
            (choice in n["gold"]) if n["gold"] else choice == NONE
            for row, n in items
            for choice in [_choice(probs[notice_key(row, n)], threshold)]
        )
        if hits > best_hits:
            best, best_hits = threshold, hits
    return best


def evaluate(data_dir: Path, split: str, services: dict[str, Path], variant: str = "") -> dict[str, Any]:
    rows = data.load(data_dir / f"{split}{variant}.jsonl")
    report: dict[str, Any] = {"split": split + variant, "methods": {}}
    report["methods"]["patterns"] = score([(r, n, pattern_choice(r, n)) for r in rows for n in r["notices"]])
    dev = data.load(data_dir / f"dev{variant}.jsonl")
    for name, path in services.items():
        records = {record["key"]: record for record in map(json.loads, path.read_text().splitlines())}
        probs = {key: record["probabilities"] for key, record in records.items()}

        def chosen(
            row: dict[str, Any], notice: dict[str, Any], threshold: float,
            probs: dict[str, dict[str, float]] = probs,
        ) -> str:
            key = notice_key(row, notice)
            return _choice(probs[key], threshold) if key in probs else NONE

        report["methods"][name] = score([(r, n, chosen(r, n, 0.0)) for r in rows for n in r["notices"]])
        threshold = _tune(dev, probs)
        tuned = score([(r, n, chosen(r, n, threshold)) for r in rows for n in r["notices"]])
        report["methods"][f"{name}-tuned"] = {**tuned, "threshold": threshold}
        if any("checks" in record for record in records.values()):

            def checked(
                row: dict[str, Any], notice: dict[str, Any], threshold: float, records: dict[str, Any] = records
            ) -> str:
                record = records.get(notice_key(row, notice))
                return checked_choice(record["probabilities"], record.get("checks"), threshold) if record else NONE

            report["methods"][f"{name}-checks"] = score([(r, n, checked(r, n, 0.5)) for r in rows for n in r["notices"]])
            # One shared check threshold, tuned on dev for accuracy.
            dev_items = [(r, n) for r in dev if r["candidates"] for n in r["notices"]]
            best = max(
                (step / 20 for step in range(10, 20)),
                key=lambda t: sum(
                    (c in n["gold"]) if n["gold"] else c == NONE
                    for r, n in dev_items
                    for c in [checked(r, n, t)]
                ),
            )
            tuned_checks = score([(r, n, checked(r, n, best)) for r in rows for n in r["notices"]])
            report["methods"][f"{name}-checks-tuned"] = {**tuned_checks, "threshold": best}
    return report


def _table(report: dict[str, Any]) -> str:
    def cell(entry: dict[str, Any]) -> str:
        return f"{entry['hits']}/{entry['n']} ({entry['rate']:.2f})" if entry["n"] else "-"

    methods = report["methods"]
    lines = [f"split {report['split']}", "| | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    for label, get in (
        ("accuracy", lambda m: cell(m["accuracy"])),
        ("recall (true notices)", lambda m: cell(m["recall"])),
        ("false explanations", lambda m: cell(m["false_explanations"])),
    ):
        lines.append(f"| {label} | " + " | ".join(get(m) for m in methods.values()) + " |")
    for kind in next(iter(methods.values()))["false_explanations_by_kind"]:
        lines.append(
            f"| false expl., {kind} | "
            + " | ".join(cell(m["false_explanations_by_kind"][kind]) for m in methods.values())
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--split", choices=tuple(data.SPLITS), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p = sub.add_parser("infer")
    p.add_argument("--data", required=True)
    p.add_argument("--splits", nargs="+", default=["dev", "test"])
    p.add_argument("--model", required=True)
    p.add_argument("--exl3", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checks", action="store_true", help="also ask the yes/no checks on the chosen option")
    p.add_argument("--variant", default="", help='file suffix, e.g. "-rewritten"')
    p = sub.add_parser("evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=tuple(data.SPLITS), default="dev")
    p.add_argument("--service", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--variant", default="", help='file suffix, e.g. "-rewritten"')
    p.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.command == "build":
        directory = Path(args.data)
        print(data.build(args.split, directory / f"{args.split}.jsonl", directory / "work", args.jobs))
    elif args.command == "infer":
        infer(args)
    else:
        services = {name: Path(path) for name, _, path in (item.partition("=") for item in args.service)}
        report = evaluate(Path(args.data), args.split, services, args.variant)
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(_table(report))


if __name__ == "__main__":
    main()
