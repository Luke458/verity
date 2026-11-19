"""Gates for the registered confirm run (docs/notice-matching.md), checked on reworded dev notices.

    python -m experiments.notices.gates lux --data reports/notices \\
        --stored reports/notices/lux-checks2-dev-rewritten.jsonl \\
        --model ~/models/decision-2.0-lux-9b --exl3 ~/models/lux-backbone-exl3-4.0      # G1
    python -m experiments.notices.gates clairic --data reports/notices \\
        --stored reports/notices/clairic-v2-dev-rewritten.jsonl --model ~/models/clairic-flash   # G2

G1: re-run Lux + checks on 20 random notices; the chosen option and every check
verdict (>= 0.5) must match the stored answers. G2: re-read 40 random notices
with unquantized BF16; the match must equal the int8 one. Seed 0 samples.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from . import data
from .match import NONE, extract_choice, lux_checks, lux_request
from .run import notice_key


def _sample(data_dir: Path, stored: Path, count: int) -> tuple[list[tuple[dict, dict]], dict[str, Any]]:
    records = {r["key"]: r for r in map(json.loads, stored.read_text().splitlines())}
    items = [
        (row, notice)
        for row in data.load(data_dir / "dev-rewritten.jsonl")
        if row["candidates"]
        for notice in row["notices"]
        if notice_key(row, notice) in records
    ]
    return random.Random(0).sample(items, count), records


def lux(args: argparse.Namespace) -> None:
    from experiments.labeller.lux import Exl3Backend

    backend = Exl3Backend(Path(args.model).expanduser(), Path(args.exl3).expanduser())
    items, records = _sample(Path(args.data), Path(args.stored), 20)
    agree = 0
    for row, notice in items:
        stored = records[notice_key(row, notice)]
        probabilities = backend.ask(*lux_request(row, notice))["match"]["probabilities"]
        best = max(probabilities, key=probabilities.__getitem__)
        verdicts: dict[str, bool] = {}
        if best != NONE:
            candidate = next(c for c in row["candidates"] if c["key"] == best)
            answers = backend.ask(*lux_checks(row, notice, candidate))
            verdicts = {name: float(a["noul"]) >= 0.5 for name, a in answers.items()}
        stored_best = max(stored["probabilities"], key=stored["probabilities"].__getitem__)
        stored_verdicts = {name: p >= 0.5 for name, p in (stored.get("checks") or {}).items()}
        same = best == stored_best and verdicts == stored_verdicts
        agree += same
        print(f"{'same' if same else 'DIFF'} {best}/{stored_best} {verdicts} {stored_verdicts}", flush=True)
    print(f"G1 Lux reproduces {agree}/20 (needs >= 19): {'PASS' if agree >= 19 else 'FAIL'}")


def clairic(args: argparse.Namespace) -> None:
    from qc.notices import extract_request

    from .clairic import load

    model = load(Path(args.model).expanduser(), "cuda", args.offload_layers, None)
    items, records = _sample(Path(args.data), Path(args.stored), 40)
    agree = 0
    for row, notice in items:
        stored = records[notice_key(row, notice)]
        bf16 = model.predict(extract_request(notice["text"]))["answers"]
        a, b = extract_choice(row, notice, stored["answers"]), extract_choice(row, notice, bf16)
        agree += a == b
        print(f"{'same' if a == b else 'DIFF'} int8={a} bf16={b}", flush=True)
    model.close()
    print(f"G2 int8 vs BF16 same match {agree}/40 (needs >= 38): {'PASS' if agree >= 38 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("lux")
    p.add_argument("--data", required=True)
    p.add_argument("--stored", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--exl3", required=True)
    p = sub.add_parser("clairic")
    p.add_argument("--data", required=True)
    p.add_argument("--stored", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--offload-layers", type=int, default=4)
    args = parser.parse_args()
    {"lux": lux, "clairic": clairic}[args.command](args)


if __name__ == "__main__":
    main()
