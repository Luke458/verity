"""Rewrite templated notices in an analyst's own words with a local LLM.

    python -m experiments.notices.rewrite --data reports/notices --split dev \\
        --model ~/models/qwen3.8-27b-exl3-11.5gb      # needs torch + exllamav3

Writes ``{split}-rewritten.jsonl``: the same rows with every notice reworded.
A rewrite must keep the facts the oracle's answer depends on: every number in
the template (store and product numbers, weeks) and the name of another
dataset when the notice is about one. A rewrite that drops one is retried with
a new seed; after three failures the template text is kept (and counted).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from . import data

INSTRUCTION = (
    "Rewrite this operations notice the way a busy retail data analyst would post it in a team "
    "chat or an email to the data team. Change the wording, order and format freely, but keep every "
    "fact: which stores, products or categories, which weeks, what changed, whether it has happened "
    "yet, and which dataset it is about if one is named. Refer to stores by number or name as you "
    "like. Reply with the rewritten notice only, in one or two sentences.\n\nNotice: {text}"
)


def _numbers(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\d+", text)}


def faithful(template: str, rewrite: str) -> bool:
    """Every number survives (as a value), and so does another dataset's name."""
    if not rewrite or len(rewrite) > 4 * len(template) + 200:
        return False
    if not _numbers(template) <= _numbers(rewrite):
        return False
    tag = re.match(r"^\[(.+?)\]", template)
    if tag:
        words = [w for w in re.findall(r"[A-Za-z]{3,}", tag.group(1))]
        if not any(w.lower() in rewrite.lower() for w in words):
            return False
    return True


def _prompt(text: str) -> str:
    # Qwen chat format with thinking disabled.
    return (
        "<|im_start|>user\n" + INSTRUCTION.format(text=text) + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=tuple(data.SPLITS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    from exllamav3 import Cache, Config, Generator, Model, Tokenizer, TopPSampler

    config = Config.from_directory(str(Path(args.model).expanduser()))
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=16384)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    sampler = TopPSampler(temperature=0.8, top_p=0.95, temperature_last=True)
    # Qwen's end-of-turn, end-of-text and start-of-turn ids, and the "<|end|>"
    # the model sometimes writes as plain text before inventing another turn.
    stop = [248046, 248044, 248045, "<|end|>", "<|im_"]

    rows = data.load(Path(args.data) / f"{args.split}.jsonl")
    pending: list[dict[str, Any]] = [n for row in rows for n in row["notices"]]
    for notice in pending:
        notice["template_text"] = notice["text"]
        notice["rewritten"] = False
    for attempt in range(3):
        todo = [n for n in pending if not n["rewritten"]]
        for start in range(0, len(todo), args.batch):
            batch = todo[start : start + args.batch]
            outputs = generator.generate(
                prompt=[_prompt(n["template_text"]) for n in batch],
                max_new_tokens=160,
                stop_conditions=stop,
                completion_only=True,
                add_bos=False,
                sampler=sampler,
                seed=1000 * attempt + start,
            )
            for notice, output in zip(batch, outputs, strict=True):
                text = output.split("<|")[0].strip()
                if faithful(notice["template_text"], text):
                    notice["text"], notice["rewritten"] = text, True
            print(f"attempt {attempt}: {start + len(batch)}/{len(todo)}", flush=True)
    kept = sum(not n["rewritten"] for n in pending)
    out = Path(args.data) / f"{args.split}-rewritten.jsonl"
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(f"wrote {out}: {len(pending) - kept} rewritten, {kept} kept as templates")


if __name__ == "__main__":
    main()
