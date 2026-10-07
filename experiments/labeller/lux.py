"""Decision-2.0-Lux-9B (vllm-sr, Apache-2.0) as a cause-labelling service.

    python -m experiments.labeller.lux shim --model ~/models/decision-2.0-lux-9b --out ~/models/lux-backbone-hf
    python -m experiments.labeller.lux caldata --model ~/models/decision-2.0-lux-9b --data reports/labeller --out cal.safetensors
    python -m experiments.labeller.lux infer --model ... --backend exl3 --exl3 ~/models/lux-backbone-exl3-4.0 \\
        --data reports/labeller --splits dev test --out reports/labeller/lux-exl3-4.0.jsonl
    python -m experiments.labeller.lux infer --model ... --backend reference --threads 4 --limit 30 ...
    python -m experiments.labeller.lux serve --model ... --exl3 ... --port 8090   # POST /v1/systemone

``reference`` runs the published runtime unmodified (CPU, fp32). ``exl3`` keeps
Lux's prompt encoding, decision head and answer normalization but takes the
backbone's final hidden states from an EXL3-quantized copy run by ExLlamaV3
(``shim`` + ``convert.py`` produce it). Output is one JSON line per row, read by
``service.ServiceLabeller``; runs resume where an output file left off.

Needs torch, transformers and (for ``exl3``) exllamav3; never imported by the
engine or by ``run.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

from . import data
from .models import CAUSES
from .service import questions, render_state, row_key, row_state, state_sha256

PREFIX = "model.language_model."


def _vendored(model_dir: Path) -> tuple[Any, Any]:
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from decision2._vendor.dev2model import decision_model, infer

    return decision_model, infer


def _tokenizer(model_dir: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def _encoded_questions(
    model_dir: Path, tokenizer: Any, state: dict[str, Any], version: int = 1
) -> list[tuple[str, dict, dict]]:
    decision_model, infer = _vendored(model_dir)
    metadata = json.loads((model_dir / "decision_config.json").read_text())
    encode = decision_model.encoder_for(metadata)
    cap = json.loads((model_dir / "MODEL_MANIFEST.json").read_text())["max_input_tokens"]
    item = {"id": "request", "state": state}
    out = []
    for cause, question in questions(version).items():
        row = infer.question_to_row(item, cause, question)
        out.append((cause, row, encode(row, tokenizer, cap)))
    return out


def _stock_dtype(key: str, tensor: Any) -> Any:
    """Lux keeps norms, conv and gate parameters in fp32; stock Qwen3.5 (and the
    ExLlamaV3 kernels) keep only ``A_log`` and the linear-attention norm in fp32."""
    import torch

    if tensor.dtype == torch.float32 and not key.endswith(("linear_attn.A_log", "linear_attn.norm.weight")):
        return tensor.to(torch.bfloat16)
    return tensor


def shim(model_dir: Path, out: Path) -> None:
    """The backbone as an HF ``Qwen3_5ForCausalLM`` directory the EXL3 converter accepts."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    backbone = model_dir / "backbone"
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((backbone / "config.json").read_text())
    # The decision model never reads logits: tie the (unused) LM head to the
    # embeddings, and drop the MTP layer the config declares but the weights lack.
    config.update(architectures=["Qwen3_5ForCausalLM"], tie_word_embeddings=True, mtp_num_hidden_layers=0)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copy(model_dir / name, out / name)
    index = json.loads((backbone / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(backbone / shard), "pt") as handle:
            tensors = {PREFIX + key: _stock_dtype(key, handle.get_tensor(key)) for key in handle.keys()}
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print(f"wrote {shard} ({len(tensors)} tensors)")
    index["weight_map"] = {PREFIX + key: shard for key, shard in index["weight_map"].items()}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def caldata(model_dir: Path, data_dir: Path, out: Path, rows: int, cols: int, seed: int = 0) -> None:
    """EXL3 calibration rows packed from Lux prompts over train-split refreshes only."""
    import torch
    from safetensors.torch import save_file

    tokenizer = _tokenizer(model_dir)
    prompts = [
        encoded["ids"]
        for row in data.load(data_dir / "train.jsonl")
        for _, _, encoded in _encoded_questions(model_dir, tokenizer, render_state(row["features"]))
    ]
    random.Random(seed).shuffle(prompts)
    stream = [token for ids in prompts for token in ids]
    count = min(rows, len(stream) // cols)
    packed = torch.tensor(stream[: count * cols], dtype=torch.long).view(count, cols)
    save_file({"input_ids": packed}, str(out))
    print(f"{len(prompts)} prompts, {len(stream)} tokens -> {count} rows x {cols} -> {out}")


class Exl3Backend:
    """Lux's head over the final hidden states of an EXL3 backbone (ExLlamaV3)."""

    def __init__(self, model_dir: Path, exl3_dir: Path):
        import torch
        from exllamav3 import Config, Model
        from safetensors.torch import load_file

        decision_model, infer = _vendored(model_dir)
        self.torch, self.infer = torch, infer
        self.model = Model.from_config(Config.from_directory(str(exl3_dir)))
        self.model.load()
        metadata = json.loads((model_dir / "decision_config.json").read_text())
        self.head = decision_model.CandidateHead(4096, metadata["head_dim"]).float().cuda().eval()
        self.head.load_state_dict(load_file(str(model_dir / "decision_head.safetensors")))
        self.tokenizer = _tokenizer(model_dir)
        self.model_dir = model_dir

    def hidden(self, ids: list[int]) -> Any:
        torch = self.torch
        params: dict[str, Any] = {"attn_mode": "flash_attn_nc"}
        x = self.model.prepare_inputs(torch.tensor([ids], dtype=torch.long), params)
        norm = self.model.modules[-2]
        # ExLlamaV3's own forward loop (model_ls.forward_ls), stopped at the final
        # norm: Lux reads normalized hidden states, never logits.
        for module, instance, _ in self.model.fwd_modules:
            params["layer_instance"] = instance
            x = module.prepare_for_device(x, params)
            if module is norm:
                return module.forward(x, params, out_dtype=torch.float32)[0]
            x = module.forward(x, params)
        raise RuntimeError("final norm not reached")

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """System One answers (Lux's own answer shapes) for any noul/choice questions."""
        torch = self.torch
        decision_model, infer = _vendored(self.model_dir)
        metadata = json.loads((self.model_dir / "decision_config.json").read_text())
        encode = decision_model.encoder_for(metadata)
        cap = json.loads((self.model_dir / "MODEL_MANIFEST.json").read_text())["max_input_tokens"]
        item = {"id": "request", "state": state}
        out = {}
        with torch.inference_mode():
            for qid, question in questions.items():
                row = infer.question_to_row(item, qid, question)
                encoded = encode(row, self.tokenizer, cap)
                hidden = self.hidden(encoded["ids"])
                candidates = hidden[torch.tensor(encoded["candidate_positions"], device=hidden.device)]
                query = hidden[encoded["query_position"]]
                logits = self.head(candidates[None], query[None])[0].float().cpu().tolist()
                out[qid] = infer.product_answer(
                    row["task_type"], encoded["keys"], logits, 1.0, [o["description"] for o in row["options"]]
                )
        return out

    def answer(self, state: dict[str, Any], version: int) -> dict[str, float]:
        return {cause: float(a["noul"]) for cause, a in self.ask(state, questions(version)).items()}


class ReferenceBackend:
    """The published runtime, unmodified (CPU, fp32)."""

    def __init__(self, model_dir: Path, threads: int):
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))
        from decision2 import Decision2

        self.model = Decision2.from_pretrained(str(model_dir), device="cpu", threads=threads)

    def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return dict(self.model.system_one(state=state, questions=questions)["answers"])

    def answer(self, state: dict[str, Any], version: int) -> dict[str, float]:
        answers = self.ask(state, questions(version))
        return {cause: float(answers[cause]["noul"]) for cause in CAUSES}


def infer(args: argparse.Namespace) -> None:
    model_dir = Path(args.model).expanduser()
    backend: Any = (
        Exl3Backend(model_dir, Path(args.exl3).expanduser())
        if args.backend == "exl3"
        else ReferenceBackend(model_dir, args.threads)
    )
    label = args.label or (f"exl3:{Path(args.exl3).name}" if args.backend == "exl3" else "reference-cpu-fp32")
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(line)["key"] for line in out.read_text().splitlines() if line.strip()}
    rows = [row for split in args.splits for row in data.load(Path(args.data) / f"{split}.jsonl")]
    if args.limit:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    with out.open("a") as handle:
        for position, row in enumerate(rows):
            if row_key(row) in done:
                continue
            state = row_state(row, args.prompt)
            record = {
                "key": row_key(row),
                "probabilities": backend.answer(state, args.prompt),
                "prompt_version": args.prompt,
                "state_sha256": state_sha256(state),
                "backend": label,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(f"{position + 1}/{len(rows)} {record['key']}", flush=True)


def serve(args: argparse.Namespace) -> None:
    """``POST /v1/systemone`` over the EXL3 backend (loopback only), for ``qc notices``."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    backend = Exl3Backend(Path(args.model).expanduser(), Path(args.exl3).expanduser())

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/systemone":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                answers = backend.ask(body["state"], body["questions"])
                payload, code = {"model": body.get("model", "lux"), "answers": answers}, 200
            except (KeyError, ValueError, TypeError) as exc:
                payload, code = {"error": str(exc)}, 400
            data_out = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data_out)))
            self.end_headers()
            self.wfile.write(data_out)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving /v1/systemone on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("shim")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("caldata")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=250)
    p.add_argument("--cols", type=int, default=2048)
    p = sub.add_parser("infer")
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=("exl3", "reference"), required=True)
    p.add_argument("--exl3")
    p.add_argument("--data", required=True)
    p.add_argument("--splits", nargs="+", default=["dev", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="random sample of this many rows (seed 0)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--label", default=None)
    p.add_argument("--prompt", type=int, choices=(1, 2, 3), default=1, help="prompt version (service.py)")
    p = sub.add_parser("serve")
    p.add_argument("--model", required=True)
    p.add_argument("--exl3", required=True)
    p.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if args.command == "serve":
        serve(args)
        return
    if args.command == "shim":
        shim(Path(args.model).expanduser(), Path(args.out).expanduser())
    elif args.command == "caldata":
        caldata(Path(args.model).expanduser(), Path(args.data), Path(args.out), args.rows, args.cols)
    else:
        infer(args)


if __name__ == "__main__":
    main()
