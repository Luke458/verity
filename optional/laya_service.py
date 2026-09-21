"""Optional localhost CPU challenger. Run only in the pinned Laya environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MODEL = "convaiinnovations/laya"
REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
LIMIT = 256 * 1024


def hashes(path):
    result = {}
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            h = hashlib.sha256()
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            result[str(file.relative_to(path))] = h.hexdigest()
    return result


def prepare(root):
    from huggingface_hub import snapshot_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_download(
        MODEL,
        revision=REVISION,
        cache_dir=root / "cache",
        allow_patterns=[
            "rl_agent_config.json",
            "model.safetensors",
            "encoder/*",
            "tokenizer/*",
        ],
    )
    target = root / "service-model"
    if target.exists():
        raise ValueError("service-owned copy already exists; choose a new directory")
    shutil.copytree(snapshot, target, symlinks=False)
    (root / "source-manifest.json").write_text(
        json.dumps(
            {"model": MODEL, "revision": REVISION, "hashes": hashes(target)}, indent=2
        )
    )
    return target


def pack(agent, state, questions):
    """Budget against upstream's exact full head, with no option truncation."""
    from laya.common import render_options

    if (
        not isinstance(state, str)
        or not isinstance(questions, dict)
        or not questions
        or len(questions) > 8
    ):
        raise ValueError("invalid state/questions")
    evidence = json.loads(state)
    if not isinstance(evidence, dict) or evidence.get("truncated"):
        raise ValueError("mandatory evidence unavailable or already truncated")
    tok = agent.tok

    def tokenize(text):
        return tok(text.replace(tok.mask_token, " "), add_special_tokens=False)[
            "input_ids"
        ]

    budget = agent.cfg.get("max_len", 512)
    for question in questions.values():
        q = agent._to_internal(question)
        options = render_options(q)
        option_lengths = [len(tokenize(" " + option)) for option in options]
        if len(options) != len(set(options)) or any(n > 48 for n in option_lengths):
            raise ValueError("distinct full option descriptions cannot fit")
        option_tokens = sum(n + 1 for n in option_lengths)
        head = len(tokenize(f"{q['t']} question: {q['ins']}"))
        head_budget = agent.cfg.get("head_max_len", 192)
        if head_budget - option_tokens < max(16, head):
            raise ValueError("full question/options exceed head budget")
        budget = min(budget, agent.cfg.get("max_len", 512) - head - option_tokens - 4)
    mandatory = {
        key: evidence[key]
        for key in ("status", "contracts", "findings", "reference")
        if key in evidence
    }
    if "findings" not in mandatory:
        raise ValueError("versioned findings required")
    mandatory["findings"] = [
        f
        for f in mandatory["findings"]
        if f.get("outcome") in ("FAIL", "CONTRACT_FAILURE")
        or (f.get("required", True) and f.get("outcome") == "UNAVAILABLE")
    ]
    packed = dict(mandatory)
    if len(tokenize(json.dumps(packed, sort_keys=True))) > budget:
        raise ValueError("mandatory evidence exceeds token budget")
    omitted = (
        ["optional_or_pass_findings"]
        if len(mandatory["findings"]) != len(evidence["findings"])
        else []
    )
    for key in (
        "historical_revision",
        "lineage",
        "temporal",
        "events",
        "version_pair",
        "reasons",
    ):
        if key not in evidence:
            continue
        candidate = {**packed, key: evidence[key]}
        if len(tokenize(json.dumps(candidate, sort_keys=True))) <= budget:
            packed = candidate
        else:
            omitted.append(key)
    return json.dumps(packed, sort_keys=True), omitted


def worker(pipe, model_dir):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import laya
    import torch

    torch.set_num_threads(4)
    model_path = Path(model_dir).resolve()
    source_manifest = json.loads(
        (model_path.parent / "source-manifest.json").read_text()
    )
    if (
        source_manifest.get("model") != MODEL
        or source_manifest.get("revision") != REVISION
    ):
        raise ValueError("model source identity mismatch")
    if (
        source_manifest["hashes"].get("model.safetensors")
        != "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
    ):
        raise ValueError("unexpected pinned checkpoint weights")
    effective_file = model_path.parent / "effective-manifest.json"
    expected = (
        json.loads(effective_file.read_text())
        if effective_file.exists()
        else source_manifest
    )
    actual = hashes(model_path)
    if actual != expected["hashes"]:
        raise ValueError("service-owned checkpoint changed since preparation")
    agent = laya.load(str(model_path), device="cpu")
    effective_file.write_text(
        json.dumps(
            {"model": MODEL, "revision": REVISION, "hashes": hashes(model_path)},
            indent=2,
        )
    )
    metadata = {
        "model": MODEL,
        "revision": REVISION,
        "device": "cpu",
        "config": agent.cfg,
        "runtime": {
            name: __import__("importlib.metadata", fromlist=["version"]).version(name)
            for name in ("laya", "torch", "transformers", "tokenizers")
        },
        "effective_hashes": hashes(model_dir),
        "production_eligible": False,
    }
    pipe.send({"metadata": metadata})
    while True:
        payload = pipe.recv()
        try:
            state, omitted = pack(agent, payload["state"], payload["questions"])
            response = agent.predict(state, payload["questions"])
            response.update(
                metadata=metadata, omitted_evidence=omitted, evidence_version=2
            )
            pipe.send(response)
        except Exception as error:  # noqa: BLE001 - worker returns explicit unavailability
            pipe.send({"error": str(error), "abstained": True})


def exchange(parent, process, payload, timeout):
    if not process.is_alive():
        raise RuntimeError("worker unavailable; restart service")
    parent.send(payload)
    if not parent.poll(timeout):
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise TimeoutError("inference deadline exceeded; restart service")
    return parent.recv()


def serve(model_dir, port=8766, timeout=60):
    import math

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=worker, args=(child, model_dir), daemon=True)
    process.start()
    if not parent.poll(timeout):
        process.terminate()
        process.join()
        raise TimeoutError("Laya startup timeout")
    metadata = parent.recv()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def respond(self, code, data):
            body = json.dumps(data, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/v1/models"):
                self.respond(200 if process.is_alive() else 503, metadata)
            else:
                self.respond(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                self.respond(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= LIMIT:
                    raise ValueError("invalid request length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request must be an object")
                if payload.get("model") != MODEL:
                    raise ValueError("only the pinned English model is available")
                answer = exchange(parent, process, payload, timeout)
                self.respond(422 if "error" in answer else 200, answer)
            except (
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                TimeoutError,
                EOFError,
            ) as error:
                self.respond(503, {"error": str(error), "abstained": True})

    try:
        HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    finally:
        process.terminate()
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    if args.prepare:
        print(prepare(args.prepare))
    elif args.model_dir:
        serve(args.model_dir, args.port, args.timeout)
    else:
        parser.error("provide --prepare or --model-dir")
