"""Clairic-flash (Cloudflare, Apache-2.0) on a 16 GB ROCm GPU, and a local Extract service.

    python -m experiments.notices.clairic smoke --model ~/models/clairic-flash [--device cpu]
    python -m experiments.notices.clairic serve --model ~/models/clairic-flash --port 8091

Clairic-flash is an extraction model: a ``state`` (text) and typed ``fields``
in, typed values with confidence, ``p_null`` and character spans out (System
One: Extract). Its published runtime targets NVIDIA (flash-attn3 on sm90,
deep_gemm, flashinfer). Off sm90 it already falls back to SDPA attention, and
its BF16 fast paths (custom matmuls and norms, CUDA graphs) are opt-in hooks;
``load`` keeps every one of them off, so the backbone runs as plain eager
Transformers in BF16, which is the runtime's own non-CUDA path at GPU precision.

The BF16 weights (19.3 GB) do not fit a 16 GB card, so ``load`` keeps the
unused vision tower and the embedding lookup on the CPU and either streams the
last ``offload_layers`` decoder layers to the GPU for each forward pass (BF16)
or quantizes the decoder's linear layers to int8 (``--quant int8``, the mode
the registered runs used; gate G2 in gates.py checks it against BF16). The
model files are read unmodified; ``--device cpu`` runs the unmodified runtime
(fp32).

Needs the runtime's own dependencies (torch, transformers 5.10.2, xgrammar);
never imported by the engine.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PUBLIC_NAME = "clairic-flash"


class _Streamed:
    """Keep a module's weights in pinned CPU memory; copy them in for each forward."""

    def __init__(self, module: Any, device: Any):
        self.device = device
        self.host = {name: p.data.pin_memory() for name, p in module.named_parameters()}
        for name, p in module.named_parameters():
            p.data = self.host[name]
        module.register_forward_pre_hook(self._load)
        module.register_forward_hook(self._unload)

    def _load(self, module: Any, _args: Any) -> None:
        for name, p in module.named_parameters():
            p.data = self.host[name].to(self.device, non_blocking=True)

    def _unload(self, module: Any, _args: Any, _output: Any) -> None:
        for name, p in module.named_parameters():
            p.data = self.host[name]


def _cpu_embedding(embedding: Any, device: Any) -> Any:
    import torch

    class CpuEmbedding(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = embedding.weight  # stays on the CPU
            self.num_embeddings, self.embedding_dim = self.weight.shape
            self.padding_idx = embedding.padding_idx

        def forward(self, ids: Any) -> Any:
            return torch.nn.functional.embedding(ids.cpu(), self.weight).to(device, non_blocking=True)

    return CpuEmbedding()


def load(model_dir: Path, device: str = "cuda", offload_layers: int = 4, quant: str | None = None) -> Any:
    """A ``clairic_model.ClairicModel``. ``device="cpu"`` is the unmodified runtime (fp32).

    ``quant="int8"``: int8 weight-only (torchao) on the decoder layers' linear
    projections, about 7 GB instead of 14, so every layer stays on the GPU and
    ``offload_layers`` is ignored. The LM head (read directly by the typed
    readout), norms, convolutions and the joint head stay BF16/fp32.
    """
    import torch

    path = str(model_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    import clairic_model

    if device == "cpu":
        return clairic_model.load_release_model(model_dir, device="cpu")[0]

    from clairic_runtime import model as cm
    from clairic_runtime import native
    from clairic_runtime.backend import ClairicFlashModel, validate_eos
    from clairic_runtime.readout import load_head
    from clairic_runtime.settings import RuntimeSettings
    from transformers import AutoProcessor, AutoTokenizer

    # The runtime's NVIDIA fast paths are optional hooks; keep them off.
    native.install_math = lambda backbone, head=None: (lambda: None)
    native.prepare_decoder_graphs = lambda backbone, max_batch: None
    native.ReadoutMath = contextlib.nullcontext

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    validate_eos(model_dir, tokenizer)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.tokenizer = tokenizer
    cm.configure_image_processor(processor, 1600000)  # as ClairicFlashModel.load
    if getattr(processor, "image_token_id", None) is None:
        processor.image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)

    # Load on the CPU in BF16, then place the text model by hand.
    engine = native.TransformersEngine.load(
        model_dir, tokenizer, device="cpu", dtype=torch.bfloat16,
        attn_implementation={"text_config": "sdpa", "vision_config": "sdpa"},
    )
    gpu = torch.device(device)
    language = engine.backbone.model.language_model
    layers = list(language.layers)
    keep = len(layers) if quant else len(layers) - offload_layers
    for layer in layers[:keep]:
        layer.to(gpu)
    if quant == "int8":
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        for layer in layers:
            quantize_(layer, Int8WeightOnlyConfig(), filter_fn=lambda m, _: isinstance(m, torch.nn.Linear))
        torch.cuda.empty_cache()
    elif quant is not None:
        raise ValueError(f"unknown quant {quant!r}")
    for layer in layers[keep:]:
        _Streamed(layer, gpu)
    language.norm.to(gpu)
    language.rotary_emb.to(gpu)
    engine.backbone.lm_head.to(gpu)
    language.embed_tokens = _cpu_embedding(language.embed_tokens, gpu)
    # Serving builds the head in BF16 and widens it to fp32 (match_bfloat16).
    engine.head = load_head(model_dir, match_bfloat16=True).to(gpu)
    engine.device, engine.dtype = gpu, torch.bfloat16

    settings = RuntimeSettings(
        model_path=path, version="1", model_id=PUBLIC_NAME, max_length=65_273, max_concurrency=1
    )
    head_config = json.loads((model_dir / "clair_head_config.json").read_text())
    runtime = ClairicFlashModel(engine, tokenizer, processor, settings, head_config)
    return clairic_model.ClairicModel(runtime, PUBLIC_NAME)


SMOKE = {
    "state": "Invoice from Northwind Supply. Issued 2026-09-02. Items: Pens; Paper. Total: $1080.00.",
    "fields": {
        "vendor": {"type": "text", "instructions": "The invoice vendor."},
        "total": {"type": "number", "decimals": 2},
        "issued": {"type": "date"},
        "items": {"type": "list", "items": "text", "instructions": "The purchased item names, in order."},
    },
}


def torch_peak() -> str:
    import torch

    return f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if torch.cuda.is_available() else "-"


def smoke(args: argparse.Namespace) -> None:
    started = time.time()
    model = load(Path(args.model).expanduser(), args.device, args.offload_layers, args.quant)
    print(f"load {time.time() - started:.1f}s", flush=True)
    for _ in range(2):
        started = time.time()
        out = model.predict(SMOKE)
        print(f"predict {time.time() - started:.1f}s", flush=True)
    print(json.dumps({k: v.get("value") for k, v in out["answers"].items()}), out["usage"])
    print(f"peak GPU memory {torch_peak()}")
    model.close()


def serve(args: argparse.Namespace) -> None:
    """``POST /v1/extract`` (System One: Extract body) on loopback, for ``qc notices``."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    model = load(Path(args.model).expanduser(), args.device, args.offload_layers, args.quant)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/v1/extract":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload, code = model.predict(body), 200
            except Exception as exc:  # noqa: BLE001 -- validation and inference errors go back to the client
                payload, code = {"error": f"{type(exc).__name__}: {exc}"}, 400
            data_out = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data_out)))
            self.end_headers()
            self.wfile.write(data_out)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"serving /v1/extract on http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("smoke", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--device", default="cuda")
        p.add_argument("--offload-layers", type=int, default=4)
        p.add_argument("--quant", choices=("int8",), default=None)
        if name == "serve":
            p.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    {"smoke": smoke, "serve": serve}[args.command](args)


if __name__ == "__main__":
    main()
