# ModernBERT text probe (frozen-encoder decision provider)

ModernBERT fits this project in two places, one of which is implemented.

## 1. Frozen-encoder probe over the evidence text (implemented)

`qc/text_provider.py` embeds the canonical evidence text with a frozen
ModernBERT-class encoder and trains a small linear head per decision field over
those embeddings. This is a third substrate beside the evidence-feature head
(`qc/training.py`) and a remote Jev-compatible provider (`qc/systemone.py`).

Why it fits:

- ModernBERT's 8192-token context holds the bounded evidence text
  (`qc/evidence_text.py`, hard-capped) without truncation;
- an encoder is one forward pass per case — no generation, CPU-friendly for a
  weekly run;
- pretrained representations should beat handcrafted features when labels are
  few, which is exactly the real-data situation we are heading into;
- a linear probe is cheap to refit as analyst labels accumulate.

```sh
# Labels carry the evidence text (they do not contain decision outputs).
qc shadow --suite-dir data/suites/demo --labels-out data/labels/demo.jsonl

# Train the probe on the real checkpoint.
qc train --labels data/labels/demo.jsonl \
  --text-embedder answerdotai/ModernBERT-base \
  --out reports/artifacts/text-probe-demo

# Use it.
qc run --scenario-dir data/suites/demo/scenario-0000 \
  --provider reports/artifacts/text-probe-demo
```

Real smoke: `answerdotai/ModernBERT-base` loads and embeds on CPU in ~16 s
(768 dimensions, L2-normalized mean pooling). A probe trained on 13 synthetic
records scored 1.000 on train and 0.625 on 4 held-out records — plumbing
evidence, not accuracy evidence.

Properties and limits:

- the evidence text deliberately excludes decision outputs and oracle labels,
  so the probe cannot read the answer from its input (`EVIDENCE_TEXT_VERSION`
  is stored in the artifact and checked on load);
- probabilities are tagged `frozen_encoder_probe`; the fitted temperature is
  not a calibration certificate;
- the artifact records the embedder id/revision, so a checkpoint change is
  visible rather than silent;
- accuracy claims still require the frozen cohort and real analyst labels:
  run it as a challenger against the rule provider, the feature head, and a
  Jev-compatible endpoint on the same held-out cases.

## 2. Analyst-note extraction and verification (candidate)

- ModernBERT-NER / GLiNER-NER checkpoints could turn analyst resolutions and
  free-text incident notes into structured tags and entities for incident
  memory (the structured path already exists; this is enrichment).
- An NLI cross-encoder (`openjev`-style, or a ModernBERT cross-encoder) could
  verify a proposed agent root cause against the cited findings before an
  outcome is accepted.

Neither is on the critical path: incidents are currently structured records
written by analysts, and outcome confirmation is a human decision.

## Where it does not fit

ModernBERT is not a replacement for the deterministic layer, and not a default
provider. It reads the evidence graph, never raw tables, and it cannot do
arithmetic or establish facts. Treat it as a measured challenger in the
semantic bake-off — the same discipline applied to TSPulse and MiniCPM.
