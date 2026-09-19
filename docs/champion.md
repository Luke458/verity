# Provider champion selection

`qc champion` answers one question with pre-registered gates: **which decision
substrate should run in production?** It trains the learned substrates on a
label set, scores every available provider on held-out cases, and selects a
champion only if one passes.

## Providers

| Provider | What it is | Requirements |
|---|---|---|
| `rule` | deterministic evidence rules with heuristic probabilities | always available for runs |
| `feature_head` | linear head over the versioned numeric feature encoder | labels with feature vectors |
| `text_probe` | linear probe over frozen encoder embeddings of the evidence text | labels with evidence text + `[text]` extra |
| `remote` | Jev-compatible `/v1/systemone` provider | endpoint and (optionally) API key |

## Flow

```sh
# 1. Collect labels.
#    Synthetic: shadow writes oracle labels (plumbing only).
qc shadow --suite-dir data/suites/demo --labels-out data/labels/demo.jsonl
#    Real: confirmed analyst outcomes from the durable store.
qc store add-run --store data/qc.db --scenario-dir ... --root-cause ... --confirmed
#    qc labels are then read directly from the store with --store.

# 2. Bake off on a held-out suite that shares no scenario with training.
qc champion --store data/qc.db \
  --suite-dir data/suites/champion-eval \
  --text-embedder answerdotai/ModernBERT-base \
  --systemone-url http://localhost:8011/v1/systemone \
  --out reports/artifacts/champion-v1
```

The report (`champion.json`, `champion.md`) contains per-provider overall and
cause accuracy, per-field breakdowns (JSON), the pre-registered gates, the
training/evaluation provenance, and the code hash.

## Rules that make the result meaningful

- **No overlap**: training and evaluation cases are keyed by `suite:scenario`
  (or `run_id` for store records); any intersection raises instead of silently
  leaking.
- **Pre-registered gates**: `min_overall_accuracy` and `min_cause_accuracy`
  are chosen before seeing results. Changing them afterwards invalidates the
  comparison.
- **Production eligibility**: a champion is `production_eligible` only when
  every training label came from confirmed analyst outcomes. Oracle-trained
  champions are plumbing validation.
- **Fail closed**: an unavailable or failing provider (no embedder, endpoint
  error, too few labels) is reported as unavailable with a note; it is never
  assumed to have succeeded.

## Reading the result

On synthetic data the rule provider is usually the strongest on cause because
the deterministic semantics encode the fault families directly; feature and
text probes are expected to close the gap only with labels that contain genuine
ambiguity. That is exactly the comparison the real-label cohort must settle:
collect confirmed outcomes, freeze a held-out cohort, run `qc champion`, and
let the gates decide. See [docs/evaluation.md](evaluation.md) for the cohort
harness and [docs/operations.md](operations.md) for the feedback loop.

### Stand-in while real analysts are unavailable

`qc simulate-analyst` writes synthetic outcomes into a store with provenance
`synthetic`, including drafts, mistakes, unknowns and corrections. This runs
every other part of the loop (store, confirmed-only labels, training, gates)
and deliberately keeps `production_eligible` False. See
[docs/synthetic-analyst.md](synthetic-analyst.md).
