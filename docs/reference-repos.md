# Reference projects: what we reused, what we did not

This project did not start from zero. This document records what was taken from
the reference implementations, what was deliberately left alone, and what is
queued next. The guiding rule: adopt interfaces and harnesses, not marketing
claims.

## Verity (our own predecessor)

`~/Projects/verity` implements the same architecture spec with a DuckDB/Delta
core and a separate model environment. It is ahead of retail-qc in several
areas and behind in others (retail-qc has typed decisions, training, incident
memory and the agent handoff; verity has TSPulse, replay and evaluation rigor).

Adopted now:

- **TSPulse adapter gates** - `qc/tspulse.py` mirrors verity's model id, pinned
  embedding revision (`tspulse-hybrid-dualhead-512-p8-r1`), 512-point
  embedding gate, daily/2048-point anomaly gate and explicit resample
  disclosure. Dimension validation (240) was added so a silent model change
  fails loudly.
- **Incident retrieval blending** - `IncidentStore.retrieve` accepts an optional
  TSPulse embedding and blends cosine features with cosine embeddings, the same
  idea as verity's `features_and_tspulse`.

Queued from verity, in priority order:

1. **Finite-sample conformal intervals** for latest-week and cross-metric
   evidence, replacing the current normal-residual percentile with
   `ceil((n+1)(1-alpha))` ranked errors and an explicit
   `INSUFFICIENT_CALIBRATION` status. Verity's own results show why nominal
   quantiles cannot be trusted at small n.
2. **Prequential (across-load) calibration**: forecast each new week from the
   previous pinned snapshot, and only admit residuals whose `available_on <
   as_of`. This is the leakage-safe way to calibrate on real refreshes; our
   current calibration is within-version.
3. **Frozen cohort evaluation**: disjoint development and held-out seeds,
   pre-registered gates (max FPR, min recall, max errors), manifest and code
   SHA-256, per-case JSONL, seed-cluster bootstrap. Our `qc shadow` scores a
   suite but has no promotion gate.
4. **Confirmed-only memory**: append-only analyst outcomes, revisioned
   registries, latest-revision reads, and retrieval that never returns model
   guesses. Our JSONL stores are simpler; the replacement is a SQLite-backed
   store with the same interface.
5. **Scoped approvals and expectations**: an approval explains exactly the
   alert it names and clears nothing else, with adversarial scope tests
   (wrong dataset, wrong week, expired, invalid approval).
6. **Bounded query contract**: `LIMIT n+1` with a `QueryLimitError` instead of
   silent truncation, plus DuckDB/Spark parity tests for the revision cube.
7. **Replay harness** with strictly increasing versions/dates and an explicit
   statement that `as_of` is a caller assertion, not authenticated.

Adopt with caution: verity's own gate results (`docs/status.md`,
`docs/validation/`) show Chronos at 42.5% FPR and ratio checks at 46.7% FPR on
its held-out panel, and TSPulse/semantic champions explicitly unqualified. The
harnesses and guards are the reusable asset, not the promotions.

## djev-spark / TypeSafe Jev

`mmastrac/djev-spark` serves `POST /v1/systemone` on a DGX Spark using
DiffusionGemma structured reads; TypeSafe's hosted Jev exposes the same shape
with `noul` / `choice` / `score` primitives. Both are decision servers, not
pipelines.

Adopted: `qc/systemone.py`.

- `build_evidence_state` compresses a run into a bounded, row-free JSON state
  (contracts, historical revision, lifecycle events, lineage, temporal flags,
  reasons).
- `SystemOneDecisionProvider` posts typed questions and maps answers back onto
  our fields. All three primitives are supported: `noul` -> boolean, `choice`
  -> choice, `score` -> ordered severity with a weighted `index` and legend.
  A plain `choice` answer for an ordinal field is accepted as a compatibility
  fallback. Reported probabilities are tagged `provider_reported`; they are
  never presented as calibrated.
- `decision_to_systemone_answers` emits the same wire format, so our decision
  set round-trips through a Jev-compatible endpoint (`answers_to_decisions` is
  the inverse).
- `FallbackDecisionProvider` uses local rules and escalates to a remote
  provider when investigation is required and local confidence is low.
- CLI: `--provider systemone:http://host:8011/v1/systemone` or
  `--provider hybrid:http://host:8011/v1/systemone`; the hosted API accepts an
  API key via the provider constructor.

So the djev container or hosted Jev can act as the semantic layer behind our
deterministic engine today, and as a benchmark comparator once real labels
exist. The vLLM patches and diffusion-read machinery stay in the serving layer
where they belong.

## CUA (trycua/cua)

CUA provides computer-use drivers, cloud desktops and benchmarks. Nothing in
its code is reusable here directly.

What it changes is the *far end* of the investigation loop: an RCA agent that
must operate Databricks notebooks, a BI tool or a warehouse UI needs CUA-style
execution infrastructure. Our `CommandAgent` already hands a structured brief
to any command and validates the JSON that comes back, so a CUA-backed agent
can be plugged in without changing the engine. CUA's benchmark-as-product
emphasis also matches the direction of our shadow/frozen-cohort harnesses.

## openjev

A Qwen3.5 cross-encoder trained on entailment/contradiction/neutral, plus
per-task heads on frozen latents. Two candidate uses, neither on the critical
path:

1. **Agent-output verification**: score whether a proposed root cause is
   entailed by the cited findings before accepting an investigation result.
2. **Trained decision provider**: an NLI-style substrate for
   `likely_cause`/`origin` once real labels exist, replacing the linear head.

Recorded as a Milestone D challenger; no code adopted yet.

## GLiNER2.5

Schema-driven extraction (entities, classification, records, relations) with
constrained label constraints. The natural fit is turning analyst notes and
incident resolutions into structured incident fields and symptom tags, and
extracting entity mentions from free-text evidence. Not required while
incidents are analyst-authored structured records.

## RLCD

Parallel constrained decoding over a broadcast KV cache on Apple Silicon.
Relevant only if we return to scoring decisions from token logits; we chose
evidence features and typed providers instead, and djev already occupies the
local serving niche. Not adopted.

## Jev (hosted)

Closed, calibrated System One models. Reachable through the same
`SystemOneDecisionProvider`. Use as an upper-bound comparator on a frozen
cohort of real cases; do not model its internals.

## Summary

| Reference | What we take | Status |
|---|---|---|
| Verity | TSPulse gates, embedding retrieval blend; conformal/prequential/cohort harnesses queued | partly adopted |
| djev-spark / Jev | `/v1/systemone` remote decision provider + fallback | adopted |
| CUA | execution layer for the investigation agent; benchmark culture | conceptual |
| openjev | NLI verifier / trained provider substrate | candidate |
| GLiNER2.5 | note and incident extraction | candidate |
| RLCD | parallel constrained decoding | not adopted |
