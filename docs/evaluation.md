# Evaluation, calibration and evidence queries

This document covers the evaluation rigor layer: frozen cohorts with
pre-registered gates, finite-sample conformal intervals, prequential
(across-load) calibration, and the bounded evidence-query tool used by the
investigation agent.

## Frozen cohorts

```sh
qc cohort --plan config/cohort.json --out reports/cohort/v1
qc cohort --heldout-seed 7101 --heldout-seed 7102 --dev-seed 1101 --out reports/cohort/v1
```

A `CohortPlan` declares development seeds, held-out seeds (disjoint, enforced),
families, controls and gates *before* running. Every case is built with the
fault oracle and scored by the full engine. The result records:

- held-out metrics: detection rate, historical false-positive rate, contract
  failure rate, mean reconstruction score, lineage first-divergence accuracy
  (controls are excluded from detection);
- development metrics for reference, never used for gates;
- per-gate actual vs threshold with pass/fail;
- `code_sha256` over the engine sources and `plan_sha256` over the plan, so a
  result can be tied to an exact revision and plan;
- `cases.jsonl`, one record per case.

`production_eligible` is always `False` for synthetic cohorts. The harness
exists so a real-label cohort can be run and judged honestly: changing the plan
after seeing results invalidates the evaluation, and generator bias is not
measured by disjoint seeds alone.

## Conformal intervals

`qc.conformal.conformal_interval` implements the finite-sample interval using
the `ceil((n + 1) * (1 - alpha))`-th smallest absolute residual. With too few
residuals the result is `INSUFFICIENT_CALIBRATION`, never a widened interval.
`leave_one_out_coverage` reports honest small-sample coverage: in-sample
coverage is trivially at least `1 - alpha` once the interval exists, so
leave-one-out is the meaningful check.

## Prequential (across-load) calibration

```sh
qc shadow --suite-dir data/suites/demo --prequential-store data/calibration.jsonl
qc prequential --store data/calibration.jsonl --as-of 104 --target-week 105 --z 2.4
```

`CalibrationPool` only admits residuals where `available_on < as_of` and
`target_week < target_week`; all calibration revisions are retained; historical queries choose the latest
revision available at their cutoff for each scope/series/target. The selected
pool is bounded. This legacy API uses numeric periods; weekly SQLite records
also carry separate observation timestamps. `percentile` and `interval` return `None` /
`INSUFFICIENT_CALIBRATION` below `min_samples`. Shadow runs append one record
per temporal series using the standardized residual that was computed before
the target was scored.

## Bounded evidence queries

```sh
qc evidence --scenario-dir data/suites/demo/scenario-0000 --query lifecycle_changes
qc evidence --scenario-dir ... --query contributors --limit 5 --json
```

`query_evidence` is an allowlisted, read-only view over a completed run:
`lifecycle_changes`, `historical_residuals`, `temporal_anomalies`,
`first_divergence`, `contract_failures`, `contributors`, `relationships`.
Unknown names raise. Every result reports `total_rows` and `truncated`, so a
consumer can tell when it is seeing a limited answer. The investigation brief
includes `tool_calls` pairing the likely cause with the first queries to run.

## Evidence benchmark, thresholds and ablation

```sh
qc evidence-bench --suite data/suites/demo \
  --thresholds 0.0005,0.001,0.002 --out reports/bench/evidence.json
```

`run_evidence_bench` runs one frozen configuration per materiality candidate
over pre-registered scenarios and scores three decision layers separately:
raw challenger recommendations, deterministic verifier decisions and effective
operational decisions. A policy override is never counted as a correct model
prediction. Qualification keeps the detection/false-positive confidence-bound
gates and adds the 95% upper confidence bound on false clearance of actionable
incidents; with too few independent incidents the bound reports
`INSUFFICIENT_EVIDENCE` rather than passing. Selection chooses the candidate
that minimizes review volume while clearing no actionable incident and passing
the bound.

The evidence ablation compares review volume and false clearance under four
evidence levels: `summary` (historical revision and contracts), `forecast`
(adds temporal screening), `hierarchy` (adds hierarchy checks and unsupported
ledger movement) and `complete` (the full policy with certificates). This shows
which evidence layer changes the operational decision rather than claiming one
aggregate model score.

## What remains

The harness is ready; the data is not. The next step is to run a cohort over
real refresh pairs with analyst outcomes, register the plan and gates, and then
retrain and recalibrate. Once confirmed outcomes exist, `qc champion` decides
which substrate runs in production; see [docs/champion.md](champion.md). Until
that happens, all semantic and threshold claims remain plumbing-validated only.

## Store-backed evaluation

Synthetic mode remains the default. Persist real shadow runs with `qc weekly`
before importing outcomes. Import dry runs use the same validation as writes.
Confirmed real analyst outcomes need explicit provenance, analyst identity,
review label and incident grouping. Never relabel synthetic outcomes as analyst.

```sh
qc cohort --source store --store data/qc.db \
  --cutoff 2026-09-01T00:00:00Z --out reports/real-cohort
qc train --cohort reports/real-cohort/store-cohort.json --out reports/artifacts/challenger
qc cohort --source store --store data/qc.db \
  --cutoff 2026-09-01T00:00:00Z --out reports/real-cohort \
  --provider reports/artifacts/challenger --split development
qc cohort --source store --store data/qc.db \
  --cutoff 2026-09-01T00:00:00Z --out reports/real-cohort \
  --provider reports/artifacts/challenger --split test \
  --selection reports/real-cohort/development-evaluation.json
```

The immutable manifest pins snapshot/run identities, evidence, features, text,
latest outcome revisions available at the observation cutoff, provenance and
split assignments. Repeated current snapshots and related incidents stay
together. Real refresh splits are chronological; interleaved groups or fewer
than four independent groups yield INSUFFICIENT_EVIDENCE, never row splitting.
Unknown incident grouping conservatively keeps a dataset together.

Feature heads, text probes and HTTP/Laya challengers consume recorded evidence.
Trainers use separate train, calibration, development and test groups. Select
one challenger on development data; final test requires its hash-verified,
passing development selection. Evaluation reports detection/review separately
from cause and severity, per-class results, UNKNOWN/abstention counts, Brier
score, log loss, reliability bins/ECE, latency, RSS and review volume. Gates
require a detection 95% lower bound >=0.90 and FPR upper bound <=0.10, complete
classes/controls and incident-grouped paired noninferiority. Confidence
intervals count independent incidents, not repeated snapshot rows. Missing evidence
cannot pass. PASS_WITH_EXPLANATION is not actionable detection.

`production_eligibility` centralizes a pinned real analyst test evaluation plus
contracts, historical isolation, migration, crash recovery, read-only operation
and provider-boundary checks. Trainers and all synthetic artifacts remain
ineligible. Pilot outputs separately report label prerequisites, engineering
readiness, evaluation completion and eligibility. Freeze data before model
selection and keep final test labels away from development operators; local
artifacts provide reproducibility, not access-control enforcement.

Rules can also be evaluated with `--provider rule` using their recorded rule
evidence. A remote provider used in durable/evaluation work must declare its
immutable model artifact, rather than identify itself only by a mutable URL:

```json
{"schema_version":1,"url":"http://127.0.0.1:8011/v1/systemone","model":"retail-v1","artifact":{"revision":"immutable-revision","weights_sha256":"<64 hex characters>"}}
```

Pass this file as `--provider systemone-config:remote.json`. The provider
operator is responsible for serving the declared revision; unlike the local
Laya wrapper, a remote service cannot be verified by hashing its local weights.

Training from `--cohort` preserves its exact split assignments and defers test
metrics until selection. Learned challengers must carry the same cohort hash
and disjoint fit-group provenance; arbitrary pre-trained artifacts cannot
silently enter a real final-test evaluation.
