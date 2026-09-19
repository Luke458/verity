# retail-qc

An automated QC and root-cause-analysis system for retail transactional data
refreshed as versioned table snapshots. The architecture is a deterministic
evidence system first and a semantic decision system last; the full
specification is in [docs/architecture.md](docs/architecture.md).

> **Validation status: synthetic only.** Every number currently produced by
> this repository is measured against the synthetic generator's own oracle or
> is plumbing. There is no real-data evidence for detection, precision, or
> calibration. See [docs/claims.md](docs/claims.md) for the per-milestone
> evidence matrix. The supported backend is local versioned snapshots
> (Parquet scenarios, delta-rs `DeltaSource`); Databricks/Spark execution is
> not implemented.

## Status: Milestones A-D (synthetic validation only)

The repository contains the synthetic retail world generator and fault oracle
(`qcgen`) and the QC engine (`qc`): data contracts, version resolution, the
revision cube, entity lifecycle, attribution, counterfactual reconstruction,
reconciliation, pipeline first-divergence, the expected-event registry,
latest-week temporal intelligence (forecasting, robust statistics, calibration),
a shadow-mode harness, typed semantic decisions, incident memory, a validated
investigation handoff to a reasoning agent, a research-only TSPulse adapter
with a suitability benchmark, entity relationship candidates, Markdown reports,
a delta-rs `DeltaSource` for assessing datalake table versions locally, and an
evaluation layer: frozen dev/held-out cohorts with pre-registered gates,
finite-sample conformal intervals, prequential across-load calibration and
bounded evidence queries for the investigation agent, plus a confirmed-only
SQLite store and drift monitoring for operations. A Jev-compatible remote
decision provider lets a local `djev-spark` container or hosted Jev act as the
semantic layer behind the deterministic engine, and a frozen-encoder text
probe (ModernBERT-class) adds a third decision substrate. Delta onboarding and
outcome import exist but have never been run against a real production table.
A synthetic analyst simulator exercises the feedback loop with honest
provenance until real analyst outcomes exist. Verity's spine is ported:
historical replay, scoped ratio expectations, independent reference controls,
a bounded RCA loop and prequential point-in-time forecasting. `qc weekly` is
the single scheduler entry point for a refresh (idempotent per version,
alertable exit codes); its idempotency and atomicity are being hardened.

"Implemented" below means the code path exists and is exercised on synthetic
data. It does not mean validated. [docs/claims.md](docs/claims.md) records the
evidence status and known weaknesses per milestone.

Synthetic data with injected faults is a valid test oracle for the
deterministic and temporal layers and validates the semantic-layer plumbing. It
is **not** evidence for semantic-model accuracy; that gate requires real
analyst labels, which the label store and training pipeline are built to
consume.

## Quickstart

New here? [docs/quickstart.md](docs/quickstart.md) has a ten-minute synthetic
walkthrough and the production-table path with **field mapping** (for example
`pfc` -> `product_id`).

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[test]"
# optional extras: ".[forecast]" (Chronos), ".[classical]" (SARIMAX),
# ".[tspulse]" (TSPulse), ".[delta]" (Delta table versions)

# Generate a suite with ground-truth faults (all pipeline stages)
.venv/bin/qcgen generate --suite demo --scenarios 13 --profile small --stages all
.venv/bin/qcgen verify --suite-dir data/suites/demo

# Run QC over a version pair (rule decisions by default)
.venv/bin/qc run --scenario-dir data/suites/demo/scenario-0012
.venv/bin/qc decide --scenario-dir data/suites/demo/scenario-0012 --json

# Score the engine against the oracle and collect labels (blind by default:
# the engine sees no ground truth; the oracle lives in a sibling vault)
.venv/bin/qc shadow --suite-dir data/suites/demo --out reports/shadow/demo \
  --labels-out data/labels/demo.jsonl

# Train and use a learned decision provider (synthetic labels: plumbing only)
.venv/bin/qc train --labels data/labels/demo.jsonl --out reports/artifacts/decision-demo

# Or a frozen-encoder text probe (optional [text] extra; ModernBERT-class)
.venv/bin/qc train --labels data/labels/demo.jsonl \
  --text-embedder answerdotai/ModernBERT-base --out reports/artifacts/text-probe-demo

# Hand a case to an investigation agent (any command returning JSON on stdout)
.venv/bin/qc investigate --scenario-dir data/suites/demo/scenario-0000 \
  --agent-cmd "my-llm-agent --json"

# Escalate semantic decisions to a Jev-compatible server (djev-spark or hosted)
.venv/bin/qc decide --scenario-dir data/suites/demo/scenario-0000 \
  --provider systemone:http://localhost:8011/v1/systemone

# TSPulse research benchmark over revision series (research-only, opt-in extra)
.venv/bin/qc tspulse-bench --suite-dir data/suites/demo --out reports/tspulse/demo.json

# Human + machine report for a run
.venv/bin/qc report --scenario-dir data/suites/demo/scenario-0000 --out reports/runs/run-1

# Assess and run against Delta table versions (no Databricks compute needed)
.venv/bin/qc delta-info --uri ./lake/fact
.venv/bin/qc delta-run --uri ./lake/fact --previous 11 --current 12

# Frozen cohort evaluation with pre-registered gates
.venv/bin/qc cohort --out reports/cohort/v1

# Bounded evidence queries for the investigation agent
.venv/bin/qc evidence --scenario-dir data/suites/demo/scenario-0000 \
  --query lifecycle_changes

# Across-load calibration pool
.venv/bin/qc shadow --suite-dir data/suites/demo --prequential-store data/calibration.jsonl
.venv/bin/qc prequential --store data/calibration.jsonl --as-of 104 --target-week 105 --z 2.4
.venv/bin/qc drift --store data/calibration.jsonl --as-of 105 --target-week 105

# Durable confirmed-only store
.venv/bin/qc store add-run --store data/qc.db --scenario-dir data/suites/demo/scenario-0000 \
  --root-cause MISSING_STORES --confirmed
.venv/bin/qc store incidents --store data/qc.db --scenario-dir data/suites/demo/scenario-0000

# Bake off decision providers and select a champion
.venv/bin/qc champion --store data/qc.db --suite-dir data/suites/champion-eval \
  --text-embedder answerdotai/ModernBERT-base --out reports/artifacts/champion-v1

# Onboard a real Delta table: profile, propose config, assess a version pair
.venv/bin/qc onboard --uri ./lake/fact --previous 41 --current 42 \
  --out config/datasets/retail.yaml
.venv/bin/qc store import --store data/qc.db --csv outcomes.csv --dry-run

# Simulate analyst feedback (provenance: synthetic; never production-eligible)
.venv/bin/qc simulate-analyst --suite-dir data/suites/demo \
  --store data/analyst.db --profile typical --seed 7

# Replay pinned version pairs with leakage guards
.venv/bin/qc replay --plan replay.json --out reports/replay/v1

# Bounded RCA investigation over a run
.venv/bin/qc rca --scenario-dir data/suites/demo/scenario-0000 --steps 3

# Point-in-time forecasting across consecutive Delta versions
.venv/bin/qc prequential-forecast --uri ./lake/fact --versions 40,41,42 \
  --min-samples 9 --out data/calibration.jsonl

# Weekly entry point for a scheduler (idempotent; exit 2 = INVESTIGATE)
.venv/bin/qc weekly --uri ./lake/fact --store data/qc.db \
  --calibration-store data/calibration.jsonl --out reports/weekly

# Tests
.venv/bin/python -m pytest
```

## Layout

```text
config/datasets/   dataset YAML configs (profiles and overrides)
qc/                deterministic engine (contracts, versions, revision,
                   lifecycle, attribution, run orchestration)
qcgen/             synthetic universe, DGP, fault injectors, oracle, CLI
data/suites/       generated scenarios (ignored by git)
docs/              architecture.md, engine.md, synthetic-data.md
```

## Milestones

"Implemented" means the code path exists and runs on synthetic data; it is not
a validation claim. See [docs/claims.md](docs/claims.md) for the evidence
status and known weaknesses of each row.

| Milestone | Scope | Status |
|---|---|---|
| A0 | Synthetic world, fault oracle, suite harness | implemented; synthetic validation |
| A | Data contracts, version pair, revision cube, lifecycle, attribution | implemented; synthetic semantics validated |
| B | Counterfactual, reconciliation, lineage, expected events, shadow mode | implemented; weak checks replaced (M2) |
| C | Chronos-2, robust statistics, forecast calibration, evidence graph | implemented; synthetic only |
| 11 | TSPulse research adapter + revision-series suitability benchmark | research; weekly-length gate unresolved |
| D | Typed decisions, labels, training, incident memory, agent handoff | implemented; real-label accuracy pending |
| C+ | Hierarchical reconciliation, entity relationships, Markdown reports, Delta source | implemented; reconciliation and relationships hardened |
| Evaluation | Frozen cohorts, conformal intervals, prequential calibration, bounded evidence queries | implemented; gate enforcement and statistical fixes pending |
| Operations | Confirmed-only SQLite store, revisioned registry, drift monitoring | implemented; provenance and atomicity fixes pending |
| Substrates | Frozen-encoder text probe (ModernBERT-class) | research; real-label bake-off pending |
| Champion | Provider bake-off with pre-registered gates and leakage checks | implemented; real-label selection pending |
| Onboarding | Delta profiling, config proposal, readiness assessment, outcome import, production field mapping | implemented; never run on a real table |
| Feedback simulation | Synthetic analyst outcomes with drafts, mistakes, corrections, provenance gates | research |
| Verity spine | Replay, scoped expectations, reference controls, bounded RCA loop, prequential point-in-time forecasting | ported; Spark/Databricks not implemented |
| Weekly run | `qc weekly` orchestrator, idempotent per version, alertable exit codes, cron script | implemented; locking/atomicity pending |
| Integration | Jev/djev-compatible remote decision provider + fallback | implemented; protocol validation pending |

See [docs/engine.md](docs/engine.md) for engine semantics and status codes,
[docs/semantic-layer.md](docs/semantic-layer.md) for decisions, training and
the agent handoff, [docs/evaluation.md](docs/evaluation.md) for cohorts,
calibration and evidence queries, [docs/operations.md](docs/operations.md) for
the store and drift monitoring, [docs/text-provider.md](docs/text-provider.md)
for the ModernBERT-class text probe, [docs/champion.md](docs/champion.md) for
provider selection, [docs/onboarding.md](docs/onboarding.md) for the first real
table, [docs/synthetic-analyst.md](docs/synthetic-analyst.md) for simulated
feedback, [docs/verity-spine.md](docs/verity-spine.md) for the ported replay,
expectations, reference and RCA pieces, [docs/weekly-run.md](docs/weekly-run.md)
for the scheduler entry point,
[docs/reference-repos.md](docs/reference-repos.md) for what was reused from
sibling projects, and [docs/synthetic-data.md](docs/synthetic-data.md) for the
fault oracle.
