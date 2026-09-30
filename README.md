# Verity

Automated QC for retail transactional tables refreshed as versioned snapshots.
Given two versions of a table, the engine checks data contracts, explains the
historical revision (which entities appeared, vanished or were reclassified,
and how much of the change that accounts for), reconciles stages, locates the
first pipeline stage that diverged, and tests the newly appended weeks against
forecasts and each entity's own share history. Every check emits a versioned
finding; one final status (`PASS`, `PASS_WITH_EXPLANATION`, `INVESTIGATE`,
`INCOMPLETE`, `DATA_CONTRACT_FAILURE`) is computed from them.

> **Validation status: synthetic only.** All evidence comes from the bundled
> generator (`qcgen`) and its fault oracle. On the registered cohort the engine
> detects 72/72 held-out faults and movements and raises no alarm on 60/60
> clean refreshes; see [docs/claims.md](docs/claims.md) for exactly what that
> does and does not show. Backends: local Parquet snapshots and Delta tables
> via delta-rs.

## Quickstart

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[test,delta]"

# Generate scenarios with ground-truth faults and verify the oracle
.venv/bin/qcgen generate --suite demo --scenarios 13 --profile small --stages all
.venv/bin/qcgen verify --suite-dir data/suites/demo

# Run QC over one version pair (human output, or --json for the machine record)
.venv/bin/qc run --scenario-dir data/suites/demo/scenario-0000
.venv/bin/qc report --scenario-dir data/suites/demo/scenario-0000 --out reports/runs/run-1

# Score the engine against the oracle (blind: the engine never sees ground truth)
.venv/bin/qc shadow --suite-dir data/suites/demo --out reports/shadow/demo

# The registered system-level gate (exit 3 if a gate fails)
.venv/bin/qc cohort --plan config/cohort.json --out reports/cohort/v1

# Point at a Delta table: inspect, propose a config (with field mapping), assess
.venv/bin/qc delta-info --uri ./lake/fact
.venv/bin/qc onboard --uri ./lake/fact --alias pfc=product_id \
  --previous 41 --current 42 --out config/datasets/retail.yaml
.venv/bin/qc delta-run --uri ./lake/fact --previous 41 --current 42 \
  --config config/datasets/retail.yaml

# Scheduler entry point: idempotent, journaled, alertable exit codes
.venv/bin/qc weekly --uri ./lake/fact --config config/datasets/retail.yaml \
  --store data/qc.db --out reports/weekly --notify sinks.json
.venv/bin/qc explain --store data/qc.db --dataset retail

# Tests and gates
.venv/bin/python -m pytest
.venv/bin/ruff check qc qcgen tests optional
.venv/bin/mypy
```

`qc weekly` exits 0 for PASS / PASS_WITH_EXPLANATION, 2 for INVESTIGATE, 3 for
DATA_CONTRACT_FAILURE, 4 for INCOMPLETE, 75 when locked and 1 on error.

## Capabilities

| Capability | Status |
|---|---|
| Synthetic world and fault oracle | validated on synthetic data |
| Data contracts | validated on synthetic data |
| Version pair and revision cube | plumbing |
| Lifecycle and attribution | validated on synthetic data |
| Counterfactual reconstruction | validated on synthetic data |
| Reconciliation | validated on synthetic data |
| Lineage first divergence | validated on synthetic data |
| Expected events, ratio expectations, reference controls | plumbing |
| Temporal QC | validated on synthetic data |
| Findings and final status | validated on synthetic data |
| Recurrence | plumbing |
| Rule cause labels | validated on synthetic data |
| Weekly orchestrator and journal | validated on synthetic data |
| Notification | plumbing |
| Delta source and onboarding | plumbing |
| Cohort evaluation | validated on synthetic data |

"Validated on synthetic data" means measured through the full engine against
the generator's oracle; it is not a real-world accuracy claim.

## Layout

```text
qc/                 engine: contracts, versions, revision, lifecycle, attribution,
                    counterfactual, reconciliation, lineage, temporal, policy,
                    weekly orchestrator, store, notify, shadow/cohort harnesses
qcgen/              synthetic retail world, fault injectors, oracle vault, verifier
config/             cohort plan (hash-pinned) and generator suite config
optional/           scale benchmark
deployment/         weekly.sh scheduler wrapper
data/, reports/     generated output (git-ignored)
```

## Docs

- [docs/engine.md](docs/engine.md): what each layer decides and the final status policy
- [docs/evaluation.md](docs/evaluation.md): cohort gate, shadow scoring, choosing thresholds
- [docs/weekly-run.md](docs/weekly-run.md): weekly runs, journal, recovery, notifications, grain and calendars
- [docs/onboarding.md](docs/onboarding.md): profiling a table and field mapping
- [docs/synthetic-data.md](docs/synthetic-data.md): the generator and fault families
- [docs/claims.md](docs/claims.md): evidence for every capability, and what is not claimed
