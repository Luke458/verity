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
> detects 72/72 held-out faults and movements and raises a false alarm on 1 of
> 60 clean refreshes. On a harder profile with category seasonality and
> late-arriving data it catches structural faults and single-week
> restatements down to 1% and single-category drops of 40%, but misses most
> drops of 10% or less, and reports per category the smallest drop it could
> have caught. Known changes (new stores, closures, category moves) are
> explained only by a registry entry a person approved; they can be drafted
> from free-text notices. The [case study](docs/case-study.md) tells how it
> got here and [docs/claims.md](docs/claims.md) records the evidence.
> Backends: local Parquet snapshots and Delta tables via delta-rs.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/detection-curves-dark.svg">
  <img alt="Detection rate against fault size for four fault families on the small and realistic profiles; numbers in docs/evaluation.md" src="docs/img/detection-curves-light.svg">
</picture>

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

# Detection curves over fault size, fault pairs and clean false alarms
.venv/bin/qc sweep --profile realistic --seeds 5001-5010 --jobs 4 \
  --out reports/sweep/realistic.json

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

# Match free-text change notices to observed changes; draft (unapproved) registry entries,
# then rerun with the entries a person approved
.venv/bin/qc notices --scenario-dir data/suites/demo/scenario-0003 --notices inbox.txt --out drafts.json
.venv/bin/qc run --scenario-dir data/suites/demo/scenario-0003 --registry approved.json

# Tests and gates
.venv/bin/python -m pytest
.venv/bin/ruff check qc qcgen tests optional experiments
.venv/bin/mypy
```

`qc weekly` exits 0 for PASS / PASS_WITH_EXPLANATION, 2 for INVESTIGATE, 3 for
DATA_CONTRACT_FAILURE, 4 for INCOMPLETE, 75 when locked and 1 on error.

## Use it in your own pipeline

Verity reads two versions of a weekly fact table and never writes to it. To
put it behind a real refresh:

1. **Make each refresh a readable snapshot.** Either a Delta table (each
   refresh is a table version) or a JSON manifest listing one Parquet file
   per refresh with its `observed_at` time ([weekly-run.md](docs/weekly-run.md)).
   The table needs an integer `week` column; derive one from the date if you
   have to.
2. **Describe the table.** `qc onboard --uri <delta table> --out retail.yaml`
   profiles a Delta table and proposes a config. Map production column names
   with `--alias pfc=product_id`. For Parquet, write the YAML by hand from
   [onboarding.md](docs/onboarding.md). Declare `restatement_weeks` if late
   transactions restate recent weeks, and a calendar if you want holiday
   evidence.
3. **Backtest before trusting it.** Assess past refresh pairs with
   `qc delta-run` (or `qc weekly` on a manifest) and read every INVESTIGATE.
   Check that your config still passes the synthetic gate:
   `qc cohort --plan config/cohort.json --config retail.yaml` (exit 3 means a
   gate failed).
4. **Schedule `qc weekly`** after each refresh commits, with `--store` for the
   journal and `--notify` for alerts, and route its exit code
   (`deployment/weekly.sh` shows one wrapper).
5. **Register known changes.** New stores, closures and category moves are
   explained only by entries in a registry file passed as `--registry`, each
   with `approved_by`, `approved_at` and `confirmed: true`. In `qc weekly` an
   approval counts only if it was recorded before the refresh was committed,
   so register known changes ahead of the refresh
   ([engine.md](docs/engine.md#expected-event-registry)).

### Prompt for an AI coding assistant

Fill in the angle-bracketed parts and give this to a coding agent working in
your repository:

```text
You are integrating Verity (https://github.com/Luke458/verity), an automated QC
engine for weekly retail fact tables refreshed as versioned snapshots, into my
data pipeline. Your job is to wire it up and measure it honestly. Do not change
how it decides a status.

My setup:
- Fact table: <location, e.g. Delta at abfss://..., or Parquet exports in s3://...>
- Columns: <week or date column, store/product/category keys, measures such as sales and units>
- Refresh: <cadence; whether late transactions restate recent weeks, and how many>
- Orchestrator: <Airflow / Databricks Jobs / cron / ...>; alerts go to <Slack webhook / email / ...>
- Python: <version, environment, how packages are installed>

Read first: README.md, docs/onboarding.md, docs/weekly-run.md, docs/engine.md
(especially "Final assessment policy" and "Expected-event registry") and
docs/claims.md (what is validated, only on synthetic data, and what is not claimed).

Steps:
1. Install Verity in an isolated environment (pip install -e ".[delta]" from a
   pinned commit). Run its tests once.
2. Make refreshes readable as snapshots. With Delta, use table versions. With
   anything else, export each refresh to Parquet and maintain the JSON manifest
   described in docs/weekly-run.md (version, observed_at, stage paths). Never
   overwrite an exported snapshot. Ensure an integer, contiguous week column.
3. Create config/datasets/<name>.yaml. With Delta, start from
   `qc onboard --uri ... --alias <production>=<canonical> --out ...` and fix
   every BLOCKER; otherwise write it by hand from docs/onboarding.md. Declare
   the grain (entity_key_columns), restatement_weeks if recent weeks are
   restated, snapshot_metrics for stock-like measures, and a calendar anchor
   and events if holidays matter. Explain every field you set.
4. Backtest on the last <N, e.g. 12> refresh pairs (`qc delta-run --json`, or
   `qc weekly --uri manifest.json --previous A --current B --out ...`). Report a
   table of status per pair, and for each INVESTIGATE list its unexplained
   findings and whether it matches a known incident or known business change.
   Do not edit thresholds to make a pair pass. If you think a setting is
   wrong, show the evidence and the before/after statuses, and ask me.
5. Run `qc cohort --plan config/cohort.json --config <your yaml>`. It must exit
   0 (exit 3 = a gate failed); report the output.
6. Add a scheduled task that runs `qc weekly --uri ... --config ...
   --store <journal.db> --out <reports dir> --registry <registry.json>
   --notify <sinks.json>` after the refresh commits. Pass storage credentials
   through the QC_STORAGE_OPTIONS environment variable, never on the command
   line. Map exit codes: 0 pass, 2 investigate, 3 contract failure,
   4 incomplete, 75 locked (retry later), 1 error.
7. Set up the registry file (format in docs/engine.md). Entries explain known
   changes (new stores with history, removals, closures, category moves) only
   once a person sets approved_by, approved_at and confirmed: true, and in
   qc weekly only if approved before the refresh is committed. Never fill in
   approvals yourself. `qc notices` reads synthetic scenario directories; for
   production, draft entries with qc.notices.draft on a run result, or by
   hand, and leave them unapproved.
8. Write a short runbook: how to read a report (`qc explain --store ...`), how
   to register a known change, what each exit code means, and how to rerun.

Ask me before you change any threshold or default, approve or confirm any
registry entry, write to any production table, or send to a real alert channel.
Deliver: the config, the scheduler task, the registry template, the runbook and
the backtest table with your reading of each INVESTIGATE.
```

To reimplement the engine in another stack instead, treat
[docs/engine.md](docs/engine.md) as the specification and the hash-pinned
`qc cohort` plan as the acceptance test.

## Capabilities

| Capability | Status |
|---|---|
| Synthetic world and fault oracle | validated on synthetic data |
| Data contracts | validated on synthetic data |
| Version pair and revision cube | validated on synthetic data |
| Lifecycle and attribution | validated on synthetic data |
| Counterfactual reconstruction | validated on synthetic data |
| Reconciliation | validated on synthetic data |
| Lineage first divergence | validated on synthetic data |
| Ratio expectations and reference controls | implemented, not measured |
| Approved known changes (backfills, closures, category moves) | validated on synthetic data |
| Temporal QC | validated on synthetic data |
| Findings and final status | validated on synthetic data |
| Rule cause labels | validated on synthetic data |
| Weekly orchestrator and journal | validated on synthetic data |
| Notification | implemented, not measured |
| Delta source and onboarding | implemented, not measured |
| Fault-size sweep and realistic profile | validated on synthetic data |
| Cohort evaluation | validated on synthetic data |
| Notice drafting (`qc notices`) | validated on synthetic notices; drafts need approval |

"Validated on synthetic data" means measured through the full engine against
the generator's oracle; it is not a real-world accuracy claim. "Implemented,
not measured" means built, wired in and tested for correct mechanics, but the
generator cannot produce the conditions to measure it (real endpoints, real
tables, independent references); [docs/claims.md](docs/claims.md) says why
for each.

## Layout

```text
qc/                 engine: contracts, versions, revision, lifecycle, attribution,
                    counterfactual, reconciliation, lineage, temporal, policy,
                    weekly orchestrator, store, notify, shadow/cohort harnesses
qcgen/              synthetic retail world, fault injectors, oracle vault, verifier
config/             cohort plan (hash-pinned) and generator suite config
optional/           scale benchmark, sweep chart renderer
experiments/        research and measurement sandboxes (cause labellers, notice matching,
                    sensitivity calibration, approvals); never on the status path
deployment/         weekly.sh scheduler wrapper
data/, reports/     generated output (git-ignored)
```

## Docs

- [docs/case-study.md](docs/case-study.md): how the engine went from alarming on everything to measured claims, and what it still cannot do
- [docs/engine.md](docs/engine.md): what each layer decides and the final status policy
- [docs/evaluation.md](docs/evaluation.md): cohort gate, sweeps, false alarms, detection limits, approvals; each with its registered predictions
- [docs/weekly-run.md](docs/weekly-run.md): weekly runs, journal, recovery, notifications, grain and calendars
- [docs/onboarding.md](docs/onboarding.md): profiling a table and field mapping
- [docs/synthetic-data.md](docs/synthetic-data.md): the generator and fault families
- [docs/claims.md](docs/claims.md): evidence for every capability, and what is not claimed
- [docs/labeller-experiment.md](docs/labeller-experiment.md): learned and zero-shot (Decision-2.0-Lux-9B) cause labellers against the rule baseline
- [docs/notice-matching.md](docs/notice-matching.md): matching free-text change notices to flagged changes with a decision model
