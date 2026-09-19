# Quickstart

Two tracks: ten minutes on synthetic data, then the same stack on a real table
with production field names mapped to canonical ones.

## Track A: synthetic data, ten minutes

```sh
git clone <repo> && cd retail-qc
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[test]"

# 1. Build a synthetic world with ground-truth faults and verify the oracle
.venv/bin/qcgen generate --suite demo --scenarios 13 --profile small --stages all
.venv/bin/qcgen verify --suite-dir data/suites/demo

# 2. Run one case: contracts, revision, lifecycle, temporal, decisions
.venv/bin/qc run --scenario-dir data/suites/demo/scenario-0000

# 3. Human + machine report
.venv/bin/qc report --scenario-dir data/suites/demo/scenario-0000 \
  --out reports/runs/run-1

# 4. Durable store, incident retrieval, developer agent handoff
.venv/bin/qc store add-run --store data/qc.db \
  --scenario-dir data/suites/demo/scenario-0000 \
  --root-cause MISSING_STORES --confirmed
.venv/bin/qc store incidents --store data/qc.db \
  --scenario-dir data/suites/demo/scenario-0000
.venv/bin/qc rca --scenario-dir data/suites/demo/scenario-0000 --steps 3

# 5. Feedback loop with simulated analysts, then a provider bake-off
.venv/bin/qc simulate-analyst --suite-dir data/suites/demo \
  --store data/analyst.db --profile typical --seed 7
.venv/bin/qc champion --store data/analyst.db --suite-dir data/suites/champion-eval \
  --out reports/artifacts/champion-demo

# 6. Whole-suite scoring and today's version of TSPulse research
.venv/bin/qc shadow --suite-dir data/suites/demo --out reports/shadow/demo
.venv/bin/qc cohort --out reports/cohort/v1
```

Orchestration note: Track A runs on Parquet-backed scenario directories;
Track B runs on Delta versions. The engine, decisions, store, reports,
expectations, replay and agent tooling are identical between them.

## Track B: a real table with production field names

Suppose the production column names are `wk`, `sty` (store), `pfc` (product),
`dol` (dollar). Canonical names are what the engine consumes: `week`,
`store_id`, `product_id`, `dollar`, ...

```sh
# 1. Inspect versions and history
.venv/bin/qc delta-info --uri ./lake/fact

# 2. Onboard with aliases; emits a config with a column_map
.venv/bin/qc onboard --uri ./lake/fact \
  --alias wk=week --alias sty=store_id --alias pfc=product_id --alias dol=dollar \
  --previous 41 --current 42 \
  --out config/datasets/retail.yaml
```

The generated config contains:

```yaml
name: fact
week_column: week
metric_columns: [dollar, units]
primary_metric: dollar
entity_columns: [store_id, product_id, banner_id, state_id]
entity_key_columns: [store_id, product_id]
report_grain: [banner_id, state_id]
column_map:
  week: wk
  store_id: sty
  product_id: pfc
  dollar: dol
```

The engine never sees `pfc`; a source decorator renames production columns to
canonical names on read, across every stage and dimension table. If a mapped
production column is missing, the source records a warning and the contract
check fails loudly rather than scoring a wrong column.

```sh
# 3. First real run against two versions
.venv/bin/qc delta-run --uri ./lake/fact \
  --previous 41 --current 42 --config config/datasets/retail.yaml

# 4. The weekly entry point (idempotent, alertable exit codes)
.venv/bin/qc weekly --uri ./lake/fact \
  --config config/datasets/retail.yaml \
  --store data/qc.db \
  --calibration-store data/calibration.jsonl \
  --out reports/weekly

# 5. Schedule it after the refresh commit (cron, Airflow, or a job)
QC_URI=./lake/fact QC_STORE=/data/qc.db \
  QC_CONFIG=config/datasets/retail.yaml \
  ./deployment/weekly.sh
```

Exit codes: `0` pass / already processed, `2` investigate, `3` data-contract
failure, `1` error. See [docs/weekly-run.md](weekly-run.md).

```sh
# 6. Collect analyst outcomes and periodically re-select the champion
.venv/bin/qc store import --store data/qc.db --csv outcomes.csv --dry-run
.venv/bin/qc store import --store data/qc.db --csv outcomes.csv
.venv/bin/qc champion --store data/qc.db --suite-dir data/suites/champion-eval \
  --text-embedder answerdotai/ModernBERT-base --out reports/artifacts/champion-v2
```

## Optional extras

```sh
uv pip install --python .venv/bin/python -e ".[forecast]"    # Chronos
uv pip install --python .venv/bin/python -e ".[classical]"   # SARIMAX
uv pip install --python .venv/bin/python -e ".[tspulse]"     # TSPulse
uv pip install --python .venv/bin/python -e ".[delta]"       # Delta tables
uv pip install --python .venv/bin/python -e ".[text]"        # ModernBERT probe
```

## Read next

- [docs/onboarding.md](onboarding.md) - profiling, aliases, readiness findings
- [docs/weekly-run.md](weekly-run.md) - the scheduler entry point
- [docs/engine.md](engine.md) - what each deterministic layer decides
- [docs/semantic-layer.md](semantic-layer.md) - decisions, training, agent handoff
- [docs/evaluation.md](evaluation.md) - cohorts, calibration, evidence queries
- [docs/architecture-delta.md](architecture-delta.md) - what was built and why
