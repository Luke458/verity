# Real-data pilot runbook

The pilot is the only path to a `production_eligible` claim. Everything before
it validates plumbing and deterministic semantics on synthetic data
([claims.md](claims.md)).

## Prerequisites

- One production-like Delta table with at least two committed versions.
- A dataset YAML from onboarding (`qc onboard --out config/datasets/<name>.yaml`).
- Analyst capacity to confirm outcomes in the durable store.
- A frozen cohort plan committed with its `.sha256` digest
  (`config/cohort.json` + `config/cohort.json.sha256`).

## Steps

1. **Onboard the table**

   ```sh
   qc onboard --uri <table-uri> --previous <v-1> --current <v> \
     --out config/datasets/pilot.yaml
   qc delta-info --uri <table-uri>
   ```

2. **Run shadow over historical version pairs** (no registry, no oracle)

   ```sh
   qc delta-run --uri <table-uri> --previous <v-1> --current <v> \
     --config config/datasets/pilot.yaml --json > reports/pilot/run.json
   ```

   Inspect `historical_revision`, `latest_week`, `lineage`, and every
   `NOT_EVALUATED`/`SKIPPED` check. Record disagreements with analysts.

3. **Import confirmed analyst outcomes**

   ```sh
   qc store import --store data/pilot.db --csv outcomes.csv --dry-run
   qc store import --store data/pilot.db --csv outcomes.csv
   ```

   Outcomes must carry `provenance=analyst` explicitly; the importer defaults
   to `imported`, never `analyst`.

4. **Check readiness**

   ```sh
   qc pilot-check --store data/pilot.db --plan config/cohort.json
   ```

   `NOT_READY` lists every blocker: label count, synthetic provenance in the
   pilot database, unpinned or changed plan. Do not proceed until `READY`.

5. **Freeze the real held-out cohort and score**

   ```sh
   qc cohort --plan config/cohort.json --out reports/cohort/pilot-v1
   ```

   The cohort run verifies the plan hash, hashes the engine source, and reports
   Wilson intervals on detection and false positives. Real-label cohort wiring
   (store-backed cases) is the remaining engineering task.

6. **Select a champion on the frozen cohort**

   ```sh
   qc champion --store data/pilot.db --suite-dir <held-out-suite> \
     --out reports/artifacts/champion-pilot
   ```

   Selection requires complete gates, homogeneous cohorts, no train/eval
   overlap, and a paired test that separates the leaders. Only then can
   `production_eligible` be true.

## Exit criteria

- `qc pilot-check` returns `READY`.
- Frozen cohort gates pass with reported confidence intervals.
- Champion selection is significant and trained on analyst-only labels.
- Every README claim in [claims.md](claims.md) for the used components moves
  from `validated-synthetic` to `validated-real` with a link to the artifact.

## What is explicitly out of scope until then

- Databricks/Spark execution. The pilot runs on delta-rs against the same
  versioned snapshots.
- TSPulse, text probe and remote Jev promotion: each has its own
  pre-registered gate and stays `research` until it passes.
