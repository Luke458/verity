# Weekly run

`qc weekly` is the single entry point for the weekly refresh. It resolves the
newest two versions, runs the whole chain, and returns an exit code a scheduler
can alert on.

## Flow

```text
version resolution (latest two versions of the table)
        |
        v
engine run (contracts, revision cube, lifecycle, attribution,
            counterfactual, reconciliation, lineage, temporal, decisions)
        |
        +--> optional reference control (pinned external totals)
        +--> optional scoped expectations (approved ratio bands)
        |
        v
calibration append (one record per temporal series)
        |
        v
drift check over the prequential pool
        |
        v
durable store (run recorded; confirmed-only retrieval untouched)
        |
        v
report (report.md, report.json) + investigation brief (brief.json when required)
        |
        v
weekly.json manifest + exit code
```

## Command

```sh
qc weekly --uri ./lake/fact \
  --config config/datasets/retail.yaml \
  --store data/qc.db \
  --calibration-store data/calibration.jsonl \
  --expectations config/expectations.json \
  --reference-uri ./lake/reference --reference-spec config/reference.json \
  --out reports/weekly
```

- `--current` / `--previous` pin versions explicitly (default: latest two).
- `--force` reruns a version whose report directory exists; the store's
  immutable run revision is reported as a note rather than overwriting history.
- `--allow-investigate` returns 0 even when the run needs investigation (for
  environments where alerts are handled elsewhere).
- `--provider` accepts `rule`, a trained artifact directory, or
  `systemone:<url>` / `hybrid:<url>`.

## Exit codes

| Code | Status | Meaning |
|---|---|---|
| 0 | PASS / PASS_WITH_EXPLANATION / ALREADY_PROCESSED | nothing to do |
| 2 | INVESTIGATE | alert: analyst or agent review required |
| 3 | DATA_CONTRACT_FAILURE | the table is not fit for QC; stop the pipeline |
| 1 | error | execution failure (details on stderr) |

## Idempotency

A version is considered processed when `reports/weekly/<dataset>/<version>/`
exists. Re-running returns `ALREADY_PROCESSED` without work, so a scheduler
retry after a failed alert is safe. Use `--force` deliberately.

## Scheduling

Cron, after the refresh commit lands:

```cron
17 6 * * 1 cd /opt/retail-qc && ./deployment/weekly.sh >> logs/weekly.log 2>&1
```

`deployment/weekly.sh` maps environment variables (`QC_URI`, `QC_STORE`,
`QC_CALIBRATION_STORE`, `QC_EXPECTATIONS`, `QC_REFERENCE_URI`, ...) onto the
command and propagates the exit code. The same script works as an Airflow
`BashOperator` or a Databricks job task; nothing in the orchestrator assumes a
particular scheduler. Spark/Databricks adapters are deliberately not ported:
the delta-rs source reads the refreshed table directly.

## What the run does not do

- It does not commit or modify the source table.
- It does not auto-confirm outcomes or promote a decision provider; those stay
  human acts (`qc store add-outcome`, `qc champion`).
- It does not refit calibration on drift; it reports `DRIFT` in the manifest
  and notes it, leaving the refit decision to policy.
- It does not paginate or notify; the exit code and `weekly.json` are the
  interface for whatever runs the alerting.
