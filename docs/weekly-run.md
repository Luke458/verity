# Weekly analyst shadow

`qc weekly` reads local Delta snapshots or a versioned Parquet manifest. It
never changes source tables or blocks publication. Rules are the default.

```sh
qc weekly --uri ./lake/fact --previous 11 --current 12 \
  --config config/datasets/retail.yaml --store data/qc.db --out reports/weekly
```

The default predecessor is relative to the selected current version. Self,
reversed and unavailable pairs are rejected. Multi-table Delta sources require
`--stage-tables`, `--dim-tables` and `--version-map` JSON mappings; equal version numbers do not imply
alignment. The primary stage mapping must equal the selected fact version;
other mapped inputs must already exist at that fact commit. Historical dimensions never fall back to latest. For an independent
control specify `--reference-uri`, `--reference-spec`, `--reference-version`
and, where necessary, `--reference-stage`. A configured missing, future, empty
or incomplete reference requires review. Reference comparison uses period and
declared grain, so offsetting errors cannot hide in a grand total.

For Parquet, pass a JSON manifest as `--uri`:

```json
{"schema_version":1,"source_id":"retail-shadow","snapshots":[
  {"version":"v1","observed_at":"2026-01-01T00:00:00Z","stages":{"warehouse":"v1.parquet","report":"report-v1.parquet"}},
  {"version":"v2","observed_at":"2026-01-08T00:00:00Z","stages":{"warehouse":"v2.parquet","report":"report-v2.parquet"}}
]}
```

Paths must stay inside the manifest directory. Include explicit `dimensions`
mappings when required. Observation time is separate from the `week` column.

## Status and review

Findings have schema versions, stable IDs, scopes, evidence references and a
required flag. Final status is calculated after all controls and approvals:

| Exit | Meaning |
|---|---|
| 0 | PASS or fully approved PASS_WITH_EXPLANATION |
| 2 | Investigation required, including additional provider review requests |
| 3 | DATA_CONTRACT_FAILURE |
| 4 | INCOMPLETE: required evidence unavailable |
| 1 | Execution failure |
| 75 | Lock contention; retry later |

Contract failures take precedence, then unexplained failures, then incomplete
required checks. Missing optional checks remain visible. Findings carry a
disposition: `HARD_FAILURE`, `UNAVAILABLE_EVIDENCE`, `UNEXPLAINED_ANOMALY`,
`STATISTICALLY_EXPLAINED`, `HUMAN_APPROVED` or `INFORMATIONAL`. A verified
explanation certificate can clear its own finding with clearance basis
`statistical`; a human approval clears with `human_approval`. Models cannot
clear policy-required review; raw model recommendations remain separately
recorded. `--allow-investigate` is deprecated and does not override these exit
codes.

## Identity, retries and recovery

SQLite is authoritative (default `reports/weekly/assessments.sqlite`). The
assessment identity hashes the complete selected snapshot content/metadata,
configuration, engine code, reference and approval inputs, observation cutoff,
and provider artifact identity. Changing an input creates another assessment.
Each execution receives a separate attempt ID. `--force` creates another
attempt for the same immutable assessment; it does not overwrite its outcome.

The journal transitions through STARTED, COMMITTED and PUBLISHED. Result,
calibration, report and the complete evidence package (`evidence.json`,
including certificates) commit together in SQLite. Reports are written
to a temporary directory and published by atomic rename. These are two durable
steps, **not** a filesystem/database transaction. A retry verifies identity and
checksums, completes publication or regenerates damaged artifacts from journal
content without duplicating logical outcomes/calibration. Damaged directories
are preserved. Cache reuse is a separate flag and retains the original QC
status and exit code; directory existence alone is never evidence of success.

Opening an older store takes a backup and migrates transactionally to schema 4.
Legacy runs, labels and provenance are retained; insufficient legacy identity
metadata prevents cache reuse. Keep the backup until recovery has been checked.
The legacy JSONL calibration option is not appended by weekly processing;
weekly calibration revisions live in SQLite. Historical selection filters them
by observation cutoff before calculating the prequential pool.

`deployment/weekly.sh` maps `QC_URI`, `QC_STORE`, `QC_REFERENCE_VERSION` and other
environment variables to CLI arguments. Schedule it after refresh commit and
route exit codes to your existing review process. No notifications, automatic
promotion, Spark/Databricks deployment or GPU serving are included.

## Declared grain and calendars

Declare entity/business keys separately from the period; never include `week`
in a grain list. `stage_keys` overrides the default analysis/report grain:

```yaml
entity_key_columns: [store_id, product_id]
report_grain: [banner_id, state_id]
stage_keys:
  warehouse: [store_id, product_id]
  report: [banner_id, state_id]
calendar: sequential_week
snapshot_metrics: [stock]
absent_entity_policy: zero
required_dimensions: []
```

`weekly_date` accepts midnight dates on one consistent weekday across snapshots.
For fiscal/retail encoded periods use `calendar: mapped` and an explicit
`period_map` from labels to unique sequential integer weeks. Fractional numeric
periods are rejected. Declare a business calendar to enable date, retail
year/week and event evidence; no anchor means no calendar evidence and the
engine never assumes a geography:

```yaml
calendar_anchor_date: 2022-01-02   # first day of calendar_anchor_week
calendar_anchor_week: 1
calendar_events:
  - name: christmas
    rule: "12-25"
    lead_weeks: 1
    lag_weeks: 1
  - name: easter
    rule: easter
    lead_weeks: 1
    lag_weeks: 1
```

Rules are `easter` or a fixed `MM-DD`; lead/lag extend the window to adjacent
weeks and 53-week retail years are handled through the ISO calendar. `qc explain`
shows an assessment's findings, ledger and certificates; `qc evidence-bench`
runs the frozen threshold/ablation workflow. Snapshot measures cannot be the primary time-aggregated
metric; stock summaries use the latest overlap period. Missing cells remain
missing evidence. The compatibility default `absent_entity_policy: zero` permits
zero only for absent entities; choose `missing` to require evidence instead.

Both snapshots and every available stage receive contracts before arithmetic:
finite measures, required columns, null/duplicate business keys, period validity
and compatible schemas. Mapping collisions fail explicitly. Required dimensions
and temporal checks that cannot run require review. `optional_checks` names
specific checks whose unavailability is visible but non-blocking; it cannot make
a failed data contract pass.
