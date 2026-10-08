# Weekly run

`qc weekly` reads local Delta snapshots or a versioned Parquet manifest. It
never changes source tables or blocks publication.

```sh
qc weekly --uri ./lake/fact --previous 11 --current 12 \
  --config config/datasets/retail.yaml --store data/qc.db --out reports/weekly \
  --registry config/registry.json   # approved known changes (optional)
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

## Status and exit codes

Findings have schema versions, stable IDs, scopes, evidence references and a
required flag. The final status is computed once from them after all checks and
approvals:

| Exit | Meaning |
|---|---|
| 0 | PASS, or PASS_WITH_EXPLANATION (every failure covered by an approval) |
| 2 | INVESTIGATE |
| 3 | DATA_CONTRACT_FAILURE |
| 4 | INCOMPLETE: required evidence unavailable |
| 1 | Execution failure |
| 75 | Lock contention; retry later |

Contract failures take precedence, then unexplained failures, then incomplete
required checks. Findings carry a disposition: `HARD_FAILURE`,
`UNAVAILABLE_EVIDENCE`, `UNEXPLAINED_ANOMALY`, `HUMAN_APPROVED` or
`INFORMATIONAL`. Only a registered, approved expected event or ratio
expectation observed before the assessment cutoff can explain a failure, and
only the exact finding IDs it covers. The rule-based cause labels in the result
cannot change the status or the exit code.

`qc explain --report <dir>` or `qc explain --store data/qc.db --dataset <name>`
prints an assessment's status and its non-passing findings with dispositions.

## Journal, identity and recovery

SQLite is authoritative (default `reports/weekly/assessments.sqlite`; set
`--store` to keep it elsewhere). It holds three tables:

| Table | Semantics |
|---|---|
| `runs` | immutable machine payload per assessed refresh; the frozen input to recurrence |
| `assessments` | content-addressed assessment identity, status, artifacts and checksums |
| `attempts` | one row per invocation with its state (STARTED, COMMITTED, PUBLISHED) and any error |

The assessment identity hashes the selected snapshot content and metadata,
configuration, engine code, reference and approval inputs, the observation
cutoff and a frozen recurrence-input manifest (distinct predecessor refreshes
strictly before the cutoff and their payload hashes). Changing an eligible
predecessor, or a refresh appearing after the cutoff, cannot silently reuse a
cached result. Each execution gets its own attempt ID; `--force` creates another
attempt for the same immutable assessment without overwriting its outcome.

Result and report artifacts commit together in SQLite; reports are then written
to a temporary directory and published by atomic rename. These are two durable
steps, **not** one transaction: a retry verifies identity and checksums and
completes publication or regenerates damaged artifacts from the journal, without
duplicating the recorded run. A cached retry keeps the original status and exit
code and never re-notifies. Opening a store from an older schema backs it up
first; retired tables from earlier schemas are left in place and never read.

Metadata-only Delta commits (OPTIMIZE, VACUUM, property and constraint changes)
are not refreshes and are skipped when choosing versions. The default pair is
the latest refresh and its predecessor; if several refreshes land between runs,
assess the intermediate pairs explicitly with `--current/--previous`.

## Notifications

```sh
cat > sinks.json <<'JSON'
{"sinks": [
  {"kind": "webhook", "target": "https://hooks.example.com/qc", "token_env": "QC_WEBHOOK_TOKEN"},
  {"kind": "file", "target": "reports/notify/weekly.jsonl"}
]}
JSON
qc weekly --uri ./lake/fact --store data/qc.db --notify sinks.json
```

A payload is sent only for actionable statuses (`--notify-status`, default
INVESTIGATE, DATA_CONTRACT_FAILURE, CONTRACT_FAILURE, INCOMPLETE) and never for a
cached retry. It is deterministic, carries a content digest and is bounded by
`--notify-max-bytes`: optional sections are dropped whole (named in `omitted`),
never truncated. Webhooks must be `https://` (plain `http://` only to loopback),
never follow redirects, and read their bearer token from the named environment
variable. Delivery is fail-soft: a failed sink is reported on stderr and cannot
change the status, exit code or report. There is no retry queue.

## Drafting explanations from notices

Change notices usually arrive as text ("S012 closed for refit from week 118").
`qc notices` matches each one to a change the refresh shows and drafts the
registry entry that would explain it:

```sh
qc notices --scenario-dir data/suites/demo/scenario-0003 \
  --notices inbox.txt --out drafts.json                 # deterministic patterns
qc notices --scenario-dir ... --notices inbox.txt --out drafts.json \
  --matcher systemone --endpoint http://127.0.0.1:8090 --model lux
```

Notices are a JSON list, JSONL (`{"id", "text"}`) or one per line. Each is
matched to one observed lifecycle change, or to none. A match becomes a draft
registry entry: a new entity with history, a removal, or history extended or
truncated is scoped to the weeks it rewrote; a closure (latest week missing)
or a category move is scoped to this refresh's latest week, and a move lists
the products observed moving. Widen `effective_to_week` when a closure is
known to last. An entity replacement drafts both its removal and its
backfill. Other matches (a new entity with no history) are annotations. One
draft is written per observed change, citing every notice that supports it.

Drafts are **unapproved** (`approved_by: null`, `confirmed: false`) and explain
nothing until a person fills in `approved_by` and `approved_at` (before the
assessment's observation cutoff) and sets `confirmed`. Only then does
`qc run --registry drafts.json` count them. They explain only the findings
that change raises: in the tests, an approved backfill draft turns the
historical-revision findings into HUMAN_APPROVED, an approved closure or move
also clears the latest-week findings of the banners or categories it moved,
and an unrelated temporal anomaly stays unexplained.

`--matcher systemone` asks any service speaking the System One protocol
(`POST /v1/systemone`, for example `experiments/labeller/lux.py serve` or a
llama.cpp server). It asks one choice question and four yes/no checks, and
keeps a match only when every check clears `--threshold`. The human output
shows each match's weakest check. Endpoints follow the webhook rules: https,
or http to loopback, and no redirects. On synthetic notices reworded by an
LLM, this found 89% of true matches against 52% for the patterns, but about
four in ten of its suggestions were wrong, mostly notices about another
dataset ([notice-matching.md](notice-matching.md)). Review every draft.

## Scheduling

`deployment/weekly.sh` maps `QC_URI`, `QC_STORE`, `QC_CONFIG`, `QC_NOTIFY`,
`QC_EXPECTATIONS` and the `QC_REFERENCE_*` variables to CLI arguments and reads
storage credentials from `QC_STORAGE_OPTIONS`. Schedule it after the refresh
commits and route exit codes to your review process.

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
restatement_weeks: 2        # late-arriving data restates the last 2 weeks
```

Declare `restatement_weeks` whenever the source restates recent weeks as late
transactions arrive. Without it, every refresh that restates more than
`week_revision_ratio` (0.5%) of a recent week is reviewed; with it, those weeks
may move by up to `restatement_tolerance` (10%) and lineage ignores them when
attributing an origin.

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
weeks and 53-week retail years are handled through the ISO calendar. Snapshot
measures cannot be the primary time-aggregated
metric; stock summaries use the latest overlap period. Missing cells remain
missing evidence. The compatibility default `absent_entity_policy: zero` permits
zero only for absent entities; choose `missing` to require evidence instead.

Both snapshots and every available stage receive contracts before arithmetic:
finite measures, required columns, null/duplicate business keys, period validity
and compatible schemas. Mapping collisions fail explicitly. Required dimensions
and temporal checks that cannot run require review. `optional_checks` names
specific checks whose unavailability is visible but non-blocking; it cannot make
a failed data contract pass.
