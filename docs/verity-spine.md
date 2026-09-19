# Verity spine, ported

Verity's implementation is the more production-shaped half of this project's
lineage. Four pieces of that spine are now ported into retail-qc, adapted to
the pandas engine and this repository's conventions: historical replay, scoped
ratio expectations, independent reference controls, and the bounded RCA loop.

## Historical replay

```sh
qc replay --plan replay.json --out reports/replay/v1
```

Plan schema (JSON):

```json
{
  "cases": [
    {
      "case_id": "2026-W10",
      "uri": "abfss://.../gold/fact",
      "previous_version": "41",
      "current_version": "42",
      "as_of": "2026-03-09"
    }
  ]
}
```

Guarantees, enforced in `ReplayPlan.__post_init__`:

- case ids are unique;
- versions must increase per source, in declaration order;
- `as_of` must not move backwards.

Leakage guards: live registries and incident retrieval are never consulted -
only the plan's own `expected_events` are passed to the engine - and every case
writes its machine output under `cases/<case_id>.json` plus a `replay.json`
index with the plan hash. A failing case is recorded as `FAILED` and does not
stop the remaining cases. `as_of` is a caller assertion, not an authenticated
time boundary; that distinction is stated in the plan itself.

## Scoped ratio expectations

An expectation explains exactly the alert it names - one dataset, one metric,
one week, one ratio band, one validity window, one approver - and clears
nothing else.

```json
{
  "expectations": [
    {
      "expectation_id": "e-2026-w06",
      "dataset": "cwt",
      "metric": "dollar_per_unit",
      "min_ratio": 0.98,
      "max_ratio": 1.02,
      "week": 6,
      "effective_from": "2026-01-01",
      "effective_to": "2026-03-31",
      "approved_by": "analyst@example",
      "note": "promotion mix change"
    }
  ]
}
```

`qc expectations --scenario-dir ... --expectations expectations.json --as-of
2026-02-01` returns `expected` and `unexpected` flags plus an audit trail with
the rejection reason for every expectation (`unapproved`, `wrong dataset`,
`wrong week`, `expired`, `not yet effective`, `outside approved band`).
Unapproved expectations are never applied.

## Independent reference controls

`run_qc(..., reference_frame=..., reference_spec=ReferenceSpec(...))` compares
the current version's totals against an independently produced snapshot:
`MATCH`, `MISMATCH` or `INCOMPLETE` (missing metrics or empty reference are
never treated as zero). A mismatch adds `reference_mismatch:<metric>` to the
reasons and forces `INVESTIGATE`; a match changes nothing else. This is the
check that catches errors inside a band that every other test would allow.

## Bounded RCA loop

```sh
qc rca --scenario-dir data/suites/demo/scenario-0000 --steps 3
```

`qc/rca.py` runs an allowlisted sequence of evidence queries (the same seven
queries as `qc evidence`), records each call with `total_rows` and
`truncated`, and stops when a selector is done or `max_steps` is reached. The
default selector maps the decision layer's `likely_cause` to a query plan
(`CAUSE_TOOL_PLANS`, shared with the investigation brief). A custom selector
(for example an LLM) may choose tools but cannot invent them or repeat one.
Every result is `confirmed=False`; analyst confirmation remains a human act.

## Prequential point-in-time forecasting

`qc.prequential.forecast_across_loads(loads, ...)` treats each weekly load as
one observation point. For every load the training history is that load's own
frame up to `target_week - 1` (causal by construction), the forecast is scored
against the load's target week, and the calibration percentile is computed only
from records that were already observable:

- target weeks must strictly increase; a repeated target raises;
- `available_on` must not move backwards;
- records are usable only when `available_on < as_of` and
  `target_week < target`, so a load that becomes observable in the same week as
  the one being scored is excluded;
- residuals are committed only after all series of a load are scored, so a
  corrupted load can never widen its own interval;
- the pool is bounded and deduplicated per `(scope, series, target)`.

`prequential_sequence(source, version_ids, ...)` builds loads from consecutive
pinned versions of a Delta table, one load per version:

```sh
qc prequential-forecast --uri ./lake/fact --versions 40,41,42 \
  --min-samples 9 --out data/calibration.jsonl
```

Early loads report no percentile until the pool reaches `min_samples`; evidence
carries `pool_records` so the caller can see exactly how much history was
available. Flags are `prequential_lower` / `prequential_upper` at the
configured percentile thresholds. `available_on` is the caller's observation
week, not an authenticated arrival time.

## Not yet ported

- **Spark/Databricks adapters and deployment**: skipped by decision. The
  delta-rs source covers local and object-store assessment, and managed compute
  is not required yet.

## Deliberately not ported

- **Revisioned ratio expectations in the SQLite store**: the versioned JSON
  file stays. Expectations are low-volume, governed, and reviewed by humans;
  a diffable file in version control is a better fit than a database table, and
  the SQLite store is reserved for run outcomes, the event registry and
  relationships where append-only history genuinely matters.
