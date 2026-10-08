# QC engine

What each layer decides, in order. Every check that can escalate emits a
versioned finding, and the final status is computed once from those findings
(see [Final assessment policy](#final-assessment-policy)).

## Pipeline

```text
previous/current snapshots
        |
        v
data contracts (report + analysis stage)
        |  fail -> DATA_CONTRACT_FAILURE (revision not attempted)
        v
version pair (overlap, new periods, shape)
        |
        v
revision cube (base grain, headline grain, week grain)
        |
        v
entity lifecycle (per entity type) + reclassification detection
        |
        v
attribution (contributors, explained delta, unexplained residual)
        |
        v
counterfactual + reconciliation + lineage evidence
        |
        v
temporal QC (forecast + share-of-parent tests, new periods only)
        |
        v
findings -> one final status -> rule cause labels -> machine output
```

## Contracts

Checked on the configured contract stage (`report` by default) and the analysis
stage, because some failures exist only upstream of the final table. A failure
returns `DATA_CONTRACT_FAILURE` explicitly rather than being treated as an
anomaly:

| Check | Meaning |
|---|---|
| `required_columns` | every configured required column is present |
| `metric_dtypes` | metric columns are numeric |
| `week_progression` | weeks are null-free and contiguous |
| `latest_week_present` | the week column is fully populated |
| `null_fraction` | null share of metrics is within the configured limit |
| `duplicate_fraction` | duplicate share of week + entity keys is within limit |

## Revision cube

For each metric and grain the cube records `previous_value`, `current_value`,
`absolute_delta`, `relative_delta` and a `period` label: `overlap` for weeks
present in both versions, `new` for weeks after the previous maximum. Revision
QC compares overlap weeks only; new periods have no previous value and are
handled by latest-week QC. Counts use distinct entity counts, not
sums.

## Entity lifecycle

| Class | Meaning |
|---|---|
| `NEW_ENTITY_RECENT` | entity appears only with the new period (no historical impact) |
| `NEW_ENTITY_HISTORICAL_BACKFILL` | entity appears with overlap-week history |
| `ENTITY_REMOVED` | entity present previously, absent now |
| `ENTITY_HISTORY_EXTENDED` | overlap history gained |
| `ENTITY_HISTORY_TRUNCATED` | overlap history removed |
| `LATEST_WEEK_MISSING` | entity traded in the previous maximum week and in at least `entity_presence_threshold` (default 0.5) of its trailing `entity_presence_window` (default 8) weeks, but is absent in the new period; the event carries its expected contribution (trailing mean, absent weeks counted as zero) |
| `POSSIBLE_RECLASSIFICATION` | product-to-commodity mapping changed between versions |

`UNCHANGED_ENTITY` events are omitted from results.

## Attribution

The residual is the part of the total overlap revision that no structural event
explains for the configured primary metric (`dollar`). Contributions are summed
at one entity level only, chosen by `explanation_entity_types` priority
(store, then banner/state, then product/commodity), so parent and child events
cannot double count.

Historical attribution is classified as follows; this is one input to the
final status policy, not the final assessment:

| Condition | Status |
|---|---|
| contract failure | `DATA_CONTRACT_FAILURE` |
| missing entities of one type, not explained by an approved closure, expected to carry more than `materiality_ratio` of the period (`lifecycle.missing_entity_impact`; re-checked like-for-like when closures are approved) | `INVESTIGATE` |
| structural events, all matched by the expected-event registry and explained fraction >= threshold | `PASS_WITH_EXPLANATION` |
| structural events are only category moves, all matched, and the residual escalates by neither rule below | `PASS_WITH_EXPLANATION` |
| structural events not matched by the registry | `INVESTIGATE` |
| material residual unexplained | `INVESTIGATE` (`unexplained_value_change`) |
| immaterial residual but changes spread across > `broad_recalculation_breadth` of rows | `INVESTIGATE` (`broad_historical_recalculation`) |
| otherwise | `PASS` |

Whole-history materiality cannot see a restatement confined to one week: it
must exceed `materiality_ratio` of *all* overlap weeks. Each overlap week is
therefore also judged on its own (`attribution.week_revisions`): its revision
minus what its structural events explain, as a fraction of that week's previous
value. Above `week_revision_ratio` (default 0.5%) it becomes a `revision_week`
finding and the run is labelled `HISTORICAL_CORRECTION`. Weeks inside the
declared `restatement_weeks` window allow `restatement_tolerance` (default 10%)
for late-arriving data.

Cross-metric evidence is recorded when the primary metric moves materially
while units do not (`dollar_change_without_units`), which is characteristic of
value/coding or warehouse transform errors.

## Assessed measures

`temporal_required_metrics` and `temporal_optional_metrics` select the measures
assessed per appended period; empty values derive required measures from
`required_columns` and optional measures from the remaining `metric_columns`.
Every present measure gets its own period forecasts, calibration pools,
findings and availability records, keyed by metric/period/hierarchy scope. An
absent optional measure is reported as informational and an absent required
measure makes the assessment `INCOMPLETE`. Measures listed in
`snapshot_metrics` (stock) are compared within a period and are never summed
across time.

An unregistered backfill, truncation, removal or reclassification always
remains `INVESTIGATE`, even at a 100% explained fraction: structure is
explained, but confirmation is a human or registry decision.

## Counterfactual reconstruction

`counterfactual` removes the per-week explained effects of the structural
events from the overlap revision and compares the reconstructed current against
the previous version. `reconciliation_score` is `1 - |reconstructed_delta| /
|raw_delta|` (1.0 when fully explained). A `PASS_WITH_EXPLANATION` now also
requires the reconstruction score to clear
`reconstruction_score_threshold` (default 0.9), so a structural label alone is
not enough if the arithmetic does not reconcile.

## Reconciliation

Invariant checks between the report and the analysis table: metric mass balance
within `reconciliation_tolerance`, per-report-group count consistency for
`store_count` / `product_count`, per-week hierarchy consistency for configured
`hierarchy_columns` (`hierarchy_tolerance`), and aggregate-marker detection:
a detail column containing values such as `TOTAL` or `ALL` fails
`hierarchy_markers:*` because aggregates mixed into detail rows double count.
Failures set `RECONCILIATION_FAILURE` on the reconciliation section. Robust
dollar-per-unit outliers (median/MAD with `price_outlier_k`) in the appended
periods, measured against the weeks before them, are findings that a scoped,
approved ratio expectation can explain; historical weeks are never re-flagged
on later refreshes.

## Entity relationships

`detect_relationships` proposes conservative replacement candidates
(`replaced_by`) between entities that disappear and newly appearing entities
whose overlap-week series correlate above
`relationship_correlation_threshold` with a volume ratio inside
`relationship_ratio_bounds`, with Benjamini-Hochberg control over the tested
pairs. Candidates are never confirmed automatically; they appear in the machine
output and the report and label the run `ENTITY_MERGE`.

## Reports and local Delta assessment

`render_markdown` / `write_report` produce a human report (`report.md`) and the
machine JSON (`report.json`) for a run; the output directory is created
exclusively, so a run is never overwritten. `qc report` writes both.

`DeltaSource` (delta-rs, optional `[delta]` extra) reads Delta tables and their
versions directly, so table versions can be assessed locally or from object
storage without Databricks compute. Single-table mode maps one table to one
stage; multi-table mode maps stage and dimension names to their own tables and
assumes commit-version alignment across stage tables. `qc delta-info` shows the
history, schema and row count; `qc delta-run` runs the engine over two table
versions. Dimension tables without the requested version fall back to latest
and record a warning.

## Lineage and first divergence

For every captured stage the overlap revision magnitude is measured against the
previous version and the row-count delta is recorded. The first stage whose
relative divergence exceeds `lineage_materiality_ratio` (or whose overlap row
count changed) is reported as `first_divergence`. Value-preserving transforms
mean this equals the injection stage for source/coding/warehouse faults.
Latest-period-only faults and mapping-only reclassifications do not diverge in
overlap history; they are surfaced by lifecycle instead. Stage summaries carry
structural fingerprints for later comparison.

## Expected-event registry

`load_registry` / `save_registry` read and write registry entries. `propose_expected_events` drafts entries from observed historical
backfills; drafts are marked `confirmed: false` and must be confirmed before
they are trusted. `qc run --registry path` uses a registry file; blind runs use none.

An entry is used only when `approved_by` is set, `confirmed` is `true`, its
`dataset` is the dataset under test and `approved_at` is no later than the
assessment's observation time. It explains an event of its `classification`
for one of its `entity_ids` (of its `entity_type`), and only within its scope:

| Classification | Scope | Also required |
|---|---|---|
| `NEW_ENTITY_HISTORICAL_BACKFILL`, `ENTITY_REMOVED`, `ENTITY_HISTORY_TRUNCATED` | every rewritten week inside `expected_history_start`..`expected_history_end` | |
| `LATEST_WEEK_MISSING` (a closure) | the latest week inside `effective_from_week`..`effective_to_week` | |
| `POSSIBLE_RECLASSIFICATION` (a category move, id `from->to`) | the latest week inside `effective_from_week`..`effective_to_week` | the moved value is conserved (`conservation_ratio` >= `explained_fraction_threshold`); if `product_ids` is given, no other product moved |

An approved closure or move also restates the refresh like-for-like before the
latest-week test (`attribution.like_for_like`): the closed entities leave every
week of both versions, and the moved products' previous rows take their new
category. A store that is present is never excluded, and products the entry
does not cover are never re-assigned. The restatement is recorded in the
machine output as `like_for_like`; each explained absence is an
`absence_event` finding with its approval. The effect on detection is
measured in [evaluation.md](evaluation.md#approving-closures-and-category-moves).
`qc notices` (`qc/notices.py`) drafts entries the same way from free-text change notices
([weekly-run.md](weekly-run.md#drafting-explanations-from-notices)).

## Shadow mode

```sh
qc shadow --suite-dir data/suites/demo --out reports/shadow/demo
```

Runs the engine over every scenario, writes one JSONL record per scenario and a
summary JSON, and (for synthetic suites) scores outcomes against the oracle:
detection rate and false-positive rate (both on the final status; faults and
genuine movements vs. clean controls), historical false-positive rate, latest-week detection and
control rates, expected-event pass rate, lineage first-divergence accuracy and
mean reconstruction score.

## Temporal QC

Latest-week QC compares the new period against a forecast trained on the
previous version only, so backfilled overlap history cannot inflate the
baseline. Series are built for every configured temporal measure (present
measures are assessed, absent optional measures are reported and absent required
measures yield INCOMPLETE) at the national level and for configured entity
columns (banner, commodity); snapshot measures are compared within a period and
never summed across time.

Evidence per series:

| Measure | Meaning |
|---|---|
| forecast quantiles | predictive distribution from the configured forecaster |
| nominal / calibrated percentile | where the actual falls in the predictive distribution, before and after empirical residual calibration |
| share shift (leaves) | the leaf's share of the national parent against its own trailing shares (Student-t prediction interval, `share_t`/`share_p`/`share_impact`) |
| robust z | deviation from the recent median/MAD (evidence only) |
| seasonal z | deviation from the same week in previous years (evidence only) |
| EWMA z | deviation from the exponentially weighted level (evidence only) |
| change point | strongest recent mean-shift t-statistic (evidence only) |

Model selection, fitting and residual calibration use only observations before
the assessment target and only the selected previous snapshot; rolling origins
never score an observation with a forecast that saw it. Every appended period is
assessed separately with its own horizon (`target - previous_max_week`), and
earlier newly appended weeks never enter training for later targets in the same
assessment. Series are dense over the observed week span: missing weeks are
explicit NaN rows with a `missing` marker instead of being compressed into
consecutive observations. Errors are standardized per origin and pooled by
metric, aggregation level, model revision and forecast horizon, so a
different-volume series cannot dominate a shared pool; pooled quantiles are
converted back through the target's own historical scale before becoming
intervals. The latest `temporal_calibration_origins` (default 12) are reserved
exclusively for calibration and are never reused for selection. `forecaster:
auto` selects on the remaining pre-target rolling-origin absolute error over
the fixed grid (seasonal-naive; ridge trend; ridge trend + annual Fourier order
1-3 + declared calendar event indicators, penalties 0.1/1/10), ties break on
interval score then the simpler model, and the choice is frozen before interval
calibration. Annual-seasonality candidates require at least 104 completed weeks.
`baseline` (seasonal-naive with residual-scale quantiles) is the fixed choice
and the default.

Calibrated prediction intervals come from pooled standardized errors scaled to
the target (`temporal_interval_alpha`, default 0.1) and are reported with
history length, observed/missing weeks, forecast error, per-series held-out
coverage and width.

Anomalies are decided once per assessment (`temporal.decide_anomalies`). Every
series of every measure and appended period is one hypothesis in a single
Benjamini-Hochberg family at `temporal_fdr_q`: national series contribute their
forecast p-value, leaves their share-of-parent p-value (or their forecast
p-value when no share test is possible). Under the global null the chance of
any false page per refresh is therefore about `temporal_fdr_q`. A significant
series is an anomaly only if its movement is also material versus the same
metric/period/scope expectation (`temporal_min_relative_residual`, default 2%;
`temporal_materiality_abs`, default 0); for a share-tested leaf the material
movement is the part attributable to the share change. Leaves are tested on
their share because most week-to-week variance in retail series is a common
market movement: a market-wide swing moves the national series, not every
leaf. Robust, seasonal and EWMA z-scores and change points ignore trend,
seasonality and multiplicity, so they are reported as evidence and never decide.
Each share-tested leaf also reports what the refresh could not have seen:
`detectable_change` (and `detectable_change_80`) is the drop in the leaf's own
value that would be flagged with 50% (80%) power as the refresh's only anomaly.
It comes from the critical t for this family's size and q, the leaf's
predictive spread and expected share, and the materiality floor.
`latest_week.sensitivity` gives the median per level and measure. It is a
conservative bound: simultaneous movements, such as dollars and units of one
category, face a more lenient BH threshold ([evaluation.md](evaluation.md#sensitivity-what-a-refresh-could-not-have-seen)).
Sparse leaves fall back to parent expectation
times a historically estimated child share and are labelled `parent_share`.
Structural additions from newly appearing entities are subtracted from the
target week before scoring, at one entity level only (store before product) to
avoid double counting; registered backfills therefore do not trigger
latest-week anomalies.

The final status is computed once, from findings, by `policy.apply_policy`;
every check that can escalate (including the opt-in distribution drift check)
emits findings, so there is no second status path to bypass.
`historical_revision.status` preserves the revision-only verdict, so the two
paths remain separable.

## CLI

```sh
qc run --scenario-dir data/suites/demo/scenario-0001
qc run --scenario-dir data/suites/demo/scenario-0001 --json
qc report --scenario-dir data/suites/demo/scenario-0001 --out reports/runs/run-1
qc delta-info --uri ./lake/fact
qc delta-run --uri ./lake/fact --previous 11 --current 12
```

The scenario adapter is development-only; `DeltaSource` implements the same
`VersionSource` interface against Delta tables.

## Validation

Unit tests assert contracts, version shapes, exact cube deltas, lifecycle
classes and attribution semantics on hand-built frames. The end-to-end suite
builds one scenario per fault family with the `qcgen` oracle and asserts the
status, classification, explained fraction and conservation ratio for each.

## Limits

- Single-node pandas; the DataFrame contract is designed to port to Spark.
- Thresholds are defaults, not validated operating points; real data requires
  calibration of `materiality_ratio`, `lineage_materiality_ratio`,
  `broad_recalculation_breadth` and the temporal z/percentile thresholds.
- Seasonal comparisons need multiple years of history; short histories widen
  forecast intervals and reduce power (the 30-week `tiny` profile cannot
  reliably detect a 12-30% single-commodity movement; the 104-week `small`
  profile can).
- Lineage reads every captured stage once per version.

## Final assessment policy

After contracts, reconciliation, references, temporal checks, recurrence and
scoped approvals, the engine collects immutable findings (every appended period
gets its own temporal findings) and computes one final status. Contract
failures win; unexplained failures yield INVESTIGATE; unavailable required
evidence yields INCOMPLETE; otherwise findings explained by an approved
expected event or ratio expectation yield PASS_WITH_EXPLANATION, and all
required successful checks yield PASS. Each finding carries a disposition
(`HARD_FAILURE`, `UNAVAILABLE_EVIDENCE`, `UNEXPLAINED_ANOMALY`,
`HUMAN_APPROVED`, `INFORMATIONAL`). Nothing is cleared automatically: only a
registered, approved event or expectation observed before the assessment
cutoff can explain a failure, and it covers the exact finding IDs it lists.

Recurrence matches unexplained findings on a serialized, period-independent
stable key (check, scope kind, measure, hierarchy level and scope) and
accumulates signed scoped impacts against a registered cumulative budget;
multiple periods inside one logical refresh count as one occurrence. Optional
missing checks remain visible.

`RuleDecisionProvider` then labels the likely cause, origin and severity, and
`likely_causes` lists every cause the evidence supports, so two simultaneous
faults can both be named. Rules run in priority order: the first match names
`likely_cause`. Structural rules (merge, missing stores or products,
reclassification, truncation, backfill) can all match. Events that are
consequences of another fault are not counted twice: products missing with
their stores, ids retired or backfilled by a merge, a category emptied by a
reclassification. At most one revision rule explains the unexplained
historical revision; when a history-rewriting structural event is present, a
source-stage revision is credited to that event. A lineage-localized coding or
warehouse revision below materiality is still named, at LOW severity. Its
`requires_investigation` is the policy verdict, so a label can neither clear
nor escalate a run. Temporal-only anomalies have UNKNOWN cause unless
independent evidence exists: the engine cannot tell a market movement from an
upstream loss it cannot localize.
