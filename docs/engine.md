# QC engine (Milestones A-D)

Implements section 73 phases 1-10, 12-15 and 18 of `docs/architecture.md`: data
contracts, version resolution, the revision cube, entity lifecycle,
attribution, counterfactual reconstruction, reconciliation, lineage, the
expected-event registry workflow, a shadow-mode harness, latest-week temporal
intelligence, the evidence graph, typed semantic decisions, incident memory and
the investigation-agent handoff. Chronos is an optional adapter; see
`docs/semantic-layer.md` for the decision and agent layers.

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
temporal intelligence (forecast + robust statistics, new periods only)
        |
        v
evidence graph + status + machine-readable output (section 77)
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
handled by latest-week QC (Milestone C). Counts use distinct entity counts, not
sums.

## Entity lifecycle

| Class | Meaning |
|---|---|
| `NEW_ENTITY_RECENT` | entity appears only with the new period (no historical impact) |
| `NEW_ENTITY_HISTORICAL_BACKFILL` | entity appears with overlap-week history |
| `ENTITY_REMOVED` | entity present previously, absent now |
| `ENTITY_HISTORY_EXTENDED` | overlap history gained |
| `ENTITY_HISTORY_TRUNCATED` | overlap history removed |
| `LATEST_WEEK_MISSING` | entity existed through the previous maximum week but is absent in the new period |
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
| any `LATEST_WEEK_MISSING` | `INVESTIGATE` |
| structural events, all matched by the expected-event registry and explained fraction >= threshold | `PASS_WITH_EXPLANATION` |
| structural events not matched by the registry | `INVESTIGATE` |
| material residual unexplained | `INVESTIGATE` (`unexplained_value_change`) |
| immaterial residual but changes spread across > `broad_recalculation_breadth` of rows | `INVESTIGATE` (`broad_historical_recalculation`) |
| otherwise | `PASS` |

Cross-metric evidence is recorded when the primary metric moves materially
while units do not (`dollar_change_without_units`), which is characteristic of
value/coding or warehouse transform errors.

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
dollar-per-unit outliers (median/MAD with `price_outlier_k`) are reported as
evidence, not as failures; they are a cheap signal for value-only changes such
as coding errors.

## Entity relationships

`detect_relationships` proposes conservative replacement candidates
(`replaced_by`) between entities that disappear and newly appearing entities
whose overlap-week series correlate above
`relationship_correlation_threshold` with a volume ratio inside
`relationship_ratio_bounds`. Candidates are never confirmed automatically.
`RelationshipStore` persists confirmed relationships (`superseded_by`,
`remapped_to`, `merged_into`, `split_into`, `alias_of`, `replaced_by`) and
looks them up per entity. Candidates appear in the machine output, the report
and the evidence graph.

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

`load_registry` / `save_registry` read and write the architecture section 11
entry shape. `propose_expected_events` drafts entries from observed historical
backfills; drafts are marked `confirmed: false` and must be confirmed before
they are trusted. `qc run --registry path` uses a registry file instead of a
scenario manifest.

## Shadow mode

```sh
qc shadow --suite-dir data/suites/demo --out reports/shadow/demo
```

Runs the engine over every scenario, writes one JSONL record per scenario and a
summary JSON, and (for synthetic suites) scores outcomes against the oracle:
detection rate, historical false-positive rate, latest-week detection and
control rates, expected-event pass rate, lineage first-divergence accuracy and
mean reconstruction score. On real data the oracle fields are simply absent and
the records become the analyst-feedback store for calibration and training.

## Temporal intelligence (Milestone C)

Latest-week QC compares the new period against a forecast trained on the
previous version only, so backfilled overlap history cannot inflate the
baseline. Series are built at the national level and for configured entity
columns (banner, commodity) from the primary metric.

Evidence per series:

| Measure | Meaning |
|---|---|
| forecast quantiles | predictive distribution from the configured forecaster |
| nominal / calibrated percentile | where the actual falls in the predictive distribution, before and after empirical residual calibration |
| robust z | deviation from the recent median/MAD |
| seasonal z | deviation from the same week in previous years |
| EWMA z | deviation from the exponentially weighted level |
| change point | strongest recent mean-shift t-statistic (history evidence, not a target-week flag) |

Model selection, fitting and residual calibration use only observations before
the assessment target; rolling origins never score an observation with a
forecast that saw it. Series are dense over the observed week span: missing
weeks are explicit NaN rows with a `missing` marker instead of being compressed
into consecutive observations. Errors are standardized per origin and pooled by
metric, aggregation level, model revision and forecast horizon, so a
different-volume series cannot dominate a shared pool. `forecaster: auto`
selects on pre-target rolling-origin absolute error over the fixed grid
(seasonal-naive; ridge trend; ridge trend + annual Fourier order 1-3 + declared
calendar event indicators, penalties 0.1/1/10), ties break on interval score
then the simpler model, and the choice is frozen before interval calibration.
Annual-seasonality candidates require at least 104 completed weeks. `baseline`
and `chronos` remain available as fixed choices; `chronos` needs the optional
`[forecast]` extra.

Calibrated prediction intervals come from held-out raw errors
(`temporal_interval_alpha`, default 0.1) and are reported with history length,
observed/missing weeks, forecast error, coverage and width. A target-week
anomaly must move materially versus the forecast median
(`temporal_min_relative_residual`, default 2%) and then either be a forecast
extreme or seasonal deviation, or carry corroborating robust-z and EWMA flags;
a lone robust-z on a low-variance series is noise and is reported without
flagging. Forecast-tail flags on leaf series pass Benjamini-Hochberg control
(`temporal_fdr_q`) before escalation. Same-direction leaf residuals are
combined within their parent level before materiality testing, so a coordinated
small movement can escalate when the individual series cannot. Sparse leaves
fall back to parent expectation times a historically estimated child share and
are labelled `parent_share`; the fallback cannot support statistical clearance.
Structural additions from newly appearing entities are subtracted from the
target week before scoring, at one entity level only (store before product) to
avoid double counting; registered backfills therefore do not trigger
latest-week anomalies.

The top-level status is the historical revision status, upgraded to
`INVESTIGATE` when a latest-week anomaly is flagged and the historical verdict
was passing. `historical_revision.status` always preserves the revision-only
verdict, so the two paths remain separable.

## Evidence graph

All deterministic and temporal observations are assembled into a canonical
graph (`qc/evidence.py`): nodes carry a stable id, type, scope, source layer
(`deterministic` / `temporal`), value, payload and optional confidence; edges
link events to the version pair and attribution. The machine output exposes it
under `evidence_graph`, which is the boundary the semantic layer will consume
in Milestone D.

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
- Seasonal comparisons need multiple years of history; short local profiles
  rely mainly on the forecast percentile and robust z.
- TSPulse remains a research adapter and is not a dependency.
- Lineage reads every captured stage; on Databricks these summaries should be
  precomputed once per version rather than recomputed per run.
- Incident memory and the decision layer are implemented in Milestones D; real
  semantic accuracy requires analyst labels (docs/semantic-layer.md).

## Final assessment policy

After contracts, reconciliation, references, temporal checks, recurrence and
scoped approvals, versioned findings determine one final status. Contract
failures win; unexplained integrity/anomaly findings yield INVESTIGATE;
unavailable required evidence yields INCOMPLETE; otherwise explained actionable
findings yield PASS_WITH_EXPLANATION, and all required successful checks yield
PASS. Each finding carries a disposition (`HARD_FAILURE`, `UNAVAILABLE_EVIDENCE`,
`UNEXPLAINED_ANOMALY`, `STATISTICALLY_EXPLAINED`, `HUMAN_APPROVED`,
`INFORMATIONAL`) and its clearance basis. A finding is cleared only by a
verified `ExplanationCertificate` for its scope or by explicit human approval;
certificates require integrity, calibrated support, an interval that contains
the observation, no contradictory evidence and residual impact below the frozen
materiality threshold. Arithmetic attribution alone never authorizes clearance.
Findings recurring in at least two of the last three refreshes escalate when
the cumulative unexplained impact is material. Optional missing checks remain
visible. Model requests may add review but cannot clear deterministic review.
Approval coverage lists exact finding IDs, and approval observation timestamps
must precede the assessment cutoff. Temporal-only anomalies have UNKNOWN cause
unless independent evidence exists.
