# Intelligent Retail Transactional QC Architecture for Azure Databricks

> Status note: this is the original design record. Implementation deviations
> and the status of every phase in section 73 are tracked in
> [architecture-delta.md](architecture-delta.md).

## 1. Purpose

Build an intelligent QC and root-cause-analysis platform for retail transactional data processed in Azure Databricks.

The primary workflow is a weekly full-history refresh:

```text
Previous table version V(n-1):
W001 ─────────────────────────────── W320

Current table version V(n):
W001 ─────────────────────────────── W320 ─ W321
│                                     │       │
└──── refreshed historical period ────┘       │
                                               └ new week
```

Each weekly load may:

- add a new week,
- rewrite all ~320 historical weeks,
- introduce historical corrections,
- add a new store with full historical data,
- add a new commodity with full historical data,
- remap products or commodities,
- remove or replace entities,
- change source coverage,
- introduce coding changes,
- change warehouse transformations,
- contain genuine source/data errors.

The system must distinguish:

```text
DATA CHANGED
```

from:

```text
DATA CHANGED FOR A KNOWN / EXPLAINABLE REASON
```

from:

```text
DATA CHANGED UNEXPECTEDLY
```

and ultimately:

```text
WHAT CHANGED?
        ↓
WHY DID IT CHANGE?
        ↓
WHERE IN THE PIPELINE DID IT FIRST CHANGE?
        ↓
DOES IT REQUIRE HUMAN INVESTIGATION?
```

---

# 2. Core Design Principles

## 2.1 Deterministic facts before AI

SQL/Spark should establish facts such as:

- which weeks changed,
- which metrics changed,
- which stores changed,
- which commodities changed,
- where entities first appeared,
- which entities explain a total discrepancy,
- whether totals reconcile,
- which pipeline stage first diverged.

Models should interpret established evidence rather than rediscover basic arithmetic.

---

## 2.2 Historical revision QC and latest-week QC are different problems

### Historical revision QC

For overlapping weeks:

```text
V(n-1) vs V(n)
```

ask:

> What changed between the two versions, and can the change be explained?

### Latest-week QC

For W321 there is no V(n-1) equivalent.

Ask:

> Given historical behaviour, is W321 plausible?

These paths converge later but should not use identical detection logic.

---

## 2.3 Explain changes before declaring anomalies

A new store arriving with 200 weeks of legitimate historical data should produce:

```text
NEW_STORE_HISTORICAL_BACKFILL
```

not:

```text
200 HISTORICAL ANOMALIES
```

The primary anomaly target should therefore be:

```text
UNEXPLAINED RESIDUAL
```

rather than raw version difference.

---

## 2.4 Use multiple independent forms of evidence

Temporal QC should not depend on one model.

Use:

```text
Chronos-2
+
TSPulse
+
robust statistics
+
change-point detection
+
deterministic structural checks
+
lineage evidence
```

Model disagreement is itself useful evidence.

---

## 2.5 Escalate intelligence gradually

Use the cheapest reliable mechanism first:

```text
deterministic rule
      ↓
specialist temporal model
      ↓
compiled classifier
      ↓
small semantic model
      ↓
large reasoning agent
```

Do not invoke an expensive reasoning agent for cases SQL can resolve exactly.

---

# 3. Target Architecture

```text
                         WEEKLY REFRESH
                               │
                               ▼
                     ┌──────────────────┐
                     │ DATA CONTRACTS   │
                     │ schema / dates   │
                     │ required fields  │
                     └────────┬─────────┘
                              │
                              ▼
                  MULTI-STAGE PIPELINE SNAPSHOTS
                              │
           source → preprocessing → coded → warehouse → report
                              │
                              ▼
                       DELTA VERSION PAIR
                         V(n-1) ↔ V(n)
                              │
                              ▼
                    DETERMINISTIC DIFF ENGINE
                              │
       ┌──────────────────────┼────────────────────────┐
       │                      │                        │
       ▼                      ▼                        ▼
 ENTITY LIFECYCLE       RECONCILIATION          REVISION SHAPES
 new entities           parent/child             contiguous changes
 removed entities       mass balance             sign patterns
 backfills               invariants               concentration
 remappings
       │                      │                        │
       └──────────────────────┼────────────────────────┘
                              │
                              ▼
                    ATTRIBUTION / EXPLANATION
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          explained component        unexplained residual
                 │                         │
                 └────────────┬────────────┘
                              ▼
                     COUNTERFACTUAL CHECK
                              │
                              ▼
                 ┌───────────────────────────┐
                 │     TEMPORAL INTELLIGENCE │
                 └─────────────┬─────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
       Chronos-2            TSPulse          Robust statistics
       forecasting          anomaly /         seasonal/MAD/
       quantiles            embeddings        change points
       covariates           retrieval
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
                    TEMPORAL EVIDENCE
                               │
                     calibration layer
                               │
                               ▼
                        EVIDENCE GRAPH
                               │
        ┌──────────────────────┼────────────────────────┐
        ▼                      ▼                        ▼
 expected-event          incident memory          entity graph
 registry                / similarity             / mappings
        │                      │                        │
        └──────────────────────┼────────────────────────┘
                               ▼
                    SEMANTIC DECISION RUNTIME
                               │
                   MiniCPM5 / compiled heads
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
           PASS            EXPLAINED         INVESTIGATE
                                                   │
                                                   ▼
                                         REASONING / RCA AGENT
                                                   │
                                           Databricks access
                                                   │
                                                   ▼
                                               ROOT CAUSE
                                                   │
                                             analyst feedback
                                                   │
                                                   ▼
                                      calibration / training /
                                         incident memory
```

---

# 4. Data Contract Layer

Run basic structural checks before deeper analysis.

Examples:

```text
required columns exist
expected datatypes are present
week/date column exists
latest week exists
date progression is valid
metrics are numeric
required entity IDs are populated
no catastrophic null increase
no catastrophic duplicate increase
expected partition exists
```

Data-contract failures are not ordinary anomalies.

They should produce explicit statuses such as:

```text
DATA_CONTRACT_FAILURE
SCHEMA_CHANGE
MISSING_LATEST_PERIOD
INVALID_DATE_RANGE
```

---

# 5. Pipeline-Wide Lineage QC

QC should eventually operate across multiple lifecycle stages rather than only the final table.

Example:

```text
SOURCE
  ↓
PREPROCESSING
  ↓
CODED
  ↓
WAREHOUSE
  ↓
REPORT
```

For each stage capture summary fingerprints:

```text
row_count

dollar
units
scripts
stock

store_count
product_count
commodity_count

min_week
max_week

null_counts
duplicate_counts

key_fingerprint
```

Then detect the earliest point of divergence.

Example:

```text
SOURCE       -17.1%
CODED        -17.0%
WAREHOUSE    -17.0%
REPORT       -17.0%
```

Possible origin:

```text
SOURCE
```

Compare:

```text
SOURCE         0.0%
CODED        -17.2%
WAREHOUSE    -17.2%
REPORT       -17.2%
```

Possible origin:

```text
CODING / TRANSFORMATION
```

This creates a first-divergence algorithm:

```text
for stage in pipeline_order:
    if anomaly appears at stage
       and did not exist at prior stage:
        likely_origin = stage
        break
```

---

# 6. Version Pair Detection

Automatically identify:

```text
previous_version
current_version

previous_max_week
current_max_week

overlap_start
overlap_end

new_periods
```

Expected normal case:

```text
previous_max_week = W320
current_max_week  = W321
new_period_count  = 1
```

Unexpected cases should be explicit:

```text
NO_NEW_WEEK
MULTIPLE_NEW_WEEKS
HISTORY_SHORTENED
HISTORY_EXTENDED
```

---

# 7. Revision Cube

Construct a normalized comparison dataset between V(n-1) and V(n).

Recommended grains:

```text
week

week × banner

week × state

week × store

week × commodity

week × product

week × store × commodity
```

These should be configuration driven.

Metrics might include:

```text
dollar
units
scripts
stock

row_count
transaction_count

store_count
product_count
commodity_count
```

For every metric/grain/week calculate:

```text
previous_value
current_value

absolute_delta
relative_delta
```

Example:

```text
run_id:             QC_20260917
grain:              commodity
entity_id:          12345
week:               2026-06-06
metric:             dollar

previous_value:     1,230,521
current_value:      1,389,344

absolute_delta:       158,823
relative_delta:        +12.9%
```

---

# 8. Entity Lifecycle Engine

Before anomaly detection, determine how entity membership changed.

Entity types:

```text
store
commodity
product
market
banner
supplier
```

For each entity calculate:

```text
exists_previous
exists_current

first_week_previous
first_week_current

last_week_previous
last_week_current

week_count_previous
week_count_current

rows_previous
rows_current

historical_weeks_added
historical_weeks_removed

historical_value_added
historical_value_removed
```

---

# 9. Lifecycle Classes

At minimum support:

```text
UNCHANGED_ENTITY

NEW_ENTITY_RECENT

NEW_STORE_HISTORICAL_BACKFILL
NEW_COMMODITY_HISTORICAL_BACKFILL
NEW_PRODUCT_HISTORICAL_BACKFILL

ENTITY_REMOVED

ENTITY_HISTORY_EXTENDED
ENTITY_HISTORY_TRUNCATED

POSSIBLE_RECLASSIFICATION

POSSIBLE_ENTITY_REPLACEMENT

UNKNOWN_STRUCTURAL_CHANGE
```

---

# 10. Example: Historical Store Backfill

Previous version:

```text
Store 9912:
absent
```

Current version:

```text
Store 9912:
2021 → current week
```

Classification:

```text
NEW_STORE_HISTORICAL_BACKFILL
```

Calculate its contribution by week:

```text
store_9912_delta_t
```

Then calculate:

```text
raw_total_delta_t
-
store_9912_delta_t
=
residual_delta_t
```

If almost all historical change disappears:

```text
explained_fraction ≈ 1
```

the event is probably legitimate.

---

# 11. Expected-Change Registry

Known changes should be registered explicitly.

Suggested logical table:

```text
qc_expected_event
```

Examples:

```text
new store onboarding
new commodity
new product range
supplier correction
historical resupply
source migration
coding change
mapping correction
banner migration
store closure
```

Suggested fields:

```text
event_id

effective_date
event_type

entity_type
entity_ids

expected_history_start
expected_history_end

expected_direction
expected_scope

description

expires_at
```

Observed changes should be matched against expected events.

Example:

```text
Observed:
Store 9912 appears with 214 historical weeks.

Registry:
Store 9912 onboarded this week.
Expected historical coverage from 2022 onwards.

Result:
EXPECTED_EVENT_MATCH
```

---

# 12. Entity / Mapping Graph

Maintain known relationships between identifiers.

Examples:

```text
Product A → Product B

Commodity 103 → Commodity 240

Barcode X → Barcode Y

Store 123 → Store 981
```

Relationship types:

```text
superseded_by
remapped_to
merged_into
split_into
alias_of
replaced_by
```

This graph helps distinguish:

```text
new data
```

from:

```text
existing data reclassified under a new code
```

---

# 13. Reclassification Detection

Example:

```text
Commodity 901:
+$41.0m

Commodity 524:
-$40.9m

National total:
+$0.1m
```

This resembles redistribution rather than genuine new sales.

Generate:

```text
positive_shift_value
negative_shift_value
net_change

conservation_ratio
```

Potential rule:

```text
IF:
    entity appears newly
    AND another entity loses a similar amount
    AND aggregate value is approximately conserved
    AND historical shapes correlate

THEN:
    POSSIBLE_RECLASSIFICATION
```

---

# 14. Revision Signature Engine

Historical refresh anomalies have temporal shapes.

For every revision series:

```python
revision_t = current_t - previous_t
```

and optionally:

```python
revision_pct_t = revision_t / previous_t
```

calculate features such as:

```text
weeks_changed
history_changed_fraction

first_changed_week
last_changed_week

largest_contiguous_block

positive_week_count
negative_week_count
sign_consistency

mean_absolute_delta
median_absolute_delta
max_absolute_delta

mean_relative_delta
median_relative_delta
max_relative_delta

net_delta
total_absolute_delta

revision_variance

top_entity_contribution_pct
entity_concentration
```

---

# 15. Revision Shape Examples

## Backfill

```text
++++++++++++++++++++++++++++++
```

Characteristics:

```text
many weeks
mostly same sign
often concentrated in a new entity
```

---

## Isolated correction

```text
000000000000+000000000000
```

Characteristics:

```text
very few weeks
localized historical impact
```

---

## Historical truncation

```text
00000000----------00000000
```

---

## Mapping migration

```text
Entity A:
----------------------------

Entity B:
++++++++++++++++++++++++++++

Total:
approximately unchanged
```

---

## Recalculation

```text
small changes across most historical weeks
```

---

# 16. Contribution Analysis

For an aggregate:

\[
\Delta Total_t = \sum_i \Delta Entity_{i,t}
\]

Rank contributors.

Example:

```text
Historical national delta:

Store 7743     +$18.1m
Store 8291      +$0.4m
Other           +$0.1m
```

If Store 7743 is a recognised backfill:

```text
explained_delta = contribution(Store 7743)
```

Calculate:

```text
raw_delta

explained_delta

unexplained_delta =
raw_delta - explained_delta
```

and:

```text
explained_fraction =
abs(explained_delta) / abs(raw_delta)
```

---

# 17. Counterfactual Reconstruction

Contribution analysis should be validated where possible with counterfactual reconstruction.

Example:

```text
Current V(n)
     ↓
remove newly introduced Store 9912
     ↓
counterfactual current version
```

Then compare:

```text
V(n)                       $412.0m

V(n) excluding Store 9912  $400.2m

V(n-1)                     $400.1m
```

Raw discrepancy:

```text
$11.9m
```

Counterfactual discrepancy:

```text
$0.1m
```

Generate:

```text
counterfactual_reconciliation_score
```

This gives stronger evidence that the identified lifecycle event genuinely explains the change.

---

# 18. Cross-Level Reconciliation

Some of the strongest QC checks are mathematical invariants.

Examples:

```text
National ≈ sum(states)

Banner ≈ sum(stores)

Commodity ≈ sum(products)

Total ≈ sum(markets)
```

Calculate:

\[
R = Parent - \sum Children
\]

Persist:

```text
reconciliation_residual
reconciliation_relative_error
```

A stable top-line total should not hide massive redistribution beneath it.

Example:

```text
National:
0%

NSW:
+$2m

VIC:
-$2m
```

The total looks unchanged, but the internal composition changed materially.

---

# 19. Cross-Metric Relationships

Compute relationships such as:

```text
dollar / unit

units / store

dollar / store

products / store

scripts / store
```

Example:

```text
Dollar      -30%
Units       -30%
Stores      -30%
Products      0%
```

This is coherent with a missing-store scenario.

Compare:

```text
Dollar      -30%
Units         0%
Stores        0%
```

This may represent:

```text
pricing/value issue
metric calculation change
currency/value transformation problem
```

Cross-metric relationships should become first-class evidence.

---

# 20. Structural Fingerprints

For every important stage/week/version generate lightweight fingerprints:

```text
row_count

distinct_key_count

null_count
duplicate_count

numeric quantiles

sum
sum_of_squares

min_key
max_key

hash_of_entity_set

hash_of_row_keys

top_n_entities
```

These can detect structural differences even when top-line totals are unchanged.

---

# 21. Temporal Intelligence Layer

Use multiple complementary methods.

```text
                        TIME SERIES
                             │
          ┌──────────────────┼───────────────────┐
          ▼                  ▼                   ▼
      Chronos-2           TSPulse          Statistical layer
      forecasting         anomaly /         robust / seasonal
      quantiles           embeddings        change points
      covariates          retrieval
          │                  │                   │
          └──────────────────┼───────────────────┘
                             ▼
                     TEMPORAL EVIDENCE
```

---

# 22. Chronos-2 Role

Chronos-2 answers:

> What should this series plausibly look like next?

Primary use:

```text
latest-week forecasting
```

For W321:

```text
W001 ... W320 → forecast W321
```

Useful series:

```text
dollar
units
store_count
product_count
commodity_count
scripts
```

Useful output:

```text
P01
P05
P10
P50
P90
P95
P99
```

Convert to features:

```text
actual

forecast_p01
forecast_p05
forecast_p10
forecast_p50
forecast_p90
forecast_p95
forecast_p99

forecast_residual
relative_residual

forecast_percentile

prediction_interval_width

below_p01
below_p05

above_p95
above_p99
```

Chronos-2 should not determine business root cause.

It supplies expected-value and uncertainty evidence.

---

# 23. Chronos Covariates

Where useful include:

```text
week_of_year

Christmas
Easter
Black Friday
EOFY

public holidays

promotion periods

catalogue periods

known store openings

known banner changes

supplier launches
```

Moving events such as Easter should not be represented solely by week number.

---

# 24. Chronos Hierarchy

Do not initially forecast every SKU × store combination.

Start at useful grains:

```text
national

banner

banner × state

banner × major commodity

major store groups

major commodities
```

Use deterministic drill-down below these.

---

# 25. TSPulse Role

TSPulse is a first-class temporal representation model.

Its jobs are different from Chronos.

Chronos asks:

> What value should come next?

TSPulse is intended to help answer questions such as:

> Does this temporal pattern look unusual?

> What does this pattern resemble?

> Can this series be represented compactly for classification or retrieval?

Use TSPulse for three possible roles.

---

## 25.1 Temporal anomaly evidence

Where sequence length and sampling frequency are suitable:

```text
business series
     ↓
TSPulse
     ↓
anomaly evidence
```

Examples:

```text
dollar
units
store_count
product_count
```

---

## 25.2 Revision-series representation

Generate:

```python
revision_t = current_t - previous_t
```

Then:

```text
revision series
      ↓
TSPulse representation
      ↓
revision classifier / similarity
```

Potential classes:

```text
normal refresh

historical backfill

isolated correction

history truncation

mapping migration

source recalculation

unknown revision
```

This complements deterministic revision-shape features rather than replacing them.

---

## 25.3 Incident similarity

Generate temporal embeddings for historical incidents.

Example:

```text
Current anomaly
      ↓
TSPulse embedding
      ↓
vector retrieval
      ↓
historically similar incidents
```

Possible result:

```text
Incident:
2026-05-14

Similarity:
high

Previous symptoms:
dollar -27%
units -28%
stores -29%
products unchanged

Confirmed cause:
supplier file omitted stores
```

This is stronger than relying only on semantic embeddings of analyst notes.

---

# 26. Important TSPulse Sequence-Length Constraint

Do not assume TSPulse zero-shot anomaly detection is automatically valid on the ~320-point weekly sequence.

The current TSPulse anomaly-detection guidance recommends substantially longer sequences for stable zero-shot anomaly detection.

Therefore:

```text
320 weekly points
```

must be benchmarked rather than assumed sufficient.

Preferred options:

### Option A — use higher-frequency history

If daily transactional data is available:

```text
320 weeks ≈ 2,240 days
```

which is much more appropriate for long-context anomaly analysis.

Potential daily series:

```text
daily dollar
daily units
daily store count
daily scripts
```

Aggregate results back to weekly QC.

### Option B — use TSPulse embeddings/classification only after evaluation

Benchmark whether shorter weekly inputs provide useful representations for:

```text
classification
retrieval
revision-pattern similarity
```

### Option C — keep Chronos + statistical detectors as the production weekly path

If TSPulse does not perform reliably on 320 weekly observations:

```text
do not force it into production
```

TSPulse should be treated as an empirically evaluated component, not a mandatory dependency.

---

# 27. Robust Statistical Layer

Use independent, cheap detectors alongside foundation models.

Potential detectors:

```text
rolling median

MAD

robust z-score

same-week-of-year deviation

rolling quantiles

EWMA

seasonal decomposition

change-point detection

trend-shift detection
```

Example evidence:

```text
Chronos percentile:
<1%

robust seasonal z:
-5.8

change-point score:
high

TSPulse:
abnormal pattern
```

Agreement should increase confidence.

Disagreement should reduce automatic confidence.

---

# 28. Forecast Calibration

Do not automatically trust nominal forecast probabilities.

Backtest forecasting models historically.

Example:

```text
model claims:
90% prediction interval

observed historical coverage:
73%
```

The interval is overconfident.

Use empirical/conformal calibration.

Conceptually:

```text
raw Chronos forecast
        ↓
rolling historical errors
        ↓
calibration
        ↓
empirically calibrated interval
```

Persist:

```text
raw_forecast_percentile
calibrated_forecast_percentile

raw_interval
calibrated_interval

historical_coverage
```

---

# 29. Model Ensemble Evidence

A useful evidence structure could be:

```json
{
  "chronos": {
    "percentile": 0.004,
    "relative_residual": -0.29
  },
  "tspulse": {
    "anomaly_score": 0.96
  },
  "statistics": {
    "robust_z": -5.8,
    "change_point_score": 0.91
  }
}
```

Do not necessarily combine these into one hardcoded scalar immediately.

Keep individual evidence available for semantic and learned downstream models.

---

# 30. Optional IBM Forecasting Challengers

The architecture should remain model-independent.

IBM Granite models such as:

```text
FlowState
Tiny Time Mixers
```

may be evaluated as champion/challenger forecasting models.

Do not make them production requirements initially.

Use the same evaluation interface as Chronos.

Example:

```python
forecast = temporal_model.predict(
    history=series,
    horizon=1,
    covariates=covariates,
)
```

This allows future model replacement without changing QC logic.

---

# 31. Evidence Graph

All prior layers should produce structured evidence.

Do not dump giant tables into the language model.

Example:

```text
                       Dollar -29%
                            │
                   Chronos < P01
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
        Units -30%                    Stores -31%
                                           │
                                    Products normal
                                           │
                                143 stores absent
                                           │
                           missing at SOURCE stage
                                           │
                      previous similar incident found
                                           │
                                           ▼
                               SOURCE COVERAGE ISSUE
```

Evidence nodes should contain:

```text
evidence_id

type

scope

entity

metric

value

source_layer

deterministic_or_model

confidence

related_evidence_ids
```

---

# 32. Expected Events + Evidence Graph

Known events should appear in the same graph.

Example:

```text
new Store 9912 detected
        │
        ▼
214 historical weeks added
        │
        ▼
expected-event registry match
        │
        ▼
99.4% raw delta explained
        │
        ▼
counterfactual matches previous version
        │
        ▼
EXPECTED HISTORICAL BACKFILL
```

This makes final decisions auditable.

---

# 33. Incident Memory

Every confirmed incident should become reusable knowledge.

Suggested fields:

```text
incident_id

run_id

symptom_signature

structured_features

temporal_embedding

root_cause

likely_origin

affected_tables

affected_entities

resolution

analyst_summary

created_at
resolved_at
```

Retrieval should combine:

```text
structured feature similarity

TSPulse temporal similarity

optional semantic text similarity
```

Do not rely only on text embeddings.

---

# 34. Semantic Decision Runtime

The semantic layer interprets evidence.

The initial local model should remain interchangeable.

Current candidate:

```text
openbmb/MiniCPM5-2B-SFT
```

The SFT/instruct checkpoint should be benchmarked against:

```text
MiniCPM5-2B-Base
MiniCPM5-2B-SFT
MiniCPM5-2B final/post-trained
```

The architecture must not depend on one specific model family.

---

# 35. Semantic Model Input

Example:

```text
QC OBJECT

Scope:
CWH AU

Week:
2026-09-12

LATEST WEEK

Dollar:
actual = 7.1m
Chronos calibrated percentile = 0.4%
relative forecast residual = -29%

Units:
calibrated percentile = 0.6%

Stores:
expected = 506
actual = 355
change = -29.8%

Products:
expected = 8170
actual = 8133
change = -0.5%

ROBUST DETECTORS

seasonal z = -5.8
change-point score = HIGH

TSPULSE

similarity to known missing-store pattern = HIGH

VERSION REFRESH

319 historical overlapping weeks

raw historical delta = $104,200
explained delta = $100,900
unexplained residual = $3,300

LINEAGE

source store count = -29.7%
coded store count = -29.8%
warehouse store count = -29.8%

first divergence = SOURCE

INCIDENT MEMORY

similar historical incident found

previous cause:
supplier extract omitted stores
```

---

# 36. Typed Semantic Outputs

The decision runtime should return typed fields rather than free-form prose.

Example:

```python
{
    "latest_week_anomaly": Boolean(),

    "historical_revision_status": Choice([
        "EXPECTED",
        "PARTIALLY_EXPLAINED",
        "UNEXPLAINED"
    ]),

    "likely_cause": Choice([
        "MISSING_STORES",
        "MISSING_PRODUCTS",
        "SOURCE_INGESTION",
        "CODING",
        "WAREHOUSE",
        "MARKET_MOVEMENT",
        "HISTORICAL_CORRECTION",
        "RECLASSIFICATION",
        "BACKFILL",
        "SCHEMA_FAILURE",
        "UNKNOWN"
    ]),

    "likely_origin": Choice([
        "SOURCE",
        "PREPROCESSING",
        "CODING",
        "WAREHOUSE",
        "REPORT",
        "UNKNOWN"
    ]),

    "severity": Choice([
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ]),

    "requires_investigation": Boolean()
}
```

Return:

```text
selected value
raw probability
calibrated probability
```

---

# 37. Semantic Decision Scoring Modes

Do not depend only on immediate A/B/C next-token logits.

Benchmark at least:

```text
1. immediate single-token scoring

2. full semantic candidate sequence likelihood

3. short-reasoning + constrained final scoring

4. ordinary instruct generation baseline

5. compiled hidden-state classifier
```

The accuracy harness should compare all modes on the exact same labeled cases.

---

# 38. Dynamic Semantic Path

For novel decisions:

```text
evidence graph
      ↓
MiniCPM instruct model
      ↓
candidate scoring
      ↓
typed probability distribution
```

Preferred scoring should support semantic labels rather than relying exclusively on token labels like:

```text
A
B
C
```

---

# 39. Short-Reasoning Fallback

If direct constrained scoring is materially less accurate than normal reasoning:

```text
evidence
   ↓
16–64 reasoning tokens
   ↓
constrained final decision
```

This remains substantially more controlled than arbitrary long-form generation while allowing the model to use a small reasoning budget.

Escalation rules should be benchmark driven.

---

# 40. Compiled Decision Heads

Frequently repeated decisions should eventually bypass language-model decoding.

Example:

```text
evidence features
      +
TSPulse embedding
      +
Chronos features
      +
optional MiniCPM hidden state
      ↓
small classifier
      ↓
typed QC decision
```

Potential recurring classifiers:

```text
missing stores

historical backfill

commodity remap

source failure

coding failure

market movement

normal
```

Conceptually expose:

```python
runtime.compile(
    field="likely_cause",
    examples=historical_cases,
)
```

The application-facing API should remain unchanged whether the backend is:

```text
dynamic language-model scoring
```

or:

```text
compiled specialist classifier
```

---

# 41. Escalation Logic

Example:

```text
deterministic evidence sufficient
        ↓
automatic resolution

otherwise:

compiled classifier high confidence
        ↓
automatic typed decision

otherwise:

MiniCPM high confidence
        ↓
typed decision

otherwise:

reasoning agent
        ↓
Databricks investigation
```

Possible thresholds should be calibrated empirically.

---

# 42. Reasoning / RCA Agent

The reasoning agent is not the primary detector.

It receives unresolved cases.

Capabilities may include:

```text
query source table

query coded table

query warehouse table

inspect mapping table

inspect entity counts

compare Delta versions

inspect pipeline metadata

retrieve incident history

inspect expected-event registry

produce root-cause report
```

Example:

```text
INVESTIGATE
     │
     ▼
RCA agent
     │
     ├── verify expected stores
     ├── inspect missing store IDs
     ├── check source file
     ├── check coded output
     └── compare previous run
           │
           ▼
CONFIRMED ROOT CAUSE
```

---

# 43. Root-Cause Localisation

The agent should begin with already established lineage evidence.

For example:

```text
Source:
143 missing stores

Coded:
same 143 stores missing

Warehouse:
same 143 stores missing
```

The agent should not waste effort investigating warehouse transformation first.

Likely search path:

```text
SOURCE
```

This is a major reason to build deterministic lineage evidence before agentic RCA.

---

# 44. QC Data Model

Recommended logical Delta tables:

```text
qc_run

qc_pipeline_stage

qc_version_diff

qc_entity_lifecycle

qc_entity_relationship

qc_expected_event

qc_reconciliation

qc_features

qc_temporal_forecast

qc_temporal_embedding

qc_evidence

qc_decision

qc_incident

qc_investigation

qc_feedback

qc_fault_test
```

---

# 45. `qc_run`

```text
run_id

table_name

previous_version
current_version

previous_max_week
current_max_week

started_at
finished_at

status
```

---

# 46. `qc_pipeline_stage`

```text
run_id

stage_name

table_name
table_version

row_count

min_week
max_week

dollar
units

store_count
product_count
commodity_count

null_summary
duplicate_summary

fingerprint

created_at
```

---

# 47. `qc_version_diff`

```text
run_id

grain
entity_id

week
metric

previous_value
current_value

absolute_delta
relative_delta

explained_delta
unexplained_delta

explanation_class
```

---

# 48. `qc_entity_lifecycle`

```text
run_id

entity_type
entity_id

exists_previous
exists_current

first_week_previous
first_week_current

last_week_previous
last_week_current

week_count_previous
week_count_current

historical_weeks_added
historical_weeks_removed

historical_value_added
historical_value_removed

classification
```

---

# 49. `qc_entity_relationship`

```text
entity_type

entity_from
entity_to

relationship

effective_from
effective_to

source
```

---

# 50. `qc_expected_event`

```text
event_id

effective_date

event_type

entity_type
entity_ids

expected_history_start
expected_history_end

expected_direction
expected_scope

description

expires_at
```

---

# 51. `qc_reconciliation`

```text
run_id

parent_grain
parent_entity

child_grain

week
metric

parent_value
children_sum

reconciliation_residual
relative_error
```

---

# 52. `qc_temporal_forecast`

```text
run_id

model_name
model_version

scope
grain
entity_id

week
metric

actual

p01
p05
p10
p50
p90
p95
p99

forecast_residual
relative_residual

raw_forecast_percentile
calibrated_forecast_percentile

raw_interval_width
calibrated_interval_width
```

---

# 53. `qc_temporal_embedding`

```text
run_id

model_name
model_version

series_type

scope
grain
entity_id
metric

start_week
end_week

embedding

anomaly_score

metadata
```

Possible `series_type`:

```text
BUSINESS_SERIES

REVISION_SERIES

DAILY_BUSINESS_SERIES
```

---

# 54. `qc_evidence`

```text
evidence_id

run_id

scope
grain
entity_id
week

evidence_type

value

source_layer

is_deterministic

confidence

related_evidence_ids

metadata
```

---

# 55. `qc_decision`

```text
decision_id

run_id

scope
grain
entity_id
week

decision_field

predicted_value

raw_probability
calibrated_probability

model_name
model_version

backend_type

created_at
```

Possible backend types:

```text
RULE

COMPILED_HEAD

MINICPM_DIRECT

MINICPM_REASONING

RCA_AGENT
```

---

# 56. `qc_incident`

```text
incident_id

run_id

symptom_signature

feature_vector
temporal_embedding

root_cause

likely_origin

affected_tables
affected_entities

resolution

human_summary

created_at
resolved_at
```

---

# 57. `qc_feedback`

```text
decision_id

predicted_label
predicted_probability

human_label

confirmed_root_cause

false_positive
false_negative

reviewed_at
```

---

# 58. Unified QC Feature Table

Potential features include:

## Version features

```text
absolute_delta
relative_delta

changed_weeks

history_changed_fraction

explained_delta
unexplained_delta
explained_fraction

counterfactual_reconciliation_score

new_store_count
new_product_count
new_commodity_count

removed_store_count
removed_product_count
```

## Revision-shape features

```text
largest_contiguous_block

sign_consistency

entity_concentration

top_entity_contribution

revision_variance

net_delta
total_absolute_delta
```

## Structural features

```text
store_count_delta
product_count_delta
commodity_count_delta

backfill_detected
reclassification_detected

history_extended
history_truncated

reconciliation_error

fingerprint_changed
```

## Temporal features

```text
Chronos percentile

Chronos calibrated percentile

forecast residual

robust z-score

seasonal z-score

change-point score

trend-shift score

TSPulse anomaly evidence

TSPulse embedding
```

## Lineage features

```text
source_delta

coded_delta

warehouse_delta

report_delta

first_divergence_stage
```

## Knowledge features

```text
expected_event_match

similar_incident_score

entity_relationship_match
```

---

# 59. Example End-to-End Case: Legitimate New Store

V(n-1):

```text
500 stores
320 weeks
```

V(n):

```text
501 stores
321 weeks
```

Store 9912 appears with 200 historical weeks.

Pipeline:

```text
raw historical differences detected
        ↓
new entity detected
        ↓
Store 9912 absent from V(n-1)
        ↓
historical backfill classification
        ↓
contribution calculated
        ↓
99.97% of raw delta explained
        ↓
counterfactual reconstruction
        ↓
V(n) excluding Store 9912 ≈ V(n-1)
        ↓
PASS_WITH_EXPLANATION
```

Do not escalate.

---

# 60. Example: Missing Stores in Latest Week

Observed:

```text
Dollar:
-29%

Units:
-30%

Stores:
-31%

Products:
-0.5%
```

Chronos:

```text
Dollar < calibrated P01
Units  < calibrated P01
Stores < calibrated P01
```

Statistical detectors:

```text
seasonal z = -5.8
change point = high
```

Lineage:

```text
SOURCE:
143 stores absent

CODED:
same stores absent

WAREHOUSE:
same stores absent
```

Incident memory:

```text
similar previous incident:
supplier extract omitted stores
```

Decision:

```text
latest_week_anomaly:
TRUE

likely_cause:
MISSING_STORES

likely_origin:
SOURCE

requires_investigation:
TRUE
```

---

# 61. Example: Commodity Reclassification

Observed:

```text
Commodity A:
-$40.9m

Commodity B:
+$41.0m

Total:
+$0.1m
```

Entity lifecycle:

```text
Commodity B newly appears historically
```

Entity graph:

```text
possible relationship exists
```

Revision signatures:

```text
A and B historical curves strongly offset
```

Counterfactual:

```text
undoing migration makes current version closely match previous
```

Decision:

```text
EXPECTED_RECLASSIFICATION
```

or:

```text
POSSIBLE_RECLASSIFICATION → INVESTIGATE
```

depending on evidence.

---

# 62. Example: Coding Error

Source:

```text
normal
```

Coded:

```text
commodity X disappears
commodity Y spikes
```

Warehouse:

```text
same pattern preserved
```

First divergence:

```text
CODING
```

Semantic result:

```text
likely_origin:
CODING

likely_cause:
RECLASSIFICATION / MAPPING_ERROR
```

The RCA agent should inspect coding/master-data logic rather than source ingestion.

---

# 63. Synthetic Fault Injection

Build a fault-injection framework.

Inject controlled problems such as:

```text
remove 20% of stores

remove one state

remove products

duplicate rows

duplicate transactions

shift commodity A → B

insert full store backfill

insert commodity backfill

truncate historical period

drop latest week

multiply dollar by 100

swap store IDs

introduce null keys

change only coded stage

change only warehouse stage
```

For every fault record:

```text
expected_detection

expected_cause

expected_origin

expected_severity
```

Then run the full architecture.

---

# 64. Synthetic Evaluation Output

Example:

```text
Fault                      Detect   Cause   Origin

Missing stores              ✓        ✓       ✓

Store backfill              ✓        ✓       N/A

Commodity remap             ✓        ✓       ✓

Duplicate records           ✓        ✗       ✓

History truncation          ✓        ✓       ✓
```

This should become a core CI/benchmark suite.

---

# 65. Historical Backtesting

In addition to synthetic faults, replay historical runs.

For each past week:

```text
history up to t-1
      ↓
forecast t
      ↓
compare to observed t
```

Measure:

```text
forecast coverage

false alert rate

model calibration

anomaly recall

seasonality handling
```

Also replay known historical incidents where available.

---

# 66. Shadow Mode

Initially run the new system beside existing QC.

Capture:

```text
existing QC flags

new architecture flags

analyst decision

confirmed root cause
```

Do not immediately allow AI decisions to suppress existing production checks.

---

# 67. Evaluation Metrics

Measure more than accuracy.

## Detection

```text
precision
recall

false positive rate
false negative rate
```

## Diagnosis

```text
cause classification accuracy

pipeline-origin accuracy

backfill recognition accuracy

reclassification recognition accuracy
```

## Probabilities

```text
Brier score
ECE
reliability curves
```

## Forecasting

```text
coverage

quantile loss

forecast error
```

## Operations

```text
investigation rate

analyst override rate

time to root cause

latency

inference cost
```

Evaluate per anomaly type rather than only overall averages.

---

# 68. Champion / Challenger Framework

Every model interface should support multiple implementations.

Examples:

```text
Chronos-2
vs
FlowState
vs
other forecaster
```

or:

```text
MiniCPM5-SFT
vs
MiniCPM5-final
vs
future instruct model
```

Persist challenger outputs without changing production decisions.

Promote based on measured performance.

---

# 69. Configuration-Driven Design

Avoid dataset-specific hardcoding.

Example:

```toml
[dataset]

name = "retail_sellout"

table = "codeddata_example_fact"

[time]

column = "TIME_PERIOD"
frequency = "W"
history_weeks = 320

[dimensions]

banner = "supplier"
state = "state_code"
store = "store_code"
commodity = "commodity_code"
product = "product_code"

[metrics]

sum = [
    "dollar",
    "units"
]

distinct = [
    "store_code",
    "product_code",
    "commodity_code"
]

[grains]

enabled = [
    "total",
    "banner",
    "state",
    "store",
    "commodity",
    "product"
]

[lineage]

stages = [
    "source",
    "coded",
    "warehouse",
    "report"
]

[temporal.chronos]

enabled = true
model = "amazon/chronos-2"

[temporal.tspulse]

enabled = true
model = "ibm-granite/granite-timeseries-tspulse-r1"

use_for_weekly_anomaly_detection = false
use_for_daily_anomaly_detection = true
use_for_embeddings = true
use_for_similarity = true

[temporal.statistics]

robust_zscore = true
seasonal = true
change_point = true

[calibration]

conformal_forecasts = true

[decision]

enabled = true
model = "openbmb/MiniCPM5-2B-SFT"

candidate_scoring = "sequence_likelihood"

short_reasoning_fallback = true

[escalation]

reasoning_agent = true
```

---

# 70. Software Module Layout

Suggested repository structure:

```text
retail-qc/
│
├── config/
│   ├── datasets/
│   └── models/
│
├── qc/
│   │
│   ├── contracts/
│   │
│   ├── versions/
│   │
│   ├── diffs/
│   │
│   ├── lifecycle/
│   │
│   ├── attribution/
│   │
│   ├── counterfactual/
│   │
│   ├── reconciliation/
│   │
│   ├── lineage/
│   │
│   └── fingerprints/
│   │
│   ├── temporal/
│   │   ├── chronos.py
│   │   ├── tspulse.py
│   │   ├── statistics.py
│   │   ├── changepoint.py
│   │   └── calibration.py
│   │
│   ├── evidence/
│   │   ├── graph.py
│   │   └── builders.py
│   │
│   ├── decision/
│   │   ├── runtime.py
│   │   ├── minicpm.py
│   │   ├── choices.py
│   │   ├── calibration.py
│   │   └── compiled_heads.py
│   │
│   ├── memory/
│   │   ├── incidents.py
│   │   └── similarity.py
│   │
│   ├── agent/
│   │   └── investigation.py
│   │
│   └── reporting/
│
├── benchmarks/
│   ├── forecasting/
│   ├── semantic_decisions/
│   ├── tspulse/
│   ├── fault_injection/
│   └── end_to_end/
│
├── tests/
│
└── docs/
    └── architecture.md
```

---

# 71. Core Interfaces

## Version comparator

```python
comparison = compare_versions(
    previous_version=previous,
    current_version=current,
    config=config,
)
```

---

## Lifecycle analysis

```python
events = classify_entity_changes(comparison)
```

---

## Attribution

```python
attribution = explain_revision(
    comparison=comparison,
    lifecycle_events=events,
    expected_events=expected_events,
)
```

---

## Chronos

```python
forecast = chronos.predict(
    history=history,
    horizon=1,
    covariates=covariates,
)
```

---

## TSPulse

```python
temporal = tspulse.embed(series)
```

and where valid:

```python
anomaly = tspulse.detect(series)
```

---

## Evidence

```python
evidence = EvidenceGraph.build(
    comparison=comparison,
    attribution=attribution,
    forecasts=forecast,
    temporal_embeddings=temporal,
    lineage=lineage,
    incidents=incidents,
)
```

---

## Semantic runtime

North-star API:

```python
result = runtime.decide(
    context=evidence,
    fields={
        "latest_week_anomaly": Boolean(),

        "likely_cause": Choice([
            "missing_stores",
            "missing_products",
            "source_ingestion",
            "coding",
            "warehouse",
            "market_movement",
            "historical_correction",
            "reclassification",
            "backfill",
            "unknown",
        ]),

        "severity": Choice([
            "low",
            "medium",
            "high",
            "critical",
        ]),

        "requires_investigation": Boolean(),
    },
)
```

Return:

```python
result["likely_cause"].value
result["likely_cause"].raw_probability
result["likely_cause"].calibrated_probability
```

---

# 72. Weekly Execution Flow

Conceptual orchestration:

```python
def run_qc(dataset_config):

    run = create_qc_run()

    validate_contracts()

    versions = resolve_versions()

    lineage = build_pipeline_stage_summaries()

    comparison = compare_versions(
        versions.previous,
        versions.current,
    )

    lifecycle = classify_entity_lifecycle(comparison)

    expected_events = load_expected_events()

    relationships = load_entity_relationships()

    attribution = explain_changes(
        comparison,
        lifecycle,
        expected_events,
        relationships,
    )

    counterfactual = reconstruct_counterfactuals(
        comparison,
        attribution,
    )

    reconciliation = run_reconciliation()

    latest_week_history = build_temporal_series()

    chronos = run_chronos(latest_week_history)

    statistics = run_statistical_detectors(
        latest_week_history
    )

    tspulse = run_tspulse_where_supported(
        latest_week_history,
        comparison.revision_series,
    )

    incidents = retrieve_similar_incidents(
        features=comparison.features,
        temporal_embedding=tspulse.embedding,
    )

    evidence = build_evidence_graph(
        comparison=comparison,
        lifecycle=lifecycle,
        attribution=attribution,
        counterfactual=counterfactual,
        reconciliation=reconciliation,
        chronos=chronos,
        statistics=statistics,
        tspulse=tspulse,
        lineage=lineage,
        incidents=incidents,
    )

    decision = semantic_runtime.decide(evidence)

    if decision.requires_investigation:
        investigation = reasoning_agent.investigate(
            evidence
        )

    persist_everything()

    generate_report()
```

---

# 73. Build Order for Codex

## Phase 1 — Core deterministic engine

Implement:

```text
configuration

Delta version resolution

overlapping/new period detection

revision cube

multi-grain aggregates

basic reporting
```

No model dependency required.

---

## Phase 2 — Entity lifecycle

Implement:

```text
new entities

removed entities

historical backfills

history truncation

history extension
```

Add unit tests using synthetic tables.

---

## Phase 3 — Attribution

Implement:

```text
contributor ranking

explained delta

unexplained residual

explained fraction
```

---

## Phase 4 — Counterfactual reconstruction

Implement:

```text
remove explained structural changes

recalculate current totals

compare reconstructed current to previous
```

---

## Phase 5 — Reconciliation + structural integrity

Implement:

```text
parent/child mass balance

cross-metric ratios

fingerprints

null/duplicate diagnostics
```

---

## Phase 6 — Pipeline lineage

Implement configurable stage summaries and first-divergence detection.

---

## Phase 7 — Expected events + entity graph

Implement:

```text
expected-event registry

entity relationships

mapping migration evidence
```

---

## Phase 8 — Chronos-2

Implement:

```text
latest-week forecasts

quantile outputs

residual features

historical backtests
```

---

## Phase 9 — Robust statistical ensemble

Implement:

```text
MAD

seasonal z-score

change-point detector

trend-shift features
```

---

## Phase 10 — Forecast calibration

Implement rolling backtests and empirical/conformal intervals.

---

## Phase 11 — TSPulse research adapter

Implement:

```text
TSPulse model wrapper

embedding extraction

similarity benchmark

daily anomaly benchmark where daily data is available

weekly 320-point suitability benchmark
```

Do not make TSPulse anomaly output a production dependency until the sequence-length experiments pass.

---

## Phase 12 — Incident memory

Persist confirmed incidents and support retrieval using:

```text
structured features

TSPulse embeddings

optional text embeddings
```

---

## Phase 13 — Evidence graph

Create a canonical structured representation of every QC case.

The evidence graph becomes the boundary between:

```text
analytics / models
```

and:

```text
semantic decision runtime
```

---

## Phase 14 — Semantic decision benchmark

Compare:

```text
MiniCPM5 Base

MiniCPM5 SFT

MiniCPM5 final
```

and scoring modes:

```text
single-token

candidate sequence likelihood

short-reasoning constrained

normal instruct generation
```

Choose based on measured accuracy/calibration/latency.

---

## Phase 15 — Semantic runtime

Implement typed Boolean/Choice API.

Do not require free-form generation.

---

## Phase 16 — Fault injection

Build synthetic corruption scenarios and end-to-end regression tests.

---

## Phase 17 — Shadow production

Run alongside existing QC.

Persist analyst outcomes.

---

## Phase 18 — RCA agent

Only after evidence quality is strong, connect a reasoning agent capable of querying Databricks.

---

## Phase 19 — Compiled heads

Train specialist classifiers from accumulated confirmed outcomes.

---

# 74. Early MVP Boundary

Do not attempt every feature simultaneously.

A useful first integrated MVP is:

```text
Delta V(n-1) ↔ V(n)
       │
       ▼
revision cube
       │
       ▼
entity lifecycle
       │
       ▼
explained / residual difference
       │
       ├───────────────────┐
       ▼                   ▼
latest-week history     historical revision
       │                   │
   Chronos-2          signature features
       │                   │
       └─────────┬─────────┘
                 ▼
            evidence object
                 │
                 ▼
        MiniCPM typed decision
```

Then add:

```text
TSPulse
lineage
incident retrieval
counterfactuals
RCA agent
```

incrementally.

---

# 75. Important Non-Goals

Do not:

```text
send millions of raw transaction rows to an LLM

treat every historical difference as an anomaly

assume model confidence is calibrated

assume TSPulse works on 320 weekly observations without testing

let Chronos determine business root cause

let MiniCPM perform basic arithmetic that SQL can perform exactly

remove deterministic QC because an AI model appears accurate

invoke an expensive reasoning agent on every run

hardcode the architecture around one model vendor
```

---

# 76. Model Independence

These roles matter more than individual model names.

```text
FORECAST MODEL
currently Chronos-2

TEMPORAL REPRESENTATION MODEL
currently TSPulse

SEMANTIC DECISION MODEL
currently MiniCPM5-2B-SFT candidate

RCA MODEL
replaceable reasoning agent
```

All model adapters should conform to stable internal interfaces.

---

# 77. Machine-Readable Output

Every run should produce structured outputs suitable for another agent.

Example:

```json
{
  "run_id": "QC_20260917",
  "status": "INVESTIGATE",

  "latest_week": {
    "anomaly": true,
    "cause": "MISSING_STORES",
    "origin": "SOURCE",
    "confidence": 0.96
  },

  "historical_revision": {
    "status": "PASS_WITH_EXPLANATION",
    "raw_delta": 104200,
    "explained_delta": 100900,
    "unexplained_delta": 3300,
    "explained_fraction": 0.968
  },

  "evidence": [
    "dollar_below_calibrated_p01",
    "units_below_calibrated_p01",
    "store_count_down_29_8_pct",
    "product_count_normal",
    "first_divergence_source",
    "similar_historical_incident"
  ]
}
```

---

# 78. Human-Readable Output

Generate:

```text
Markdown
PDF
Excel
```

or existing reporting formats.

Example summary:

```text
QC RUN
V1050 → V1051

LATEST WEEK
2026-09-12

STATUS
INVESTIGATE

Dollar:
below calibrated 1st percentile

Units:
below calibrated 1st percentile

Stores:
-29.8%

Products:
-0.5%

Chronos:
strong negative forecast deviation

Robust statistics:
seasonal z = -5.8

Likely cause:
MISSING_STORES

Likely origin:
SOURCE

Confidence:
96%

HISTORICAL REFRESH

319 overlapping weeks compared.

Raw historical delta:
$104,200

Explained:
$100,900

Explanation:
new-store historical backfill

Unexplained:
$3,300

Historical status:
PASS_WITH_EXPLANATION

SIMILAR INCIDENT

Previous pattern:
supplier source extract omitted stores

Recommended investigation:
compare expected store list against current source delivery
```

---

# 79. End-State Learning Loop

The complete system should improve from each resolved run.

```text
weekly load
    ↓
detection
    ↓
decision
    ↓
investigation
    ↓
confirmed root cause
    ↓
analyst feedback
    ↓
incident memory
    ↓
calibration dataset
    ↓
specialist classifier training
    ↓
better future decisions
```

---

# 80. Long-Term Architecture

The eventual fast path may look like:

```text
              structured QC features
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
       Chronos      TSPulse     lineage
       features     embedding   features
           │           │           │
           └───────────┼───────────┘
                       ▼
                  fusion vector
                       │
                       ▼
               compiled classifier
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      normal        known issue     uncertain
        │              │              │
       pass          explain           ▼
                                semantic runtime
                                       │
                                  uncertain?
                                       │
                                       ▼
                                    RCA agent
```

At that point MiniCPM is no longer required for every QC object.

It becomes a semantic fallback and router for novel or ambiguous situations.

---

# 81. North-Star Behaviour

The system should progress from:

```text
Sales changed by 8%.

FLAG.
```

to:

```text
Historical sales increased 8.3%.

95.2% of the change is attributable to Store 9912,
which is newly present in the current version with
240 weeks of historical data.

The expected-event registry confirms that Store 9912
was onboarded this week with historical coverage.

A counterfactual current version excluding Store 9912
closely matches the previous table version.

Another 3.9% of the change is consistent with a commodity
reclassification.

0.9% remains unexplained.

Historical revision:
PASS_WITH_EXPLANATION.


The newly arrived week is independently anomalous.

Dollar:
below calibrated 1st percentile.

Units:
below calibrated 1st percentile.

Active stores:
-29%.

Products:
normal.

The same store deficit is already present in the source
snapshot and persists through coded and warehouse stages.

A similar historical temporal pattern is associated with
a previous source delivery that omitted stores.

Latest-week status:
INVESTIGATE.

Likely cause:
MISSING_STORES.

Likely origin:
SOURCE.

Confidence:
96%.

Recommended first investigation:

Compare the expected store population against the current
source extract and identify stores present in the prior week
but absent from the current source delivery.
```

---

# 82. Central Architectural Thesis

The project should not be thought of as:

```text
an LLM that performs QC
```

It should be thought of as:

```text
a deterministic + temporal + semantic evidence system
for automated data-quality diagnosis
```

where:

```text
SQL / Spark
establish facts

Chronos-2
models expected temporal behaviour

TSPulse
provides temporal representations,
anomaly evidence and similarity where appropriate

statistical models
provide independent anomaly evidence

the evidence graph
combines established observations

MiniCPM / compiled heads
turn evidence into typed semantic decisions

the RCA agent
investigates only unresolved cases

human outcomes
continually improve calibration, memory and classifiers
```

That is the concept to build.