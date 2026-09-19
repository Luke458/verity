# Semantic decision layer, incident memory and agent handoff

Milestone D turns the evidence graph into typed decisions, retrieves similar
historical incidents, and - when investigation is required - hands a structured
brief to a reasoning agent. The deterministic and temporal layers remain the
source of truth; the semantic layer never replaces arithmetic with inference.

## Decision fields

| Field | Kind | Classes |
|---|---|---|
| `likely_cause` | choice | MISSING_STORES, MISSING_PRODUCTS, SOURCE_INGESTION, CODING, WAREHOUSE, MARKET_MOVEMENT, HISTORICAL_CORRECTION, RECLASSIFICATION, BACKFILL, SCHEMA_FAILURE, UNKNOWN |
| `likely_origin` | choice | SOURCE, PREPROCESSING, CODING, WAREHOUSE, REPORT, UNKNOWN |
| `severity` | score (ordered) | LOW < MEDIUM < HIGH < CRITICAL |
| `requires_investigation` | boolean | False, True |

Score fields carry an `index`, the probability-weighted level index with the
lowest level at 0, in addition to the most probable level and its
distribution.

## Jev wire mapping

| Our kind | Jev primitive | Mapping |
|---|---|---|
| boolean | `noul` | probability of True |
| choice | `choice` | canonical option order preserved |
| score | `score` | `index` weighted level, `legend` index-to-level, probabilities keyed by level index |

`decision_to_systemone_answers` emits this format; `answers_to_decisions`
parses it, accepting probabilities keyed by level index or level name, falling
back to a one-hot at the nearest index, and accepting a plain `choice` answer
for an ordinal field with the rank as its index. A decision set therefore
round-trips through a Jev-compatible endpoint, and older servers that only
speak `choice` remain compatible. Provider-reported probabilities stay tagged
`provider_reported`.

Every decision carries probabilities, `probability_kind`, `strategy` and
evidence references. `heuristic` probabilities come from evidence strength
rules; `temperature_scaled` probabilities come from a trained head. Neither is
a calibration certificate.

## Rule provider

`RuleDecisionProvider` is the default. It mirrors the deterministic semantics:

| Evidence | Cause | Requires investigation |
|---|---|---|
| contract failure | SCHEMA_FAILURE | yes |
| `LATEST_WEEK_MISSING` (store) | MISSING_STORES | yes |
| `LATEST_WEEK_MISSING` (product) | MISSING_PRODUCTS | yes |
| registered backfill | BACKFILL | no |
| unregistered backfill | BACKFILL | yes |
| truncation / removal | HISTORICAL_CORRECTION | yes unless registered |
| reclassification | RECLASSIFICATION | yes unless registered |
| unexplained material change, first divergence coded | CODING | yes |
| unexplained material change, first divergence warehouse | WAREHOUSE | yes |
| unexplained material change, first divergence source | SOURCE_INGESTION | yes |
| broad immaterial historical change | HISTORICAL_CORRECTION | yes |
| temporal-only anomaly | MARKET_MOVEMENT | no (movement is an explanation) |

The rule provider classifies all eleven synthetic fault families exactly. That
is a test of the mapping, not evidence about real refresh data.

## Trained provider

`TrainedDecisionProvider` fits one linear softmax head per field over the
versioned feature encoder (`qc/decisions.py`, `FEATURE_VERSION`) and fits a
scalar temperature per field on a held-out split.

```sh
# Collect labels while shadow-running a suite (oracle labels for synthetic data)
qc shadow --suite-dir data/suites/demo --labels-out data/labels/demo.jsonl

# Fit and evaluate
qc train --labels data/labels/demo.jsonl --out reports/artifacts/decision-demo

# Use the artifact
qc run --scenario-dir data/suites/demo/scenario-0000 --provider reports/artifacts/decision-demo
```

Artifacts record `label_sources`, `families`, `n_records` and a warning when
they were trained on synthetic labels only. Feature-version or feature-name
mismatches are rejected at load time.

## Frozen-encoder text probe

ModernBERT-class encoders can embed the canonical evidence text and drive the
same fields through a trained linear probe (`qc/text_provider.py`): one forward
pass per case, long context, CPU-friendly, and strong when labels are few. It
is a challenger substrate, not a default, and its probabilities are tagged
`frozen_encoder_probe`. See [docs/text-provider.md](text-provider.md).

## Label store and the accuracy gate

Synthetic oracle labels exist to validate the plumbing. A provider trained on
them will look strong because the fault families are cleanly separable.
**Accuracy claims for real data require real analyst labels.** The shadow
harness already records every run's features and outcomes; analyst feedback
(`qc investigate ... --save-draft`, later confirmed) turns those records into
training data. Until then, the rule provider is the operational default and
the learned provider is a measured experiment.

## Incident memory

`IncidentStore` persists confirmed incidents with their feature vector, symptom
tags, root cause, origin and resolution. Retrieval ranks by cosine similarity
over the versioned features with a symptom-tag overlap bonus. Incidents saved
by an agent are drafts (`confirmed=False`) until an analyst confirms them.

```sh
qc incidents list --store data/incidents.jsonl
```

## Agent handoff

When `requires_investigation` is true, `build_investigation_brief` packages:

- candidate causes with probabilities,
- findings from contracts, lifecycle, attribution, counterfactual, lineage and
  temporal evidence,
- open questions derived from the cause and any unregistered structural change,
- recommended first queries (SQL skeletons parameterised by week and entities),
- similar historical incidents with resolutions,
- evidence ids, read-only constraints and the expected response schema.

```sh
qc investigate --scenario-dir data/suites/demo/scenario-0000 \
  --incident-store data/incidents.jsonl \
  --agent-cmd "my-llm-agent --json"
```

`CommandAgent` runs any command with the brief JSON on stdin and parses the
result JSON from stdout, so no provider is built into the engine. `NullAgent`
returns the brief without invoking anything. Responses are validated: a
missing key, malformed JSON, non-zero exit or a confidence outside [0, 1]
raises rather than being treated as success.

Result schema:

```json
{
  "root_cause": "string",
  "confidence": 0.0,
  "evidence_ids": ["..."],
  "recommended_actions": ["..."],
  "summary": "string",
  "follow_up_questions": ["..."]
}
```

## SARIMAX

SARIMAX is available as an optional classical challenger
(`pip install -e ".[classical]"`, `forecaster: sarimax`) for covariate-adjusted
expected values and independent comparison on stable low-count series. It is
not a default: per-series order selection and slow fits do not pay off across
thousands of series when a foundation forecaster and robust statistics already
provide quantiles and independent agreement.
