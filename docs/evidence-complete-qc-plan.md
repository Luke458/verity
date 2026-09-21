# Evidence-complete retail QC and dependable review decisions

> **Implementation status.** Delivered across the four increments below.
>
> 1. *Historical correctness and evidence contracts*: target-isolated
>    rolling-origin fitting, model selection and calibration; explicit missing
>    weeks; residual pools by metric/level/horizon/revision; declared business
>    calendar with retail year/week and event windows; versioned
>    `AssessmentEvidence`/`ExplanationCertificate` schema 1
>    (`qc/calendar.py`, `qc/temporal.py`, `qc/evidence_package.py`).
> 2. *Statistical expectations and explanation ledger*: hierarchy aggregates and
>    coverage checks, sparse parent-share support, symmetric quantity/unit-value
>    decomposition with conservation and separate arithmetic/support labels
>    (`qc/hierarchy.py`, `qc/ledger.py`).
> 3. *Review policy and evaluation*: finding dispositions, certificate-only
>    statistical clearance, Benjamini-Hochberg leaf control, coordinated
>    residual combination, recurrence escalation, evaluation metrics and
>    confidence-bound gates (`qc/policy.py`, `qc/recurrence.py`,
>    `qc/evaluation.py`).
> 4. *Operational integration and cleanup*: `qc explain`, `qc evidence-bench`,
>    machine report schema 3, finding schema 2, evidence package schema 1,
>    journaled `evidence.json`, calendar/evidence-policy identity and
>    hierarchy aggregate caching (`qc/cli.py`, `qc/evidence_bench.py`,
>    `qc/weekly.py`, `qc/store.py`).
>
> Acceptance fixtures: `tests/test_qc_acceptance.py`,
> `tests/test_qc_evidence_package.py`, `tests/test_qc_evaluation.py`,
> `tests/test_qc_evidence_bench.py`, `tests/test_qc_explain.py`. The claims
> matrix records these as synthetic-only; real-data accuracy is measured
> separately and is not assumed from history length or synthetic results.

## 1. Goal and agreed behaviour

Build an evidence-first assessment that answers: **“Is this movement consistent with established business behaviour, and is any material change still unexplained?”**

Use the available 320-week history, calendar, declared hierarchy, multiple measures and pipeline controls. A deterministic verifier makes the operational review decision. Laya and other models receive the same evidence as measured challengers.

Agreed defaults:

- Validated statistical explanations may clear their specific anomaly without a new human approval.
- Hard integrity failures and missing required evidence still require review.
- Small product/commodity anomalies remain visible; material impact, persistence and coordinated patterns determine escalation.
- Start with a versioned calendar and existing transactional/dimension data. Promotion, delivery and business-event records remain optional extension points.
- Calibrate thresholds provisionally, freeze them before evaluation, and keep them configurable.
- Preserve local CPU, Parquet/delta-rs, read-only shadow operation and existing artifacts. Real-world qualification remains pending real analyst evidence.

## 2. Build the statistical evidence engine

### Correct historical isolation first

Fix the confirmed defect in `qc/temporal.py`: current target observations enter the calibration backtest. Forecast fitting, model selection and calibration must use only evidence available before the assessment target.

Assess every newly introduced period, not just the latest. Preserve missing weeks explicitly instead of compressing sparse histories into consecutive observations. Separate residual pools by metric, aggregation level, forecast horizon and model revision.

Use rolling-origin evaluation; each forecast must precede its evaluated observation. This follows the historical isolation described in [Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html).

### Calendar and expected behaviour

Add an immutable calendar input mapping business periods to dates, retail year/week and event windows. Initially support Christmas, Easter and financial-year-end, including adjacent weeks and 53-week years. Synthetic fixtures supply an explicit Australian calendar; real datasets must declare their calendar rather than inherit an assumed geography.

Implement two lightweight CPU candidates:

- The existing seasonal-naive baseline.
- Ridge regression using trend, annual Fourier terms and calendar-event indicators. Use a fixed candidate grid: Fourier order `{1, 2, 3}` and standardized-feature ridge penalty `{0.1, 1, 10}`.

Select using historical rolling-origin absolute error; break ties by interval score, then simpler model. Freeze selection before interval calibration. Calendar indicators and seasonal terms follow the approach described in [useful forecasting predictors](https://otexts.com/fpp3/useful-predictors.html).

Use at least 104 completed weeks for annual-seasonality explanations. Calibrate prediction intervals from held-out forecast errors; insufficient calibration prevents automatic explanation. Report history length, forecast error, interval coverage and width alongside each prediction.

### Hierarchy and measures

Build reusable aggregates for national, banner, state, commodity, product and store scopes, plus declared commodity-by-banner/state intersections.

- Assess sales and units; assess scripts where present.
- Treat stock as a snapshot measure, never accumulated sales.
- Calculate effective unit values only where units make the calculation meaningful.
- Run inexpensive coverage and residual checks at all declared levels. Reserve expensive fitting and detailed reconstruction for aggregate series and flagged descendants.
- Use parent expectations with historically estimated child shares for sparse series. Label this fallback explicitly; unsupported sparse leaves cannot create an automatic explanation.
- Keep each hierarchy internally reconcilable. Treat geographic and commercial breakdowns as alternative views, not additive explanations.

## 3. Turn observations into verified explanations

### Canonical evidence package

Introduce versioned `AssessmentEvidence` and `ExplanationCertificate` objects shared by rules, reports, feature heads, text probes and HTTP providers.

Each package records:

- Actual versus expected values and intervals.
- Statistical support and calibration availability.
- Coverage, reconciliation and lineage results.
- Contributions, cross-metric consistency and contradictory evidence.
- Net and gross unexplained residuals.
- Snapshot/calendar/model/configuration identities and observation cutoffs.
- Evidence IDs and any omitted evidence.

Preserve the complete package as an artifact. Provider-specific views may compress it, but must retain mandatory evidence or abstain.

### Quantified explanation ledger

Keep snapshot revisions separate from new-period business movement. Explain movements through:

- Seasonal/calendar expectations and trend.
- Like-for-like versus changing store/product coverage.
- Matched-item quantity and unit-value effects, using symmetric decomposition so their contributions sum exactly.
- Entry, exit, assortment and classification changes.
- Independently approved business events when available.

Separate **arithmetic attribution** from **support for a legitimate explanation**. Identifying missing stores or higher prices does not, by itself, authorize clearing the anomaly.

Allocate each contribution once. Keep positive and negative residuals visible so offsetting errors cannot disappear. A certificate must identify its exact findings, supporting evidence, coverage and unexplained remainder.

### Review policy

Extend finding dispositions to distinguish:

- Hard failure.
- Unavailable required evidence.
- Unexplained actionable anomaly.
- Statistically explained movement.
- Human-approved exception.
- Informational finding below escalation thresholds.

Automatic statistical clearance requires all applicable integrity checks to pass, historically calibrated support, no contradictory evidence, and residual impact below the frozen materiality threshold. The explained fraction alone is never sufficient. Wide or unqualified intervals cannot justify clearance.

Start threshold selection from the existing aggregate materiality ratio of 0.1%. Evaluate `{0.05%, 0.1%, 0.2%}` on development cases; choose the configuration minimizing review volume while passing safety gates. Monetary limits remain configurable rather than inventing a currency-specific default.

For small findings:

- Escalate material aggregate or local effects.
- Escalate recurrence in at least two of three refreshes when cumulative impact becomes material.
- Combine coordinated, same-direction residuals within declared parent groups before testing materiality.
- Apply false-discovery control to leaf anomaly screening; it never suppresses hard failures.

Keep existing overall statuses and exit codes. Extend `PASS_WITH_EXPLANATION` to verified statistical certificates, recording a distinct clearance basis from human approval. No model can override the verifier.

## 4. Evaluation, provider comparison and repository improvements

### Measure evidence quality separately from model quality

Extend the common harness to report:

- Raw challenger decisions.
- Deterministic verifier decisions.
- Effective operational decisions.
- Detection, false clearance, review volume, cause/severity accuracy, abstention and explanation coverage.
- Forecast coverage, interval width, calibration diagnostics, latency and peak memory.

Do not count policy overrides as correct model predictions. Exclude final decisions, approval outcomes and oracle labels from challenger inputs; retain the underlying check evidence.

Preserve chronological, incident-grouped train/calibration/development/test splits. Freeze thresholds, evidence versions and one challenger before final test. Add an evidence ablation comparison: current summary, forecast evidence, hierarchy/multi-metric evidence, then the complete package.

Engineering acceptance requires zero incorrect outcomes on deterministic policy fixtures. Statistical qualification retains the existing detection/FPR confidence-bound gates and adds a **95% upper confidence bound of 1% on false clearance of actionable incidents**. Insufficient independent incidents remain insufficient evidence; synthetic success never grants production eligibility.

### Make the repository easier to operate and maintain

- Preserve the current uncommitted implementation and establish it as the regression baseline.
- Extract shared evidence construction, explanation verification and serialization from CLI/provider handlers into typed services.
- Add `qc explain` for an assessment’s ledger/certificates and `qc evidence-bench` for the frozen benchmark workflow.
- Introduce machine-report schema 3, finding schema 2 and evidence-package schema 1. Retain readers for legacy artifacts; do not invent missing historical evidence.
- Include calendar, statistical artifacts and evidence-policy versions in assessment/cache identity. Persist new artifacts through the existing SQLite journal and recovery path.
- Add aggregate caching and column projection. Benchmark temporal processing enabled on 320-week histories, including full synthetic profile and one-/five-million-row scale fixtures.
- Keep core dependencies lightweight, Delta tests mandatory and model environments isolated.
- Update README, claims, architecture, evaluation and recovery runbooks together. Reproduce numerical claims through artifact-generating commands.

## 5. Acceptance scenarios and rollout

Add independent, hand-authored fixtures alongside generator scenarios:

| Scenario | Required result |
|---|---|
| Expected post-Christmas decline with complete integrity evidence | Verified explanation; no review |
| Same decline outside seasonal expectations | Material unexplained change; review |
| Moving Easter or a 53-week year | Correct calendar alignment |
| Missing ingestion hidden inside an otherwise seasonal decline | Integrity/coverage failure; review |
| Small isolated product fluctuation | Visible finding; no automatic dataset escalation |
| Persistent or coordinated small errors | Aggregate/persistence escalation |
| Opposing commodity errors with unchanged national total | Gross residual detected |
| Price/mix change | Exact attribution; clearance only with supporting evidence |
| New products, sparse histories, stockouts or missing weeks | Explicit support level; no invented evidence |
| Current/future observations or corrected labels added | Earlier forecasts, calibration and decisions unchanged |
| Challenger says “no review” against a hard failure | Operational review preserved; challenger error counted |
| Mandatory evidence exceeds Laya’s budget | Explicit abstention; verifier result preserved |

Deliver in four ordered increments:

1. **Historical correctness and evidence contracts:** fix leakage, add calendar/provenance interfaces and regression fixtures.
2. **Statistical expectations and explanation ledger:** hierarchical forecasts, cross-metric attribution and certificate verification.
3. **Review policy and evaluation:** automatic scoped clearance, materiality/persistence handling, independent metrics and ablations.
4. **Operational integration and cleanup:** journaled artifacts, CLI/reporting, CPU benchmarks, provider comparisons and documentation.

The resulting claim is precise: complete evidence can produce a reproducible, testable review decision under an explicit policy. Real-data accuracy is measured separately and is not assumed from history length or synthetic results.
