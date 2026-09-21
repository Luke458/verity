# Repair QC clearance, historical reproducibility and evaluation

> **Superseded (v0.22.0).** The follow-up review in
> [qc-second-remediation-plan.md](qc-second-remediation-plan.md) tightened
> certificate binding, extended assessment to every configured measure and
> replaced the permissive qualification contract; see that plan for the current
> status.

> **Implementation status (v0.21.0).** All four increments are implemented;
> every acceptance scenario in section 5 has regression coverage in
> `tests/test_qc_remediation_acceptance.py`, and the full suite, lint and type
> gates pass. Automatic statistical clearance is disabled unless a pinned
> qualification artifact (`qc evidence-bench --out ...` writes one) covers the
> exact model/metric/level/horizon and the assessment provenance. Synthetic
> qualification cannot authorize real-data clearance. Measured 320-week
> benchmarks are recorded under `reports/benchmarks/` (one and five million
> rows); they are single-machine CPU measurements, not scale qualification.
> Real-world accuracy remains unresolved until independent analyst-labelled
> data exists.

## 1. Outcome and implementation order

Fix all eight review findings while preserving existing reports, assessment history and analyst labels. Rules remain authoritative; statistical explanations may clear only their exact, eligible anomalies. Laya remains an optional challenger.

Deliver in four increments:

1. Safe certificate binding and complete period assessment.
2. Correct forecasting, calibration and historical identity.
3. Shared provider evidence and independent evaluation.
4. Migration, regression verification and corrected documentation.

First convert every demonstrated failure into a regression test. Disable automatic statistical clearance in the repaired runtime until its qualification checks are implemented; evidence generation and deterministic review continue.

## 2. Findings, certificates and complete assessment coverage

**Separate evidence construction from final policy.** Run checks, create immutable findings, build evidence, verify explanations against those findings, then compute final status once. Provider recommendations remain separate.

- Introduce versioned structured scopes containing dataset, stage, metric, period and hierarchy coordinates. Reference identifiers are identifiers, never interchangeable with hierarchy scopes.
- Bind certificates to exact finding IDs, assessment identity, evidence digest and policy version. Reject empty coverage, mismatched identities, altered evidence and unsupported certificate versions.
- Allow statistical clearance only for explicitly eligible temporal anomalies. Contract, reconciliation, reference, lineage, hierarchy-integrity and required-evidence failures cannot receive statistical clearance. Historical revisions require independent attribution or scoped approval.
- Normalize check outcomes through shared types. Remove string comparisons between incompatible contract-status vocabularies.
- Certificate eligibility requires successful applicable integrity checks, complete mandatory evidence, supported history, qualified calibration, finite intervals, no scoped contradictions and residual impact below scoped materiality.
- Annual explanations require at least 104 completed, observed historical weeks. Sparse parent-share estimates remain informational and cannot authorize clearance.
- Require a pinned qualification artifact specifying supported model/metric/level/horizon combinations, coverage requirements and interval-width limits. Missing qualification prevents automatic clearance. Width limits must be selected on development evidence and validated on untouched test evidence; no permissive implicit default.
- Keep synthetic qualification explicitly restricted to synthetic assessments. It cannot authorize real-data clearance or production eligibility.

Assess **every appended period and every configured QC measure**, respecting additive versus snapshot aggregation rules. Build forecasts from the selected previous snapshot; earlier newly appended observations must not enter training for later targets in that same assessment.

Create period-specific findings, forecasts, ledger entries and certificates. Derive totals from matching period-specific entries, preserving gross unexplained movement so offsetting errors remain visible.

For temporal findings, calculate materiality against the same metric, period and scope’s expected magnitude, with configured absolute floors. Never use the accumulated historical total as a weekly denominator. Revision materiality remains tied to the affected historical comparison scope.

## 3. Forecasting, calibration and historical reproducibility

**Use explicit, disjoint time partitions.**

- Reserve the latest configured calibration origins exclusively for calibration.
- Select candidates using earlier rolling origins, requiring at least the existing minimum selection count. Fit each origin using only observations available before its forecast origin.
- Freeze the selected specification before evaluating calibration targets. Final forecast fitting may use all eligible previous-snapshot observations.
- Return insufficient evidence when required partitions cannot be formed; never reuse selection observations as calibration.
- Derive horizons from configured business periods, preserving missing periods. Add assertions recording training endpoint, forecast origin and target period to prevent the existing off-by-one error.

**Correct residual units.**

- Pool standardized errors only across compatible metric, hierarchy-level, model-revision and horizon groups.
- Estimate each target’s scale from its historical inputs and transform pooled residual quantiles back into that target’s units.
- Keep raw residuals for diagnostics only. Quantiles and intervals must use the same documented units.
- Evaluate qualification using held-out forecast outcomes; do not treat pooled sample counts or in-sample coverage as independent target-series support.

**Freeze recurrence before computing assessment identity.**

- Select distinct logical predecessor refreshes strictly before the assessment’s observation cutoff, also respecting business-period order.
- Select the latest eligible historical revision for each predecessor at that cutoff. Exclude retries and alternative attempts as duplicate evidence.
- Persist a recurrence-input manifest containing selected assessment/revision IDs, evidence hashes and cutoff; include its digest in assessment identity before cache lookup.
- Match recurrence using a separate stable check/metric/entity key, since exact finding IDs now include periods. Attribute impact to matching findings rather than the whole dataset.
- Missing required predecessor evidence produces explicit unavailability. An initial assessment with no predecessor history is a recorded cold start.

Changing eligible recurrence inputs creates a new assessment; future inputs cannot affect historical replay. Retries of an existing assessment reuse its frozen manifest.

## 4. Provider evidence and trustworthy evaluation

**Route every provider through one evidence contract.**

- Replace legacy System One, Laya, text and feature representations with adapters over the recorded evidence package.
- Exclude final QC status, certificate verdicts, effective review decisions and outcome labels from provider inputs. Include underlying check results, calendar, forecast uncertainty, scoped ledger, cross-metric evidence and provenance.
- Build evidence independently of whether automatic clearance is enabled.
- Preserve failed checks, missing required evidence and decision-relevant predictions during packing. Record optional omissions; abstain if mandatory evidence and complete question/options cannot fit.
- Version feature encoders and reject incompatible trained artifacts rather than silently mapping new evidence to old features.
- Persist the canonical evidence hash and exact provider payload hash. Provider failure preserves deterministic results and records challenger unavailability.

**Replace evaluation shortcuts.**

- Freeze incident membership and chronological train/calibration/development/test assignments in an evaluation manifest. Related snapshots remain together.
- Use development cases to select materiality from the existing `{0.05%, 0.1%, 0.2%}` candidates and select one challenger. Evaluate the frozen choice once on untouched test cases.
- Hash data identities, labels and revisions, cutoffs, splits, configuration, engine, evidence schema, provider artifacts and candidate grid—not just the suite file.
- Calculate confidence bounds using independent incident groups. For safety gates, a group fails when any actionable member is falsely cleared; report case-level metrics separately.
- Gate detection using its 95% lower confidence bound, false-positive rate using its 95% upper bound, and false clearance using the agreed 95% upper bound of 1%.
- Missing controls, required classes or sufficient independent groups yield `INSUFFICIENT_EVIDENCE`, never a pass. Missing rates remain null with reasons.
- Compare deterministic fixtures against independent oracle labels, not agreement between two implementation outputs.
- Apply the same gate implementation to synthetic and store-backed evaluation. Central eligibility remains dependent on real analyst provenance and operational qualification.

Implement genuine evidence ablations: rerun each provider on independently constructed summary, forecast, hierarchy/multi-metric and complete evidence views. Keep cases, provider configuration and measurement conditions fixed. Label these as challenger experiments; operational policy continues using complete evidence.

## 5. Compatibility, verification and release acceptance

Bump evidence, certificate, finding, report and policy versions where semantics change. Preserve legacy artifacts for inspection, but do not reuse old certificates or cached assessments as newly qualified results. Back up SQLite before transactional migrations; retain run-to-label relationships and provenance.

Required regression acceptance:

| Scenario | Required result |
|---|---|
| Doubled reference totals named `national` | Investigation; no statistical clearance |
| 60% drop in the first of two appended weeks | First week detected; both weeks assessed |
| Certificate with short history, missing inputs, huge interval or failed integrity | Rejected with explicit reasons |
| Larger sibling added to a normalized calibration pool | No interval inflation caused solely by sibling units/scale |
| Future-dated recurrence inserted before historical replay | Identical historical identity and result |
| Eligible recurrence evidence changes | New identity; no stale cache reuse |
| Repeated copies of one incident, or no negative controls | Insufficient evidence |
| Candidate selection and calibration | Disjoint target IDs and correct forecast horizons |
| Model request capture | Required evidence present; final policy answers absent |
| Mandatory evidence exceeds budget | Explicit provider abstention |
| Legacy-store migration and crash recovery | Preserved labels; no duplicate logical assessments |

Run the full suite, required Delta integration, protocol tests and end-to-end onboarding → assessment → label import → frozen evaluation. Verify actual provider payloads with lightweight adapters; run the pinned Laya CPU integration separately when available and report any unavailable check explicitly.

Add reproducible 320-week benchmarks at one million and five million rows, recording runtime, peak RSS and snapshot-read counts. Treat incomplete benchmark runs as limitations, not scale qualification.

Update the saved plan, README, claims matrix and runbooks together. Replace blanket “delivered” claims with implemented, verified, unavailable and unqualified statuses, linking reproducible artifacts. Completion requires all eight review defects to have regression coverage and passing checks; real-world accuracy remains unresolved until independent analyst-labelled data exists.
