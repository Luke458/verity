# Second remediation: close unsafe passes and qualify the complete decision path

> **Implementation status (v0.22.0).** All three increments are implemented and
> verified through the real orchestration paths. Certificate binding is strict
> (identity, evidence digest, finding coverage, schema, policy and qualification
> digests; exact scope/metric/level/period match) and verification evaluates all
> applicable contract, reconciliation, reference, lineage, hierarchy and
> required-input failures before granting clearance. Every configured measure is
> assessed per period with snapshot measures compared within a period. The
> qualification artifact is schema 2: source assessments, incident groups,
> cutoffs, splits, model identity, calendar/config hashes, independent held-out
> coverage with group-confidence lower bounds, registered gates, no test
> tolerance and analyst-label linkage for real provenance. It is resolved and
> validated before assessment identity, so replacement, revocation or loss at
> the same path changes identity. Provider views keep mandatory failed-check
> evidence (compressed to references) and abstain instead of falling back.
> Recurrence uses an explicitly serialized stable key with metric and hierarchy
> scope, eligible unexplained predecessors, signed/gross impacts and a
> registered cumulative budget. Synthetic, store, cohort and qualification
> evaluation share one gate implementation with the 1% false-clearance bound.
> Contract versions: findings 4, machine 5, evidence/certificate 3,
> qualification 2, recurrence 3, recurrence input 2, evaluation/bench 3,
> features 3, evidence text 3, SQLite 6, engine 0.22.0. Permanent regressions
> live in `tests/test_qc_second_remediation.py`; the previous 426 tests were
> preserved (444 total). The one-million and five-million-row benchmarks were
> re-run with the multi-metric pipeline (1M: 8.2 s / 1.07 GiB; 5M: 53.3 s /
> 4.58 GiB; 4 snapshot reads each; dollar and units assessed, scripts and stock
> reported absent, 1 period, 4 hierarchy checks). Verification, compatibility
> and completion checks in section 5 pass except where documented as a
> limitation in `docs/reliability-limitations.md`.

## 1. Assessment of the current implementation

**The implementation has improved, but it is not complete enough to trust automatic clearance.** The review covered the uncommitted changes present on 2026-09-21, preserving them unchanged. All **426 tests passed**, Ruff passed, and mypy passed across 60 source files.

Multi-period assessment, normalized residual pooling, selection/calibration separation and grouped evaluation gates are substantive improvements. However, several acceptance tests verify helper outputs rather than the complete operational behavior.

Confirmed remaining defects:

| Priority | Finding | Reproduction or code evidence |
|---|---|---|
| P1 | Secondary measures remain unchecked temporally | A **99% units drop**, with unchanged dollar behavior and consistent report totals, returned **`PASS`**, without review. Only dollar predictions were produced. |
| P1 | Certificate binding remains permissive | Removing a certificate’s evidence digest still allowed **`PASS_WITH_EXPLANATION`**. |
| P1 | Certificate verification still ignores applicable failures | Evidence containing failed reconciliation and a cross-metric contradiction produced **`VERIFIED`**, with no rejection reasons. |
| P1 | Qualification changes do not invalidate cached assessments | Replacing a qualification artifact with `UNQUALIFIED` at the same path returned the same assessment ID and cached result. |
| P1 | Mandatory provider evidence can disappear | Packing a failed contract with a long detail returned a usable payload with **no contracts**. The contract omission was absent from the reported omissions. |
| P1 | Qualification lacks independent support | Nine duplicate development records and one test record produced **`QUALIFIED`**. Coverage diagnostics also evaluate errors against an interval estimated using those errors. |
| P2 | Recurrence matching fails after serialization | `stable_key` is a property omitted by `asdict`; consecutive-period findings therefore fall back to different finding IDs and do not match. |
| P1 | Evaluation paths still diverge | Store-backed evaluation lacks the new false-clearance gate; evidence ablation evaluates a bespoke rule rather than the actual challenger providers. |

The permissive certificate filter is in [qc/policy.py](../qc/policy.py) (`apply_policy`, around line 450 at review time). The provider fallback behavior is in [qc/systemone.py](../qc/systemone.py) (`build_provider_state`, around line 41). The new qualification implementation is in [qc/qualification.py](../qc/qualification.py).

## 2. First increment: repair operational safety

**Make statistical authorization fail closed.**

- Require explicit, nonempty assessment identity, evidence digest, finding coverage, certificate schema and policy version. Remove compatibility defaults that treat missing fields as current.
- Bind certificates to immutable evidence and qualification digests. Reject mismatched metric, period, scope or qualification identity.
- Evaluate all applicable contract, reconciliation, reference, lineage, hierarchy and required-input findings before granting clearance. Use structured dependency references rather than matching contradiction strings by suffix.
- Ensure hierarchy and other integrity results become policy findings, rather than appearing only in reports.
- Reject unknown check outcomes, unsupported schemas, invalid numeric limits and nonfinite evidence. Zero materiality remains zero; it must not silently select a broader fallback threshold.
- Preserve automatic clearance only for exact eligible temporal findings. Hard failures remain unconditionally reviewable.

**Assess every configured measure and period.**

- Introduce explicit required and optional temporal metrics, derived initially from existing required columns and configured metric columns.
- Assess present configured measures independently; report absent optional measures explicitly and mark missing required measures incomplete.
- Key forecasts, calibration, findings, ledgers, availability and certificates by metric, period and hierarchy scope.
- Respect snapshot-measure aggregation rules. Compare stock within a period without summing stock across time.
- Use scoped metric units and materiality throughout. Do not apply dollar thresholds to units, scripts or stock.
- Verify offsetting errors, coordinated deviations and parent/child results without double counting.

Until these checks pass, qualification artifacts cannot enable automatic clearance in the repaired release.

## 3. Second increment: make qualification and recurrence reproducible

**Replace qualification’s permissive artifact contract.**

- Require source assessment IDs, incident groups, observation cutoffs, split assignments, model artifact identity, calendar/configuration hashes and evidence-policy version.
- Reject overlapping development/test groups, duplicate observations, unsupported provenance and missing evaluation references.
- Freeze candidate coverage and width policies using development evidence; validate the frozen choice on untouched test observations.
- Require sufficient independent groups on both sides. Determine sufficiency from the registered confidence-bound requirements, not a development-row count plus one test observation.
- Remove the implicit test-coverage tolerance and fixed permissive width defaults. Thresholds must be explicitly registered and included in the artifact identity.
- Distinguish calibration diagnostics from independent coverage validation. Coverage calculated using errors that fitted the interval cannot serve as qualification evidence.
- Require the common false-clearance, detection and false-positive gates before an artifact can authorize statistical clearance.
- Preserve synthetic qualification for synthetic assessments only. Real qualification requires linked analyst-labelled evaluation evidence.

**Freeze qualification before cache lookup.**

Resolve and validate the artifact once, before assessment identity is calculated. Include its content digest, policy, provenance and availability state in the manifest. Pass that loaded immutable artifact through execution instead of rereading its path.

Historical assessments select only artifacts available at their observation cutoff. Replacing, revoking or losing an applicable artifact changes identity; retries reuse their original frozen inputs.

**Repair recurrence serialization and semantics.**

- Serialize the stable recurrence key explicitly, with metric and hierarchy scope; derive it through one shared function.
- Match only eligible unexplained predecessor findings. Exclude cleared, approved and passing observations.
- Preserve signed and gross scoped impacts. Do not fall back from a legitimate zero scoped impact to dataset-wide movement.
- Group multiple periods within one logical refresh without counting them as independent refreshes.
- Include informational unexplained deviations when testing persistence. Use a registered cumulative materiality budget; summing each observation’s threshold cannot detect a sequence whose members all remain individually below threshold.
- Order predecessors using snapshot/business-period metadata, not inferred ordering of version strings. Missing required historical inputs produce explicit unavailability.

## 4. Third increment: unify provider evidence and evaluation

**Use one canonical, versioned evidence input for every provider.**

- Include every assessed metric and period, applicable integrity results, missing-required-input records, uncertainty, ledger contributions and provenance.
- Define mandatory evidence by decision dependencies. Failed checks and their compact structured evidence cannot be dropped.
- Compress verbose details into evidence references before removing optional content. Record every omission after packing completes.
- Abstain when mandatory evidence cannot fit. Remove the silent complete → forecast → summary fallback from operational provider requests.
- Reserve tokenizer budgets for complete questions and distinct options in Laya. Character limits remain an additional transport bound.
- Do not use legacy final-status-bearing payloads when canonical evidence is unavailable. Mark those historical cases incompatible or rebuild evidence from their frozen inputs.
- Build feature and text representations exclusively from the same canonical evidence. Remove derived anomaly verdicts and policy answers from learned features.
- Persist exact payload hashes and evidence identities for HTTP, Laya, feature and text providers.

**Make evaluation genuinely common.**

- Use one gate implementation for synthetic benchmarks, store cohorts, qualification and production eligibility.
- Require the registered classes and controls; enforce grouped confidence bounds and the 1% false-clearance upper bound everywhere.
- Preserve original feature/evidence versions during cohort import. Reject incompatible records rather than relabelling them with the current feature version.
- Freeze complete input and label identities, not only suite metadata and selected label summaries.
- Execute only development selection before freezing the candidate; then execute one final test configuration.
- Run actual provider adapters for each evidence ablation. Keep a bespoke rule baseline explicitly labelled as such.
- Include all periods and metrics in qualification and ablation extraction; remove first-package-only shortcuts.
- Report abstention, detection, review volume, attribution, severity and resource usage separately. A benchmark or qualification failure must remain visible rather than falling back to a candidate that appears selected.

## 5. Verification, compatibility and completion

Add permanent regressions for every reproduction above, then test their integration:

- Consistent warehouse/report data with a 99% units drop requires review.
- Blank, stale, wrong-scope and unsupported certificates cannot clear findings.
- Failed reconciliation, hierarchy integrity or missing required inputs prevent statistical authorization.
- Qualification replacement or revocation changes assessment identity; future qualification cannot affect historical replay.
- Oversized failed-check evidence causes compression with preserved mandatory content or explicit abstention.
- Duplicate incidents, overlapping splits, one test observation and missing classes cannot qualify.
- Consecutive-period recurrence survives serialization, storage and replay.
- Actual provider captures contain all required evidence and no final policy answers.
- Store and synthetic evaluation produce identical gates from equivalent labelled cases.
- Legacy records retain their versions, labels and provenance through migration.

Run the full suite, lint, type checks, required Delta integration and an end-to-end assessment → persistence → label import → frozen evaluation workflow. Exercise provider failures and journal recovery after qualification/identity changes.

Re-run the existing one-million and five-million-row benchmarks with the corrected multi-metric pipeline. Record assessed metrics, periods and hierarchy counts alongside runtime, memory and read counts; the earlier measurements do not establish the cost of this expanded path.

Preserve authored changes and legacy artifacts. Version the changed evidence, certificate, qualification, recurrence and evaluation contracts. Replace “all increments implemented” claims with verified capabilities and remaining limitations.

**Completion means these behavioral checks pass through the real orchestration paths—not merely through isolated helpers. Real-world accuracy and production eligibility remain unestablished without independent analyst-labelled evidence.**
