# Architecture delta and phase status

`docs/architecture.md` is the original design record and remains the source of
intent. This document records where the implementation deliberately diverges,
why, and the current status of every phase in section 73 of that document.

## Deviations

| # | Spec position | What changed | Why |
|---|---|---|---|
| D1 | §34-39, §73 phases 14-15: MiniCPM next-token typed decisions | The semantic layer is evidence-feature decisions: a rule provider, a trained linear head over a versioned feature encoder, and a Jev-compatible remote provider with local fallback. No LM token scoring is on the critical path. | The predecessor `system-one` experiments showed prompt-scored next-token decisions from a frozen general LM are order-sensitive and uncalibrated (50-65% field accuracy, confidently wrong cases). Evidence features are measurable, versioned and retrainable; remote Jev-compatible models remain available for escalation. |
| D2 | §73 phase 19: compiled heads over LM hidden states | Realised as the trained decision provider over evidence features (`qc/training.py`), not as probes on MiniCPM hidden states. | Same capability (fast repeated decisions, small artifact) without coupling to one backbone's representations. |
| D3 | §73 phase 11: TSPulse as temporal representation | Research-only adapter with a closed production gate. Weekly histories are shorter than the 512-point context; embeddings require an explicit linear resample. Real benchmark: nearest-centroid family accuracy 0.667, intra/inter similarity 0.954/0.837, and latest-week-only faults have zero overlap revision by construction. | The weekly-length question is unresolved; promoting it would violate the project's own gate discipline. |
| D4 | §30, §73 phase 9 forecasting challengers | Chronos is an optional adapter; a seasonal-difference baseline is the default; SARIMAX is an optional classical challenger (`[classical]` extra). | A dependency-free baseline keeps tests and local runs fast and doubles as the champion/challenger reference. SARIMAX is useful for stable low-count series but not across thousands of series. |
| D5 | §73 phase 18 / §41: automatic escalation to a reasoning agent | The agent handoff is explicit, not automatic. When `requires_investigation` is true, a structured brief is built and can be handed to any command; reports are polled by a scheduled agent instead. | Operations prefer a scheduled agent reading report artifacts over an in-run side effect; the decision gate is still computed on every run. |
| D6 | §77-78 outputs | Machine JSON, a Markdown report and the Jev-compatible answer format are implemented; PDF/Excel are not. JSON is the contract; Markdown is for humans. | Avoid heavy rendering dependencies until a concrete consumer asks. |
| D7 | §70 module layout | Added modules not in the spec: `labels.py`, `training.py`, `incidents.py`, `agent.py`, `systemone.py`, `tspulse.py`, `tspulse_benchmark.py`, `reporting.py`, `relationships.py`, `delta.py`. | These are the label store, training loop, memory, handoff, wire-compatibility, research adapter and platform adapters that the phases implied but the layout did not name. |
| D8 | §71 interfaces: `runtime.decide(context=evidence, fields=...)` | `run_qc` produces the evidence graph and decisions in one call; decisions are also available from any configured provider through the same `decide(result)` interface. | Keeps one deterministic entry point per run while preserving the provider abstraction. |
| D9 | Deployment | The engine reads Delta tables directly through delta-rs (`qc/delta.py`); Databricks compute, jobs and Unity Catalog wiring are out of scope for local assessment. | The immediate need is assessing datalake table versions, not running on managed compute. |
| D10 | §34 semantic substrate | Added a frozen-encoder text probe (ModernBERT-class) as a third decision substrate beside evidence features and remote Jev. | Encoder classification avoids generation, fits the bounded evidence text, is CPU-friendly and is a strong few-label challenger; accuracy remains gated on real labels. |
| D11 | Coverage | Synthetic faults extended to product-level omissions (`missing_products`) and entity merges (`entity_merge`), with the `ENTITY_MERGE` cause class. | `MISSING_PRODUCTS` and relationship candidates existed in the decision layer but had no oracle coverage. The synthetic-data question is coverage and difficulty, not volume; volume does not substitute for real labels. |
| D12 | Onboarding | Added Delta profiling, config proposal, readiness assessment and analyst-outcome import (CSV/Delta). | With no production data yet, the highest-value work is making day one mechanical: profile, propose, flag blockers, and import outcomes without manual entry. |
| D13 | Feedback simulation | Added a synthetic analyst simulator (drafts, mistakes, unknowns, corrections, latency, investigation disagreement) writing provenance-tagged outcomes. | Real analyst data does not exist yet, but the feedback loop can still be exercised end to end. Provenance guarantees simulated labels never unlock `production_eligible`. |
| D14 | Canonical repository | retail-qc is canonical; Verity's spine is ported here (replay, scoped expectations, reference controls, bounded RCA loop, prequential point-in-time forecasting). Spark/Databricks adapters and deployment are deliberately skipped; ratio expectations deliberately stay in a versioned JSON file. | One spec, two engines was a drift risk. The pandas engine plus delta-rs covers assessment; managed compute is not required yet, and diffable approvals beat a database table for low-volume governed data. |
| D15 | Operations | Added `qc weekly` as the single scheduler entry point: version resolution, engine, reference control, expectations, calibration append, drift check, store, report and brief; content-addressed immutable assessments, separate attempts, SQLite journal recovery and explicit review exit codes. | Every step existed as a command, but a refresh needed an operator to run nine of them in order. A scheduler needs one call with a status contract. |
| D16 | Field mapping | Added a config `column_map` (canonical -> production) applied by a source decorator, onboarding `--alias` inference, and fail-loud behaviour when a mapped column is absent. | Real tables never use the synthetic names; the engine must see one vocabulary and never silently score a wrong column. |

## Phase status

| # | Phase | Status | Notes |
|---|---|---|---|
| 1 | Core deterministic engine | done | contracts, version pair, revision cube |
| 2 | Entity lifecycle | done | incl. latest-week gap and reclassification |
| 3 | Attribution | done | contributors, explained/unexplained residual |
| 4 | Counterfactual reconstruction | done | per-week reconstruction and score gate |
| 5 | Reconciliation + structural integrity | done | mass balance, counts, hierarchy and aggregate-marker checks, ratio outliers; revision signatures in fingerprints |
| 6 | Pipeline lineage | done | first divergence per stage |
| 7 | Expected events + entity graph | done | registry, detected replacement candidates, confirmed relationship store |
| 8 | Chronos-2 | done (adapter) | real checkpoint smoke on CPU; default Chronos-2 id untested |
| 9 | Robust statistical ensemble | done | MAD, seasonal z, EWMA, change point |
| 10 | Forecast calibration | done | empirical residual calibration, finite-sample conformal intervals, prequential across-load pool |
| 11 | TSPulse research adapter | done (research) | closed production gate, benchmark harness |
| 12 | Incident memory | partial | retrieval + embedding blend; revisioned store with historical observation cutoffs implemented |
| 13 | Evidence graph | done | |
| 14 | Semantic decision benchmark | partial | label store, training and shadow scoring done; real-label bake-off pending |
| 15 | Semantic runtime | done | rule/trained/remote providers, typed score fields |
| 16 | Fault injection | done | thirteen families with an oracle, including product-level omissions and entity merges |
| 17 | Shadow production | partial | synthetic only; no real refresh yet |
| 18 | RCA agent | done (boundary) | brief, bounded evidence-query tool and validated command handoff; the reasoning agent itself stays external |
| 19 | Compiled heads | done (reinterpreted) | trained provider over evidence features |

## What still decides success

1. A first real table: onboard it, fix the blockers, run a real version pair,
   then collect confirmed analyst outcomes. The harness, store and champion
   selection are ready; they need the data.
2. Operational governance: who may change thresholds and promotion gates, when
   calibration is refit, and how drift alerts are handled. The store and drift
   checks exist; the process does not.

See [reliability acceptance and limitations](reliability-limitations.md) for the
current engineering evidence and remaining qualification work. Phase completion
means an implemented path, not production validation.

## Reliability release (schema versions)

Machine reports use schema 3, findings schema 2, the evidence package schema 1
and SQLite schema 4. Migrations back up and preserve legacy runs/outcomes;
legacy identity is never silently reused, and readers tolerate schema-2
artifacts because no missing historical evidence is invented. `qc.policy`
computes final status after contracts, references, temporal checks,
recurrence and scoped approvals, and can clear a specific finding only through
a verified explanation certificate. `qc.assessment` and `qc.weekly` journal
immutable assessment content (including `evidence.json`) before atomic
filesystem publication. `qc.store_cohort` freezes analyst revisions and
chronological incident groups for separate training, calibration, development
and test sets. `qc.laya` is a lightweight HTTP adapter; heavy model
dependencies stay in a separate optional service. Rules remain the default, and
every current synthetic artifact is ineligible.
