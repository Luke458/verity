# Next-phase plan: optimise the semantic layer, unblock real validation

Status: proposed (v0.22.0, after the second remediation cycle).
Scope: where the repository stands today, what current research says we should
change, and an ordered set of falsifiable increments.

### Progress

| Item | State |
|---|---|
| C1 quickstart reproducibility | **done** — see the defect below |
| C2 fault-family doc drift | **done** — `tests/test_docs_sync.py` now diffs the table against `FAMILY_SPECS` |
| B4 metric surface | **done** — macro-F1, MCC, G-mean, per-class table, confusion matrix and cost-weighted score in `qc train`, `qc champion` and `label_metrics` |
| B1 decomposition | **done — gate FAILED.** See "B1 result" below. Reported as failed; no threshold moved |
| B3 calibration hardening | **done** — Platt/isotonic/reliability bins; the plan's own assumption about isotonic was wrong and is corrected in "B3 result" |
| B2, B5–B7, Track A, Track C | not started |

**Defect found and fixed while doing C1.** `qc shadow --labels-out` was
documented as "append oracle labels" but silently wrote *placeholder* labels
(`UNKNOWN` / `PASS`, `family: null`) tagged `source: "oracle"` whenever the
oracle vault was absent for a scenario — `shadow.py` used
`vault.read(scenario_id) or {}` and fed the empty payload to
`oracle_labels_for_result`. That is a provenance lie of exactly the kind the
rest of the codebase refuses (`docs/claims.md`: provenance CHECK with no analyst
default), and it let `qc train` fit a provider to noise while reporting it as
trained on ground truth. `run_shadow` now refuses to write labels unless every
scenario has an oracle payload (`tests/test_qc_shadow.py`). Separately,
`--labels-out` **appends**, so re-running the quickstart after a feature-version
bump accumulated incompatible records and `train_decision_provider` rejected the
file; regenerating from a clean slate is required, and `qcgen generate` writes
the vault that `qc shadow` needs.

**A live measurement now available.** With the regenerated demo suite and its
vault, `qc train` on 13 oracle records reports `likely_cause` accuracy
**0.000** on both validation and test while `requires_investigation` scores
1.000. The B4 metric surface makes the picture sharper than accuracy could:

- `severity` on test: accuracy 0.333 but **MCC −1.000** — perfectly
  *anti*-correlated. This is the wrong-direction bias of §2.4, visible now
  instead of hidden, and it is exactly the case where temperature scaling
  cannot help.
- `requires_investigation`: accuracy 1.000 but **MCC 0.000** — it predicts one
  class constantly. Accuracy flatters it completely.
- `likely_cause`: accuracy 0.000, macro-F1 0.000, G-mean 0.000. It never finds
  a single class correctly.

None of this is evidence about real performance — it is 13 records. It is
evidence that the current feature/label formulation does not work, which is
what B1 and B2 are for, and that the metric surface was worth building first.

This document inherits the claims discipline of [claims.md](claims.md): nothing
in it confers `production_eligible`, and every number quoted from external
research is a number *about someone else's benchmark*, never a claim about this
system.

---

## 1. Where the repository stands

### 1.1 What is complete and verified

The deterministic evidence system is done and heavily tested:

| Layer | State |
|---|---|
| Contracts / versions / revision cube / lifecycle / attribution | complete, adversarially tested |
| Counterfactual, reconciliation, lineage first-divergence, events, fingerprints | complete |
| Temporal forecasting + finite-sample conformal intervals + prequential pooling | complete, leakage-tested |
| Evidence package, policy clearance, certificate binding, per-measure ledgers | complete (schema 3) |
| Store (SQLite schema 6), assessment/attempt journal, atomic publication | complete |
| Evaluation harness: cohorts, conformal p-values, grouped CIs, paired champion selection, plan-hash pinning | complete |
| Weekly orchestration, onboarding, shadow, replay, drift, RCA | complete as code paths |
| Oracle isolation (blind runs, `qc` cannot import `qcgen`, vault separation) | enforced by test |

Everything above is `validated-synthetic` or `plumbing-only` in
[claims.md](claims.md). **Nothing is `validated-real`.** The project is honest
about this and that honesty is the asset most worth protecting.

Live baseline at time of writing: **444 tests, all passing** on `.[test,delta]`
in the local `.venv`. Ruff and mypy gates are green per CI. The engineering
surface is not the constraint.

### 1.2 What is weak, stale, or incomplete

1. **The two real gates are external and unfilled** — `docs/architecture-delta.md`
   "What still decides success": (a) a first real table with confirmed analyst
   outcomes and a frozen cohort, (b) an operational governance process for
   thresholds, calibration refits and drift response. No amount of code closes
   either. Phases 12, 14 and 17 are `partial` for exactly this reason.
2. **Every learned substrate currently loses to the rule provider.**
   `reports/artifacts/champion-demo/champion.json` selects `rule` (1.00) over
   `feature_head` (0.75) and `text_probe` (0.6875); against simulated-analyst
   labels the learned providers collapse (0.31 and 0.25 overall, **0.0 cause
   accuracy**). This is the concrete optimisation target.
3. **Label volume is 13.** `data/labels/demo.jsonl` holds 13 oracle records. Any
   accuracy number at that size is noise — the docs say so correctly.
4. **Stale artifacts that break the documented quickstart.**
   - `data/labels/demo.jsonl` carries `feature_version: 1`; `qc/decisions.py`
     has `FEATURE_VERSION = 3`, and `train_decision_provider` rejects the
     mismatch. The README `qc train` snippet currently cannot run.
   - `docs/synthetic-data.md`'s fault-family table disagrees with
     `qcgen/spec.py` (declared single source of truth) on `expected_event`,
     `null_duplicate_storm` and `market_movement`.
5. **Metric surface is thin for the problem class.** `qc/training.py` reports
   per-field accuracy + Wilson CI, multiclass Brier, top-label ECE, overall
   accuracy. For an 11-way imbalanced cause taxonomy that is not enough to make
   a promotion decision.
6. **Calibration is temperature scaling only.** `fit_temperature` is a scalar
   grid search over softmax temperature (see §2.4 — this cannot correct a
   wrong-direction bias).
7. **Unresolved research questions:** TSPulse's scenario-holdout gate is not
   estimable with one scenario per family; evidence ablation reruns the rule
   adapter only; capacity beyond the 5M-row single-process envelope is untested.

---

## 2. What current research says (and why it changes our plan)

Research conducted September 2026 into "JeV-like" / System One models and
BERT-family classifiers. Five findings bear directly on this codebase.

### 2.1 "Jev" resolved — and we already integrate the right abstraction

**Jev** is TypeSafe AI's proprietary *System One Model* (early access 15 Sep
2026; named after W. Stanley **Jevons**, model class after Kahneman's
"System One"). It is a **non-generative typed-decision model**:
unstructured state in → typed probabilistic decisions out, `output_tokens: 0`.
Three primitives — **noul** (one probability), **choice** (argmax over ≤255 IDs
+ distribution), **score** (expected index over ≤50 ordered levels). Training
is *RLCD — Reinforcement Learning for Calibrated Decisions*, 100% synthetic
data. There is **no paper, no parameter count, no model card**.

Our `qc/systemone.py` already speaks exactly this wire protocol
(`POST /v1/systemone`, `noul`/`choice`/`score`) and `qc/laya.py` pins
`convaiinnovations/laya` — which the reproduction literature identifies as one
of the six open Jev clones (ModernBERT-large + 2 decision layers). **The
substrate selection is correct and current.** What follows is about how we use
it.

### 2.2 🔴 Decomposition beats model scale — the single highest-value change

From `jev-phishing-bench` (2,000 PhishNChips emails, Jev 1.13 vs Claude Haiku
4.5):

| Configuration | Accuracy | AUROC | ECE (10 bins) |
|---|---|---|---|
| Jev, **one broad question** | 62.6% | 0.689 | 0.154 |
| Haiku 4.5 | 81.3% | 0.837 | 0.097 |
| **Jev, five narrow signal nouls + logistic regression** | **95.0%** [93.5, 96.2] | **0.982** | **0.027** |
| *control:* two-line regex on URL features | 91.8% | — | — |

Splitting one broad question into narrow *signal* questions and learning a small
combiner on top moved accuracy **+32.4 points** and cut ECE **5.7×** — and beat
the frontier-model baseline while being 27× cheaper and 5× faster.

**Our current encoding is exactly the losing configuration.** `_questions()` in
`qc/systemone.py` emits four questions:

```
likely_cause            -> choice over 11 values      (one broad question)
likely_origin           -> choice over 6 values       (one broad question)
severity                -> score over 4 levels
requires_investigation  -> noul
```

`likely_cause` is an 11-way argmax over an imbalanced taxonomy — precisely the
"broad question" that measures 62.6%.

### 2.3 🔴 Jev's calibration claim does not survive independent audit

Two independent evaluations measured **ECE 0.107–0.154** for a model marketed on
calibration; TypeSafe has published no calibration figure. Worse: **questions
have no guaranteed joint distribution** — TypeSafe's own worked example returns
0.7 for a noul and 0.7 for its negation (sums to 1.19). Failure modes
documented for Jev 1.13: weak at arithmetic/counting, reads dates as text,
degrades as unrelated content grows in `state`, treats input as non-hostile.

Consequences for us:
- we must **measure** per-question calibration on our own distribution rather
  than trust `frozen_encoder_probe` / provider confidence (the codebase already
  refuses to treat temperature as a calibration certificate — good, keep it);
- we should **coherence-check** noul answers and their negations;
- we must keep the evidence state tightly scoped (we already exclude status,
  findings, certificates, decisions and labels — the "degrades as unrelated
  content grows" failure mode argues for keeping it that way and for keeping
  the 32 KB budget tight).

### 2.4 🔴 Temperature scaling cannot fix a wrong-direction bias

The phishing bench reports post-hoc temperature scaling on the broad question
made calibration **worse**. A scalar temperature rescales confidence; it cannot
flip a systematically inverted class. Guo et al. 2017 (`arXiv:1706.04599`) is
the origin of the recipe, and the standard remedies for a biased head are
**Platt / isotonic / Dirichlet** calibration — which our `fit_temperature` does
not implement. Given that our learned providers score **0.0 cause accuracy**
against simulated-analyst labels while still emitting confident probabilities,
this is not academic: our heads are currently in exactly the regime where
temperature scaling misleads.

**Measured here (B3, `qc/calibration.py`).** The intuitive remedy list above
needs a correction, and it was a test that produced it rather than an argument.
It is tempting to write "isotonic fixes bias". Two things are true instead:

- **Per-class monotone maps do not preserve argmax.** Only a *shared* monotone
  transform does. Per-class maps have independent shapes and can reorder across
  classes — demonstrated in
  `tests/test_qc_calibration.py::test_per_class_monotone_maps_do_not_preserve_argmax`.
- **But isotonic still cannot rescue an inverted head.** On an anti-correlated
  target the one-vs-rest labels are *decreasing* in the score, so the
  non-decreasing least-squares fit collapses to the constant class prior. It
  becomes uninformative rather than correct. On the fixture every fitted map is
  flat at 1/3 and prediction degenerates to a single class.

**Only Platt scaling with a free-sign slope recovers it** — a fitted negative
slope inverts the class. On the deliberately wrong-direction fixture: Platt
reaches accuracy 1.000 with every slope negative; temperature stays at 0.000 at
any temperature from 0.05 to 50; isotonic collapses to chance and predicts one
class. `select_calibrator` picks Platt on NLL as it should.

The practical rule that follows: use isotonic against *overconfidence*, and
Platt when direction is in question. Check the fitted slopes first — if they are
all near +1 the head was never inverted, and if one is negative that is worth
investigating rather than silently calibrating away.

### 2.5 The low-label regime is a solved, documented playbook

We are heading into real data with few labels. The 2024–2026 evidence:

| Method | Labels | Result | Source |
|---|---|---|---|
| **SetFit** (contrastive ST + LR head) | **8–64 / class** | AG-News **0.87 on 32 total examples**; competitive with fine-tuned RoBERTa-Large on full CR | `arXiv:2209.11055`, philschmid reproduction |
| Fine-tuned DeBERTa-v3 / ModernBERT | ~1k–50k | accuracy ceiling (SST-2 94.8); **3.6–4.3 ms/sample** | `alexjacobs08/beatingBERT` (32 controlled experiments) |
| Embedding + logistic regression | 50–5k | good, well-behaved probabilities, **add classes without retrain** | MTEB classification column |
| Zero-shot NLI (DeBERTa-v3-MNLI) | 0 | bootstrap only; brittle to hypothesis wording | `arXiv:1909.00161` |
| 1–2B LLM zero-shot | 0 | SST-2 93.8 but **loses to DeBERTa**, 20× slower, 445 ms @ 2k context | `alexjacobs08/beatingBERT` |
| Teacher→student distillation | 0 human | FineWeb-Edu pattern; `simple-jev` ships this as **RFDT** | `featherless-ai/simple-jev` |

Three controlled findings worth internalising:
1. **Fine-tuned encoders beat prompted LLMs** when labels exist
   (`arXiv:2406.08660`, EMNLP-2025 Findings `aclanthology.org/2025.findings-emnlp.1033`).
2. **Few-shot in-context examples are a mixed bag** — Qwen2.5-1.5B collapsed on
   RTE 78.7→53.4 with k=5. Do not assume k-shot helps.
3. **A rule/regex baseline beat every learned signal** on a rule-separable task
   (91.8% regex vs 89.4% best Jev signal). Our `RuleDecisionProvider` being
   champion is not a failure — it is the correct baseline, and it must stay in
   every bake-off.

### 2.6 Recommended metric surface

For imbalanced multi-class with asymmetric cost (missed defect ≫ false alarm):
**macro-F1**, **MCC** (preferred over accuracy/F1 under imbalance — Chicco &
Jurman, *BMC Genomics* 2020), **ECE + Brier + reliability diagram** wherever a
threshold gates an action, **per-class recall + confusion matrix**, and
**coverage-at-accuracy** as the operating metric for any auto-clearance path.
Caveat: MCC/F1 degrade under *extreme* imbalance
(`arXiv:2404.07661`) — pair with G-mean and an explicit cost matrix.

### 2.7 Nearest open references

Nothing in open source does retail transactional QC the way this repo does. The
closest artefacts are `featherless-ai/simple-jev` (prefill-only logit scoring,
shared-prefix KV reuse, **RFDT distillation pipeline and JevBench harness** —
the most reusable external code), the six Jev clones catalogued by Latent Space
19 Sep 2026 (Laya, Bespoke Nimble, DiffusionGemmaJev, SemIf/OpenJev, Jevlike,
Kev-0.5B), `dcarpintero/pangolin-guard` (ModernBERT-large matching frontier
models on narrow judgment tasks — the existence proof for our text probe), and
`SaiBhavesh/Product-Quality-Intelligence-Engine` (complaint → failure-mode
extraction blueprint). For metrics and calibration methodology,
`alexjacobs08/beatingBERT` is the best-controlled public comparison.

---

## 3. The plan

Three tracks. **Track B is the work this research actually unlocks** and it can
proceed entirely on synthetic data against pre-registered gates. Track A is
external and cannot be accelerated by code. Track C is process.

Ordering principle: every increment has a *falsifiable* acceptance gate, a named
measurement artefact, and must not degrade the claims discipline.

---

### Track A — Real-data unblocking (external, prepare the runway)

Cannot be closed here. What code *can* do is remove friction so the first real
table converts into evidence fast.

| # | Increment | Acceptance gate | Effort |
|---|---|---|---|
| A1 | **Real-table onboarding dry run on a *non-production* real table.** Take any real retail-shaped table the owner controls (even a public grocery/retail dataset), run `qc onboard`, resolve blockers, run one version pair end to end. | `qc weekly` reaches a terminal state (even `INCOMPLETE`) on non-synthetic data; blockers enumerated in `reports/onboarding-feedback.json`. | S |
| A2 | **Label collection instrument.** A one-page analyst outcome form mapped onto the existing outcome schema (cause, origin, severity, requires_investigation, confirmed/disputed) plus `qc store import` round-trip. | Import of 20 hand-entered outcomes passes provenance CHECK with `source: analyst`. | S |
| A3 | **Governance doc** — who may change thresholds and promotion gates, when calibration is refit, how drift alerts are triaged. Explicitly a process decision; write it down. | `docs/governance.md` accepted by the owner; referenced from `docs/operations.md`. | S |
| A4 | **Capacity envelope beyond 5M rows / concurrent runs.** Reuse `optional/scale_benchmark.py`. | Measured RSS and wall clock at 20M rows; documented concurrency limits or a stated "not supported". | M |

> **Why A1 first:** it is the only item that converts `plumbing-only` →
> `validated-real`, and everything in Track B is calibrated on synthetic data
> that we *know* is easier than reality.

---

### Track B — Semantic-layer optimisation (the core work)



Goal, stated falsifiably: **make at least one learned substrate beat
`RuleDecisionProvider` on held-out synthetic scenarios under a pre-registered
gate**, with calibration reported and honest. Today none does. That is a
measurable target achievable without real data, and it is the precondition for
the real-label bake-off ever selecting a learned provider.

#### B1 — Decompose the decision questions into narrow signals 🔴 highest value

Split the two broad `choice` questions into **narrow binary `noul` signal
questions** and learn a small calibrated combiner over them.

Concretely, replace the single `likely_cause` 11-way choice with ~13 signal
nouls aligned to the existing fault families in `qcgen/spec.py`:

```
sig_missing_stores          "Is there evidence of absent stores?"
sig_missing_products        "Is there evidence of absent products?"
sig_coding                  "Is there a mapping/coding discontinuity?"
sig_warehouse_transform     "Is there a warehouse transform discontinuity?"
sig_recalculation           "Is there an approved history correction?"
sig_schema_failure          "Is the delivery contract violated?"
sig_entity_merge            "Were entities consolidated?"
sig_commodity_remap         "Did categories move between products?"
sig_new_store_backfill      "Is this historical onboarding backfill?"
sig_history_truncation      "Is observed history shorter than declared?"
sig_null_duplicate_storm    "Is there a null/duplicate integrity storm?"
sig_market_movement         "Is there independent market evidence?"
sig_expected_event          "Is this a declared calendar event?"
```

...plus a short set of **cross-cutting** nouls (`is_actionable`,
`first_divergence_at_source`, `affects_reported_totals`, `confidence_is_low`).
Then a linear/monotone combiner maps signal probabilities → the four typed
fields (and can emit `UNKNOWN` when no signal clears threshold).

Why this is the right change here, not just generically:
- **It is the measured +32.4-point move** in the nearest public benchmark
  (§2.2), with a 5.7× calibration improvement.
- It matches our label structure exactly: the oracle and simulated-analyst
  labels are *per-fault-family*, i.e. they supervise the signals natively.
- Signal probabilities are far more useful to the **policy/clearance layer**
  than a single argmax label — a 0.7 "warehouse transform discontinuity" is
  actionable evidence; a 0.7 "WAREHOUSE" class is not.
- It is **provider-agnostic**: the same decomposition serves `systemone.py`
  (remote Jev), `laya.py` (pinned clone), and gives the local `feature_head` /
  `text_probe` many narrow targets instead of one hard 11-way one. Narrow
  targets are exactly what makes low-label learning work (§2.5).
- It keeps `likely_origin` as a secondary 6-way choice *conditioned on* the
  chosen cause signal, which is a much easier question than 6-way from scratch.

Also implement the coherence check from §2.3: flag when a noul and its negation
sum outside `[0.9, 1.1]`, and normalise where safe.

**Acceptance gate (pre-registered, add `config/decomposition-gate.json` alongside
`config/cohort.json`):** on the frozen cohort held-out split, the decomposed
combiner beats the current 11-way `likely_cause` on macro-F1 **and** ECE, with
the paired bootstrap CI excluding zero. Measured separately for each substrate
(rule-over-signals, feature-head-over-signals, text-probe-over-signals,
Laya-over-signals).
**Effort: M–L. Impact: high. Risk: low** (additive; the old flat question set
stays as a bake-off arm).

#### B1 result: the gate failed, and the negative result is informative

`config/decomposition-gate.json` was written before any result existed. The
comparison ran as registered (112 synthetic oracle records, `tiny` profile,
seed 7, frozen split 39/34/17/22) via
`optional/decomposition_benchmark.py`. Report:
`reports/benchmarks/decomposition.json`.

**Result: FAIL.** Both arms produced **identical hard predictions on all 22
test cases across all four fields** (`records_disagreeing = 0/22` everywhere,
paired macro-F1 delta exactly 0.0 with interval exactly [0, 0]).

| field | flat macro-F1 | decomposed macro-F1 | flat ECE | decomposed ECE | flat Brier | decomposed Brier |
|---|---|---|---|---|---|---|
| `likely_cause` | 0.8519 | 0.8519 | 0.0491 | **0.0412** | **0.0748** | 0.0854 |
| `likely_origin` | 1.0 | 1.0 | 0.0 | 0.0015 | 0.0 | 0.0 |
| `severity` | 0.8963 | 0.8963 | 0.0508 | **0.0490** | 0.1543 | **0.1479** |
| `requires_investigation` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |

The probability distributions *do* differ (ECE and Brier move) — only the
argmax agrees. So this is not a measurement artifact: decomposition reaches the
same decisions by a different route, and does not reach better ones.

**Two flaws in the gate, owned and not retro-fitted.** Requiring the paired
interval to exclude zero *on every field* is unsatisfiable by construction where
both arms score 1.00 (`likely_origin`, `requires_investigation`) — nothing can
beat a perfect score. A gate that cannot be passed is not a test. Separately, a
20-scenario smoke run of the harness revealed results before the registered
run; the gate was not changed afterwards, but the sequence should have been
smoke-on-a-dummy-target, then register, then run.

**What this says.** The +32-point decomposition win in the literature came from
a *model* being asked a broad versus a narrow question — the model's zero-shot
judgment was the bottleneck. Here the bottleneck is not the classifier: the
handcrafted feature encoder over the deterministic evidence graph already
separates these synthetic faults cleanly (both arms 0.85–1.00), so there is no
uncertainty left for the question formulation to recover. That is consistent
with this project's own architecture thesis — the deterministic layer is the
strong part — and it redirects the semantic work.

**Where decomposition should still be tried.** The corpus has a ceiling effect:
clean single injected faults on a 30-week `tiny` profile. Decomposition is
expected to pay when the *reader* of the evidence is uncertain, so the honest
follow-ups are (a) a harder corpus — noisier signals, multi-fault scenarios,
smaller effect sizes, ambiguous boundary cases; (b) real analyst labels, where
ambiguity is the norm; and (c) the **remote provider** path, where the
decomposition drives the Jev wire protocol (22 narrow `noul` questions instead
of 4 broad ones) and the model's own uncertainty is the thing being shaped. The
question set and reconstruction rule (`qc/decomposition.py`) are correct and
tested regardless; what failed is the claim that they improve a linear head
over already-separable features.

#### B2 — SetFit-style contrastive training for the text probe

Replace "frozen ModernBERT + linear head on 13 records" with **contrastive
fine-tuning of the sentence encoder on generated pairs + logistic head**
(SetFit recipe, `arXiv:2209.11055`). This is the documented 8–64
examples-per-class method and we are at the extreme low end.

- Reuse the existing `ModernBertEmbedder` and `EVIDENCE_TEXT_VERSION` gating.
- Emit the artifact through the same `TrainedDecisionProvider` contract so
  `qc champion` and `qc train` need no new CLI.
- Report **coverage-at-accuracy** alongside accuracy — with few labels the
  honest question is "how much can we auto-handle at fixed reliability", not
  "what is the accuracy".

**Acceptance gate:** on the same frozen cohort held-out split, the SetFit
variant beats the frozen-probe baseline on macro-F1 with paired CI excluding
zero, at n ∈ {8, 16, 32, 64} per class. Learning curves must be plotted — the
interesting result is the *slope*, because that predicts the real-data regime.
**Effort: M. Impact: high in the low-label regime. Risk: low.**

#### B3 — Calibration hardening

- Add **Platt (per-class sigmoid)** and **isotonic** as calibration challengers
  beside `fit_temperature`; select on the calibration split only, report on
  test only. (§2.4: temperature alone cannot correct a wrong-direction bias,
  and our heads currently exhibit one.)
- Report **ECE, Brier, reliability diagram, MCC, macro-F1, per-class recall,
  confusion matrix** for every substrate in `qc train`, `qc champion`,
  `qc evidence-bench`.
- Keep the existing rule that a fitted temperature is *not* a calibration
  certificate. Extend it: no calibrated artifact may be promoted without a
  pinned real calibration set.

**Acceptance gate:** a synthetic adversarial test that constructs a
wrong-direction head and asserts Platt/isotonic recover it where temperature
does not. Plus: ECE reported for every artifact in `reports/`.
**Effort: S–M. Impact: medium (high for any confidence-gated clearance).
Risk: low.**

#### B3 result: implemented, and the plan's own assumption was wrong

`qc/calibration.py` adds three calibrators sharing one interface —
`TemperatureCalibrator`, `PlattCalibrator` (per-class free-sign logistic
scaling), `IsotonicCalibrator` (per-class PAVA monotone step map) — plus
`select_calibrator` (fit all, pick lowest NLL on the calibration split) and
`reliability_bins` (machine-readable reliability-diagram data, returned as
numbers rather than drawn). numpy only; no scikit-learn.

The acceptance gate was "a synthetic adversarial test that constructs a
wrong-direction head and asserts Platt/isotonic recover it where temperature
does not". **Half of that gate was wrong**, and the test said so
(`tests/test_qc_calibration.py`, 16 tests):

| calibrator | on a deliberately inverted head | mechanism |
|---|---|---|
| temperature | accuracy **0.000** at every temperature 0.05–50 | one positive scalar preserves every argmax |
| isotonic | **collapses to the flat 1/3 prior**, predicts one class | monotone fit on decreasing labels flattens to the mean |
| Platt | accuracy **1.000**, every fitted slope negative | free-sign slope can invert |

Isotonic genuinely helps where the plan expected it to — it reduces ECE on a
deliberately *overconfident* head — so it stays, but as a shape repair. The
corrected rule is in §2.4: **isotonic for overconfidence, Platt when direction
is in question**, and inspect the fitted slopes before trusting either.

Also worth keeping: `calibrator_report` stamps `calibrator_version` and repeats
that a fitted calibrator is not a calibration certificate. That rule is
unchanged from the plan and is the one that matters for promotion.

#### B4 — Metric surface and asymmetric cost

Implement the §2.6 metric set in `qc/training.py`, and add an explicit **cost
matrix** for `likely_cause` (missing a real fault ≫ false alarm) so champion
selection can optimise something other than flat accuracy. Add **G-mean** as a
guard for the rare classes.

**Acceptance gate:** `qc champion` output includes the full metric surface and
the cost-weighted score; a test pins the matrix so silent changes are visible.
**Effort: S. Impact: medium. Risk: low.**

#### B5 — Teacher→student label multiplication (RFDT / FineWeb-Edu pattern)

The binding constraint is 13 labels. Fix it without pretending synthetic labels
are real ones:

1. Generate a **large synthetic corpus** (the `qcgen` profiles already do this —
   `full` is 320 weeks).
2. Use a **teacher** (a strong LLM, or the remote Jev endpoint, or even the
   oracle) to label many evidence texts.
3. Train the student probe on the multiplied labels.

This is `featherless-ai/simple-jev`'s RFDT path and the FineWeb-Edu economics.
**Critical constraint:** teacher-labeled and oracle-labeled data must be
provenance-tagged distinctly and **must never unlock `production_eligible`**,
exactly as `simulate-analyst` is tagged `synthetic` today. The value is *feature
quality and decomposition sanity*, not a promotable artifact.

**Acceptance gate:** student trained on 1k teacher labels beats student trained
on 13 oracle labels on the *held-out oracle* set. If it does not, the teacher is
not informative and we stop.
**Effort: M. Impact: medium. Risk: medium** (teacher bias). Deprioritise behind
B1–B3.

#### B6 — Optional open substrate: prefill-only logit scoring

Add a 5th substrate using the `simple-jev` technique (prefill the answer slot,
read one token's logits over the candidate set, shared-prefix KV reuse) over a
small open model. Rationale: it is an open, self-hostable, ~free-latency
typed-decision substrate with **no network dependency**, and it would make the
bake-off 5-way. Goedecke measured 2–3× speedup on Qwen2.5-1.5B doing exactly
this. **Defer until B1 lands** — decomposition matters more than substrate
count.

**Acceptance gate:** protocol parity with `decision_to_systemone_answers`, and
it beats `RuleDecisionProvider` on the frozen cohort or is reported as a failed
challenger.
**Effort: M. Impact: low–medium. Risk: low.**

#### B7 — Explicitly **not** recommended

- **Fine-tuning ModernBERT on domain text now.** Our evidence text is short,
  bounded and templated; re-annealing (`AnswerDotAI/ModernBERT` intermediate
  checkpoints) needs a real corpus and >1k labels. Revisit after Track A.
- **A bigger LLM as the classifier of record.** Loses to DeBERTa on labelled
  tasks (§2.5), 20× slower, and the decoder's context cost scaling (57→445 ms
  from 256→2048 tokens) is hostile to a weekly CPU run. Keep it as a teacher and
  as an appeal tier only.
- **kNN over embeddings as a primary classifier.** Fails on rule-separable
  decisions (§2.2 control). Use logistic/prototype heads instead.
- **ColBERT / late interaction.** A retriever, not a classifier. Only relevant
  if we build an exemplar library of past incidents — a Track C idea, not now.
- **ALBERT-style parameter sharing** for latency. Fewer params ≠ faster.

---

### Track C — Residual engineering and hygiene

| # | Increment | Acceptance gate | Effort |
|---|---|---|---|
| C1 | **Fix the stale demo labels.** Regenerate `data/labels/demo.jsonl` at `feature_version: 3` so the README `qc train` snippet runs. | README quickstart executes top to bottom. | XS |
| C2 | **Fix `docs/synthetic-data.md`** to match `qcgen/spec.py`, or add a test that diffs the doc table against `FAMILY_SPECS` so it cannot drift again. | Doc/test agrees with `FAMILY_SPECS`. | XS |
| C3 | **TSPulse scenario-holdout estimability.** The gate needs ≥2 scenarios per family. Either generate a multi-scenario-per-family suite and estimate it, or formally retire the gate with the reason recorded. | Gate is *estimable and passed* **or** explicitly withdrawn in `claims.md`. No third state. | S |
| C4 | **Provider ablation on labels.** `qc evidence-bench` currently reruns the rule adapter only; extend to rerun the actual provider (already noted in `reliability-limitations.md`). | Ablation reports per-substrate evidence sensitivity. | S |
| C5 | **Crash/power-loss tests** for the journal and publication seams. | Fault-injection tests for the named seams. | S |
| C6 | **Incident exemplar memory (Phase 12 completion).** Retrieval over past confirmed incidents as a feature for the combiner in B1. | Recall@k on held-out incidents reported. | M |

---

## 4. Suggested sequence

```
Week 1   C1, C2            (unbreak the quickstart; stop doc drift)        [XS]
         B4                (metric surface — needed to measure everything) [S]
         A3                (governance doc — cheap, and a named gap)        [S]

Week 2   B1                (decomposition — the highest-value change)      [M-L]
         + pre-register config/decomposition-gate.json BEFORE running

Week 3   B3                (calibration: Platt/isotonic + reliability)     [S-M]
         B2                (SetFit probe + learning curves)                [M]

Week 4   C3, C4, C5        (close the named engineering holes)             [S ea]
         A1, A2            (real-table dry run + label instrument)         [S ea]

Later    B5, B6, A4, C6    (only if B1-B3 land and real labels are arriving)
```

The pre-registration discipline matters more than the schedule: freeze
`config/decomposition-gate.json` and the split **before** looking at held-out
results, exactly as `config/cohort.json` and `config/tspulse-gate.json` already
do. Do not move the gate to fit the result.

---

## 5. What success looks like

Short term (synthetic, achievable now):
- a learned substrate **beats `RuleDecisionProvider`** on the frozen cohort
  held-out split under a pre-registered gate, with macro-F1, MCC, ECE and
  per-class recall reported and paired CIs excluding zero;
- decomposition beats the flat 11-way question on every substrate;
- the README quickstart runs clean.

Medium term (needs real data):
- `docs/claims.md` gains its first `validated-real` row;
- `qc champion` selects a substrate on real analyst labels;
- statistical clearance is enabled against a pinned real qualification artifact.

Non-goals, restated so they stay non-goals: no production-eligible learned
artifact without a pinned real test set; no synthetic or teacher label ever
confers eligibility; the rule provider stays in every bake-off; the
deterministic evidence layer remains the source of truth and no model may clear
a deterministic finding.

---

## Sources

Primary for the Jev claims: `typesafe.ai/blog/introducing-system-one-models-and-jev`,
`docs.typesafe.ai`, `sebastianraschka.com/blog/2026/jev-classification-generalization.html`,
`seangoedecke.com/jev-means-structured-output-is-interesting-again/`,
`docs.litellm.ai/blog/jev-auto-router-benchmark`, `github.com/anisselbd/jev-phishing-bench`,
`github.com/featherless-ai/simple-jev`, `latent.space/p/ainews-here-are-6-clones-of-jev-in`.
Note: `jevai.net` / `jevtypesafe.org` are unofficial fan sites — do not cite.

Primary for the classifier claims: `arXiv:2412.13663` (ModernBERT),
`arXiv:2111.09543` (DeBERTaV3), `arXiv:2507.11412` (Ettin),
`arXiv:2209.11055` (SetFit), `arXiv:2205.13147` (Matryoshka),
`arXiv:1909.00161` (zero-shot TC), `arXiv:2406.08660`,
`aclanthology.org/2025.findings-emnlp.1033`, `arXiv:2410.01627`,
`arXiv:2407.12813`, `arXiv:1706.04599` (temperature scaling),
`link.springer.com/article/10.1186/s12864-019-6413-7` (MCC),
`arXiv:2404.07661` (metrics under extreme imbalance),
`github.com/alexjacobs08/beatingBERT`, `github.com/dcarpintero/pangolin-guard`.

**Evidence caveats.** MTEB leaderboard figures quoted in the underlying research
come partly from third-party aggregators; the board splits into three
non-comparable leaderboards (legacy / v2 / MMTEB) — re-verify before quoting any
specific number. All Jev results other than TypeSafe's own launch post are ≤9
days old at time of writing and rest on three evaluations over synthetic or
narrow corpora. The speed and cost result is well replicated; **the accuracy and
calibration results are not settled**, and one strong benchmark has Jev losing
to Haiku 4.5 on accuracy by 18.7 points. Plan accordingly: treat Jev/Laya as
measured challengers, never as trusted authorities.
