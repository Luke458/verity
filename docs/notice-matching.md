# Experiment: matching free-text notices to findings

**Status: done; test results below, then a registered confirm run of an extraction model (Clairic-flash) at the end. Shipped as `qc notices`**
([weekly-run.md](weekly-run.md#drafting-explanations-from-notices)); the sandbox
is `experiments/notices/`.
Research sandbox; nothing here changes how the engine decides a status.

## The gap

The engine can clear a structural change only through a registered, approved
expected event (classification, entity, week range; `attribution._match_expected`).
In practice the information arrives as text, in the days before a refresh:

> Chain 4 store S012 closed for refit from week 118, reopening in about six weeks.
> Products in Cough & Cold moving to Respiratory from wk 120 (range review).
> Store S900 (new, Burwood) loaded with 12 weeks of history.

Someone has to read the notices, work out which flagged change each one
describes, and write the registry entry. That is a reading task. The cause
labeller experiment ([labeller-experiment.md](labeller-experiment.md)) showed
that a decision model recognizes direct evidence well but applies conditional
rules poorly. Matching a notice to one of a few observed changes needs the
first skill, not the second.

## The task

For one assessed refresh and one notice, the candidates are the refresh's
observed lifecycle changes, grouped by classification (for example "latest
week missing: stores S012, S044" or "new entity with history: store S900,
weeks 109-120"), plus **none of these**. A decision model answers one
`choice` question per notice: which candidate does the notice describe? The
candidates are exactly the options Decision-2.0-Lux scores jointly.

A match becomes a **draft** registry entry (classification, entity IDs, week
range, notice text as provenance), never an approved one. A person approves
it, and only then does the deterministic engine use it on a rerun. The
status path is unchanged, and so is the rule that nothing clears itself.

## Ground truth and data

Generated alongside each scenario (`qcgen`), so the oracle knows the answer:

- **True notices** for the injected change: store closures (missing stores),
  new stores with history (backfill), history removed (truncation), category
  moves (reclassification), store merges (entity merge). Several templates
  per change type, with varied ID formats ("S012", "store 12"), week phrasing
  and irrelevant detail.
- **Distractors**, the dangerous case:
  - the right change type for an entity not in this refresh;
  - the right entity with the wrong change ("S012 refit" against a backfill
    of S012);
  - the right entity and change outside the observed weeks;
  - notices about another dataset, and unrelated operational chatter.
- **Clean refreshes** with only distractors.

A later, harder set rewrites the templated notices with a local LLM
(Qwen3.8-27B on the development GPU) so wording is not template-regular.

## Measurement

Per notice: the chosen option against the oracle's (a candidate or none).
Reported separately:

- **False explanation rate:** a distractor matched to a real change. This is
  the error that could get a real fault approved away, so it is the headline.
- **Recall:** true notices matched to the right change.
- By distractor kind, and templated versus LLM-rewritten.

Baseline: a deterministic matcher that extracts entity IDs, weeks and
change-type keywords with patterns and matches them to candidates. On
templated notices it should be hard to beat. The question is whether it
degrades on rewritten text while the model does not.

Splits by seed as in the labeller (train for any threshold tuning, dev for
iteration, test scored once), and predictions registered before the test
run.

## Dev results (iteration)

Dev split: 100 refreshes (seeds 8101-8110), 366 notices on refreshes with at
least one observed change. Methods:

- **patterns**: written before any result and never changed.
- **lux**: one `choice` question, answering the most probable option.
- **lux + checks**: the model is asked only direct questions, and code combines
  them. After the choice, four yes/no checks on the chosen change: about this
  dataset? already happened by the latest week? same kind of change? weeks
  cover the observed weeks? The match stands only if every check clears 0.5
  (or one shared threshold tuned on dev).

The rewritten set is the same notices reworded by Qwen3.8-27B
(`rewrite.py`). A rewrite must keep every number and any other dataset's
name, or the template text is kept (40 of 510 on dev). A hand audit of 40 random
rewrites found 39 that keep their answer, and one borderline case.

| Dev | patterns | lux | lux + checks | lux + checks, tuned |
|---|---|---|---|---|
| Templated: accuracy | 0.98 | 0.70 | 0.80 | 0.87 |
| Templated: recall | 56/60 | 60/60 | 60/60 | 21/60 |
| Templated: false explanations | 1% | 30% | 20% | 3% |
| Rewritten: accuracy | 0.87 | 0.70 | 0.90 | 0.90 |
| Rewritten: recall | 34/60 | 60/60 | 56/60 | 45/60 |
| Rewritten: false explanations | 6% | 30% | 9% | 6% |

- **Templated notices:** the patterns know the template vocabulary by
  construction and win easily.
- **Rewritten notices:** the patterns lose 22 of their 56 matches.
- **A single choice question** finds every true notice but falsely explains
  30%: it recognizes the entity and the change and ignores the condition
  (another dataset, a future date). That is the cause labeller's weakness
  again.
- **Asking the conditions as separate yes/no questions** fixes the weeks
  (46% to 2% false explanations on rewritten notices). Other datasets remain
  the weak spot for both methods, especially when the notice names the other
  data source only in passing.

## Registered predictions (before the test run)

Frozen: patterns, lux + checks at 0.5, and lux + checks with the dev-tuned
threshold. On the test split (200 refreshes, seeds 8201-8220):

1. **Templated:** patterns beat lux + checks (0.5) on accuracy and on false
   explanations.
2. **Rewritten:** lux + checks (0.5) recall ≥ 0.85, and patterns recall ≤ 0.70.
3. **Rewritten:** every method's false explanation rate is between 3% and 12%.
4. **Rewritten:** other-dataset notices are the largest source of lux + checks'
   false explanations.

## Test results

Test split: 200 refreshes (seeds 8201-8220), 720 notices on refreshes with at
least one observed change. 940 of 1,020 notices were rewritten; the other 80
kept their template text.

| Test | patterns | lux + checks | lux + checks, tuned | lux, one question |
|---|---|---|---|---|
| Templated: accuracy | **0.97** | 0.77 | 0.86 | 0.68 |
| Templated: recall | 114/120 | 116/120 | 42/120 | 120/120 |
| Templated: false explanations | **2%** | 22% | 4% | 32% |
| Rewritten: accuracy | 0.83 | **0.87** | **0.87** | 0.70 |
| Rewritten: recall | 62/120 (0.52) | **107/120 (0.89)** | 89/120 (0.74) | 120/120 |
| Rewritten: false explanations | 9% | 11% | 9% | 30% |
| Rewritten: false explanations, other-dataset notices | 48/125 | 60/125 | 49/125 | 120/125 |

**Registered predictions** (methods = the three frozen ones):

1. **Templated, patterns win: held.** Accuracy 0.97 against 0.77; false
   explanations 2% against 22%.
2. **Rewritten recall: held.** Lux + checks 0.89 (needed ≥ 0.85); patterns 0.52
   (needed ≤ 0.70).
3. **Rewritten false explanations in 3-12%: held**, at 9%, 11% and 9%. The
   single-question model, never a frozen method, was 30%.
4. **Other datasets dominate Lux + checks' false explanations: held**, 60 of
   82.

**Two analyses not registered in advance** (rewritten test set):

- **Suggestion precision.** Of the matches each method proposes, 107/189
  (0.57) are right for Lux + checks and 62/127 (0.49) for patterns.
- **Notices already scoped to the dataset** (other-dataset notices removed):
  false explanations fall to 22/595 (3.7%) for Lux + checks and 17/595 (2.9%)
  for patterns.

## Conclusion

On realistically worded notices, a decision model asked direct questions
finds 89% of the notices that explain a flagged change, against 52% for
hand-written patterns, at a similar false explanation rate. The patterns win
only on the templates they were written for. This is the first role in this
repo where the model beats the deterministic alternative, and it is the one
the deterministic engine cannot fill: reading text.

It is a **suggestion aid, not an approver.** About four in ten of its
suggestions are wrong, mostly notices about another dataset that mention the
same stores. Its matches belong in draft registry entries that a person
approves, which is the only way the engine accepts an explanation anyway.
Notices should be scoped to their dataset upstream (by channel or metadata),
where both methods' false explanations fall to about 3%.

What the result does not show: the notices, rewrites and distractors are
synthetic. Real notices are messier and refer to things this generator does
not model (banners, regions, promotions, partial weeks).

Results: [`results/notices-test.json`](results/notices-test.json) and
[`results/notices-test-rewritten.json`](results/notices-test-rewritten.json).

## Clairic-flash: an extraction model reads, the rules decide (2026-11-19)

**Status: confirm run done; 5 of 6 predictions held (results below).** The registration text that follows is as written before the run (sha256 of the file then: `6e0606a5`).

The patterns lose recall on reworded notices because their *reading* breaks
(regexes for cues, tense, tags and weeks), not because their matching rules
are wrong. [Clairic-flash](https://huggingface.co/Cloudflare/clairic-flash)
(Cloudflare, 9B, Apache-2.0, released 2026-11-18, revision `36ad68b`) is an
extraction model: text and typed fields in, typed values with confidence,
`p_null` and character spans out (System One: Extract). The new method asks it
to fill the fields the patterns extract, `qc.notices.NOTICE_FIELDS` (data
source, kind of change, planned, stores, products, categories, first and last
week), and passes its reading to the **same** deterministic rules
(`qc.notices.decide`, shared with `PatternMatcher`, whose 3,060 dev and test
answers are unchanged by the refactor). The model never sees the candidates,
so it cannot pick a change; it can only misread the notice. Shipped as
`qc notices --matcher extract` (`ExtractMatcher`, any Extract endpoint:
`experiments/notices/clairic.py serve` on loopback, or Workers AI with
`QC_EXTRACT_TOKEN`).

**Running it on the development GPU** (16 GB, ROCm): the published runtime
targets NVIDIA (flash-attn3, deep_gemm, flashinfer). Off sm90 it already uses
SDPA, and its BF16 fast paths are opt-in hooks that `clairic.py` leaves off,
so the backbone runs as plain eager Transformers. BF16 (19.3 GB) does not fit,
so the runs use int8 weight-only quantization (torchao) of the decoder's
linear layers (peak 13.2 GiB; LM head, norms and the joint head unquantized).
No 8-bit release exists. On the README example the int8, BF16 and unmodified
CPU fp32 runtimes return identical values and token counts. A larger fp32
reference was not run: a notice costs several passes over about 5,000 prompt
tokens (estimated minutes per pass on this CPU, not measured).

### Dev iteration (100 refreshes, 366 notices with observed changes)

| Dev, reworded | patterns | Clairic v1 | **Clairic v2** | Lux + checks |
|---|---|---|---|---|
| accuracy | 0.87 | 0.92 | **0.95** | 0.90 |
| recall | 34/60 | 42/60 | **47/60** | 56/60 |
| false explanations | 23/366 (6%) | 10/366 (3%) | **4/366 (1%)** | 33/366 (9%) |
| false expl., other dataset | 19/64 | 7/64 | **1/64** | 22/64 |

| Dev, templated | patterns | Clairic v1 | **Clairic v2** | Lux + checks |
|---|---|---|---|---|
| accuracy | 0.98 | 0.89 | **0.98** | 0.80 |
| recall | 56/60 | 43/60 | **56/60** | 60/60 |
| false explanations | 4/366 | 25/366 | **5/366** | 73/366 |

v1 is the first schema, scored with the final rules. Changes from v1 to v2,
all made on dev: a `[tag]` and bare words (data, history, trading) in the
data-source instruction and the generic-word list; "planned" excludes a later
reopening or resupply, and stated weeks override tense (a change starting by
the latest week is in effect); "sending no sales" added to the closure
option; fuller list instructions. Adding the patterns' id regexes to the
model's ids gained one notice and was dropped. The int8 model agreed with BF16
on 142 of 144 fields over the first 18 dev notices (neither difference changed
a match).

### Registered for the confirm split (written before any method ran on it)

New split `confirm`: seeds 8301-8320, 200 refreshes, 732 notices on refreshes
with an observed change (120 true, 128 other-dataset, 128 wrong-entity, 128
wrong-change, 100 wrong-weeks, 128 chatter), reworded by the same
`rewrite.py` and Qwen3.8-27B. Frozen methods: **patterns** (unchanged);
**Lux + checks** at 0.5 (the 2026-10-07 method: Decision-2.0-Lux-9B revision
`1dcd1e7`, whose weights are the 2026-10-03 release, on the same EXL3 4.0 bpw
backbone); **Clairic** (schema v2, `NOTICE_SCHEMA_VERSION = 2`, int8, model
ids only, `decide`).

Gates, checked before scoring:

- **G1.** Lux reproduces its stored dev answers (its head was re-downloaded):
  same chosen option and same check verdicts on ≥ 19 of 20 reworded dev notices.
- **G2.** Clairic int8 against BF16 on 40 random reworded dev notices (seed 0):
  the same match on ≥ 38.

Predictions:

1. **Reworded:** Clairic recall ≥ 0.70; patterns recall ≤ 0.65.
2. **Reworded:** Clairic false explanations ≤ 4%, and fewer than either other method.
3. **Reworded:** Lux + checks recall is higher than Clairic's (the model that
   sees the candidates finds more, at a cost).
4. **Reworded:** Clairic suggestion precision (right / proposed) ≥ 0.85;
   Lux + checks ≤ 0.70.
5. **Templated:** Clairic accuracy ≥ patterns' − 0.03, and its false
   explanations ≤ 3%.
6. **Reworded:** Clairic falsely explains ≤ 8% of other-dataset notices.

### Confirm results

200 refreshes (seeds 8301-8320), 732 notices on refreshes with an observed
change; 941 of 1,020 notices were reworded, 79 kept their template text. Both
gates passed first: Lux reproduced its stored dev answers on 20 of 20 notices
(G1), and int8 Clairic matched BF16 on 40 of 40 (G2; only 6 of those 40 were
matches, so G2 is weak on the matches themselves). The run was interrupted
once (the agent session ended) and resumed from its output file; no notice
was read twice.

| Confirm, reworded | patterns | **Clairic** | Lux + checks |
|---|---|---|---|
| accuracy | 0.85 | **0.94** | 0.88 |
| recall | 62/120 (0.52) | **96/120 (0.80)** | 106/120 (0.88) |
| false explanations | 50/732 (6.8%) | **20/732 (2.7%)** | 75/732 (10.2%) |
| suggestion precision | 0.55 | **0.83** | 0.59 |
| false expl., other dataset | 39/128 | **9/128** | 50/128 |
| false expl., wrong change | 6/128 | 11/128 | 19/128 |

| Confirm, templated | patterns | **Clairic** | Lux + checks |
|---|---|---|---|
| accuracy | **0.975** | 0.966 | 0.777 |
| recall | 113/120 | 110/120 | 118/120 |
| false explanations | **11/732 (1.5%)** | 15/732 (2.0%) | 161/732 (22%) |

Wilson 95% intervals, reworded: Clairic recall 0.72-0.86 against patterns
0.43-0.60 and Lux 0.81-0.93; false explanations 1.8-4.2% against 5.2-8.9% and
8.3-12.7%. Suggestion precision is right matches / proposed matches.

**Registered predictions:**

1. **Clairic recall ≥ 0.70, patterns ≤ 0.65: held** (0.80 and 0.52).
2. **Clairic false explanations ≤ 4% and fewer than either other method: held**
   (2.7%, against 6.8% and 10.2%).
3. **Lux + checks recall higher than Clairic's: held** (106 against 96; the
   intervals overlap).
4. **Clairic suggestion precision ≥ 0.85, Lux + checks ≤ 0.70: failed.**
   Clairic 0.83 (96 right of 116 proposed); Lux 0.59. Dev had 0.92 (47/51), a
   small-sample estimate the prediction leaned on.
5. **Templated: Clairic accuracy ≥ patterns' − 0.03 and false explanations ≤
   3%: held** (0.966 against 0.975; 2.0%).
6. **Clairic falsely explains ≤ 8% of other-dataset notices: held** (9/128, 7.0%).

**What the errors are** (looked at after scoring; nothing was changed). Of
Clairic's 20 reworded false explanations, 11 are merge notices about the
wrong id ("S900 replaced by S969" against a refresh where S900 arrived as a
new store with history). The shared rules accept a merge cue against a
new-store backfill that shares an id, whichever direction the merge runs; the
patterns have the same rule and trip it less only because they read fewer
merge cues. The other 9 are other-dataset notices whose source sits
mid-sentence ("Heads up on the PBS script volumes data: ...") and was not
extracted. Of the 24 missed true notices, 9 had their change cue misread
(closures, deleted history), 7 were rejected as another dataset because a
non-name was read as the source, and 8 found no observed change (ids or weeks
misread).

### Conclusion

Letting the model read and the rules decide is the best trade-off measured
here on reworded notices: recall between the patterns' and Lux's (0.80
against 0.52 and 0.88) at less than half the patterns' false explanations
and about a quarter of Lux's, so about five in six suggestions are right
(Lux: three in five). On templates it matches the patterns. The model never
sees the candidates, and every decision is a deterministic rule a reviewer
can read; each draft carries the extracted values, confidences and character
spans. It is still a suggestion aid behind human approval.

The remaining errors point at two fixes for a later registered test: the
merge rule should require the notice's direction to match the observed one,
and the data-source field misses sources named mid-sentence. The cost is
latency: about 6-8 s per notice on the 16 GB development GPU (int8, ROCm),
against well under a second for the patterns. That is fine for a weekly inbox
of notices, but not for bulk text. As before, the notices, rewrites and distractors are
synthetic.

Results: [`results/notices-confirm.json`](results/notices-confirm.json) and
[`results/notices-confirm-rewritten.json`](results/notices-confirm-rewritten.json)
(methods `clairic`, `patterns`, `lux-checks`; `clairic+ids` and the tuned
variants were not registered).

**After the confirm run:** `--matcher systemone` (`SystemOneMatcher`) and
`experiments/labeller/lux.py serve` were removed from the shipped tool. On the
error this experiment treats as the headline, false explanations, Lux + checks
was the worst method on both variants (10.2% reworded, 22% templated); its
only advantage was 10 more true notices out of 120. Its request builders
(`choice_request`, `check_request`) moved unchanged to
`experiments/notices/match.py`, so every Lux number here still reproduces.

### Merge direction (post hoc, after the confirm run)

`decide` now requires a merge notice to run the observed way: it explains a
new store with history only through the id it names as the new one, and a
removal only through an id it names as the old one. The patterns read the
direction from the ids after a forward phrase ("merged into", "replaced by",
"now trades as", "rebranded to", "consolidated into"); with no such phrase,
the old rule (any shared id) applies.

Measured only on spent splits, and designed after seeing the confirm errors,
so biased in its favour. False explanations of the patterns, before and
after, with recall unchanged on every split:

| Patterns | dev | test | confirm |
|---|---|---|---|
| templated | 4 -> 1 | 14 -> 4 | 11 -> 0 |
| reworded | 23 -> 23 | 65 -> 61 | 50 -> 46 |

**Correction to the error analysis above.** The fix does not reach
Clairic's 11 merge false explanations. Its reading has no new-id field, and
for most merge notices it lists one store (the old id), sometimes filing the
other id under products, so the direction cannot be read and the old rule
applies. Those errors are a reading gap as well as a rule defect. Closing it
needs a new-id field in `NOTICE_FIELDS`, which changes the request and needs
its own registered test. Until then `--matcher extract` is exactly the
registered method; its confirm answers are unchanged.

### A merge drafts the replacement (2026-11-20)

When a merge notice fits several of a refresh's changes, `decide` now
prefers the replacement ("S005 replaced by S900") over the removal of S005
or the backfill of S900. `qc notices` drafts a replacement as both its
removal and its backfill. Before, it drafted only the removal, so approving
the draft still left S900's backfilled history unexplained. Every measured
count is unchanged on every stored split (both methods, both variants), as
the gold answers accept any of the three. Between 11 and 33 merge notices
per split now pick the replacement instead of the removal.

**What approving a merge covers.** Approving both drafts explains the whole
history revision of the two stores, including any value the merge gained or
lost. The registry checks conservation for closures and category moves, not
for replacements. The generator's merge fault carries history across scaled
by 0.9-1.1; in the test's scenario (0.92) the refresh stays INVESTIGATE only
because the latest-week share test flags the change
(`tests/test_qc_notices.py`). A conservation check for replacements is not
built.
