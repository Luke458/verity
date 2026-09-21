# Optional Laya CPU challenger

Rules remain System One's default. Laya is an optional integration experiment;
its CPU smoke and synthetic comparison are not evidence of retail accuracy.
The upstream [documentation](https://github.com/NandhaKishorM/laya) describes
training-sensitive typed decisions. Its [runtime](https://github.com/NandhaKishorM/laya/blob/main/laya/agent.py)
loads model/tokenizer assets; this wrapper controls the download and local copy.

Use a separate Python 3.12 environment. Torch and Transformers are absent from
the core runtime. The lock includes CPU Torch and every installed dependency:

```sh
python3.12 -m venv .venv-laya
.venv-laya/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r optional/laya-requirements.lock
.venv-laya/bin/python optional/laya_service.py --prepare .laya/run-1
.venv-laya/bin/python optional/laya_service.py \
  --model-dir .laya/run-1/service-model --port 8766 --timeout 60
```

The explicit English checkpoint is `convaiinnovations/laya` at revision
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`. There is no automatic checkpoint
routing. Preparation downloads an immutable snapshot and copies it to a
service-owned directory. Source hashes and post-load effective hashes are
recorded, accounting for upstream tokenizer changes. Weight hash:
`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
Do not reuse a preparation directory for another model revision.

The service binds localhost and exposes `/health`, `/v1/models` and
`/v1/systemone`. It supports strict `choice`, `score` and `noul` answers.
Requests are byte-bounded and socket/inference deadlines are enforced; a timed
out worker is terminated and requires service restart. The core records
challenger unavailability and preserves deterministic results.

Tokenizer-aware packing reserves complete questions and distinct options,
prioritizes failed/required checks, and records omitted optional evidence. It
abstains if mandatory evidence cannot fit or upstream head/option limits would
truncate it. The checkpoint's small context will therefore reject some larger
retail cases. Retail cause/origin rubrics include UNKNOWN. Raw predictions,
provider probabilities and model/evidence identity remain visible alongside
the effective policy decision. Probabilities are not domain-calibrated.

```sh
.venv/bin/qc decide --scenario-dir data/suites/demo/scenario-0000 --provider laya
.venv/bin/python -m optional.laya_smoke
.venv/bin/python -m optional.laya_compare
```

Generated local evidence: [CPU protocol smoke](../reports/laya-smoke.json) and
[hand-authored synthetic comparison](../reports/laya-comparison.json).
These ignored artifacts are reproduced by the commands above. The smoke
exercises all three typed interfaces; the comparison includes doubled report
totals and verifies the deterministic review floor. In one smoke the model
assigned review probability below 0.5 to a failed check, illustrating why
model recommendations cannot clear deterministic findings. Fine-tuning,
domain calibration and promotion require real analyst evidence; a synthetic
benchmark result never changes the default provider.
