# Production-only setup protocol

The independent runner `scripts/benchmark_native_raw_pair_production_setup.py`
uses the frozen general-policy worker without changing its runtime. It requires
explicit paths to completed strict model qualification receipts. CPU preflight
checks every receipt's source hash and common execution fingerprint. The new
worker process independently compares those receipts with its actual current
model/device/native-table fingerprint before activation.

The intersection of observed qualifying batches on both TP ranks determines the
eligible graph sizes. Requested batches outside that intersection remain native.
The development-frozen shape policy still applies; offline qualification does
not replace physical-prefix admission or permit previously untested batch sizes.
Actual train26 receipts currently cover all B6..21 on both ranks and pass the CPU
preflight against current frozen sources.

Only production candidate graphs are captured. Diagnostic Q/O buffers are not
allocated, no witness/checked graph is captured, and no per-session qualification
generation occurs. A setup receipt verifies these facts plus the exact offline
provenance gate. The existing native graph collection remains for fallback.

`--setup-only` starts a fresh model process and measures native initialization,
additional candidate configuration, and production graph capture. It ends with
`completed_setup_only` and makes no generation or speed claim. Default execution
runs natural-EOS native/candidate/candidate/native phases; `--phase-kind fixed`
uses 128-token control, and `both` runs both. Natural output tokens, finish
reasons, cache hits, and actual graph trajectories must match across ABBA.

For full runs, first-use component sums combine measured common input/native
initialization, candidate-only setup (including offline preflight), and ABBA
cold-document task means. They are explicitly **estimates**, not independent
cold-service measurements. Experimental export/control costs are separately
recorded. The first A arm can include JIT effects; native AA stability remains
visible and must be assessed before interpreting speed. Candidate execution
and exact-output gates remain required, including when the policy falls back
for an entire short-prefix document.

Example (root assigns GPUs):

```sh
.venv-engine/bin/python scripts/benchmark_native_raw_pair_production_setup.py \
  --protocol configs/native_natural_cohort16plus_training_v1.json \
  --document-id DOCUMENT_ID \
  --offline-qualification-records \
    results/native_raw_pair_general_policy_qual_v1/train26_bf16/rank0/qualification_raw_pair.json \
    results/native_raw_pair_general_policy_qual_v1/train26_bf16/rank1/qualification_raw_pair.json \
  --setup-only --output results/native_raw_pair_production_setup_v1/DOCUMENT_NAME
```

This runner has passed syntax/CLI checks and actual-receipt CPU preflight. It has
not yet run GPU setup or production generation at this source freeze.
