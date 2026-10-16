# Explicit training-shape transfer on the fixed next eight documents

This independent version transfers the combined direct-input/device-cut implementation's **training-shape** qualification. It does not label new documents as locally qualified. The original document-indexed strict and production loaders remain unchanged. Neither constituent implementation's receipts nor the old grouped-hint receipts are accepted.

The frozen selection preserves all eight original documents. The six inputs fitting the unchanged 163840-token context become three fixed two-document tasks; original indices 5 and 7 retain their context failures. No replacement, truncation or complete-eight-document success claim is permitted. The exact inputs, order, hashes and fixed policy are in `configs/raw_pair_direct_multidoc_next8_shape_transfer_energy_v2.json`; its only change from planning v1 is explicit setup energy accounting.

The worker validates both ranks' exact-source, model/device/table-extent and all-layer strict receipts from the two frozen training documents. Per-rank unions of qualified batch shapes are intersected across ranks and restricted to B6..19. Missing shapes use native attention. Evaluation documents do not change this set. Every new document still clears request/slot/overwrite ownership, uploads its cut at a synchronized empty-scheduler boundary, binds physically identical prefix pages, preserves original native S and aliases native lengths/tables. One production graph bank is captured for the engine and used for both documents.

Four natural arms run in fixed native/candidate/candidate/native order, with both documents in the same order in every arm. All outputs are retained before final identity gating. Output tokens, finishes, APC and request/position trajectories, native AA and candidate BB must agree. Divergence invalidates the affected speed claim and is not a reason to change the input list or dispatch policy. Full-document quality, caps and preemptions are reported. This is measured trajectory equivalence under an explicitly stated transfer protocol, not a proof of all-state numerical equivalence.

## Timing and energy

Read-only NVML counters are matched to actual worker CUDA UUIDs and initialized **before** candidate configuration. Boundaries bracket configuration, the single capture and common precision export. Counter initialization/identity RPC and precision export are common control costs charged to both arms. Candidate offline provenance validation plus configuration/capture and their three counter samples are charged once to candidate setup.

The first-task primary uses the sum of two-document **full phase** ABBA means, including switches, prefix reset, cold anchors, group control, generation, worker export and all six per-document NVML samples. One-time common controls and candidate setup are then added. This is a component estimate from one shared engine, not an independent cold-service benchmark. An additional cold component adds measured input and native-engine initialization equally.

Task Joules include measured candidate configuration/capture and common precision export once. Common engine and NVML initialization precede the first energy counter; offline CPU validation also has no GPU boundary. Thus this is not full-process startup energy. Unsupported, invalid or UUID-unverified counters remain null. Exclusive physical GPUs are necessary for board-energy attribution.

## Execution prerequisites and commands

Root must first finish the new combination's cross-cut/FP64, real counterexample, memcheck, full-model strict training and development production identity gates. The standalone-gate results do not substitute for model qualification.

CPU contract checks (no visible GPU):

```bash
CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/check_raw_pair_training_shape_transfer_cpu.py
CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/benchmark_raw_pair_direct_next8_transfer.py --task-index 0 --training-qualification-summary results/raw_pair_direct_multidoc_model_qual_v1/summary.json --prepare-only --output results/raw_pair_direct_next8_transfer_prepare_task0_v1
```

After successful current-source preparation, root may assign each task index 0, 1, 2 its own exclusive GPU pair:

```bash
.venv-engine/bin/python scripts/benchmark_raw_pair_direct_next8_transfer.py --task-index 0 --training-qualification-summary results/raw_pair_direct_multidoc_model_qual_v1/summary.json --output results/raw_pair_direct_next8_transfer_energy_v1/task0
```

A 2400-second limit per task is a conservative initial allocation; timeout/failure remains a result. `--setup-only` is an optional diagnostic that does not establish task benefit. The CPU test verifies provenance rejection (including the actual old qualification), transfer coverage, clearing ownership, unchanged computational hooks/capture, energy ordering, complete cost arithmetic and null preservation. It uses mock numerical receipts only and does not fabricate GPU evidence.
