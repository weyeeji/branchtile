# BranchTile feasibility experiments

Started 2026-10-16 (Asia/Shanghai). The source hypothesis and full acceptance
criteria remain in `proposal-iscas.md`. This log distinguishes qualification
evidence from a completed application evaluation.

## Constraints and environment

- All commands run as uid 1000 (`qcjz`), never sudo/root.
- Python: `.venv/bin/python`. The project venv uses the existing
  `/home/qcjz/miniconda3/envs/sglang-vllm-m120` packages read-only via
  `--system-site-packages`; any extra packages are installed into `.venv`.
- Eight RTX 5090 GPUs (32 GB each), CUDA capability 12.0, 170 SMs,
  reported 96 MiB L2. GPUs 0–3 and 4–7 are two NUMA groups; no NVLink.
- PyTorch 2.11.0+cu130, Triton 3.6.0, TileLang 0.1.9,
  FlashInfer 0.6.12, Transformers 5.12.1; nvcc 13.2.
- No Slurm binaries are available here. Use explicit per-process GPU assignment,
  check for other workloads before launching, and record UUIDs and durations.
- P1 qualification budget: 24 allocated GPU hours (includes load/failures).
  P2 qualification budget: 8 allocated GPU hours. No fabricated busy-work to
  raise utilization; independent useful experiments use available cards.

## Initial evidence and deviations

- The initial repository has the proposal and proxy helper only. No implementation,
  old evaluation manifest, or claimed previous result is present in this checkout.
- Targeted local searches have not recovered the original 8/100/167 DocFinQA
  split. A replacement exploratory split must be labeled as new, deterministic,
  and unsuitable for claiming the proposal's frozen formal evaluation.
- Locally available Qwen3-30B-A3B is the older BF16 model, not the specified
  Thinking-2507 FP8 checkpoint. Do not silently substitute it in results.
- A sandbox launcher failed before any experiment; subsequent commands use the
  normal user after the permission profile changed. This did not grant/use root.

## Work in progress

1. Implement exact same-mask fixed-row and changing-row on-chip attention,
   then validate against a high-precision reference before timing.
2. Fetch official datasets and selected checkpoints; capture natural query-aware
   supports with provenance rather than inventing favorable masks.
3. Verify 2026 primary literature and strongest applicable execution controls.
4. Qualify tensor parallel transport and native FlashInfer on this hardware.
5. Randomize paired timing, include split merge/preparation, check cold/hot cache,
   and retain unfavorable shapes. Qualify before full model integration.

Git commits checkpoint executable code, small raw measurements, configuration,
and conclusions; weights, captures, venv and compiler caches are ignored.

## Qualification checkpoint (2026-10-16)

- Native FlashInfer shared-source paged decode is correct against FP64. Its
  returned LSE uses log2; validation explicitly converts to natural log.
- Version 1 dynamic M16/M32 and register-resident fixed-row control pass 17
  tests, including empty rows, partial tiles, split merge, changing runtime
  row identities and the four fused/partial mechanisms. Compute Sanitizer
  memcheck passed three selected cases with zero reported errors.
- Eight separate GPU jobs completed the initial 480-run screening: 420 valid
  measurements, 60 candidate N64 launches rejected for excessive shared memory
  (102656 bytes). Raw failures are retained. After independently selecting the
  fastest configuration, v1 loses to the fixed-row control in all 30 tested
  synthetic shapes. These are hot-cache screening results; the selections
  span devices and are not a final paired-performance claim.
- V2 explicitly reuses buffers between mutually exclusive M16/M32 branches and
  separates Q/state loads to avoid conservative compiler barriers. Its N64
  compiled shared memory decreases to 64768 bytes including static allocation.
  The same-shape fixed control uses 47104 bytes. V2 is undergoing numerical
  checks and same-device comparisons with native FlashInfer; no v2 speedup is
  established yet.
- Nsight Compute reports `ERR_NVGPUCTRPERM` for uid 1000. We do not change driver
  permissions or use root. Actual hardware-counter/DRAM attribution remains
  unavailable; compiled resource counts and measured latency are separate
  evidence. All eight NVML cumulative energy counters are readable, but energy
  improvements have not yet been measured.
- Local old Qwen3-30B-A3B BF16 will be used for an explicitly labeled natural
  GQA8 diagnostic while official 2507 weights download. Three-card layer
  sharding is instrumentation, not a substitute for the proposed TP2 engine.
  First loading attempt failed on missing `accelerate`; this dependency was
  installed only in the project venv, with the failed attempt retained.

## Paired comparison checkpoint (2026-10-16 13:14 CST)

V2 passed all 17 FP64/memcheck cases and runtime-changing M16/M32 racecheck.
Configurations are tuned independently for hot and cold cache, then measured
in 30 randomized/reversed paired blocks with fixed A/A and block bootstrap.
The following are complete producer-plus-merge times in microseconds, not
application speedups. Raw records are in `results/paired/`.

| Synthetic shape | Cache | Native FlashInfer | Fixed region loop | Dynamic v2 |
|---|---|---:|---:|---:|
| cross, 16K, GQA8 | hot | 10.48 | 14.74 | 16.45 |
| cross, 16K, GQA8 | cold | 24.58 | 26.62 | 28.67 |
| cross, 128K, GQA8 | hot | 82.74 | 92.47 | 94.38 |
| cross, 128K, GQA8 | cold | 106.50 | 100.35 | 106.46 |
| mixed, 64K, GQA8 | hot | 27.60 | 39.39 | 52.98 |
| mixed, 64K, GQA8 | cold | 57.34 | 63.49 | 77.82 |
| cross, 64K, GQA4 | hot | 26.32 | 26.58 | 29.84 |
| cross, 64K, GQA4 | cold | 57.38 | 57.34 | 59.39 |

The first older-BF16-30B natural replay (QMSum group 0, four preselected layers,
4K document budget, step 1, including full system prefix/private suffix) also
loses: hot native 7.93–7.97 us, fixed 21.22–21.61 us, dynamic v2 26.22–27.41 us.
These are serially captured operator inputs, not simultaneous ready requests,
not the specified 2507 main checkpoint, and not quality measurements.

A bounded v3 attempt keeps canonical FP32 output state in registers and reuses
K/V shared storage during exact-bit state transfers. It achieves 48,384 bytes
shared memory, 240 registers, and no local/spill allocation, but still fails:
fixed/candidate = 0.738 (16K hot), 0.993 (128K hot), both below the approximately
1.15 progression threshold. Five FP64/memcheck cases and dynamic-map racecheck
pass. The conservative allocation bound is 0.153 GPU hours. No further dynamic
layout iteration is justified by these measurements; see
`results/kernel_ablation_v3/qualification.json`.

The 48-case strict 2×2 comparison is retained in `results/kernel_ablation_v2/`.
Region-partial execution is often faster despite writing full canonical rows;
the current evidence does not support a general performance benefit from
cross-region state residency. This partial implementation is not an optimized
packed E3 reproduction.

An additional fixed-row masked union can tile across owner-region boundaries,
eliminating their individual short-tail padding. It passes six FP64 tests and
memcheck (zero errors). It is a stronger existing-mechanism control, not a new
contribution; five independent comparisons are running. All eight frozen QMSum
development documents have older-30B captures; official 4B weights are now
verified against immutable HF hashes. Official 30B-FP8 weights are still
downloading. A separate `.venv-engine` now imports official vLLM 0.31.0 and
Torch 2.13.0+cu130 successfully; actual target TP2 model deployment is not yet
qualified.

The goal remains active. Evidence currently argues against the core performance
hypothesis. The missing target-checkpoint/application/quality, fixed-partition,
and exact hierarchy evaluations must not be described as completed.
