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

## Initial work plan

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

At this checkpoint the goal remained active. Evidence argued against the core performance
hypothesis. The missing target-checkpoint/application/quality, fixed-partition,
and exact hierarchy evaluations must not be described as completed.

## Final bounded feasibility decision (2026-10-16 13:52 CST)

The current dynamic-row implementation is a **no-go for full integration** under
the proposal's operator progression gate. This completes the bounded feasibility
investigation, not the unperformed full application/quality study. The Chinese
report `FEASIBILITY.md` records the final evidence and limitations.

- Official 4B and older-30B diagnostic captures each cover eight exploratory
  QMSum documents, four questions, four layers and four steps (1024 principal
  snapshots total). Dense trajectories hit the 256-token instrumentation cap;
  they are not complete-answer or sparse-quality results.
- FlashInfer 0.7.2 (released October 15) is qualified in `.venv-latest` with
  unchanged Torch/TileLang. Eight concurrent repeat jobs retain the dynamic
  negative result and initially show a fixed-union gain versus separate decode.
- A stronger native joint-query paged-prefill wrapper reuses the same physical
  payload with an exact custom mask. Four official-4B paired comparisons show
  native joint raw hot 9.55/10.17/12.43/11.80 us versus our fixed union
  10.15/10.25/13.07/12.19 us. Including native log2-to-ln conversion leaves only
  approximately 3–10% hot speedup for our union, below the 15% progression gate.
- Native-union qualification retains 4 passed and 3 failed small-shape tests.
  Direct native single-prefill reproduces all three LSE tolerance failures;
  inputs, references and tolerances were not relaxed. Actual performance
  configurations are timed only after passing the frozen FP64 checks.
- Dynamic/fixed validation in the latest environment passes 23 tests. V3 and
  union sanitizer evidence remains in `results/validation/`. This is not a
  claim that the entire repository test suite passes: the native failures are
  deliberate retained evidence of unsupported numerical cases.
- All 69 ledger jobs ended, including failed attempts; P1 ledger time is
  0.937 GPUh, P2 3.093 GPUh. Early manually timed probes are incomplete, so
  additional reserves are explicitly reported rather than invented exact costs.
  See `results/budget_audit.json`. All GPUs and the monitor were released.
- The formal 30B-2507 FP8 download was stopped with resumable files retained
  after the no-go decision. No TP2 main-model, application speedup, quality
  noninferiority, energy reduction, or universal impossibility is claimed.
- Literature and raw measurements remain versioned; an existing fixed-union
  mechanism is not relabeled as a new idea. Strong native joint attention is
  now a required baseline for any future pivot.

## Goal resumed: full paper evidence (2026-10-16 14:40 CST)

The user requested continued exploration until paper experimental requirements
are met. The earlier bounded no-go remains valid but is not the completion
criterion for this expanded goal. `PAPER_EXPERIMENTS.md` tracks every required
evidence category and explicitly marks missing work.

Official 4B QMSum train8 development screening now includes natural-EOS A0/A1
with 8192 tokens per question (joint B-times cap), complete outputs and failures.
All 64 question answers across the two arms parsed, no cap was hit. A single
seed and eager engine do not establish optimized production timing or formal
quality. Diagnostic ROUGE uses rouge-score0.1.2; equivalence to original QMSum
scoring remains unverified. Joint answers have higher overlap scores but usually
longer group completion; neither arm is declared quality-cost dominant.

A Transformers5 BatchEncoding mismatch caused the first pilot to fail; three
newly started sibling jobs were stopped once diagnosed. All allocation charges
and original snapshots remain, with `results/application_screen/failure_audit.json`.
The corrected tokenizer path was checked on all eight complete documents before
the four successful parallel retries. The step observer receives only final
outputs; its traces cannot establish scheduler ready membership.

Main-model TP2, executed E1/E4, stronger selector profiling and exact protocol
recovery remain active. No new successful paper mechanism is claimed yet.

## Public protocol and executable controls (2026-10-16 15:12 CST)

The user explicitly waived the old-data requirement. The new public DocFinQA
protocol freezes development8/evaluation100/training167 from verified official
sources, with text/question leakage checks, conservative identity components,
label-conflict sensitivity rules and every overlength document retained. This
does not assert the old company's split or company-held-out generalization.

The official Qwen3-30B-A3B-Thinking-2507-FP8 deployment passed a complete-report
TP2 probe with BF16 KV. Actual worker capture checked hooked K/V against paged
cache bit-for-bit in 200 snapshots and exported 24 layer/rank/anchor groups from
the same forward. Last two document tokens cross the shared full-page boundary
and remain in each private cache; they are never silently omitted. Capped256
instrumentation is not natural task evaluation. Full-EOS public development
evaluation at 262K context is now running separately.

E1 all15 fixed partitions and E4 all15 exact binary trees passed 984 initial
configuration checks plus small-shape FP64 qualification. E4's synthetic-tree
gain over full union shrinks to about2.8% against strong E1. Natural initial
shapes do not support a new mechanism. These are explicit exact mechanism ports,
not a complete QUILT reproduction. Official B3/GQA8 main-model replay is active.

The selector branch qualified native CUB stable-set selection, ties/repeats and
CUDA graphs. FlashInfer's strict deterministic-order flags are unsupported on
SM120; preserving only the stable selected set is qualified separately. Native
BF16 LSE changes one selected token in two of the initial four captures, while
the official86K main-model point preserves the set. MiniKV already establishes
LSE reuse; the speedups are a required strong baseline, not novel research.
Source: `results/selector/summary.json` and the pivot literature dossier.

QMSum now has20 development configurations and640 natural-EOS answers with no
empty final answers or cap hits: each official4B Thinking/Instruct model has
dense, joint, and eight hybrid-RAG settings. All original answers, token IDs,
preparation costs and grid results are retained. The calibrated original Perl
ROUGE scorer uses author-linked settings and matches the279-example published
fixture within0.011 score points; the remaining discrepancy is documented.
Google whole-summary ROUGE-L was diagnostic and is not interchangeable.

Instruct dense/joint macro ROUGE-L is29.15/30.67 and eager total group JCT across
eight documents is35.60/55.58s. Thinking dense/joint is26.12/29.40 and200.61/286.40s.
The tested Instruct RAG grid does not exceed dense on this exploratory
ROUGE/cost comparison after measured retrieval preparation. These different
model-recommended sampling configurations are application alternatives, not a
controlled model-only ablation. They still require factual review and formal
test evaluation; no noninferiority claim follows from train8.

One native CUDA-graph profile on QMSum group3 (roughly21K prompt tokens, B4)
records32 actual decode iterations: attention including split merge is47.11%
of summed GPU kernel time. Profiler timings are not uninstrumented latency,
and the dense fraction cannot be carried over to a sparse engine. Graph A/A
on group0 produced identical tokens, but first-call residual JIT caused16.65s
versus7.65s on the second call; steady execution and startup must be separated.

All new work remains ordinary uid1000 and project virtualenvs. All failed GPU
attempts and allocated compilation time are charged. Long native CUB AOT was
subsequently performed with no visible GPU and its706.5s CPU cost recorded
separately. Current parallel work also includes genuine sparse natural-generation
qualification; the full-paper goal remains active and incomplete.
