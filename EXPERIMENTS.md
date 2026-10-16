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


## 2026-10-16 15:29 — repeatability and complete sparse development cohort

The graph Instruct4B A/A cohort contains eight full documents, three cold
repeats and one warm repeat per dense/joint arm. All40 cold request sequences
are identical across repeats. Only18/32 dense and3/8 joint warm sequences
match cold, so warm/cold latency ratios are not same-workload speedups.
Uniform steady cold totals across eight documents are21.659s dense and30.991s
joint, excluding separately recorded initialization and first-call overhead.
Calibrated cold0 ROUGE-L is29.791/29.512; these are development results, not
factual correctness or formal test noninferiority. Raw Perl output names now
include the report stem to prevent eager/graph collisions; the four overwritten
eager raw files were restored from git before isolated graph rescoring.

The genuine HF QC-SnapKV reference finished all128 train8 answers at10/25/50/
100% document budgets with natural EOS. Its sampling/backend differs from the
vLLM application experiments; only the four HF arms are paired compression
comparisons. Main-model paged sparse execution qualification is proceeding
independently. Exact output and LSE tests remain separate and LSE failures
are retained; a good attention output does not certify the LSE contract.

All8 public DocFinQA development reports now have full-source RAG plans,
including reports too long for dense generation. The main-model RAG arm is
pre-specified as1024-token chunk target/8192-token context/hybrid reranker before
its answers; the boundary Instruct model runs all eight grid variants. This
is exploratory development, not a claim that one variant is already optimal
for financial QA. The complete full-report main-model dense/joint run continues
on a separate TP2 pair. Both use the frozen strict-FINAL metric with all failure
and unserviceable questions kept in the denominator.


## 2026-10-16 15:49 — main-model complete-report screening and graph attribution

The official FP8 main model completed public DocFinQA development8 dense/joint
screening at262K context: five documents/17 questions were serviceable and three
documents/seven questions remained unserviceable. All serviced generations ended
naturally; no cap hits. Both arms score2/24 under the frozen exact FINAL metric;
mandatory gold-precision rounding gives7/24 dense and9/24 joint. Full denominator
and grammar/unit errors are retained. This is neither official program-execution
accuracy nor adequate evidence of useful application quality. The two arms'
serviced generation totals are1067.13/1095.43s in eager mode; startup, tokenization
and1.22863 allocated GPUh are separately recorded. Main-model RAG is ongoing.

Native CUDA-graph TP2 profiling on one86.8K-token B3 document captures32 actual
decode steps per rank. Attention including split merge accounts for53.14/53.19%
of summed kernel time; NCCL communication5.03/5.08%. Kernel timeline spans
474.33/473.31ms. These are instrumented attribution windows, not unprofiled
latency or an application speedup claim. Both raw traces are hashed, each rank
is summarized separately, and capped256 tokens are explicitly instrumentation.
The main graph engine runs successfully; complete natural graph A/A now uses
three cold repeats plus one warm repeat on the same frozen development cohort.

Full-budget paged sparse execution passed all local output checks and short
greedy sequences, but logits drift across layers persists in both native SIMT
and Tensor Core variants. The drift is under investigation and is not accepted
by retroactively loosening tolerances. Eager reference wrappers are not a strong
production speed baseline. A2 FastKVzip-NVIDIA exact-gate/mask/full-budget
qualification also passed; its shared query-agnostic development generation
runs separately. Both A2 and A4 still require factual review and formal testing.


## 2026-10-16 16:14 — factual coverage, source annotations and measurement repairs

The completed budget-blind A4 audit has128/128 source-grounded judgments. At
10/25/50/full retention, factual-pass counts are12/12/13/13 of32; adequate coverage
is7/10/12/18, and both hold for4/8/8/9. This is one LLM-assisted judge, not human
validation or a formal noninferiority result. The identity map was joined only
after all judgments were committed; no verdict was revised during unblinding.
FastKVzip-NVIDIA global-mask reference also completed128 natural-EOS answers.
Its full32 answers match A4 token-for-token; only96 novel answers need review.
Reference masking does not measure physical cache compression or the author's
optimized chunked-prefill implementation.

The Instruct4B financial RAG grid completed192 natural answers, serving all24
questions under every setting. Best exact FINAL is6/24; best frozen rounding
is15/24. Full-report graph dense serves10 questions and joint6, with all failures
retained in24-question denominators. The best rounded development RAG setting
is being repeated in native graph mode, explicitly as a development selection.
An independent source-only audit matched all24 questions to25 official FinQA
records, retained source ambiguity, and identified8 semantic/unit/text concerns.
No model output informed that audit and no primary gold or metric was changed.

Cost fields now add the previously omitted shared generator-input tokenization
to measured cold phases; old composed fields remain defined as before. Startup,
warmup and harness overhead remain separate, so these are not integrated E2E
measurements. The revised plot uses the corrected phase sum. Auxiliary FINAL-only
trace names collided across model directories:70 historical references cannot
be recovered. Primary JSON answers, token IDs, per-request metrics and JCT are
intact. Verified traces are archived by hash and future paths include the parent
report directory; missing traces cannot support step analysis. Even valid FINAL
completion traces never established attention-ready membership.


The pre-specified main-model financial RAG arm finished all24 requests:23 natural
EOS and one8192-token cap, counted as a failure despite retained raw text. Exact
FINAL is6/24 and frozen rounding14/24. Measured phases total2620.219s, generation
2543.329s, allocated1.44117 GPUh. This eager result does not establish speedup;
native graph repeats are running with the same retrieval variant.


## 2026-10-16 16:29 — graph baselines, complete blind comparison and budget coverage

Main-model full-report graph A/A completed all64 configured runs. Five documents
and17 questions are serviceable, with all other questions retained. All17 dense
and5 joint cold request sequences match across three repeats; no warm sequences
match. Steady-cold totals are320.242/204.910s, with startup/first-call/preparation
separate. Cold0 exact FINAL is3/24 dense,2/24 joint; rounded both9/24. Instruct4B
selected-RAG graph repeats also match all24 cold requests, with6/24 exact and
15/24 rounded. None of these cross-output timings establish same-workload speedup.

The96 novel FastKVzip-mask judgments are complete. Their32 full answers reuse
prior judgments only after exact source/question/gold/answer identity verification.
10/25/50/full factual-pass counts are12/14/15/13, adequate coverage8/15/15/18,
and both6/8/8/9. The comparative figure retains differing global/per-head budget
policies and one-judge limitations. A power-button wording ambiguity is documented
for versioned cross-candidate review; already-unblinded v1 judgments stay intact.

The main natural operator budget extension covers8K/16K/32K, all85 configurations
at each point and60 paired observations per cache. Dynamic v2/v3 still lose; at
32K, exact hierarchy is about19.41us versus21.63us full union in hot cache. It must
remain a strong control. All larger-budget jobs together cost0.014603 P2 GPUh.
MSA exact-tail gather adaptation passes numerical qualification but is slower
than native E0 by4.46x/8.72x hot on the main/QMSum points, primarily from gather.
Two additional main-model documents, including true B4, are being captured.

Real FULL-graph sparse qualification now includes2016 actual-query/cache shadow
checks, explicit candidate graph captures, exact E0 graph A/A logits and no dense
pure-decode calls. CPU metadata preparation is80–181ms/step and added graph memory
is explicit. This is successful integration qualification with a severe cost
bottleneck, not production speedup. Native plan reuse and GPU metadata updates
will be tested as engineering controls; ordinary caching is not claimed novel.


## 2026-10-16 16:47 — broader negative evidence and absolute-budget audit

Main-model RAG graph repeats match all24 cold token sequences, including one
capped answer retained as failure. Exact/rounded FINAL is4/24 and12/24; steady
cold JCT230.503s. Eager output differs, preventing a same-workload speedup claim.
Two further full financial reports provide true B3/B4 captures,456 bitwise cache
checks and48 same-forward ready groups. All680 operator configurations qualify;
dynamic v2 loses all8 hot/cold points at4K/16K (best-control/dynamic0.324–0.832).

Shared native plans across48 layers plus one GPU metadata mapping pass1344
actual graph shadow checks with exact E0 A/A logits. Main preparation host cost
is0.325–0.609ms/step; additional validation still costs about45ms/step. This is
baseline engineering, not a new method or uninstrumented production speedup.

Absolute4K/8K/16K/32K/full factual-pass counts are14/14/14/13/13, adequate
coverage14/16/17/18/18, and both9/9/10/9/9, each over32 questions.56 novel
reviews are separate from104 nominal reused references to32 full outputs.
Four identical outputs independently reviewed in different batches by the same
judge agree on all three answer-level fields; this is not inter-rater evidence.
A verdict-independent32-answer second assisted judge sample is now frozen.

Completed allocation totals are8.39856 P1 and4.09085 P2 GPUh, with all starts
matched and no anomalies; historical uncertainty reserves remain separate.


## 2026-10-16 17:00 — natural B8 extension rejected

The first two eligible training reports were frozen before capture, with eight
natural questions sorted by SHA, complete120K/145K prompts and no evaluation
selection. Both achieve actual B8 same-forward decode;960 cache checks and48
rank/layer/step ready groups verify provenance. The old exporter failed on one
report because warm requests also share a question prefix. Recovery now requires
an actually captured Q with identical token prefix and the same physical cached
page, including origin request IDs; failed attempts remain in the cost ledger.

The bounded M16/M32/M64 extension passes32 FP64/changed-input graph checks,
memcheck with zero errors and representative shared-memory racecheck with zero
hazards.136 configurations on two documents at4K/16K all qualify. Dynamic hot
calls293–490us lose to native controls11–24us; cold also loses. Hundreds of owner
classes cause fragmentation; even optimistic tile opportunity is often worse
than the exact combinatorial fixed-partition model. The predeclared stop rule
ends this layout search. GPU partition/hierarchy enumeration at B8 remains
incomplete; the existing qualified faster controls suffice for rejection, never
for a positive competitive claim.


## 2026-10-16 17:06 — second assisted judge agreement

The32 preselected unique answers were judged against all4139 source lines by a
second conversation without first verdicts or identity mapping. Agreement is
26/32 factual pass,29/32 coverage,24/32 uncertainty,30/32 factual-and-adequate;
descriptive unweighted kappa is.571/.810/.500/.796. The first origin in the
frozen mapping defines the primary comparison; other prior versions remain
explicit. Neither verdict set is changed. Same model-family judges and only
8 documents/7 higher source groups limit reliability and do not establish human
or formal noninferiority evidence.


## 2026-10-16 17:10 — observer boundary correction, historical results retained

The old TP-rank exporter used max(0,prompt_length-32), incorrectly including
some document Q when the question suffix was shorter than32. Production
FrozenQCSelector and the Transformers reference already used suffix-only Q.
New `_suffix_v2` exports retain the same actual suffix Q bitwise and remove
backfill in120/208 rank/layer/question records across five captures. The88
already-correct score/mask records are bitwise unchanged. Historical operator
and capacity observations remain valid for their saved masks but represent a
different observer variant, not the specified QC-SnapKV semantics. Corrected
pre-specified operator cells are queued; no old artifact is overwritten.

All eight GPUs now run four TP2 natural-generation configurations in parallel:
full-budget plus the pre-existing8K/16K/32K grid, with qualified capacity and
B1/B2/B3 transitions. Full-budget executor controls distinguish potential
floating-point effects from token removal; no post-hoc quality equivalence claim.


## 2026-10-16 17:24 — corrected replays, natural budgets, and stronger XQA control

All1156 configurations at12 corrected suffix-only cells qualify. Dynamic still
loses all24 hot/cold points. The B8 extension remains reject-only, lacking full
partition/hierarchy enumeration. Additional32769-token N32/split64 long-region
fixtures pass changed-input FP64 checks, zero-error memcheck and zero-hazard
racecheck for compact and union. Historical observer-variant artifacts remain.

The8K/16K/32K natural grid completed without forcing or shadows. E0 outputs
6721/4289/3424 tokens with22.49/15.08/10.40s generation; union outputs
6530/4789/3351 with27.87/18.95/13.56s. Dense is identically3000 tokens across all
five budget runs; the three new dense generation times are14.4–14.7s. Rounded FINAL counts
are E0 1/2/2, union3/2/2, dense2/2/2, each of three questions. Full budget yields
the same three numeric endpoints across methods but different token trajectories.
This is one development document/seed, not quality noninferiority or same-work
speedup. Initialization, selector preparation, capture and graph memory remain
separate costs. The historical4K first-dense sampler-JIT caveat is retained.

Official native FLASHINFER actually selects BF16 XQA, page16/LBNHC, with native
B1/B2/B3 FULL graphs. Two cold repetitions match all3066 output tokens and actual
step sequences, finish all questions, and score1/3 exact,2/3 rounded. Generation takes
12.954/12.879s (23.684/23.649s including the cold anchor), faster than the prior default FLASH_ATTN baseline. Any32K benefit
must be assessed against this strong control, with full costs and quality.

A separately frozen CPU-only joint support exchange screen keeps each request's
budget/protected positions and caps pooled-score loss. Epsilon0 changes no masks;
1% yields17.65–22.99% or18.12–22.87% logical common-source union reduction in two
fixed orders. All132 rows validate; physical source capacity released is zero.
Actual held-out decode attention fidelity and independent prior-art review are
next, not implied by this score proxy. No candidate is yet novelty-qualified.

Completed allocation totals are9.41855 P1 and4.46901 P2 GPUh; every start has a
terminal record and no ledger anomaly. Early unknown-cost reserves stay separate.


## 2026-10-16 17:44 — native-XQA expansion and joint-selection limits

The official FI bridge passes864 SIMT and864 same-Q native-XQA comparisons,
with exact E0 graph A/A logits. Four TP2 jobs use all8 GPUs for original32K and
second-document16K/32K/full, completing36 natural EOS answers at0.525397 P1 GPUh.
The second-document dense token sequences match across all three budgets.
Rounded FINAL is native2/3, E0 0/1/1 and union1/1/2 across16K/32K/full. The
original32K E0 is3/3 versus native/union2/3. Full E0 also changes the answer:
backend floating-point trajectories prevent attributing all changes to pruning.
Two documents/one seed do not demonstrate quality-qualified cross-document gains.
New raw, figures and CSV separately retain generation, cold anchor, preparation,
graph capture and memory; the strict primary endpoint is also preserved.

Joint support actual-query screening retains1080 rows/8640 head metrics.240
zero-epsilon groups match O/LSE/indices bitwise. At1% pooled-score loss the worst
actual retained mass declines6.50pp at4K and4.92pp at16K; worst single-head
relative L2 increases.13347/.04476. This refutes a proxy-loss-to-quality guarantee.
Classical reverse-delete/refill and independent relaxed top-prefix controls are
frozen and recorded separately; the latter reduces union only1.95–4.13% at1%,
leaving logical coordination opportunity. There is no demonstrated new algorithm,
physical cache release or natural quality gain. Saturated greedy and minimal
system-mechanism checks continue before any larger experiment.

Completed allocation totals are10.05249 P1 and4.47272 P2 GPUh. All starts have
terminal records, no ledger anomaly; historical uncertainty reserves stay separate.
