# BranchTile feasibility study

Experiments for `proposal-iscas.md` on eight RTX 5090 GPUs. The current dynamic
query-row implementations are numerically validated but have **not** met the
proposal's performance threshold for full application integration. See
[Chinese feasibility report](FEASIBILITY.md), [experiment log](EXPERIMENTS.md), [kernel audit](results/kernel_audit.json), and
the raw JSON results. Negative results and failed configurations are retained.

The continued paper investigation is tracked in [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md).
It is **not complete**: official FP8 main-model serving, native XQA, shared-plan
E0/union and optional joint-support installation have engineering qualifications,
but no candidate has established quality-qualified net application gains or a new
paper contribution. The original dynamic-row stop decision remains in force.

Current operator continuation: [Chinese owner-task eight-GPU report](docs/operator_owner_8gpu_report_20261016.md).
Ten complete performance waves contain 74 clean jobs and 4,532 FP64-qualified
configurations across 25 distinct real development settings and three documents.
Owner-task packed partials pass 480 dedicated numerical phases and improve the
old dynamic operator. An enlarged grid finds one approximately 4.8% hot win;
fixed-parameter confirmation wins one of eight correlated inputs by about 5.4%.
Other inputs and preparation amortization still fail the research gate. Eight
legacy-compilation jobs were stopped early and remain charged as incomplete.
The [frozen summary](results/operator_8gpu_exploration_summary_v2/summary.json),
independent audits, all failures and stage diagnostics are retained. The
[fixed-parameter confirmation figure](results/operator_owner_confirmation_figure_v1/fixed_owner_paired_speedup.pdf)
shows all eight correlated cells and the coarse cold-event intervals. The
[earlier register-bank report](docs/operator_8gpu_exploration_report_20261016.md)
and same-wave [figure](results/operator_lane_wave16_audit_v1/paired_time_ratios.pdf)
remain historical evidence; no formal paper or application gain is established.

Scope is single-user multi-agent reuse. The [comparison protocol](docs/operator_comparison_protocol_20261016.md)
separates identical-work replay from natural-EOS quality and application latency.
Different natural output lengths are retained. No formal paper-success claim.

Recent evidence:

- [Physical KV alias control](results/native_physical_alias_v1/README.md):
  eight paired cells, 48 bitwise numerical checks and 384 address-sensitive
  checks pass. Existing sparse E0 benefits from shared addresses by 1.22–1.77×
  in cold calls; this is a baseline property, not a new executor gain. Hardware
  traffic and full-model speed are not inferred from these operator timings.
  [Figure](results/native_physical_alias_figures_v1/native_physical_alias_hot_cold.pdf)
  · [Independent audit](results/native_physical_alias_independent_audit_v1/README.md).
- [Corrected suffix-only replay](results/suffix_v2_replay/): 1,156 qualified
  configurations; dynamic execution loses at all 24 tested hot/cold points.
- [Native-FI natural development matrix](results/main30b_sparse/fi_natural_expansion_summary.json):
  two documents, all 36 answers reach EOS; quality varies across budgets and
  backends, including full-history controls. This is not formal evaluation.
- [Joint selection and classical controls](results/joint_selection_controls_summary.json),
  [held-out attention fidelity](results/joint_attention_fp64_v1/README.md), and
  [native E0 mechanism screen](results/joint_e0_gate_v1/README.md): smaller union
  is not a task-quality, physical-capacity or deployment-speed guarantee.
- [Amortization decision](docs/joint_e0_amortization_review_20261016.md) and
  [current accounting](results/budget_audit.json): preparation, failed runs and
  all allocated GPU time remain charged. The [real 48-layer check](results/joint_real48_replay_v1/README.md)
  retains a small positive signal but fails the performance/amortization gate;
  the current joint-E0 application branch is stopped.

- [Official FP8 KV qualification](results/native_fp8_kv_unit_confirmation_v1/README.md) and
  [natural precision controls](results/native_precision_natural_development_v1/README.md):
  all four TP2 jobs exit normally, with exact cold A/A trajectories. FP8 has
  lower descriptive time and different output lengths; one rounded-score loss
  comes from a missing percent sign. No formal quality noninferiority is shown.
  The fixed KV pool stores about twice as many tokens without halving process memory.
- [Post-stop research assessment](docs/post_stop_research_decision_20261016.md):
  no qualified replacement contribution has emerged; ordinary quantization,
  index ordering or weighted attention variants are not relabeled as new ideas.

The [input-capacity audit](docs/native_context_capacity_review_20261016.md),
[calibrated scalar-loader audit](results/native_fp8_calibration_interface_audit_v1/README.md),
and [register-bank residual review](docs/register_bank_residual_review_20261016.md)
record remaining boundaries without launching another GPU sweep or claiming a
qualified replacement idea.
The [query-residual certificate review](docs/query_residual_certificate_review_20261016.md)
and [shared-prefill review](docs/shared_prefill_application_gap_audit_20261016.md)
also close ordinary Taylor/fallback and leader/follower proposals using current
primary literature and reproducible CPU checks. A mature TinyGEMM router control
has completed its [bounded qualification](results/tinygemm_gate_qualification_v1/README.md):
small rounding differences prevent conservative bitwise substitution; candidate
performance was not measured. The [large-workset CPU audit](results/natural_capacity_boundary_v1/README.md)
finds a remaining BF16 capacity boundary but no qualified new acceleration mechanism.

## Environment

All runs use uid 1000 and a virtual environment; no root or driver permission
changes are required. This machine's working setup is:

```bash
/home/qcjz/miniconda3/envs/sglang-vllm-m120/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install accelerate==1.15.0 socksio==1.0.0 pyarrow==26.0.0 matplotlib==3.11.2
export PATH="$PWD/.venv/bin:/home/qcjz/miniconda3/envs/sglang-vllm-m120/bin:/usr/local/cuda-13.2/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.2
export PYTHONPATH="$PWD/src"
```

The parent environment provides Torch 2.11.0+cu130, Triton 3.6.0, TileLang
0.1.9, FlashInfer 0.6.12 and Transformers 5.12.1, and is used read-only.
The paths are machine-specific; on another host create an equivalent venv.
`results/environment_packages.txt` records the final package inventory.
Model serving has a separate `.venv-engine` with official vLLM 0.31.0 and
Torch 2.13.0+cu130; its inventory is recorded separately. Native-engine results
must not be mixed with operator results from the other environment.
The final native controls use `.venv-latest`: the same Torch/TileLang base with
FlashInfer core/cubin 0.7.2. See `results/platform/latest_flashinfer_environment.json`
for immutable version details and `configs/native_union_paired.json` for commands.
The final package inventories describe the final environment, not a historical
lockfile for each earlier run.

## Reproduce a bounded experiment

Check card availability first. Each manifest assigns disjoint GPUs and limits
job duration. Do not run independent manifests on overlapping GPUs.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q tests/test_attention.py tests/test_union.py
.venv/bin/python scripts/run_jobs.py configs/union_paired.json
.venv/bin/python scripts/summarize_feasibility.py
```

`run_jobs.py` appends allocated wall time, including initialization and failures,
to `results/job_ledger.jsonl`. It enforces per-manifest disjoint GPU assignments;
the operator must also check global allocations and the proposal's aggregate
24 GPU-hour P1 / 8 GPU-hour P2 limits. The initial eight-card screen, v2/v3
checks, and official-4B replay each have separate configs under `configs/`.

For one complete same-mask comparison with a common natural-log LSE contract:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compare_attention.py \
  --tokens 16384 --pattern cross --gqa 8 --include-union \
  --normalize-native-lse --output results/local_comparison.json
```

For additional native operator controls, use `.venv-latest/bin/python` and add
`--include-native-raw --include-native-union` to this command. The native joint
paged-prefill control expresses the exact union mask without copying KV payload.

Every configuration is checked against FP64 before timing. Hot and cold cache
have independent tuning, paired randomized/reversed orders, fixed A/A, and
paired-block bootstrap intervals. CUDA events time the producer and split merge;
CPU planning/JIT and cache flushes are outside the measured kernel interval.
See each raw file for the scope and source provenance. Early native results
return native log2 LSE; newer runs with `--normalize-native-lse` also time the
conversion to natural log. No application speedup follows from these timings.

## Data and weights

Dataset/model revision and SHA256 records live in `results/data_*` and
`data/manifests/`. The user explicitly excluded the unavailable historical split.
New public-source development8/evaluation100/training167 manifests are frozen;
the evaluation set is not used to select the method or its configuration.
Full documents are retained, including documents that exceed model context.
Weights, raw corpora, generated tensor captures, virtual environments and JIT
caches remain local and are excluded from git.

`scripts/prepare_data.py` prepares the source data; `scripts/capture_model.py`
captures real post-RoPE tensors and question-conditioned supports. The early HF
captures use dense generation with a 256-token instrumentation cap and serial
question execution. Later official TP2 engine captures establish actual B3/B4/B8
same-forward readiness and physical cache identity; their bounded capture output
still does not establish task quality. Separate natural-output experiments retain
EOS/cap status, complete generated text, preparation and per-request quality.
Corrected `_suffix_v2` exports exclude document Q from short suffix observation
windows; historical exports remain a separate observer variant.
The older BF16 30B diagnostic is explicitly distinguished from the intended
Thinking-2507 FP8 main model.

Nsight Compute hardware counters are unavailable to the normal user on this
host (`ERR_NVGPUCTRPERM`). Resource allocations are compiler evidence, logical
tile counts are analytical estimates, and neither is reported as measured
DRAM traffic or board energy savings.

## Numerical qualification limits

The current dynamic/fixed/union implementations pass their documented FP64 and
sanitizer checks. The additional native-joint tests retain **4 passes and 3 LSE
precision failures**; direct FlashInfer single-prefill reproduces these failures.
The suite is intentionally not advertised as entirely green. Actual timed
configurations must pass the original output and LSE checks; failed configs
remain in the raw JSON. See `results/platform/native_union_qualification_status.json`.
The final native joint control removes the large hot-cache union advantage seen
against separate native decode. This is a bounded negative feasibility result,
not a completed main-model application evaluation.
