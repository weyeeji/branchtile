# BranchTile feasibility study

Experiments for `proposal-iscas.md` on eight RTX 5090 GPUs. The current dynamic
query-row implementations are numerically validated but have **not** met the
proposal's performance threshold for full application integration. See
[experiment log](EXPERIMENTS.md), [kernel audit](results/kernel_audit.json), and
the raw JSON results. Negative results and failed configurations are retained.

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

Every configuration is checked against FP64 before timing. Hot and cold cache
have independent tuning, paired randomized/reversed orders, fixed A/A, and
paired-block bootstrap intervals. CUDA events time the producer and split merge;
CPU planning/JIT and cache flushes are outside the measured kernel interval.
See each raw file for the scope and source provenance. Early native results
return native log2 LSE; newer runs with `--normalize-native-lse` also time the
conversion to natural log. No application speedup follows from these timings.

## Data and weights

Dataset/model revision and SHA256 records live in `results/data_*` and
`data/manifests/`. The original proposed historical split was unavailable;
the replacement eight-document development sets are explicitly exploratory.
Full documents are retained, including documents that exceed model context.
Weights, raw corpora, generated tensor captures, virtual environments and JIT
caches remain local and are excluded from git.

`scripts/prepare_data.py` prepares the source data; `scripts/capture_model.py`
captures real post-RoPE tensors and question-conditioned supports. Capture
configs preserve model identity and device assignments. Captures use dense
generation with a 256-token instrumentation cap and serial question execution;
they do not establish sparse-generation quality or simultaneous ready lifetime.
The older BF16 30B diagnostic is explicitly distinguished from the intended
Thinking-2507 FP8 main model.

Nsight Compute hardware counters are unavailable to the normal user on this
host (`ERR_NVGPUCTRPERM`). Resource allocations are compiler evidence, logical
tile counts are analytical estimates, and neither is reported as measured
DRAM traffic or board energy savings.
