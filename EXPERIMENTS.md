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
