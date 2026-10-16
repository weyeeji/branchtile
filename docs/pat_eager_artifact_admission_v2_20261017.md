# PAT artifact admission after CPU-only build guard failure

The original `results/pat_prior_build_v2.json` remains **failed**. Compilation,
linking and module loading produced `_prefix_attn.so`, but the subsequent
`assert not report['CUDA_initialized']` failed. This is not a successful CPU-only
build. The parent is tracking the possible GPU-context/overlap effect separately.

The independent `scripts/benchmark_pat_raw_pair_eager_v2.py` preserves the v1
benchmark and admits only the exact compiled artifact SHA256
`046e5b5488be33dfd872dae8da8eb1d1b5d37a7a233d7f8a5665204a5fae2444`
when the original receipt has `CUDA_initialized=true` and the exact final
assertion signature. Original/copied source hashes and the on-disk module hash
are still verified. Other failed builds are rejected. The output retains the
original status, error and an explicit admission limitation. Artifact admission
is permission to run a new numerical GPU qualification, not numerical success.

Use the v1 protocol's command with script name replaced by
`benchmark_pat_raw_pair_eager_v2.py`. All numerical, layout and timing contracts
remain unchanged. CPU preparation checks used `CUDA_VISIBLE_DEVICES=''` and
only standard-library inspection/py_compile; no CUDA module was imported.
