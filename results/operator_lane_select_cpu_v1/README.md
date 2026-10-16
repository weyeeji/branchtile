# Historical single-buffer lane-select CPU proof

This report refers to source commit `b7c7666`, not the current split-bank source.
The CPU audit reads those two candidate files through `git show b7c7666:path`;
it never resets the working tree. Source hashes identify the exact historical
bytes. Reproduce with:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_lane_select_cpu.py
```

The historical source's integer routes and source-expression mathematics pass,
but the subsequent root-owned wave11 GPU jobs failed compilation:
TileLang BufferUseDefCollector rejects one `T.Parallel` accessing different
indices of the same CU/out buffer. This CPU proof does not certify compiler
legality and does not override that failed qualification. No failure is retried
or deleted by this audit. The changed source is separately committed as
`6c3a98f`, with a separate CPU report under `operator_lane_select_cpu_v2`.
