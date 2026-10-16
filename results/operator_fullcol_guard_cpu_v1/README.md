# FullCol warp-policy control

This is an independent control for the fixed-register v4 mechanism. It starts
with the v2 empty-shard-guard implementation and changes exactly its two
`T.gemm` policies from `FullRow` to `FullCol`. It retains the original shared
CM/CL/CU canonical-state gather/scatter, state layout, RowMap, region order,
uniform empty-shard guard, split geometry, softmax math, and inherited merge.
It does not implement v4's canonical-U register residency.

Interface matches v2:

```python
from branchtile.attention_fullcol_guard import Attention
op = Attention(plan, splits=64, block_n=64, threads=128,
               compact=True, region_partials=False)
output, lse = op(q, k, v)
```

`compact=False` remains the original unmodified fixed-producer control, so use
`compact=True` for this ablation. Compare v2 → emptyguard to isolate skipping;
compare emptyguard → fullcol_guard to isolate warp policy; compare this control
with the qualified v4 under the same numerical and timing conditions to assess
the additional state-residency mechanism. The generated CUDA layout can change
with warp policy even though canonical state still goes through shared memory.

CPU-only source audit:

```bash
.venv/bin/python scripts/check_fullcol_guard_source.py
```

It changes the two AST attributes back to `FullRow` and proves the entire kernel
AST equals the emptyguard baseline; the wrapper differs only by its producer
import. It also rechecks emptyguard's single-guard delta from v2. Python syntax
and diff checks pass. No GPU compilation, numerical qualification or timing was
run for this control. Root-owned budgeted GPU qualification is required.
