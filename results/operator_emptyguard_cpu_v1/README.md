# Independent empty-shard guard ablation

This candidate changes only v2's empty-region-shard handling. It is independent
of the fixed-register v4 candidate and must be reported as its own condition.
No GPU compilation, numerical qualification or timing was run by this audit.

Interface:

```python
from branchtile.attention_emptyguard import Attention
op = Attention(plan, splits=32, block_n=64, threads=128,
               compact=True, region_partials=False)
output, lse = op(q, k, v)
```

The constructor, output buffers, graph-compatible call and merge are identical
to `attention_v2`. `compact=False` retains v2's original unmodified
`fixed_producer` control; the ablation is the `compact=True` condition.
`scripts/bench_kernel_emptyguard.py` adapts the existing benchmark protocol,
but any execution must use the root-owned GPU job ledger and frozen run plan.

Inside the uniform region loop, the guard computes the exact same begin/end
split formulas as `do_region`, then calls the unchanged macro only when
`shard_begin < shard_end`. This skips the query/state gather and state scatter
for empty shards. It also covers empty shards of a nonempty region. All lanes
share `h/s/r`, so the conditional encloses the complete macro uniformly.

For an empty shard the original tile-loop has zero iterations: its gather and
scatter are an identity on canonical CM/CL/CU. Skipping it leaves existing
partial state intact. An entirely empty CTA still writes the initialized
CM=-inf, CL=0, CU=0, after which the unchanged inherited merge's empty-row
contract is zero output and -inf LSE. GPU behavior remains to be qualified.

CPU audit command:

```bash
.venv/bin/python scripts/check_emptyguard_source.py --output results/operator_emptyguard_cpu_v1/audit.json
```

The audit proves by AST comparison that removing the uniform guard restores
the complete v2 source, and that the wrapper differs only in its kernel import.
It checks 105264 uneven split shards (2448 empty) and 42 empty gather/scatter
identity cases. Python syntax checks and `git diff --check` passed.

`tests/test_attention_emptyguard.py` contains six not-yet-run GPU qualifications:
all-empty inputs, an empty head, many uneven/empty shards, each with combined
or region-partial execution. They require bitwise agreement with v2 plus
independent FP64 tolerance checks and explicit no-NaN/empty-state assertions.
Passing CPU source/algebra checks does not establish those GPU results.
