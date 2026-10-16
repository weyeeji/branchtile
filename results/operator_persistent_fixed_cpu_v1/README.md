# Fixed M32 persistent-state control

Independent strong ablation for B3/B4, GQA8, D128. Every owner region executes
the same 32 canonical query rows, with both GEMMs using FullCol. Q is loaded
once into shared memory. FP32 out (32×128), maximum (32) and denominator (32)
fragments are initialized once before the region loop and remain live across
regions. Runtime owner Labels mask score rows; no RowMap, canonical state
gather/scatter, owner-specific macro expansion, or shared CM/CL/CU is used.
Only Q/K/V/probability tiles use shared memory in the written kernel.

This is a fixed-row control, not a novelty claim or a replacement for v4.
The existing `fixed_producer` already uses the general persistent-fixed-row
principle; this control provides FullCol, exact Labels interface and empty-shard
guard under the same wrapper as v4. It deliberately spends M32 MMA work even
when only one request owns a region. Any comparison must account for that work
versus the state movement eliminated; it is not a change of state location with
all arithmetic work held equal.

Interface:

```python
from branchtile.attention_persistent_fixed import Attention
op = Attention(plan, splits=64, block_n=64, threads=128)
output, lse = op(q, k, v)
```

The wrapper AST equals `attention_v4` apart from its producer import. Output
buffers and inherited split merge are unchanged. Only `compact=True` and
`region_partials=False` are supported, as for v4. Plan metadata preparation
still creates the standard plan, but this kernel consumes only its indices,
offsets and the uploaded label vector.

For an unowned row, all current scores are -inf. The online maximum retains
the previous maximum (`clear=False`); a previously nonempty row has rescale
factor exactly one and receives a zero PV update. An empty row remains zero
with maximum -inf/denominator zero. Empty shards skip the whole update.
The inherited merge returns zero output/-inf LSE for an entirely empty row.
B3's extra eight M32 rows have zero Q and masked scores and are not exported.

CPU-only audit:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_persistent_fixed_cpu.py
```

AST checks cover fixed M32 allocations and two FullCol policies; one-time Q
load and state initialization; no RowMap or shared canonical state; exact
owner mask and shard formulas; softmax rescale/normalization expressions
matching the previous FullCol control; unchanged v4 wrapper and merge; no
racecheck/compiler-pass override. Python syntax and diff checks pass.

All 64 float64 CPU cases pass (B3/B4 × all masks/unequal/empty/one-token ×
splits 1/3/8/64 × N32/N64), including 35072 exact unowned-row state identities
and 5352 skipped empty shards. The simulator compares split-merged results
against independent dense masked attention at atol/rtol 2e-13 and checks empty
row and B3 padded-row behavior. It does not model BF16 probability rounding,
compiled register layouts, hardware races, GPU correctness, or speed.

No Torch or TileLang import, GPU compilation, GPU qualification, performance
measurement, or GPU allocation occurred in this implementation task. Default
TileLang race checking remains enabled. Root-owned budgeted GPU qualification
must retain failures and check FP64 output/LSE, empty shapes, graph replay,
mutable Q/KV, generated state traffic and actual compiler resources before
timing or making any state-residency claim about generated code.
