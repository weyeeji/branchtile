# Bounded same-native E0 development gate

This protocol/script are committed before GPU execution. Six single-GPU jobs
cover all three corrected suffix-v2 captures at rank 0/layer 15/anchor 64,
both KV heads together, and budgets 4096/16384. Actual request prediction
steps and complete prefix/private histories are frozen in `protocol.json`.

Methods are the original independent support, the already locked epsilon
0.01/two-round/ascending exchange, and the already locked classical full-V
saturated greedy cover. CPU reconstruction must exactly match earlier support
hashes. No held-out Q or performance result changes these masks. Selector and
native plan preparation are measured separately. The cover helper's temporary
CPU-only CUDA visibility change is restored before any CUDA initialization.

The same FlashInfer 0.7.2 `NativeSparseDecode` is used with tensor cores,
automatic backend, default split and page size one, without tuning. Original
shared K/V payload is aliased by all methods. Per-native-batch token counts and
resolved plan scheduling must match across masks. Each method is qualified
against its own FP64 attention: output atol 0.006/rtol 0.015; natural-log LSE
atol/rtol 0.00005. Native log2 LSE conversion is validation-only. Different
masks are not required to produce equal outputs. Any failed qualification
prevents a timing claim for the cell and is retained.

**Primary batch order is production BH**: request-major/head-minor native
batch B*H, queries shaped `[1,B*H*G,D]`, K/V a pure `[1,H*N,D]` view, indices
`head*N+token`. The exported HB query conversion is a separately reported
preparation operation; engine inputs are already BH. The prior HB native
batch order is retained only as a sensitivity arm. Restoring output layout
does not make HB scheduling equivalent to BH.

Two contracts are kept separate: O-only (`return_lse=False`) and O+original
raw log2 LSE. Each has two separately reported execution scopes: native GPU
execution and complete output-layout restoration to `[B,H*G,D]` (and LSE
`[B,H*G]`). BH restoration is a view; HB includes its required preallocated
copy kernels. CUDA graphs time GPU work, excluding host dispatch, selection
and planning. This is not host scalar-call or application latency.

Every method and independent A/A graph participates in 20 randomized paired
blocks and the reverse of each block. Hot graphs repeat 20 calls; cold graphs
run once after a read/write flush larger than twice device L2 and at least
256 MiB, outside the timing interval. Paired-block bootstrap keeps random and
reverse measurements clustered. Seeds and all raw measurements are retained.
Independent A/A is a separate graph of the same function and buffers.

Jobs use GPUs 2–7 and a 75-second timeout each. Including run_jobs' 15-second
termination grace, worst-case P2 allocation is exactly 0.15 GPUh. No automatic
timeout extension or unrecorded retry is allowed. Partial state, errors and
all six cells remain visible. The shared run_jobs ledger is authoritative.

This is a development mechanism gate on one layer/rank and a contiguous
export, not evidence of complete 48-layer noncontiguous engine-cache reuse,
natural quality, physical capacity release, throughput, or novelty. Primary
BH and HB sensitivity, cache regimes, and output contracts cannot be pooled
or selected after observing performance. If this cheap screen shows no
stable same-kernel benefit, recommend stopping more expensive mechanism
work; a favorable point still needs engine-layout and quality validation.

```bash
.venv/bin/python scripts/run_jobs.py results/joint_e0_gate_v1/jobs.json
```
