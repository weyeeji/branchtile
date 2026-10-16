# Lane-select independent-bank source and CPU route proof

Candidate source commit `6c3a98f` uses runtime register selects instead of v4's
7/15 owner-specialized GEMM copies. One generic region macro has exactly two
call sites, M16 and M32. This is an experimental candidate, with no performance
or novelty conclusion in this CPU report.

Interface:

```python
from branchtile.attention_lane_select import Attention
op = Attention(plan, splits=64, block_n=64, threads=128)
output, lse = op(q, k, v)
```

B3/B4, GQA8, D128 only; external allocation/merge contract matches v4. The
producer additionally consumes standard plan RowMap/Count/Owner tensors plus
the uploaded Labels vector. Plan metadata must remain mutually consistent and
keep active requests in ascending request order, as standard `make_plan` does.

Four independent 8×128 FP32 CU fragments carry the canonical request banks.
Each active output slot gathers with an explicit four-way select, and every
CU buffer reference is `[i,d]`; runtime metadata never indexes a register array.
After the large MMA output update, separate `T.Parallel(8,dim)` loops extract
the two/four slots into independent OUT0/1/2/3 fragments. Scatter selects these
independent buffers into a fixed CU request bank. M16 only reads OUT0/1.
B3 initializes its fourth CU bank but statically excludes Owner[24] and does
not export the padded bank. CM/CL retain shared RowMap gather/scatter, while
QK/softmax/PV and final split merge keep their original numerical contract.

The separate-bank representation addresses the known single-buffer
BufferUseDefCollector restriction. Default race checking stays enabled;
an explicit whole-CTA barrier remains at each region boundary. Source legality
and intended thread layouts still require actual compiler/GPU qualification.

Reproduce the CPU-only proof:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/check_lane_select_split_banks_cpu.py
```

The proof extracts and executes select, slot-copy and scalar gather/scatter
expressions from the actual source, comparing them with an independently
constructed ascending-request RowMap and popcount slot definition. All 22
nonempty B3/B4 owner labels pass:

- 45056 active query-head/dimension coordinates have correct gathers and
  identical source/destination lane identities.
- 180224 complete CU state cells pass identity/changed scatter checks,
  including banks that must remain unchanged.
- 896 active/dummy CM/CL gathers and 2816 complete scalar-state scatter cells
  pass, including -inf maximum/zero denominator states.
- 3840 coordinates verify the replicated row layout for all five scalar
  fragments at M16 and M32.

AST checks prove one index per CU/OUT/out buffer within each Parallel loop,
constant-shaped bank accesses, exactly two macro calls, M16 bounds, B3 Owner
bounds, no shared CU, matching v4 MMA/scalar layouts, and the identical complete
tile arithmetic of the compact FullCol control. The bank layout preserves the
thread map while each independent bank uses eight FP32 slots/thread.

No Torch or TileLang was imported. No GPU was launched by the proof. It does
not test compiled races, spills, numerical output, JIT duration or latency;
those remain the root-owned qualifications' responsibility. Fewer source
specializations may reduce compile/code costs, while explicit selects and
extra live slot fragments may increase execution cost or register pressure.
Both outcomes must be measured; no speedup is presumed.
