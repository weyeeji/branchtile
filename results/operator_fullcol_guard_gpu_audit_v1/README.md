# FullCol control independent compiled-artifact audit

CPU-only review of four completed wave3 FullCol guard GPU qualifications.
See [the report](../../docs/operator_fullcol_guard_gpu_audit_20261016.md) and
[machine-readable evidence](audit.json).

- 4 clean job exits, 20 cases, 120 numerical/mutable-graph checks passed.
- All saved CUDA/cubin/SASS hashes match the qualification receipts.
- Fresh CPU cuobjdump resource/SASS outputs match every saved artifact.
- All 20 cases have LOCAL=STACK=0 and no SASS LDL/STL.
- Generated CUDA confirms canonical U remains in shared memory and is gathered
  and scattered through runtime RowMap indices at each nonempty region.
- AST comparison confirms two FullRow→FullCol GEMM policy changes; other
  producer semantics and wrapper behavior match emptyguard.

The original `resource_review_pending` JSON status is preserved. This audit
finishes that independent review; it does not turn a numerical pass into a
failure or claim any speedup. Static SASS sites are not dynamic traffic counts.
No GPU work, source changes, compilation, or original-artifact writes occur.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_fullcol_guard_gpu_artifacts.py
```
