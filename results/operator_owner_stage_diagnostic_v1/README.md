# Wave23 owner-stage diagnostic: independent CPU audit

Eight clean P2 jobs used 0.02044224234375482 GPU-hours. This audit used zero GPU-hours and did not import torch or change runtime/experiment files.

The auditor verifies frozen source hashes, exact producer/merge launch AST equality, ledger start/end identity and allocation non-overlap, 32 recorded complete-output checks, the 48 actually consumed capture-file hashes, selected chunk/provenance identity, and every randomized/reverse measurement order. All 64 medians reproduce exactly from 640 rows / 2,560 timing values. The prior wave is used only for configuration and provenance, never as a timing denominator.

- [audit.json](audit.json): hashes, per-cell summaries, A/A block-bootstrap intervals, raw-record identities, and eight ledger receipts.
- [stage_medians.csv](stage_medians.csv): all hot/cold complete, producer-only, merge-only and complete-A/A medians, task counts and logical partial payload.
- [Interpretation](../../docs/operator_owner_stage_diagnostic_20261016.md): table, limits and next experiments.

Stage-only graphs change residency, dependencies and scheduling. Their sum/difference is not removable merge/launch cost or a prediction of fused performance. Artificial cold merge starts with flushed partials. Cold A/A quantization differences remain visible. There is no competitor, whole-model, energy, DRAM-traffic or new-method result here.

Reproduce:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/audit_operator_owner_stages_wave23.py
```
