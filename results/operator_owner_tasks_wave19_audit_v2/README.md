# Wave19 expanded owner granularity: independent resource audit

The eight new qualification jobs ended cleanly and passed **40 cases × 6 phases = 240 FP64 O/natural-LSE checks**. This separate snapshot preserves the original wave17 audit and receipts. Both the frozen expanded qualifier and its protocol/manifest match commit `b7927fe`.

The auditor independently normalizes the expanded qualifier's CLI chunk choices and self-source filename, then proves its AST equals the original wave17 qualifier. The owner wrapper, pure CPU task planner, and producer/merge kernel hashes remain unchanged since `aac3e85`. The original five fixed-topology phases and sixth rebuilt-plan/new-graph phase are retained; no check was disabled.

| B | Chunk tokens | N | Synthetic tokens | Real budget |
|---:|---:|---:|---:|---:|
| 3 | 64 | 32 | 8,197 | 32,768 |
| 3 | 128 | 64 | 8,197 | 32,768 |
| 3 | 2,048 | 32 | 8,197 | 32,768 |
| 3 | 4,096 | 64 | 8,197 | 32,768 |
| 4 | 64 | 64 | 8,197 | 32,768 |
| 4 | 128 | 32 | 8,197 | 32,768 |
| 4 | 2,048 | 64 | 8,197 | 32,768 |
| 4 | 4,096 | 32 | 8,197 | 32,768 |

These eight points alternate N32/N64 across the new granularity extremes; they are not the full sixteen-point B × chunk × N factorial. The larger synthetic input permits multiple chunks at chunk4096. Later performance configurations still require their own FP64 checks before timing.

All **144 launched roles** have zero Triton spills, cubin LOCAL/STACK, and SASS LDL/STL sites. The **576 PTX/cubin/resources/SASS artifact hashes** match, and fresh CPU `cuobjdump` reproduces each saved dump. Sixteen empty producer roles are correctly `not_launched` without invented artifacts; their merge kernels pass empty semantics. Producers retain 123/128 registers and 19,456/29,696 B total shared for N32/N64; merge retains 40/42 registers and 3,072 B total shared. Total shared includes the 1,024 B static cubin shared allocation plus Triton dynamic shared.

All 40 stale host operators explicitly reject the topology mutation, and none replays the stale graph. Correctness after changed owners uses a newly constructed operator and newly captured graph. Direct `graph.replay()` bypasses the host check. `splits=64` remains a compatibility parameter; task partitioning uses `chunk_tokens`.

The eight terminal jobs consumed **0.0448788451 P1 GPUh** (161.563842 allocated GPU seconds), with zero failures/timeouts. The CPU audit consumed zero GPUh. It verifies 52 per-job capture provenance checks against 19 unique immutable files and the earlier capture audit. Original `resource_review_pending` statuses remain unchanged and are not numerical failures.

This evidence establishes qualification of the expanded granularity points. It does not establish speed, occupancy, application quality, natural-EOS latency, energy savings, or a pure packing contribution. Diagnostic preparation/rebuild/recapture costs remain separately recorded in `audit.json`.

Reproduce with a new output path, normal uid and project virtual environment:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/audit_owner_tasks_wave19.py --output results/operator_owner_tasks_wave19_audit_recheck
```
