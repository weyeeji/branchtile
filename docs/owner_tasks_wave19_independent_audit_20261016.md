# Expanded owner granularity wave19 qualification

The new eight-point granularity qualification passed independently audited numerical, graph-topology and resource gates: eight clean jobs, 40 cases and 240 FP64 O/natural-LSE phases. Synthetic inputs have 8,197 tokens, with one real 32K capture per job. The new chunks are 64, 128, 2,048 and 4,096, alternating N32/N64 across B3/B4; this is not a complete sixteen-point factorial.

An independent AST comparison confirms the expanded qualifier changes only accepted chunk choices and its own source filename. The runtime owner wrapper, task planner, producer and merge remain the wave17 implementation. Source/protocol/manifest match frozen `b7927fe`; original wave17 data and audits remain unchanged.

All 144 launched producer/merge artifact roles are spill-free, with zero saved and independently redumped LOCAL/STACK/LDL/STL. Sixteen empty producer roles correctly have no launch/artifacts. Preparation and topology rebuild/recapture costs remain explicit; changed topology is checked only with a rebuilt operator and new graph. No stale graph is replayed, and direct graph replay still bypasses host validation.

The eight jobs cost 0.0448788451 P1 GPUh. The audit used no GPU. Full hashes, source-equivalence proof, resource summaries, all 240 checks, immutable capture provenance, topology receipts and ledger accounting are saved in [audit.json](../results/operator_owner_tasks_wave19_audit_v2/audit.json) and [README](../results/operator_owner_tasks_wave19_audit_v2/README.md). The new auditor is [scripts/audit_owner_tasks_wave19.py](../scripts/audit_owner_tasks_wave19.py). Qualification supplies no acceleration or application-quality result; each subsequent performance configuration retains its own FP64 gate.
