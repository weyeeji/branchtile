# Owner-task wave17 numerical and resource review

The saved wave17 qualification is complete: eight clean terminal jobs, 40 cases and all 240 FP64 O/natural-LSE checks passed. The independent CPU auditor verifies frozen source `aac3e85` and protocol `d7ace33`, all saved artifact and capture hashes, and complete ledger accounting. It reruns `cuobjdump` on 38 unique cubins, matching all 144 launched resource roles. Sixteen empty producer roles correctly have no launch or artifacts.

No launched cubin has Triton spills, LOCAL/STACK allocation, or SASS LDL/STL instruction sites. Producers use 123/128 registers; total shared memory is 19,456/29,696 B for N32/N64, including both static and dynamic shared allocation. Merge uses 40/42 registers and 3,072 B total shared. These observations remove a spill-related qualification concern; they do not prove good occupancy or any speedup.

The graph contract is narrower than previous mutable-offset kernels. Q/K/V and original Indices values change under one graph. Topology changes reject the old host operator and require a new plan/operator and graph recapture. All 40 cases exercise rejection and rebuilt-graph correctness; none replays the stale graph. Direct graph replay bypasses the host guard. The receipt's original `resource_review_pending` label remains unchanged and is not a numerical failure.

The eight jobs consumed 0.0477035164 P1 GPUh; this audit used no GPU. Diagnostic preparation/rebuild costs remain separate from steady-state timing. Six numerical phases per case are repeated checks, not independent documents. The next evidence must come from a complete producer-plus-merge comparison against all strong controls, with preparation visible and immutable-topology timing stated explicitly.

Full machine-readable evidence and reproduction command are in [the audit receipt](../results/operator_owner_tasks_wave17_audit_v1/audit.json) and [its README](../results/operator_owner_tasks_wave17_audit_v1/README.md). The auditor is [scripts/audit_owner_tasks_wave17.py](../scripts/audit_owner_tasks_wave17.py). No original wave17 result was edited.
