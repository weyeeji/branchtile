# Complete16-cell real-capture PAT eager comparison

The independent audit `results/pat_fixed_broad_independent_audit_v1.json` combines the14 predeclared new cells with the two original multistream cells: train1/train7 × rank0/rank1 × layers0/15/31/47. All16 pass the same frozen source/FP64/RawPair-bitwise gates, comprising1472 four-arm numerical checkpoints. Each has30 randomized paired timing blocks and verified CUPTI full-call kernel coverage. No failed cell was excluded or per-input single/multistream winner selected.

The hot caller-stream event speedup of grouped-hint RawPair over locally repaired multistream PAT lies between1.427× and1.461×. This is still prepared-metadata eager operator timing, on two shared-prefix documents and one GPU architecture. The wider layer/rank coverage reduces selected-layer concerns but is not16 independent workloads, a natural end-to-end result, or evidence that the eager speedup survives graph replay. A matched-graph test is prepared specifically to check the host-overhead explanation.

The local PAT changes, allocation/setup accounting, paired-bootstrap intervals, separate wall times and cold-AA uncertainty remain as described in `pat_fixed_scheduler_eager_results_20261017.md`. Raw data and the complete independent per-cell audit retain the exact values and source hashes.
