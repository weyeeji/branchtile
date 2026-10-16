# TinyGEMM router qualification

The bounded TP2 job exits normally, costing 0.0546374953 P1 GPUh. It does not pass the predeclared conservative bitwise direct-replacement gate. No candidate performance timing or natural generation with TinyGEMM was performed.

All 576 original same-input graph reproductions, 288 cold A/A tensor points, and 12 poison/overwrite receipts pass. Of 147,456 logit elements across both phases and TP ranks, 36 differ by one adjacent BF16 encoding. These repeats reduce to nine changed (batch,layer) points on rank0/full_A. Every selected expert set stays unchanged; one unique row swaps experts 18 and 83 after a tie, and some normalized weights differ. Neither implementation is uniformly more accurate against the own-input FP64 reference. This establishes a floating-point execution difference, not incorrect TinyGEMM mathematics, measured quality loss, or performance failure.

Actual native inputs/logits/routing tensors are included as deterministic gzip archives with original/compressed SHA receipts. Candidate full logits were not saved, so a CPU audit cannot independently regenerate every candidate FP64 error without executing TinyGEMM again. Raw qualification records retain all candidate routing outputs and descriptive error metrics. The live model always used its original router.

Run `../../.venv/bin/python analyze_result.py` from this directory, or `.venv/bin/python results/tinygemm_gate_qualification_v1/analyze_result.py` from the repository root, to recheck source hashes, full denominators, repeats, archive integrity and the frozen stop rule. Official gate weights are identified by tensor SHA in rank configuration receipts; model weights remain external artifacts.

This is an existing-library baseline check, not a new research contribution. A non-bitwise replacement requires a separately frozen model-level numerical and quality protocol. Paper requirements remain unmet.
