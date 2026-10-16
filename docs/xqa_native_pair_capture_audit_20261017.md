# Native XQA pair packing: real-capture audit, 2026-10-17

The corrected four real-capture screens pass bounded bitwise equivalence. BF16 shows a substantial prefix-only opportunity. FP8 shows a small improvement. Neither establishes private-tail support, model equivalence or end-to-end benefit.

Reproducible CPU auditor: `scripts/audit_xqa_pair_native_capture_v2.py`. Machine-readable result: `results/xqa_pair_native_capture_audit_v2/audit.json`. Frozen GPU runner: commit `f19e6f5`. Results: `results/xqa_pair_padding_native_capture_v2/`.

| Actual capture | Native median µs | Fixed-split pair median µs | Median within-block speedup | Paired bootstrap 95% interval |
|---|---:|---:|---:|---:|
| train1 BF16 rank0 L15 | 203.096 | 130.202 | 1.560× | 1.552–1.566× |
| train7 BF16 rank0 L47 | 245.091 | 152.366 | 1.609× | 1.605–1.615× |
| train1 FP8 rank0 L15 | 128.594 | 123.339 | 1.045× | 1.031–1.056× |
| train7 FP8 rank0 L47 | 151.486 | 145.477 | 1.046× | 1.035–1.053× |

All 20 randomized timing blocks remain in each estimate. Bootstrap resamples complete paired blocks, 20,000 times with a fixed seed. These intervals describe within-run variability, not device, document or independent-run uncertainty. Native/native-AA paired ratios are 1.003–1.012 and all their intervals include one. Native-AA provides a separately reported comparison; no warm-up blocks were selectively dropped. Candidate timing includes Q packing and output restoration.

Each case passes six native-AA and six fixed-split exact BF16-bit comparisons, including Q mutation, pair member exchange, physical page reversal and a shared non-page-aligned boundary. All six auto-split controls change output bits. The direct public native call matches the production convenience entry. Source and capture SHA256 hashes match, actual int32 page tables are checked, reconstructed original-stride cache pages round-trip byte exactly, and unit scales are explicit.

The independent CPU FP64 reference covers 16 sampled query rows/heads over the complete actual prefix. Native and fixed-pair errors match exactly. Their maximum relative L2 error across the four cases is 0.004704; maximum absolute error is 0.002328. The reference evaluates mathematical attention and does not emulate native probability/split rounding. It is an input/output diagnostic, not an additional model quality result.

CUPTI traces confirm native grid `[10,2,8]`, fixed pair `[10,2,4]`, and auto pair `[21,2,4]`. Native and fixed pair have the same measured kernel register and shared-memory requirements in each case. Trace occupancy estimates claim zero active blocks despite executed kernels; these estimates are not usable as hardware occupancy evidence.

## Invalid earlier real-capture runs

The two real FP8 files in `results/xqa_pair_padding_v1/` used CPU-exported int64 page tables without converting them. The native wrapper reinterprets page table pointers as `KVCachePageIndex=int32_t` (`mha.h`), producing an invalid address interpretation. Their output mismatches cannot reject pair packing or characterize numerical error. Preserve those files as invalid-ABI diagnostics. The two synthetic v1 cases already used int32 tables and remain valid. V2 fixes the ABI and passes both real FP8 cases.

## Evidence boundary

The public GQA16 adapter is an existing-kernel diagnostic, not a new algorithm. It shares the complete prefix, has no independent private suffix, and is tested on four captures with six mutations each. Broader rank/layer tests and natural membership subsets are separate runs. Subset v3 selects initial captured Q rows using observed survivor membership; it does not claim those Q values came from the later natural step.

## Complete BF16 matrix and survivor subsets

A second independent auditor, `scripts/audit_xqa_pair_bf16_broad_subset.py`, verifies all 16 actual BF16 captures (two documents × two TP ranks × layers 0/15/31/47) and all eight subset cases. Result: `results/xqa_pair_native_capture_audit_v2/broad_subset_audit.json`. Source/capture hashes, page ABI, actual launch grids, mathematical reference consistency and negative controls pass. Across these 24 cells there are 144 fixed-pair bitwise checks, 144 native-AA checks and 144 auto-split output-changing checks.

All 16 B8 cells improve: paired speedups range from 1.560× to 1.619×. This removes the original two-layer selection concern within the captured matrix; it does not establish generalization beyond the two documents or sampled layers.

| Survivor batch | train1 paired speedup [95% within-run interval] | train7 paired speedup [95% within-run interval] |
|---|---:|---:|
| B2 | 0.945× [0.943, 0.948] | 0.961× [0.959, 0.964] |
| B4 | 1.173× [1.159, 1.185] | 1.178× [1.166, 1.187] |
| B6 | 1.484× [1.474, 1.499] | 1.512× [1.503, 1.522] |
| B8 | 1.565× [1.559, 1.573] | 1.606× [1.603, 1.609] |

B2 loses in all 20 paired blocks for both documents. B4/B6/B8 win in every block. Therefore an unconditional pair dispatch is contradicted by these results. Native split counts are recalculated for the actual batch: B2=42, B4=21, B6=14, B8=10; the corresponding paired call preserves that same count.

The observed natural traces also limit the opportunity: B≥6 occupies 43.7% of train1 decode steps and 16.6% of train7 steps; B≤2 occupies 33.0% and 70.4%, respectively. These are step counts, not wall-time fractions. Large-batch steps cost more, so counting steps alone cannot predict total savings. Odd batches were not screened here; treating them as equivalent to the next even batch would be unsupported. Real private tails, odd survivors, activation of fallback and end-to-end natural trajectories remain necessary.
