# Stop the Q+PV components hypothesis

The bounded original-page experiment passes its numerical gates but does not establish useful acceleration. Three of four cases are slower than their matched Q-only control, and every case is substantially slower than mature FlashInfer FA2 shared-prefix attention. Do not expand this into a tile sweep or integrate it into the model.

| Case, M64 | XQA O-only, us | FA2 O+LSE, us | Q-only control, us | Q+PV components, us | Paired control minus candidate, us |
|---|---:|---:|---:|---:|---:|
| Synthetic 8K | 17.38 | 14.77 | 51.65 | 53.68 | -2.053 |
| Train1 L15, 120,528 tokens | 127.99 | 52.46 | 72.14 | 76.23 | -4.054 |
| Train1 L47, 120,528 tokens | 128.75 | 53.29 | 72.59 | 76.66 | -3.990 |
| Train7 L15, 144,576 tokens | 151.64 | 60.10 | 80.85 | 80.36 | +0.605 |

Each arm uses the same original Q and fused physical cache storage. The candidate includes both Q decomposition and P decomposition inside the producer. Twenty paired blocks each contain ten graph replays. The paired median 95% block-bootstrap intervals are respectively [-2.150, -1.939], [-4.160, -3.939], [-4.382, -3.451], and [+0.205, +0.741] us. The first three candidates win 0/20 blocks; the last wins 17/20. The last result is a small positive incremental measurement, not evidence of a competitive implementation: its median is still 34% slower than FA2. Timing blocks are not independent workloads.

All 36 recorded checks pass their preregistered elementwise tolerances: two QK arithmetic probes, four complete-prefix outputs, and three base-two LSE checks per case. Real candidate output relative L2 error versus FP64 is .00162–.00180, almost unchanged from the matching BF16-P control. Candidate/control LSE errors coincide as expected because their QK and pre-cast denominator paths match. These observations qualify only the recorded prefix inputs. They establish neither bitwise O identity nor a model-level quality guarantee; QK probes alone do not qualify PV.

Runtime producer resources explain why removing V conversion need not improve the kernel: candidate dynamic shared memory rises from 24 KiB to 32 KiB and reported spills rise from zero to two. Candidate registers are 254 versus control 254–255. Offline compilation with a different Q row stride previously reported 242 versus 239 registers and no local memory; those figures do not describe the actual measured specialization. The new runner's actual resource receipts take precedence. No causal attribution to spills alone is possible without a profile, and no such extra profiling is needed to make the stop decision.

Native XQA's much higher long-prefix latency does not establish this numerical technique's value: mature sharing already improves it more. The experiment excludes private tails, live metadata maintenance, model sampling, and end-to-end wall time. XQA emits O only while the three split-attention producers emit O plus LSE.

`analyze_paged_probability_components_gpu.py` verifies every raw result's source hashes against the frozen files before computing paired summaries. The JSON audit is `results/platform/paged_probability_components_gpu_analysis_20261017.json`. Raw evidence remains in `results/paged_probability_components_gpu_v1/`.
