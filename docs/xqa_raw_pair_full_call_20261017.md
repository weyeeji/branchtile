# Complete native-preserving paired call: positive operator result

All 16 BF16 real capture states (two training documents × two TP ranks × layers 0/15/31/47) completed the frozen five-arm benchmark. Native PDL-on is the main baseline. Native-AA, native PDL-off, unshared checkpoint/resume and the complete paired call are interleaved in 30 randomized blocks. All initial and final graph outputs match native by int16 views. Earlier dynamic-input correctness tests are retained separately.

Native / complete-pair latency ranges are **1.396–1.446× hot** and **1.229–1.286× cold**. Hot samples replay ten times; cold samples flush 256 MiB before timing one replay. Each paired call includes Q packing, pair page-table and length copies, raw-state writes/reads, private continuation and native final merge. Unshared checkpoint/resume is slower than native: pausing alone provides no gain. PDL-off native has similar timing, so disabling native PDL is not the source of the reported advantage.

These are attention-call GPU measurements. Standalone host physical-prefix validation and length preparation take approximately 62–70 μs per call and are reported separately, as is graph capture. The whole-model adapter must charge preparation once per step, plus its actual per-layer Q/O copies and all metadata work. The entire model is not expected to achieve the same speed ratio. No natural task gain is claimed here.

Independent audit and plot: `scripts/audit_plot_xqa_raw_pair_perf.py`, `results/xqa_raw_pair_perf_audit_v1/audit.json`, and `complete_call.pdf`. Source, qualification and actual capture hashes match. The audit retains every block and recomputes paired-ratio bootstrap intervals with 20,000 resamples. The initial private tails are short; longer generation and changing active batches still require model and additional runtime qualification.

The next gate is 128 native-trajectory decode states in the actual 48-layer model, comparing all attention Q/O, current KV writes, hidden states and logits by bytes. Only after that gate can fixed-length and natural-EOS ABBA trials establish useful task-level results.
