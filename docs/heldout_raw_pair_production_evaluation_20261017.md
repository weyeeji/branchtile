# Frozen production evaluation of raw-pair sharing

The primary evaluation was frozen in commit `e5b58c1`, before the first heldout GPU invocation. Its machine-readable contract is `results/native_raw_pair_heldout_production_launch_v1.json`; it fixes source hashes, input hashes, offline numerical receipts, all eight selected documents, six runnable job manifests, and the aggregate analysis. The earlier strict nine-phase manifests remain unexecuted alternatives and are not candidates for choosing the better reported result.

## Inputs and denominators

The input rule is the first eight documents with at least six natural deduplicated questions in the frozen public-evaluation ordering. All natural questions and the complete original document are retained. Selection preceded tokenization and did not read model outputs. The eight documents have 101 questions. Evaluation indices 8 and 10 fail the frozen 163,840-token context / 8,192-output-token feasibility check. They are not replaced, truncated, or assigned a smaller output cap. The six feasible documents contain 71 questions; runtime failures, if any, also remain in the report.

## Runtime and numerical scope

Every feasible document runs a fresh process with the frozen production-only driver from `0521958` and the general-policy worker from `debcb81`. The process constructs only production graphs and executes four natural-EOS phases in native/candidate/candidate/native order. It retains real output lengths, cap hits, preemptions, quality, phase wall time, and actual candidate replay coverage. The frozen development policy admits actual batch size at least 6, nominal common prefix at least 65,536 tokens, and maximum private length at most one eighth of that prefix. Unsupported or ineligible states execute native attention and remain valid results.

Offline training-document-26 receipts supply both TP ranks' numerical qualification for batches 6 through 21. Exact source and execution fingerprints must match. Qualification costs are reported separately; no new heldout numerical qualification is hidden in setup. Training receipts verify per-layer Q/O, hidden states, logits, current KV writes, metadata, and native repeatability. Heldout acceptance requires identical complete generated token sequences, APC metadata, finish reasons, and graph input trajectories across ABBA arms. It does **not** claim that every heldout hidden value was compared bitwise.

## Predeclared latency accounting

For each document, the primary ratio is:

```
mean native (anchor + natural generation)
-------------------------------------------------------------
mean candidate (anchor + natural generation)
  + offline receipt preflight
  + candidate configuration
  + actual production graph capture
  + setup receipt export
```

All four measured setup components are charged once to that document, with no amortization across documents. The graph setup is performed using that document's actual shared prefix. Ready-generation, anchor-plus-generation, and complete phase-control ratios are also reported. Common model initialization and input preparation are excluded from warm-service document latency. Separate first-use component sums are estimates, not independently timed cold-service runs.

The frozen `scripts/summarize_raw_pair_heldout_production.py` reports both geometric mean speed ratios and ratios of summed times over paired completed documents. It cannot silently drop a slow native-fallback case, replace a failed document, infer timing for a failed pair, or finalize while a feasible document remains pending. Timing aggregates are explicitly conditional on paired completion; all eight documents and 101 questions remain in coverage and failure denominators. Two same-session repeats do not establish a population confidence interval.

## Environmental caveat and retained evidence

`results/pat_build_visibility_guard_audit_v1.json` records an earlier attempted CPU build whose FlashInfer import could initialize CUDA on visible devices. Actual GPU activity was not measured. The conservative exposure intervals and charges remain in the ledger; overlapping development timings are exploratory pending clean repeats. A successful numerical audit and clean process exit establish neither exclusive resource access nor freedom from timing interference. Formal heldout runs start after those recorded build intervals. Any later overlap must be reported alongside their timing.

## Reproduction

Root schedules the six frozen manifests `configs/native_raw_pair_heldout_production_eval{2,3,5,6,7,9}_jobs.json` onto free TP2 pairs. Their initial GPU assignments are placeholders, not authorization to run them concurrently on the same pair. After all six terminate, CPU-only aggregation is:

```bash
CUDA_VISIBLE_DEVICES='' .venv-engine/bin/python scripts/summarize_raw_pair_heldout_production.py \
  --output results/native_raw_pair_heldout_production_final_audit_v1.json
```

An interim report requires `--allow-incomplete` and a distinct output path. The analyzer refuses to overwrite an existing report. Sources and the original selection must remain frozen; necessary corrections require a new explicit version and preserved original evidence.
