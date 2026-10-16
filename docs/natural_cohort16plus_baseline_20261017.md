# Complete natural B21/B19 native baselines

The selection was fixed before measuring outputs: take the first two entries in frozen `docfinqa_public_training167_v1.json` order whose natural question count is at least 16, and retain every deduplicated exact-context question and the complete original document. This selects training indices 26 and 33, with 21 and 19 questions. No question was copied, no document or question cohort was truncated, and no later document replaced these selections. Frozen development/evaluation manifests and the existing B8 protocol remain unchanged.

`data/manifests/docfinqa_training_natural_cohort16plus_v1.json` preserves source question occurrences and source parquet hashes. `configs/native_natural_cohort16plus_training_v1.json` preserves the same sampling settings, including natural EOS and the 8,192-token maximum. CPU tokenization was independently reproduced in `.venv-engine`: every prompt hash and complete-document boundary matched. Prompts are 135,400–135,429 tokens for B21 and 123,837–123,886 for B19, within the existing 163,840 context limit including the output cap.

## Actual native measurements

Both TP2 BF16-KV jobs completed cleanly. The model remains Qwen3-30B-A3B-Thinking-2507-FP8, with the existing FlashInfer/XQA native attention and complete history. Graph sizes cover every actual batch size from 1 to the selected cohort size. No alternative attention, cached-output intervention, profiler, or output cap reduction was used.

| Measurement | index26, B21 | index33, B19 |
|---|---:|---:|
| Actual KV token capacity per rank | 274,048 | 274,128 |
| Actual KV pool GiB per rank | 12.5449 | 12.5486 |
| Native initialization including graph setup, seconds | 94.604 | 93.942 |
| Cold anchor prefill plus one output token, seconds | 24.323 | 21.056 |
| Complete natural group generation, seconds | 105.191 | 70.298 |
| Cold anchor plus natural group, seconds | 129.515 | 91.354 |
| Longest natural output, tokens | 5,551 | 3,254 |
| Request preemptions | 0 | 0 |
| Length-cap hits | 0/21 | 0/19 |
| Internal strict FINAL exact accuracy | 4/21 | 7/19 |
| Internal strict FINAL rounded accuracy | 10/21 | 10/19 |

There was no additional unmeasured natural warmup. Engine initialization includes its normal profiling/capture preparation; any remaining first-use prefill or JIT cost occurs inside the measured anchor or generation interval. These are first-session natural task measurements, not a claim of warmed steady-state performance. Native request statistics and CPU-only identity/step records are enabled and charged. Request scheduled-to-first-token timings are retained but are not isolated GPU prefill time.

## Cohort survival

The audit verifies identical step and graph trajectories across the two TP ranks. It counts actual prepared pure-decode batches, including every decrease in active requests. High-batch thresholds are cumulative and must not be added together.

| Active batch threshold | B21 steps / all pure-decode steps | B19 steps / all pure-decode steps | B21 observed CPU boundary span | B19 observed CPU boundary span |
|---|---:|---:|---:|---:|
| B≥4 | 3,438 / 5,551 (61.93%) | 2,169 / 3,254 (66.66%) | 79.396s (78.26%) | 56.832s (85.11%) |
| B≥8 | 1,833 / 5,551 (33.02%) | 1,534 / 3,254 (47.14%) | 54.777s (53.99%) | 46.085s (69.02%) |
| B≥16 | 876 / 5,551 (15.78%) | 786 / 3,254 (24.15%) | 32.136s (31.68%) | 26.143s (39.15%) |

The span columns assign time between adjacent CPU input-preparation boundaries to the earlier scheduled batch. Their percentages use the observed pure-decode boundary span as denominator. They include model, scheduler, and host effects; they are not GPU kernel or critical-path attribution. First-entry and final-completion gaps are excluded. Recorded all-batch boundary spans are 103.893s and 69.217s, with 2.439s and 2.442s attributed to prefill or mixed batches.

These baselines demonstrate naturally occurring larger cohorts that remain scheduled for substantial portions of the observed execution. They do not demonstrate a candidate speedup or prove real application arrival concurrency. Any proposed sharing mechanism still needs full-model comparison, including fallback at small B, setup, complete private history, and quality.

The CPU worst-case estimate remains relevant: if every request stayed active until all 8,192 output tokens, BF16 KV payload would require 14.114 and 12.837 GiB per rank, above the actual pools. These natural runs avoided that worst case through earlier EOS; zero observed preemptions is not a universal capacity guarantee. The selection and cap remain unchanged.

Quality is better than some earlier B8 cases but still limited. All parse failures remain in the denominator; index33 includes one unparsed gold answer. The internal FINAL metric is not official program-execution accuracy. No task-success or publication claim follows from these two baseline runs.

## Artifacts

- `results/native_natural_cohort16plus_v1/train26_bf16/summary.json` and `analysis.json`
- `results/native_natural_cohort16plus_v1/train33_bf16/summary.json` and `analysis.json`
- `results/natural_cohort16plus_feasibility_v1/audit.json` and `engine_tokenizer_verification.json`
- Reproducible native runner: `scripts/benchmark_native_natural_cohort.py`
- Independent audit: `scripts/audit_native_natural_cohort.py`

The audit rechecks source hashes, clean-exit ledger receipts, all natural questions, both rank trajectories, actual cache capacity, preemptions, EOS/caps, wall arithmetic, and frozen FINAL scores. Metric internals use tuples which JSON serializes as lists; the independent rescore comparison normalizes this representation without changing scoring or tolerances.
