# Independent relaxed-budget top-prefix result

**At a 1% pooled-score loss allowance, independently shortening each request's protected top-prefix reduces common-source union by 1.95–4.13% (median 2.77%), and total selected entries by 2.37–4.49% (median 2.93%).** This gives substantially less logical union reduction than frozen joint exchange's approximately 18–23% on these cells. It does reduce per-request work, which exact-k exchange does not. Neither outcome establishes task quality, GPU performance, or physical memory savings.

This arm uses **at most k, not exactly k**. The score floor, epsilon grid, first4/last32 protection and corrected suffix-only inputs match exchange v1. The two feasible sets must not be merged into an exact-k algorithm comparison. For every request it takes the shortest score-sorted optional prefix passing the original actual `fsum` loss conditions; it never reads another request's demand to choose its mask.

The script and protocol were frozen in commit **e06c531**, after toy self-tests and before natural computation. The single run completed without a retry or implementation change: 48 distinct document/budget/head/epsilon cells, containing 160 request checks. Inputs cover all three prior documents, rank0/layer15, budgets4096/16384, both KV heads, and B3/B3/B4. These are correlated cells from three documents, not 48 independent quality samples.

| Allowed pooled-score loss | Common-source union reduction min / median / max | Sum selected entries reduction min / median / max | Lower union than exact-k exchange, ascending / descending |
| ---: | ---: | ---: | ---: |
| 0 | 0 / 0 / 0% | 0 / 0 / 0% | 0 / 0 of 12 |
| 0.001 | 0.124 / 0.308 / 0.423% | 0.226 / 0.285 / 0.458% | 0 / 0 of 12 |
| 0.005 | 0.970 / 1.395 / 2.067% | 1.184 / 1.467 / 2.277% | 0 / 0 of 12 |
| 0.01 | 1.951 / 2.774 / 4.133% | 2.368 / 2.928 / 4.490% | 0 / 0 of 12 |

At epsilon zero every natural mask remains the original. For positive epsilon, all masks are subsets of their own original mask; all protected document-tail positions remain unchanged. The maximum actual 1% relative score loss is 0.00999953522479884. All 160 requests satisfy both `loss <= epsilon * original_score` and `loss / original_score <= epsilon`, without tolerance. Independent sorting reconstructs each selected-index hash, and removing the final optional token violates the same floor. Toy cases separately cover mandatory positions, stable ties, a positive threshold, and zero-valued optional tokens where zero epsilon can legitimately use fewer tokens under the stated arithmetic.

The common-source interval and request-private partial-page document tail are counted separately. Full source KV are still allocated: **physical capacity released is zero**. No source K/V, held-out decode Q, reference answers or natural output were used. No O/LSE or natural-quality comparison was run for this relaxed arm. The separately reported exchange attention-fidelity study does not qualify these different masks.

Post-import, post-self-test run time was **5.306 CPU seconds / 5.306 wall seconds**, with peak process RSS 534,564 KiB (522.04 MiB). Per-epsilon selection time summed over 12 cells was 0.642–0.650 seconds. This CPU reference repeats sorting across epsilon/budget cells and includes audit work in overall run time; it is not an optimized production selector or a repeated timing distribution. CUDA stayed uninitialized and GPU usage was **0 GPUh**.

`validation.json` records count, minimality, score, mask-subset, private-tail and selected-index checks. A separate post-run JSON read additionally verified 48 unique keys, output SHA, sums of per-request cardinalities, owner histogram union/entry arithmetic, every per-request score bound and zero physical capacity; all passed.

The comparison supports a limited conclusion: in the frozen pooled-score geometry, coordinating selected identities can reduce the union considerably more than simply shortening independent prefixes. It does not establish exchange novelty or superiority over classical cover/reverse-delete, and no method is quality-matched merely by sharing this proxy threshold. See `PROTOCOL.md` for the exact reproduction command and accounting scope.
