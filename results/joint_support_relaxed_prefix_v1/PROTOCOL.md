# Independent shortest top-prefix control

This control deliberately relaxes exact cardinality to **at most k**. It is a deployment control at the same per-request pooled-score floor, not an exact-k optimizer or a quality-matched conclusion. Every other input, protected position, epsilon, score floor and identity comes from the frozen joint-support exchange v1 protocol.

For each request independently, sort optional tokens by its original score and lower original token ID, then find the shortest prefix that satisfies both actual `fsum` loss comparisons. Mandatory first4/last32 remain. A final independent sort/count reconstruction checks identity and a prefix shorter by one must fail. No cumulative-sum approximation or numerical tolerance is used in acceptance. Only pooled scores/supports and suffix provenance are consumed; no future Q, source KV, answers or natural output is inspected.

Freeze the script, this file and `protocol.json` before the natural run. Toy self-tests may precede freeze. Execute from the project root, without root or GPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 timeout 120 .venv/bin/python scripts/probe_joint_support_relaxed_prefix.py
```

The 48 rows are 3 documents × 2 budgets × 2 KV heads × 4 epsilons. Report all rows and both fixed exchange-order comparisons, without selecting the more favorable comparison. Record union and sum of per-request selected entries separately. Full source remains resident, so physical capacity released is zero. Timing excludes imports/toy self-tests and is a single CPU run, not production selector latency or a timing distribution.
