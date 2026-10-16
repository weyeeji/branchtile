# Second blind factual audit: development 32

This is a second, independently conducted **LLM-assisted** audit of the 32 anonymous candidates in `development32_blind_queue.jsonl`. It is not human evaluation. The reviewer used a separate conversation from the first judge but belongs to the same model family, so correlated judgment errors remain possible.

During this audit the reviewer read only the frozen rubric, this anonymous queue, its eight complete source documents, and the reviewer's own outputs. General project context was inherited. The reviewer did not read first-judge item judgments, summaries or unblinding results; candidate identity mappings; method or budget results; implementation; or Git history. No candidate identity was inferred, and no method-level aggregation or inter-rater comparison was performed.

## Protocol

The scoring definitions are from `results/factual_audit/rubric_v1.json`, rubric ID `train8-blind-factual-audit-v1-20261016`. The task explicitly supplies a new 32-candidate queue: the rubric's historical queue path and expected count of 128 do not describe this batch. Only these batch metadata fields are overridden; the definitions of major claims, support, contradiction, unverifiability, coverage, factual pass and uncertainty are unchanged.

All eight sources were read in consecutive numbered chunks, totaling 4,139 lines. Their SHA-256 hashes were checked against the anonymous queue. The summary records each source path, hash, byte count, complete line range and actual reading chunks. Question-specific coverage key points were established from the question, reference and complete source. Gold references were treated as coverage cues, not independent factual authority; source-supported facts take precedence.

Each substantive major claim was judged against the source. Conjunctions with differing factual support were separated, and repeated equivalent propositions were counted once. Explicit factual conflicts are labeled `contradicted`; missing support or unresolved material attribution is labeled `unverifiable`. The latter does not by itself establish a contradiction. Factual pass requires at least one substantive claim and all major claims supported. Coverage is a separate 0–2 judgment, so an accurate but incomplete answer can pass factuality while receiving coverage 1. Material uncertainty is retained with a reason.

## Overall results

| Measure | Count |
| --- | ---: |
| Completed / expected candidates | 32 / 32 |
| Complete source documents | 8 |
| Major claims | 171 |
| Supported claims | 134 |
| Contradicted claims | 17 |
| Unverifiable claims | 20 |
| Answers passing factuality | 8 |
| Answers failing factuality | 24 |
| Coverage 0 / 1 / 2 | 0 / 17 / 15 |
| Candidates with material uncertainty | 16 |

These are overall descriptive counts, not method comparisons. Claim decomposition, attribution and coverage remain interpretive. Individual evidence ranges and rationales are preserved so later reviewers can examine disagreements without treating this judge's labels as ground truth.

## Files and validation

- `development32_claims_v1.jsonl`: 32 records sorted by anonymous ID, with atomic claims, labels, source evidence ranges, label counts, coverage, factual-pass status and uncertainty.
- `development32_summary_v1.json`: aggregate counts, rubric/queue/output hashes, full-source reading provenance and validation results.
- `development32_README_v1.md`: this protocol and limitation statement.

Validation ran as ordinary UID 1000 using the project `.venv/bin/python`; no GPU was used. It confirmed exactly 32 unique output IDs matching the queue, all eight source hashes and full line ranges, all evidence ranges within source bounds, claim-label totals, the frozen factual-pass rule and explicit reasons for every uncertain candidate. Python only checked structure, hashes and arithmetic after the judgments; it did not assign claim labels or substitute lexical overlap for factual review. No first-judge comparison, agreement statistic or human validation is included.
