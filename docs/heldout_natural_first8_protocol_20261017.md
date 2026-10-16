# Outcome-blind heldout natural cohorts

Selection was committed in `42f9be5` before tokenization. It takes the first eight entries in frozen `docfinqa_public_evaluation100_v1.json` order with at least six exact-context natural questions, preserving every deduplicated question and the complete original document. Raw train/validation/test parquet hashes were checked; canonical answers and conflicting source answers remain recorded. No model outputs, quality, lengths, or performance affected selection. The selected documents do not overlap the training/development document hashes.

The manifest is `data/manifests/docfinqa_evaluation_natural_first8_cohort6plus_v1.json`, SHA256 `9cb3b29e50be30dc649f3cd765e801af8dbacdcd671dd4b22e0ebf9a17fd5b67`. It contains 101 natural questions. The separate CPU tokenizer used the actual `.venv-engine` tokenizer with `CUDA_VISIBLE_DEVICES=''`; no GPU or model inference ran.

| Frozen index | Document ID suffix | Natural B | Full prompt tokens | Context + 8192 cap |
|---|---|---:|---:|---|
| 2 | fe116609ddf687d5 | 14 | 113312–113325 | fits |
| 3 | 7a820a65d1141c0b | 8 | 128802–128816 | fits |
| 5 | fbb136a8d8b926a6 | 16 | 97943–97969 | fits |
| 6 | 11104b2bd6d8f14d | 10 | 153575–153587 | fits |
| 7 | fa7b5e32de51fddd | 14 | 133688–133703 | fits |
| 8 | 08846aeb2394e885 | 21 | 434059–434111 | failed |
| 9 | b63e8014d3fb9ba2 | 9 | 139585–139607 | fits |
| 10 | b6f597f905eedb2e | 9 | 162071–162086 | failed |

The engine limit is 163840 tokens. The final document's prompt alone fits, but its unchanged 8192 output allowance does not. Both failures remain selected and counted; do not truncate text, reduce the output allowance, substitute later documents, or silently remove them from task-completion denominators. Report any conditional speed statistic explicitly as conditional on feasible cases alongside the 6/8 context-feasibility rate. No speed or quality observation exists for this selection yet.

CPU BF16 KV payload estimates at all-request 8192 outputs range from 8.91 to 11.39 GiB per rank for the six context-feasible documents. These estimates exclude weights, graph/workspace storage, allocator effects and prefill transients; they do not guarantee admission or absence of preemption. The protocol retains all eight documents and uses per-document natural B without replicated requests.

Formal heldout execution must wait until training has frozen mechanism source, enabled-batch dispatch, numerical qualification, setup scope and analysis rules. The current configuration is an input/scenario specification, not permission to tune on heldout outcomes. Record every context/OOM/qualification/timeout/cap failure, actual candidate coverage, all output tokens, preemptions, strict task quality and complete task wall time. Preserve native/candidate output equality and report startup/qualification costs separately and inclusively. Test the frozen method once; revisions motivated by these outcomes invalidate the untouched-heldout interpretation of subsequent reuse.

Artifacts: `configs/native_natural_evaluation_first8_cohort6plus_v1.json` and `results/evaluation_natural_first8_feasibility_v1/audit.json`. The original evaluation manifest and existing B8/training protocols were not changed.
