# Owner-task qualification: fixed topology and explicit rebuild

2026-10-16. [qualify_owner_tasks.py](../scripts/qualify_owner_tasks.py) is a new
qualification schema, `owner-task-qualification-v1`. It must not be reported as
the earlier v4 same-graph mutable-offset qualification. Only CPU checks and
Python compilation were run when this script was frozen; GPU results remain
pending until separately allocated jobs finish.

The module is fixed to `branchtile.attention_owner_tasks`, threads=128,
N=32/64 and chunk_tokens=256/512/1024 (default512). `splits` remains an explicit
compatibility parameter and does not divide the work; fixed chunk_tokens
determines task descriptors. Ordinary uid, a virtual environment, a single
visible GPU and new nonexisting output/artifact paths are enforced. No retry or
overwrite occurs. Allocation/timeout truth remains in the job ledger; a hard
kill may leave the JSON running, preserving the original partial evidence.

Every invocation runs all four original fixtures (`all_masks`, `unequal`,
`empty`, `one_token`), plus an optional fifth corrected same-forward real
capture. Capture source/mask/decode hashes are recorded and checked unchanged.
Output and natural-log LSE tolerances remain O atol .006 / rtol .015 and LSE
atol .0003 / rtol .0001 against the existing FP64 reference.

Five checks use one unchanged graph topology: eager, graph baseline, changed Q,
changed K/V, changed Indices. Before replay, `validate_plan()` is called;
only these values may change in place. Then the qualification changes owner
assignments, constructs updatedPlan, copies its offsets into the old plan and
requires `old_op.validate_plan()` to raise RuntimeError. Copying identical
offset values still changes the tensor version, so the empty fixture exercises
the guard too. **The old graph is never replayed after topology mutation.**

Offsets/counts/row_map/labels define fixed descriptors and require Attention
rebuild plus graph recapture. `graph.replay()` bypasses the host check and
cannot automatically protect a caller who ignores that contract. After the
explicit old-op rejection, a new Attention(updatedPlan) and new graph are
constructed. The sixth numerical check compares that new graph with the
changed-owner FP64 reference. It is named
`rebuild_and_new_graph_changed_owners`, not `graph_changed_offsets_and_indices`.

Plan construction, old-plan mutation/rejection, Attention rebuild and
warmup/JIT/new capture/first replay wall times are recorded separately and in
total. They are diagnostic lifecycle costs, not a kernel speed benchmark;
artifact export and FP64/hash auditing are outside that sequence. Operator
prepare_ms, metadata, structure hashes and workspace details are retained.

Both initial and rebuilt operator instances export their Triton producer and
merge CompiledKernel PTX/cubin, CPU cuobjdump resource/SASS, reported n_regs /
n_spills and metadata. Zero-task producers are explicitly `not_launched` with
no fabricated file; the merge is still launched and exported. Nonzero spill
evidence is recorded for independent review rather than silently assumed away.
The final `resource_review_pending` status is distinct from numerical failure.
If artifact export fails, failure_stage and the completed numerical checks are
preserved; a resource failure is not evidence of a numerical mismatch.

CPU verification before freeze: B3/B4 × three chunk sizes × two N sizes,
12 argument combinations and 48 fixture/planner checks at tokens=2053, with no
Torch/Triton import, plus `py_compile`. Example:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/qualify_owner_tasks.py --cpu-check --batch 4 --chunk-tokens 512 --block-n 64 --tokens 2053 --output results/tmp/unused.json
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m py_compile scripts/qualify_owner_tasks.py
```

The CPU check prints hashes and creates no output file. GPU execution must use
the central allocator; full qualification source and planner hashes are saved
in each resulting receipt and checked again before successful completion.
