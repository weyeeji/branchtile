# Device-resident cascade qualification: independent audit (2026-10-17)

The frozen implementation has passed its bounded numerical and replay qualification. This removes a correctness blocker for controlled model integration; it does not establish model quality, speedup, or novelty. GPU execution was allocated by the root agent; this audit only reads source and completed evidence on CPU.

## Evidence and independent checks

`analyze_device_cascade_qualification.py` verifies every recorded source SHA256 against the frozen files, all O/LSE pass flags, the expected check counts, valid replay receipts, unchanged tail-plan topology, explicit rejection reasons, and v2 sticky-flag/bind-rejection checks.

| Evidence | O comparisons, candidate + native | Candidate merged LSE checks | Valid graph replays | Explicit prefix binds | Rejected cases |
|---|---:|---:|---:|---:|---:|
| `dense_cascade_device_qual_v1` (4 jobs) | 480 | 240 | 120 | 40 | 56 |
| `dense_cascade_device_qual_v2` (4 jobs) | 1044 | 522 | 169 | 32 | 61 |

All comparisons passed their recorded tolerances. O uses `atol=.006, rtol=.015` against independent FP64 attention that reads original physical pages, not the candidate's compact indices. Merged LSE uses `atol=.0003, rtol=.0001` after converting log2 LSE to natural log. These are combined absolute/relative tests; an absolute LSE error above .0003 is not by itself a failure. In v2, the largest candidate O relative L2 across jobs is .0036171; largest absolute O error is .0039276. Output values are BF16, so this is tolerance qualification, not bitwise identity.

v1 tests BF16, typed FP8, uint8-backed FP8 with nonunit K/V scales, and a longer prefix. However, its actual cache stride is `(16384,4096,256,1)`, not the engine's native `(8192,256,512,1)`, and tail capacity 160 produces only one metadata CTA per request. Those gaps motivated v2; v1's synthetic storage must not be described as proving the native ABI.

v2 asserts native cache stride `(8192,256,512,1)` and native-width contiguous int32 block tables. Three two-layer jobs test BF16, FP8, and uint8 FP8 with scales K=.5/1.25 and V=1.25/.5. Their fixed batches are 8/4/3/2/1, tails cross 16-, 4096-, and 8192-token boundaries and reach 8448 with row offsets. Capacity requires three metadata CTAs per request. A fourth job tests all 48 layers, batch 8, FP8, a 16384-token prefix, and tail cases 1/4097/8193/8446 plus row offsets. Public prefix/tail `.plan()` methods are replaced with exceptions after construction, so any subsequent public host planning fails the test.

Within a fixed-batch graph, Q and sequence lengths change each replay. The last valid case reverses the shared physical prefix, rotates private page tables, changes the entire KV payload, and explicitly rebinds the prefix without recapturing the graph. V2 also rejects a uniformly changed prefix without rebind, divergent per-row prefixes, and tails below/above capacity; it verifies errors remain sticky after restoring valid inputs and that divergent host binding raises.

## Why one-time tail planning is coherent in this restricted contract

The installed FlashInfer FA2 scheduler (`attention/scheduler.cuh`, around lines 637–652) uses one chunk when `disable_split_kv=True`; its work decomposition depends on fixed request/query shape, not changing private-tail page counts. The runtime prefill kernel (`attention/prefill.cuh`, around lines 3890–3906) obtains runtime KV length and uses it as the chunk end when split-KV is disabled. Thus changing indptr/last-page length can preserve this fixed scheduling topology. The qualification checks both actual outputs and the recorded plan topology.

This reasoning applies to fixed B, one current query per private request, no speculative future KV, and the current no-split-KV tail plan. It does not authorize arbitrary plan reuse with other query layouts, split-KV schedules, attention windows, or a different FlashInfer version. B changes construct a new plan and graph. The test's survivor batches are separate fixed shapes, not one graph whose B changes.

## Remaining exact-mapping check

Loose attention tolerances cannot reliably detect a single missing page in a very long context. The separately frozen `qualify_device_cascade_metadata_exact.py` (commit `6cec300`) therefore tests the actual GPU metadata kernels plus graph replay, comparing every indptr, last-page length and packed page ID to an independent CPU oracle. It checks all unused slots and trailing guards remain sentinel values. Its 30 cases are B=1/2/3/4/7/8 times tails=16/17/4096/4097/8192, a 120528-token prefix, three CTAs per request, and native int64 contiguous positions/table strides. This report does not claim that test passed until its result exists and is independently checked.

## Integration obligations and limits

- Physical page IDs must be allocator-valid. The flags check logical tail bounds and bound-prefix identity, not physical address ranges. Deliberately invalid physical IDs must not be submitted to attention.
- Positions must be contiguous 1D device values in the tested ABI. `_update_tail` addresses `Position + lane` and does not accept an arbitrary position stride.
- Flags are accumulated on device and read at a phase boundary after attention executes. Any invalid phase is rejected; its output must not be committed or counted as qualified performance. This is error detection, not prevention of an invalid memory access.
- Prefix binding includes an explicit host-visible identity check and must be charged at cohort admission/rebinding. No claim of free online construction is justified.
- Metadata is prepared once before the first attention layer and consumed on the same stream. All 48 layer caches must share the allocator's logical-to-physical page mapping, as in the tested layout.
- Synthetic per-layer qualification is not full-model sampled-generation qualification. The root's separate original-cache model and natural-output tests supply that evidence.
- Device planning, graph-stable metadata, and shared-prefix cascade have direct prior art; see `device_metadata_prior_review_20261017.md`. Passing these tests establishes a usable control implementation, not a new paper contribution.

Machine-readable audits: `results/platform/device_cascade_qualification_v1_audit_20261017.json` and `results/platform/device_cascade_qualification_v2_audit_20261017.json`.
