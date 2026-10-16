# Minimal uniform/chunk × rectangular/packed experiment

This is a CPU-only design, not a completed experiment or a new runtime implementation. The current owner task implementation covers chunk + packed only. Comparing it with the old `region_partials` hook changes scheduling, storage, merge scanning and the TileLang/Triton implementation together. It cannot identify the cost of packing or canonical-state transport by itself.

## Four arms with one arithmetic engine

Use the same immutable RegionPlan, original Q/K/V/Indices, compact M16/M32 producer math, BF16 QK/PV, FP32 online U/M/L, N tile, 128 threads and streaming GROUP16 merge. A task always handles one head, owner cohort and nonempty token interval. No arm keeps cross-owner canonical state. All arms return the same BF16 O and natural FP32 LSE, including O=0/LSE=-inf for empty requests.

| Arm | Partition within each owner region | Partial address space |
|---|---|---|
| U-R | Uniform S shards, empty shards removed | `T * rows` canonical row slots |
| U-P | Identical U-R task intervals and ordering | `sum_t active_rows[t]` packed slots |
| C-R | Fixed C-token chunks, empty regions removed | `T * rows` canonical row slots |
| C-P | Identical C-R task intervals and ordering | `sum_t active_rows[t]` packed slots |

For a region `[begin,end)`, uniform task s is `[begin + length*s//S, begin + length*(s+1)//S)`, s=0…S-1, dropping zero-length intervals. Chunk tasks are `[begin+j*C, min(begin+(j+1)*C,end))`. Task descriptor order is deterministic head/region/interval order. Storage choice must not change this list, task count, launch geometry, M choice, Q reload count or tile loop.

Use a common producer with a compile-time address choice only. A rectangular destination is `task_id*rows + source_query_row`; a packed destination is `prefix_active_rows[task_id] + active_local_row`. Both store only actual active rows. This needs no full-row U initialization or shared canonical-U buffer. The same GROUP16 merge kernel can be used unchanged: a per-request base is `task_id*rows + request*GQA` for rectangle, or `packed_base + request_slot_within_owner*GQA` for packed. Both add the GQA group-head offset.

All four arms use the same per-query valid task list; only the stored physical row base differs. Thus rectangular unused row holes are neither initialized nor read. This first 2×2 isolates address/storage layout and its capacity/cache/coalescing effects from task partitioning. It deliberately does not reproduce a weak rectangular merge that scans every unrelated owner. The old dense `region_partials` hook remains a separately labelled legacy control. If the research question is specifically dense scanning versus valid-list merge, add a separate dense-scan control at fixed task boundaries; do not fold that extra factor into “pure packing.”

## Primary and tuned comparisons

A minimal initial matrix is S=64 versus C=512, storage R/P, at N32 and N64: eight independently qualified jobs, convenient for eight GPUs. The numeric S and C are distinct scheduling parameters; equal values do not mean equal partitions. Declare this matched implementation matrix before timing. Each storage pair must have identical task boundaries. A task-count-matched scheduling sensitivity analysis is optional and must be declared separately rather than chosen per test capture to favor an arm.

The existing development observation that chunk optima hit a grid edge motivates the expanded owner grid; it does not justify selecting a new primary C on test outcomes. For a tuned family comparison, retain all measured S and C settings with equal search accounting and report the tuning oracle separately. Current development candidates include S32/64/128 and C64/128/256/512/1024/2048/4096. A fixed primary matrix and a broader tuning matrix answer different questions. Pick which to run using development evidence and available budget, before consuming the confirming captures.

Use unchanged complete-call FP64 qualification before selecting winners: all owner masks, full/disjoint/cross/tree/skewed supports, unequal region lengths, empty and one-token cases, tile/shard/chunk boundaries, and real complete private suffixes. Verify every original query-token edge appears once and that rectangular and packed addresses reconstruct identical per-query contributions. Same-graph Q/K/V/original-Indices value changes must pass; changed topology must reject the old host operator, then rebuild and recapture. Every arm has the same topology/replay contract. No producer-only timing or relaxed arm-specific tolerance is permitted.

## Cost model and measurements

Let `T` be the total number of nonempty tasks across heads, `rows=B*GQA`, `a_t` the active row count and `P=sum_t a_t`. Rectangular partial allocation is `4*T*rows*(D+2)` bytes; packed partial allocation is `4*P*(D+2)`. Since the two storage forms write and read only the same active rows in this design, their logical U/M/L producer writes and merge reads are both `4*P*(D+2)` bytes. Capacity reduction alone must not be described as reduced runtime bytes. Hardware transactions may differ and require a profiler; a formula is a logical proxy, not measured GDDR traffic.

Report descriptors (six int32 fields per task), valid-list indices/counts, common RegionPlan, output/LSE, allocator/graph scope and any alignment/capacity surplus separately. The currently padded valid-list table costs `4*H*B*maxContrib` plus `4*H*B` counts, not a free CSR. CPU construction, D2H planning reads, H2D uploads, allocations, first JIT/call, graph capture, and topology-change rebuild/recapture must be explicit; only a declared steady-state graph excludes them. Do not compare partial-only memory to another arm's all-owned memory.

For full owner-label coverage with every uniform shard nonempty, B3 has 96 active versus168 rectangular row slots per head/shard; B4 has256 versus480. The 1.75×/1.875× reduction describes allocation capacity in this exact comparison. Skew and empty shards change the actual counts. Packed task partials may still exceed a fused across-owner implementation's `H*S*rows` partials.

Also record union KV visits `sum_t(end-begin)`, query-row reloads `sum_t a_t`, task/partial-row counts, per-query contribution distribution, per-task token lengths, padded QK/PV work `sum_t M_t*N*ceil(length_t/N)` and generated register/SMEM/stack/spill observations. Uniform versus chunk changes task count, Q reloads, tail padding and load balance; those are the partition mechanism, not an accidental storage effect. Producer/merge stage diagnostics may explain a complete-call result but never replace it.

For a capture, let L_U_R etc. denote complete-call latency. Report storage ratios `L_U_R/L_U_P` and `L_C_R/L_C_P`, partition ratios `L_U_R/L_C_R` and `L_U_P/L_C_P`, plus interaction `log(L_U_R/L_U_P)-log(L_C_R/L_C_P)`. Use synchronized random/reverse paired blocks and an A/A control. Within-capture block bootstrap measures timing noise only; confirming cross-capture results require pairing on the same natural capture and clustering by document/request trajectory. Preserve all four cells, errors and incomplete jobs. Keep the complete native/E1/fixed-union/hierarchy/FullCol/wide/persistent/v4/lane controls for the performance decision: explaining an E3 mechanism does not establish that it beats the best execution strategy.
