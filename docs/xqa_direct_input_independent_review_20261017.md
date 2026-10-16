# Independent CPU review of native-input pair continuation

Reviewed frozen `src/xqa_raw_direct_input_build.py`, `src/xqa_raw_direct_input_runtime.py`, inherited padded/general runtime, and generated source identity `c28391a037f29141`. No blocking address, ABI, or schedule discrepancy was found within their declared BF16, head128, page16, KV-head2, GQA8, no-beam/no-speculation scope. GPU results and memcheck reported by the parent are separate evidence; this document is a CPU source review.

For pair group g, KV head h in {0,1}, and logical prefix row r=8m+a (member m in {0,1}, local head a in [0,7]), the new Q source head is:

`32*g + 8*h + 16*m + a = 16*(2*g+m) + 8*h + a`.

This exactly addresses the corresponding request/head in the original contiguous `[B,16,128]` Q. Tail phase3 keeps the original request/head addressing. This is a nontrivial head permutation over a **contiguous original Q**; inherited construction still rejects arbitrary-stride Q. A future model adapter cannot assume its own Q canonicalization copy has also disappeared.

All prefix cache length and four K/V page-loader request indices become `2*g`, equivalent to the former `table[::2]` and `lengths[::2]` copies. The unchanged inherited prepare contract checks each pair's physical shared prefix. Lengths remain native full-batch data; the new wrapper's originalBatch comes from `seqLen.size(0)`, independently of the prefix grid's pair count. Prefix launcher SM argument `S*groups*2` with launch batch `groups` and KV-head2 preserves S; tail uses `S*B*2`. Sequence cap and original stride-S tile schedule remain unchanged. The phase1 raw-state store returns before normalized output/scratch merge. The grouped-hint control correction is inherited unchanged.

For an odd final pair, the partial async-copy specialization declares only eight available heads. The four producer warps cover disjoint groups of four logical rows, and the second member's rows receive zero source bytes. `copyPartialHeadsAsync` passes srcSize=0 to `cp.async` for every invalid row, which zero-fills its shared destination; every valid row maps to a live original request. Pointer arithmetic still forms addresses past the last Q row for the masked member, so source review alone is not a memory-safety qualification. The parent reports odd B7/B19 memcheck with zero errors and changed-input/native-bit gates; those bounded tests are consistent with the intended zero-fill path, not a universal compiler proof.

The new run path calls two XQA wrappers and omits Q packing, every-second page-table copy, and every-second length copy. It still inherits unused packed/padded buffers from the old constructor; reducing resident allocations is separate work. Small B<=4 retains the previous native fallback. This review does not authorize new scale formats, head geometry, tensor strides, or direct integration without same-state model qualification.
