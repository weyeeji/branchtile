# Independent raw XQA self-resume audit, 2026-10-17

The frozen self-resume implementation passes its five reported GPU cases. Independent CPU review found no missing live state in the declared BF16/GQA8/head128/SM120 contract. This supports the next bounded pair-mapping qualification; it establishes no sharing performance or whole-model result.

Evidence: `results/xqa_raw_selfresume_gpu_v1/qualification.json`; independent auditor `scripts/audit_xqa_raw_selfresume.py`; audit output `results/xqa_raw_selfresume_audit_v1/audit.json`. All two runner/builder hashes, original `mha.cu` hash and 26 generated-source hashes match. The current shared-library SHA is separately recorded for artifact identity.

## State and control flow

Each original CTA exports four PV warps × 32 lanes × 35 FP32 values. Slots 0–31 preserve the original `WarpAcc` register fragment, slot 32 the consumer's row maximum, slot 33 its row sum, and slot 34 the particular producer warp's shared-memory maximum hint. This retains unreduced PV fragments and four independent hint arrays. At B8/Hkv2/S10, storage is 2,867,200 bytes.

The producer waits for the final X tile to be consumed before reading its hint. Consumers export their own accumulator/statistic fields and return before output normalization or split scratch/merge. Producer and consumer writes occupy disjoint fields. The next kernel launch provides the inter-kernel dependency; candidate runtime PDL is disabled. Resume reconstructs barriers but restores the original value fields, continuing each split at its first original native tile on or after the checkpoint.

The continuation formula is correct for the original strided tile sequence. Exhaustive CPU enumeration checks 237,615 combinations covering S=1/2/3/10/14/21/42, lengths through 69 tiles and checkpoint boundaries both inside and beyond the sequence. Prefix tiles followed by resumed tiles equal the original ordered sequence with neither omission nor duplication. This is an index proof over enumerated cases, not a GPU arithmetic proof.

Changing the final normalization condition to use `originalSeqIterInit` is necessary: a split that completed all its work before the checkpoint still must normalize and participate in the original final merge. Preserving the original final merge also retains its BF16 partial-output casts.

## Pair mapping obligations

For native M16 MMA fragments, a lane owns row `lane/4 + 8*i` and column `8*n + 2*(lane%4) + j`. The native GQA8 valid rows use `i=0`. Packing a second request into the upper eight rows means its native accumulator indices `4*n+j` must read paired indices `4*n+2+j` from the same lane and warp. This differs from the row-scalar/hint mapping: valid native lanes 0–7 read paired lanes 8–15 for the second member. Preserve the PV warp identity and its head-dimension slice; do not combine the two warp groups.

An independent symbolic CPU check exhausts 4,096 active accumulator coordinates for both pair members, four PV warps and their row/column slots with no missing or duplicate logical coordinates. It verifies the proposed mapping, not the future generated pair implementation. Exact GPU qualification is still required.

## Coverage gaps to close in the next independent runner

- Existing `torch.equal` comparisons operate on BF16 values, so +0 and -0 compare equal. Use `.view(torch.int16)` for the next strict bitwise gate. This does not imply an observed signed-zero mismatch in the frozen result.
- All five GPU cases use B8/S10 and roughly 4K histories, with at most two sequence iterations per split in the nonempty-checkpoint cases (zero-checkpoint cases reach three). Prefix 16 is rounded to the same zero-tile checkpoint as prefix 0. Add prefix 256 or 768 to exercise a mixture of empty and nonempty checkpoints, and actual 120K history for many hint updates.
- The graph replay uses unchanged inputs once. Mutate Q, physical page bindings and unequal lengths at the same addresses, then compare to fresh native output. This catches stale state and metadata dependency errors.
- Existing cases include unequal tails, checkpoints on a 256-token boundary, a prefix only 16-token aligned and splits with no remaining tail. Keep these in the pair path, which introduces a different mapping and shared/private address contract. Add an all-shared/no-tail case and the first private tile crossing a 256-token boundary if not covered there.
- Runtime candidate PDL is disabled. Native PDL-on/off matches in the five cases, but candidate PDL-on is unqualified. Retain the qualified mode until separately tested.

The generated-module API itself does not validate arbitrary caller dtype, shape, phase or state capacity. Future integration must enforce the declared contract before launch. The current synthetic runner supplies valid arguments.
