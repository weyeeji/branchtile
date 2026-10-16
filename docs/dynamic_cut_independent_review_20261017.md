# Independent review of dynamic-cut ABI and qualification

Reviewed parent commit `6fa2944` without modifying its source or launching GPU
work. The generated tree `e10856d7e20ea436` matches all 26 hashes in
`results/xqa_raw_dynamic_cut_build_v1.json`. No GPU result existed at review time.

## Valid-input ABI

The binding and wrapper change the penultimate scalar argument to `TensorView`;
the wrapper passes its device address as `uint32_t const*`. Both launch forwarding
paths pass that pointer into the kernel, which loads the current cut before
computing `resumeOffset` and `loopEnd`. The Python shim replaces precisely the
penultimate argument used by both padded prefix and native tail calls. Positive
int32 tile counts have the same bits as the CUDA uint32 view. S, split identity,
tile stride, arithmetic and original final merge remain unchanged.

This supports a fixed graph with a stable cut address whose contents change
**between** replays on the same stream. It does not permit concurrent updates
while any prefix/tail kernel is consuming the scalar. The stream and ownership
contract must survive the eventual engine integration.

## Concrete integration risks

1. `src/xqa_raw_pair_dynamic_cut_runtime.py:19` accepts an externally supplied
   `device_cut` after checking only its device, dtype and scalar shape. It neither
   initializes nor verifies its value against the host `prefix_tokens`. A stale
   or uninitialized shared scalar can therefore make the kernel use a different
   cut from the physical-prefix validation. The current qualifier always creates
   the scalar internally and does not exercise this branch. Before engine use,
   define one owner that initializes/updates the shared scalar and validates the
   host contract, and test wrong/negative/stale initial scalar handling.
2. `change_prefix`, lines33–35, mutates the host prefix and GPU cut before
   inherited `prepare` checks `seq_lens <= max_len`. A rejected over-capacity
   change leaves a partially updated plan. Slicing `table[:, :pages]` also clamps
   to available columns, so an over-wide prefix does not itself fail that slice.
   Validate prefix/table extent and every new length before any mutation. A
   negative-control test should confirm that rejected updates leave cut, host
   prefix and length buffers unchanged before replaying the previous valid state.

These are unsupported-input/shared-owner hazards, not evidence that the current
qualifier's valid-input path fails. No running source was changed.

## What the frozen gate establishes

The synthetic gate covers B6/7/8/9/14/19/21, ten cut changes per fixed graph,
256-token and merely16-token alignment, cut growth/shrinkage, random physical
remapping, changing Q/KV, unequal private lengths up to8192, native int32 length
aliasing, and poisoned old raw state. Full O bits and native AA are compared;
FP64 checks the last row at three selected cuts. This is a useful pointer/graph
plumbing test.

Before engine integration, the smallest missing tests are shared-scalar reuse
across two plans with different B/S, negative update rejection as above, explicit
finite-output checks at every state, and memcheck on capture/replay. The current
script initially captures with length1 but does not explicitly gate that dummy
output. Existing dummy-state evidence helps, but the new pointer path should be
included in its own sanitizer run. Complete FP64 for all rows is useful on one
case; the inherited native comparison is the main full-output gate.

None of these synthetic tests resolves the held-out eval2 token divergence. That
requires the actual first-divergent native state, per-layer Q/O and KV provenance,
raw hidden/logits comparison and native AA under the same state. A new device cut
is a graph-parameter compatibility change, not evidence of restored numerical
replacement equivalence.
