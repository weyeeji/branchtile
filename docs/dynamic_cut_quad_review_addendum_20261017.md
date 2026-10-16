# Dynamic-cut fix and GQA4 quad review addendum

The two API hazards in the preceding dynamic-cut review are repaired by
`c227a18`, before that experiment's GPU launch. An external scalar is checked
against the declared initial tile cut. All prefix/length/physical-page checks and
new length-tensor allocation precede updates. The added negative controls reject
a wrong external value and two over-capacity changes, checking that host prefix,
device cut and lengths remain unchanged. The one-time `.item()` belongs to setup.
Cross-plan shared-scalar reuse remains a useful additional small test.

For `7e21975`, I independently reconstructed the quad mapping from logical row/
column coordinates and the installed `mma.cuh` operand register order. For each
member j, native live rows0..3 map to packed rows4j..4j+3 at the same column. The
required source is lane+16(j mod2), register+2(floor(j/2)); all1024 source slots in
a 32-lane×32-register PV fragment are covered once. No mapped source exceeds
lane31/register31. Scalar max/sum/backward-hint source rows reach at most15.
The accumulator layout follows the documented BF16 m16n8k16 fragment mapping.
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/archive/11.5.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float).

The runtime packs query axes as group, KV head, member, GQA4 head. Its raw-state
request index uses floor(request/4), while the final GQA4 tail keeps original B
and S. Zero dummy requests occupy only unused prefix rows. These source and
coordinate checks found no blocking mapping error. The report is
`results/dynamic_cut_and_quad_cpu_review_v1.json`.

This establishes address/layout consistency, not floating-point replacement
equivalence. Filling formerly unused M16 rows can still expose assumptions in
producer hints, masks, stores and pipeline state. The frozen GPU gate compares
native/AA/PDL-off/unshared/pair/quad, includes two-row FP64, dummy-length capture,
changed Q/lengths, request reordering and poisoned state. It is an appropriate
first operator test. Separate physically different prefixes between groups,
rejected within-group mismatches, broader real activations and actual model
trajectories remain unproved. No GPU work or frozen-source edits were performed
by this reviewer.
