# Independent CPU review of the GQA4 model adapter

Reviewed the frozen second-model adapter introduced by `e4eb8c2`, its
`LayerSharedGQA4RawPairBank`, `RawGQA4PairPlan`, precision probe and offline setup
validator. This is a source-contract review, not a new GPU qualification.
The owner independently reports both TP ranks passed eight actual B8 states;
this review does not replace or broaden that experiment.

No new blocking issue was found in the geometry migration:

- The worker and precision probe require 36 layers, Hq16/Hkv4/D128 and original
  BF16 fused page-size16 storage. Every layer must be present.
- The offline validator checks the new 36-layer required fields, all layer Q/O
  checks, checked/production graph modes and captured-layer coverage. It rejects
  the old 48-layer contract. Source hashes and execution fingerprint, including
  model path, bind a receipt to its model and implementation.
- One workspace and one raw-state allocation are reused serially across all 36
  layers. `run_layer` refreshes **all four** K/V and transposed wrapper views from
  the current layer before calling the plan. Thus wrapper pointers do not remain
  bound to layer0.
- Original actual B8, full table width and Hkv4 determine native S. Prefix and
  tail calls derive their SM parameters from that same S; sharing does not
  silently change the split schedule. Only B8 descriptors are captured/enabled.
- The native int32 GPU length buffer is aliased as uint32; per-step pointer,
  shape and stride checks enforce the existing native metadata contract. The
  first cohort binds actual shared physical pages; later request/slot/overwrite
  epoch guards prevent a newly admitted request from inheriting that binding.
- Q canonicalization, pair packing, pair table/length copies and final O copy
  stay within the candidate call. KV stays in original storage.

The valid scope remains B8, unit scales and one serial execution stream. Smaller
survivor batches fall back to native. An identical head geometry does not let a
Thinking-model checkpoint reuse another model's strict qualification: its model
fingerprint changes, so a fresh qualification is required. The owner already
uses this separate-model protocol.

This review did not edit running sources, import CUDA modules, or launch GPU
work. It does not certify arbitrary prefix reallocation outside the verified
engine's overwrite-epoch contract, concurrent graph execution, or other batches.
