# Frozen policy wrapper for the general model adapter

Entry: `scripts.vllm_native_raw_pair_general_policy_worker.NativeRawPairGeneralPolicyWorker`.
The prior general worker remains frozen and unchanged. RPCs, buffer bank, native
GPU length aliases, and strict qualification checks are identical.

The production dispatch rule comes from the frozen development fit:
B >= 6, P >= 65536, and (max actual length - P) * 8 <= P. P is the declared
16-token-aligned common prefix used by that development grid; the actually
shared complete native tiles floor(P/256)*256 are reported separately. Using
the rounded tile prefix as the fitted variable would silently change the rule.
All 160 development cell decisions match the fitter exactly (77 admitted);
four integer threshold boundary checks also passed on CPU.

Strict qualification may test legal shapes outside this performance rule, up
to eight real states per naturally observed enabled B. Each prepare record says
whether the state satisfies the production policy. Candidate performance phases
require both the frozen policy and previously passed model qualification for
that B. An empty qualified-batch set is allowed as an all-native phase; no
candidate graph can execute in that case. Short-prefix documents are valid
native-fallback inputs, not malformed or grounds to change the heldout manifest.

The worker records the policy JSON SHA and parameters. No runtime timing or
future-output length participates in the decision. This remains a development
policy, not a heldout/E2E performance result. The separate dummy-length CUDA
capture gate must pass before full model use; parent scheduling records it.
