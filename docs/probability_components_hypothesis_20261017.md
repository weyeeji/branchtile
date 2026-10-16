# Bounded next hypothesis: retain FP8 V in a two-component PV product

The original-page Q-only result is negative. This separate follow-up tests whether removing the remaining V-to-BF16 conversion can compensate for its extra probability conversion and layout cost. This is one bounded implementation experiment, not a claim of a new numerical method. Old Q-only sources remain unchanged.

## Arithmetic and scale safety

First round the online unnormalized probability P to BF16, matching the control. Let sh=2^-8 and sl=2^-13:

```
ph = E4M3(P_bf16 / sh)
pl = E4M3((P_bf16 - sh * ph) / sl)
```

Maintain a single logical FP32 numerator accumulator A in units sh. At each tile, after the usual online-softmax correction alpha:

```
A = alpha * A
A = MMA(ph, V8, A)
A = 32 * A
A = MMA(pl, V8, A)
A = A / 32
```

The final numerator written to the split workspace is A/256. The denominator retains pre-cast probabilities. Both MMA calls request `max_num_imprecise_acc=0`; this setting is not itself proof of hardware numerical fidelity. Rescaling uses powers of two, but accumulation ordering still changes and must pass independent GPU checks. No second logical M×D accumulator is required by the algorithm; compiler lowering may introduce additional state.

The initially considered sl=2^-14 is unsafe: P=.96875 can produce residual magnitude .03125, which scales to 512 beyond E4M3's finite range. Full CPU enumeration finds eight BF16 P values with out-of-range residuals for that scale. The selected sl=2^-13 has no such overflow for BF16 P in [0,1]. It exactly reconstructs all such P≥2^-15; smaller values remain approximate with maximum absolute error 2^-23. This is not universal exact BF16 representation.

## CPU evidence and cost warning

On the two real native-FP8 captures, rank0/layer15, all 128 query vectors and all prefix tokens, CPU simulation of online P rounding gives additional output relative L2 error of 7.7e-6–1.97e-5 versus BF16-P. A single FP8 P with scale 256 gives .0049–.0102; equal-scale dual components give .000217–.000358. The simulation uses FP64 QK and accumulation to isolate P representation. It does not predict actual tensor-core accumulation error or end-to-end quality.

Offline SM120 compilation passes. The new same-source Q-only control uses 239 registers and the Q+PV variant 242, with no static local memory or stack. Dynamic shared memory increases from 24 KiB to 32 KiB. Thus fewer conversions do **not** imply lower overall resource use; FP8 P/V layout changes may eliminate the benefit. The control's register allocation differs from the prior frozen Q-only source because V conversion is now deferred until the PV expression. Only the new paired control is appropriate for attributing this change.

## Prior work and scope

[SageAttention2](https://proceedings.mlr.press/v267/zhang25ae.html) already quantizes P and V to FP8 and addresses accumulation precision. [P-Cast Precision in FP8 Attention](https://arxiv.org/html/2606.06521v1), June 2026, analyzes scale 256 and reverse traversal, explicitly identifying both as existing FlashAttention engineering choices. Neither static scaling nor FP8 PV is new. Multi-component low-precision products also have earlier precedent. The only hypothesis here is a particular same-accumulator dataflow retaining the preexisting unit-scale FP8 cache and BF16-P rounding boundary; novelty remains unestablished.

The new runner `scripts/qualify_paged_probability_components_gpu.py` retains original page aliases and compares XQA, mature FA2, new Q-only control, and Q+PV. FP64 complete-prefix output and LSE checks gate a fixed M64, 20-block, 10-replay test. QK checks do not qualify PV; complete attention output does. No tile sweep, no model integration before qualification, and no end-to-end claim. The root scheduler assigns GPUs. If the variant fails accuracy or cannot approach the mature same-page FA2 baseline, stop this hypothesis.
