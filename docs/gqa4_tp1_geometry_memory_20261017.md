# Thinking4B TP1 opportunity: geometry and capacity before a probe

CPU-only calculation from the local Qwen3-4B-Thinking-2507 configuration and
weight index. TP1 has 36 layers, Hq32/Hkv8, head128 and GQA4; TP2 has
Hq16/Hkv4 per rank. No imported model, CUDA initialization, kernel patch or
GPU launch was used. Details are in
`results/gqa4_tp1_geometry_memory_analysis_v1.json`.

Keeping the actual native long-context split rule on 170 SMs gives:

| B | TP2 S | TP2 native / quad CTAs | TP1 S | TP1 native / quad CTAs |
|---:|---:|---:|---:|---:|
| 4 | 10 | 160 / 40 | 5 | 160 / 40 |
| 5 | 8 | 160 / 64 | 4 | 160 / 64 |
| 6 | 7 | 168 / 56 | 3 | 144 / 48 |
| 7 | 6 | 168 / 48 | 3 | 168 / 48 |
| 8 | 5 | 160 / 40 | 2 | 128 / 32 |

TP1 therefore does not fix quad's low CTA coverage; B8 coverage is lower.
There are twice as many KV heads per device, and smaller S means longer
per-CTA sequence walks. The useful independent change is removing tensor
parallel communication and changing the model/attention balance on a single
GPU. Neither improvement in speed nor native-output agreement with TP2 can
be inferred from that change. Current GQA4 adapters hardcode Hkv4 and TP2,
so a TP1 extension requires fresh scope/ABI and real-state qualification.

BF16 KV costs 144KiB per token across all 36 layers at TP1, versus 72KiB per
rank at TP2. The 120528-token shared prefix alone is 16.552GiB, and the weight
index reports 7.493GiB of tensor payload. On the existing TP2 run the eight
questions had 315 private prompt tokens and generated 15743 tokens total.
If this same token footprint occurred on TP1, KV would be 18.757GiB and
weights plus KV 26.250GiB. The configured 90% of the observed 31.36GiB device
is about 28.224GiB, leaving only 1.974GiB for activations, CUDA/framework
allocations, graph pools, workspace and page rounding. This is plausible but
not a capacity guarantee. TP1 outputs may have different lengths.

If all eight requests reach the 8192-token cap, weights plus KV alone reach
33.088GiB and exceed physical memory. A real probe must report KV capacity,
preemptions, graph allocation and actual outputs; it must not silently lower
the task's cap or discard long samples to fit. The raw-state arena is not the
main concern: at B8 its prefix state is 573440B at TP1 versus 716800B at TP2,
and the shared workspace remains 64MiB. Native Q/O tensors double per rank
but are directly aliased, rather than copied into 36 extra resident buffers.

Proceed only with the already planned original-task native TP1 probe and
capture. Do not claim a new mechanism or a memory-safe B8 configuration from
these estimates alone.
