# Qwen3-4B TP2: GQA4 continuation extension

This is a bounded second-geometry qualification, not evidence of second-model end-to-end speedup. Both local `Qwen3-4B-Instruct-2507` and `Qwen3-4B-Thinking-2507` configs specify 32 query heads, 8 KV heads, head dimension128,36 layers,BF16 and no sliding window. TP2 therefore gives16 query/4 KV heads per rank, GQA4. Config hashes are in the CPU audit receipt.

Native `mha.cu` gives GQA4,8 and16 the same M16×N64 warp tile and four PV warps, with two independent PV warp groups. Thus a two-request GQA4→8 prefix can reuse the raw-state continuation interface; the native tail remains GQA4 with original B, Hkv4 and original split count. It leaves eight M16 rows unused. Four-request GQA4→16 could use all rows but is deliberately deferred until this smaller mapping passes.

The existing GQA8→16 mapping is **not** reusable unchanged. In the native m16n8 accumulator, row=`lane//4 + 8*((register%4)//2)` and column=`8*(register//4)+2*(lane%4)+register%2`. GQA4 valid accumulators have lane<16 and register%4<2. Paired member1 lives four rows higher, requiring source lane+16 at the same register. Max,sum and producer backward hints instead use source scalar row+4. All padded native rows are initialized to the original safe max/zero and never stand in for live rows. The independent generator makes these distinctions explicit; a CPU coordinate enumeration checks every live accumulator and scalar row.

The raw state remains `[group,Hkv,S,4,32,35]` and includes every consumer fragment plus four independent producer hints. At SM170 and B8/Hkv4, native S=5 (not the former GQA8 S=10). Prefix forced SM is S×groups×4 and tail forced SM is S×B×4. Full original table capacity determines S, exactly as native. Prefix only traverses complete shared256-token tiles; mixed boundary and all private tokens remain in the original tail sequence.

Independent generated CUDA modules `iscas_xqa_raw_pair_gqa4_g{4,8}_bc64af60df12f914` compiled with GPUs hidden and `torch.cuda.is_initialized()==false`; no site-packages or frozen GQA8 sources changed. See `results/xqa_raw_gqa4_build_v1/audit.json`.

The frozen GPU gate `scripts/qualify_xqa_raw_gqa4.py` uses synthetic independent queries/private KV with physically shared prefixes. Cases B8/P256,B8/P768,B8/P120528,B6/P120528 cover splits without a prefix checkpoint, mixed boundaries and many iterations. It compares int16 output bits to original production PDL-on XQA, checks native PDL-on/off and AA, checks FP64 for first/last request, then replays one graph after Q/length updates, tail growth to8192,dummy length1, state poisoning, physical remapping and KV changes. Native tolerances remain atol.006/rtol.015 for FP64; bitwise parity is mandatory. No timing win can be claimed from this gate.

Command (GPU assigned by root):

```bash
PATH=/home/qcjz/iscas/.venv-engine/bin:$PATH FLASHINFER_CUDA_ARCH_LIST=12.0 CUDA_VISIBLE_DEVICES=6 .venv-engine/bin/python scripts/qualify_xqa_raw_gqa4.py --output results/xqa_raw_gqa4_gpu_v1/qualification.json
```

A failure retains its full receipt and prevents model integration. A pass licenses actual second-model capture and complete-call measurement, not direct reuse of the GQA8 dispatch policy. The changed KV-head count, split count and half-filled prefix tile require their own cost evidence.
