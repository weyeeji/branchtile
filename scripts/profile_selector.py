"""Known-prior selector controls on unchanged natural Q/K captures.

Each row is one layer/question snapshot, not an independent task sample. All
LSE generation, score computation, stable selection, and preparation costs are
identified explicitly. This is not an application or novelty measurement.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
import flashinfer
from selector_baseline import (FusedKnownLSEScorer, qc_scores, reference_scores,
                               scores_from_lse, select_support, streaming_lse)


def load_case(path, layer, question, device):
    start = time.perf_counter()
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    source = torch.load(path / f"source_l{layer}.pt", weights_only=True)
    sel = torch.load(path / f"selector_q{question}_l{layer}.pt", weights_only=True)
    dec = torch.load(path / f"decode_q{question}_l{layer}_s1.pt", weights_only=True)
    q = sel["observation_q"].to(device)
    positions = sel["observation_positions"].to(device)
    # Decode step1 adds its current generated token. Trim exactly at the last
    # observation token to recover suffix-prefill selector full history.
    length = int(sel["observation_positions"][-1]) + 1
    k = torch.cat([source["prefix_k"], source["document_k"], dec["private_suffix_k"]], dim=2)[:, :, :length].contiguous().to(device)
    v = torch.cat([source["prefix_v"], source["document_v"], dec["private_suffix_v"]], dim=2)[:, :, :length].contiguous().to(device)
    torch.cuda.synchronize()
    info = {"capture": str(path), "layer": layer, "question": question,
            "model_revision": meta["model_revision"],
            "document_sha256": meta["document_sha256"],
            "document_range": meta["document_token_range"],
            "q_shape": list(q.shape), "k_shape": list(k.shape),
            "load_cpu_concat_transfer_ms": (time.perf_counter()-start)*1000,
            "load_cost_is_capture_instrumentation_not_runtime_selector_cost": True}
    return q, k, v, positions, sel, info


def bench_graph(fn, repeats=2, blocks=15):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeats):
            out = fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(blocks):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b)*1000/repeats)
    del g, out
    gc.collect()
    torch.cuda.empty_cache()
    return {"median_us": statistics.median(samples), "samples_us": samples,
            "graph_calls": repeats, "blocks": blocks, "cache_regime": "warm"}


def score_error(a, b):
    a, b = a.double(), b.double()
    diff = a-b
    return {"max_abs": float(diff.abs().max()),
            "relative_l2": float(diff.norm()/b.norm().clamp_min(1e-30))}


def mask_error(a, b):
    return {"equal": bool(torch.equal(a, b)),
            "symmetric_difference_per_head": (a != b).sum(-1).tolist(),
            "jaccard_per_head": ((a & b).sum(-1)/(a | b).sum(-1).clamp_min(1)).tolist()}


def case(args, specification):
    case_start = time.perf_counter()
    path, layer, question = specification.rsplit(":", 2)
    q, k, v, positions, saved, info = load_case(path, int(layer), int(question), "cuda")
    ds, de = info["document_range"]
    budget = min(args.budget, de-ds)
    result = {"input": info, "budget": budget, "timing": {}, "qualification": {}}
    torch.cuda.reset_peak_memory_stats()
    lse = streaming_lse(q, k, positions)
    original_scores = qc_scores(q, k, positions, ds, de)
    original_mask = select_support(original_scores, budget)
    fp64_scores, fp64_lse = reference_scores(q, k, positions, ds, de)
    result["qualification"]["fp32_lse_vs_fp64"] = score_error(lse, fp64_lse)
    result["qualification"]["original_scores_vs_fp64"] = score_error(original_scores, fp64_scores)
    result["qualification"]["original_vs_saved_scores"] = score_error(original_scores, saved["pooled_scores"].cuda())
    if args.budget in saved["supports"]:
        result["qualification"]["original_vs_saved_mask"] = mask_error(original_mask, saved["supports"][args.budget].cuda())
    start = time.perf_counter()
    native_topk_mask = select_support(original_scores,budget,"flashinfer")
    torch.cuda.synchronize()
    result["first_native_topk_wall_ms_including_any_jit"] = (time.perf_counter()-start)*1000
    result["qualification"]["native_topk_same_scores"] = mask_error(original_mask, native_topk_mask)
    assert result["qualification"]["native_topk_same_scores"]["equal"], "Native stable-top-k semantics changed"
    preparations, scorers, tunes, first_runs = {}, {}, {}, {}
    for bn in (32, 64, 128):
        start = time.perf_counter()
        scorers[bn] = FusedKnownLSEScorer(q,k,positions,ds,de,bn)
        torch.cuda.synchronize()
        preparations[bn] = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        scores = scorers[bn].run(lse)
        torch.cuda.synchronize()
        first_runs[bn] = (time.perf_counter()-start)*1000
        torch.testing.assert_close(scores.double(), fp64_scores, atol=2e-8, rtol=3e-5)
        tunes[bn] = bench_graph(lambda: scorers[bn].run(lse), blocks=7)
    bn = min(tunes, key=lambda b:tunes[b]["median_us"])
    scorer = scorers[bn]
    result["fused_tuning"] = tunes
    result["fused_block_n"] = bn
    result["prepare_ms"] = preparations
    result["first_fused_call_wall_ms_including_any_jit"] = first_runs
    fused_scores = scorer.run(lse).clone()
    result["qualification"]["fused_scores_vs_fp64"] = score_error(fused_scores,fp64_scores)
    result["qualification"]["fused_vs_original_mask"] = mask_error(
        select_support(fused_scores,budget,"flashinfer"),original_mask)
    result["qualification"]["fp64_vs_original_mask"] = mask_error(
        select_support(fp64_scores,budget),original_mask)
    start = time.perf_counter()
    nq = q[0].transpose(0,1).contiguous()
    nk, nv = k[0], v[0]
    torch.cuda.synchronize()
    result["native_prefill_query_layout_prepare_ms"] = (time.perf_counter()-start)*1000
    def native_lse():
        _, value = flashinfer.single_prefill_with_kv_cache_return_lse(
            nq,nk,nv,causal=True,kv_layout="HND",backend="fa2")
        return (value.transpose(0,1).reshape(k.shape[1],-1) * math.log(2)).contiguous()
    start = time.perf_counter()
    native_denominator = native_lse()
    torch.cuda.synchronize()
    result["first_native_prefill_wall_ms_including_any_jit"] = (time.perf_counter()-start)*1000
    result["qualification"]["native_lse_vs_fp64"] = score_error(native_denominator,fp64_lse)
    native_scores = scorer.run(native_denominator).clone()
    result["qualification"]["native_lse_scores_vs_fp64"] = score_error(native_scores,fp64_scores)
    result["qualification"]["native_lse_vs_original_mask"] = mask_error(
        select_support(native_scores,budget,"flashinfer"),original_mask)
    result["qualification"]["native_lse_vs_fp64_mask"] = mask_error(
        select_support(native_scores,budget,"flashinfer"),select_support(fp64_scores,budget))
    methods = {
        "original_two_pass_argsort": lambda:select_support(qc_scores(q,k,positions,ds,de),budget),
        "two_pass_native_topk": lambda:select_support(qc_scores(q,k,positions,ds,de),budget,"flashinfer"),
        "known_fp32_lse_torch_score_native_topk": lambda:select_support(scores_from_lse(q,k,positions,ds,de,lse),budget,"flashinfer"),
        "known_fp32_lse_fused_score_native_topk": lambda:select_support(scorer.run(lse),budget,"flashinfer"),
        "known_native_lse_fused_score_native_topk": lambda:select_support(scorer.run(native_denominator),budget,"flashinfer"),
        "full_fp32_lse_fused_score_native_topk": lambda:select_support(scorer.run(streaming_lse(q,k,positions)),budget,"flashinfer"),
        "lse_full_history_fp32_only": lambda:streaming_lse(q,k,positions),
        "native_prefill_attention_and_lse": native_lse,
        "known_lse_torch_score_only": lambda:scores_from_lse(q,k,positions,ds,de,lse),
        "known_lse_fused_score_only": lambda:scorer.run(lse),
        "argsort_protect_and_mask_only": lambda:select_support(original_scores,budget),
        "native_topk_protect_and_mask_only": lambda:select_support(original_scores,budget,"flashinfer"),
    }
    order = list(methods)
    random.Random(20261016).shuffle(order)
    for name in order:
        result["timing"][name] = bench_graph(methods[name],blocks=args.blocks)
        print(json.dumps({"case":specification,"method":name,"median_us":result["timing"][name]["median_us"]}),flush=True)
    result["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["case_elapsed_seconds"] = time.perf_counter()-case_start
    result["scope"] = {"known_lse_incremental_arms_exclude_lse_generation":True,
        "full_fp32_arm_charges_complete_history_lse":True,
        "native_prefill_arm_returns_attention_and_lse_not_lse_alone":True,
        "preparation_and_jit_excluded_from_graph_timing_reported_separately":True,
        "pool7_includes_halos_and_zero_padding":True,"protect_after_pool":True,
        "topk_lower_position_boundary_tie":True,"original_capture_selector_is_not_a_serving_baseline":True,
        "application_speedup_or_novelty_claim":False}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case",action="append",required=True,help="capture/path:layer:question")
    p.add_argument("--budget",type=int,default=4096)
    p.add_argument("--blocks",type=int,default=15)
    p.add_argument("--output",required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {"schema":1,"status":"running","uid":os.geteuid(),"virtual_environment":sys.prefix,
              "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch":torch.__version__,"flashinfer":importlib.metadata.version("flashinfer-python"),
              "gpu":torch.cuda.get_device_name(),"start_epoch":time.time(),"cases":[]}
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    for spec in args.case:
        result["cases"].append(case(args,spec))
        output.write_text(json.dumps(result,indent=2)+"\n")
        gc.collect(); torch.cuda.empty_cache()
    result.update(status="completed",end_epoch=time.time())
    output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
