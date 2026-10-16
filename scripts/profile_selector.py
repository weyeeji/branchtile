"""Known-prior selector controls on unchanged natural Q/K captures.

Each row is one layer/question snapshot, not an independent task sample. All
LSE generation, score computation, stable selection, and preparation costs are
identified explicitly. This is not an application or novelty measurement.
"""
from __future__ import annotations

import argparse
import csv
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
                               scores_from_lse, select_support, streaming_lse,
                               protected_count)


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
    prefix_length, shared_document_length = source["prefix_k"].shape[2], source["document_k"].shape[2]
    private_length = dec["private_suffix_k"].shape[2]
    assert length <= prefix_length+shared_document_length+private_length, "Capture omits required full-history K"
    assert torch.equal(sel["observation_positions"],torch.arange(length-q.shape[2],length)), "Native causal LSE requires the final consecutive observation positions"
    ds,de = meta["document_token_range"]
    assert prefix_length == ds
    private_document_tail = de-prefix_length-shared_document_length
    assert 0 <= private_document_tail <= private_length, "Document tail missing from private source"
    k = torch.cat([source["prefix_k"], source["document_k"], dec["private_suffix_k"]], dim=2)[:, :, :length].contiguous().to(device)
    v = torch.cat([source["prefix_v"], source["document_v"], dec["private_suffix_v"]], dim=2)[:, :, :length].contiguous().to(device)
    torch.cuda.synchronize()
    info = {"capture": str(path), "layer": layer, "question": question,
            "model_revision": meta["model_revision"],
            "document_sha256": meta["document_sha256"],
            "document_range": meta["document_token_range"],
            "q_shape": list(q.shape), "k_shape": list(k.shape),
            "tensor_parallel_size":meta.get("tensor_parallel_size"),
            "actual_tp_rank":meta.get("actual_tp_rank"),
            "actual_shared_document_range":meta.get("actual_shared_document_token_range",[ds,de]),
            "private_document_tail_tokens_included":private_document_tail,
            "full_history_tokens":length,
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


def summarize(files, output):
    rows = []
    names = ["original_two_pass_argsort", "two_pass_native_topk",
             "full_fp32_lse_fused_score_native_topk",
             "known_fp32_lse_fused_score_native_topk"]
    for filename in files:
        data = json.loads(Path(filename).read_text())
        assert data["status"] == "completed"
        for c in data["cases"]:
            q = c["qualification"]
            rows.append({"source":filename,"capture":c["input"]["capture"],
                "layer":c["input"]["layer"],"question":c["input"]["question"],
                "document_tokens":c["input"]["document_range"][1]-c["input"]["document_range"][0],
                "private_document_tail_tokens_included":c["input"]["private_document_tail_tokens_included"],
                "native_topk_mask_equal":q["selected_topk_backend_same_scores"]["equal"],
                "fp32_fused_mask_equal":q["fused_vs_original_mask"]["equal"],
                "original_equals_frozen_mask":q["original_vs_saved_mask"]["equal"],
                "native_lse_mask_symmetric_difference":sum(q["native_lse_vs_original_mask"]["symmetric_difference_per_head"]),
                "native_lse_max_abs_error":q["native_lse_vs_fp64"]["max_abs"],
                "fused_prepare_ms":c["prepare_ms"][str(c["fused_block_n"])] if str(c["fused_block_n"]) in c["prepare_ms"] else c["prepare_ms"][c["fused_block_n"]],
                **{name+"_us":c["timing"][name]["median_us"] for name in names}})
    summary = {"status":"completed_exploratory_selector_controls","rows":rows,
        "scope":"Single-question/layer snapshots. Warm graph medians, no paired/A-A inference, no application speedup or novelty claim.",
        "known_lse_arms":"Exclude denominator production; full_fp32_lse arm charges the entire causal history denominator.",
        "strict_native_flags":"FlashInfer0.7.2 deterministic=True,tie_break=1,dsa_graph_safe=True is unsupported on SM120. Original three failures retained.",
        "tested_native_flags":{"deterministic":False,"tie_break":1,"dsa_graph_safe":True,
            "reason":"CUB output index order is irrelevant to bool-mask scatter; lower-index boundary set matches stable argsort on repeated tie fixtures and real scores."},
        "prior_art_not_novelty":{"lse_reuse":"https://aclanthology.org/2025.findings-acl.952/",
                                 "native_topk":"https://docs.flashinfer.ai/api/topk.html"},
        "qualification":"10 semantic/fused-FP64/graph cases plus 3 CUB tie/repeat/graph cases passed; 1 separate test proves strict-flags unsupported.",
        "native_lse_warning":"FA2 BF16 LSE changes one selected token in each of two boundary snapshots. Preserve these differences; do not call native-LSE masks universally identical."}
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2)+"\n")
    with output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(len(rows)); width=.2
    fig, ax = plt.subplots(figsize=(11,4.6))
    labels = ["Two-pass + stable sort", "Two-pass + native top-k",
              "Full FP32 LSE + fused + native top-k", "Existing FP32 LSE + fused + native top-k"]
    for i,(name,label) in enumerate(zip(names,labels)):
        ax.bar(x+(i-1.5)*width,[r[name+"_us"] for r in rows],width,label=label)
    ax.set_xticks(x,[f"{'Main FP8 TP2' if 'main30b' in r['capture'] else 'Old 30B' if 'old30b' in r['capture'] else 'Official 4B'}\nL{r['layer']}, {r['document_tokens']:,} tokens" for r in rows])
    ax.set_yscale("log");ax.set_ylabel("Complete selector GPU latency (microseconds, warm graph)")
    ax.set_title("Known-prior selector controls; denominator availability is explicit")
    ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    ax.legend(fontsize=8,loc="upper left")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"),dpi=200)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def case(args, specification):
    case_start = time.perf_counter()
    backend = args.topk_backend
    path, layer, question = specification.rsplit(":", 2)
    q, k, v, positions, saved, info = load_case(path, int(layer), int(question), "cuda")
    ds, de = info["document_range"]
    budget = min(args.budget, de-ds)
    result = {"input": info, "budget": budget, "timing": {}, "qualification": {}}
    torch.cuda.reset_peak_memory_stats()
    lse = streaming_lse(q, k, positions)
    original_scores = qc_scores(q, k, positions, ds, de)
    original_mask = select_support(original_scores, budget)
    candidate_scores = original_scores.clone()
    candidate_scores[:, :min(4,de-ds)] = -torch.inf
    candidate_scores[:, max(0,de-ds-32):] = -torch.inf
    ranked = candidate_scores.sort(dim=-1, descending=True, stable=True).values
    count = budget-protected_count(de-ds)
    if 0 < count < de-ds-protected_count(de-ds):
        boundary, following = ranked[:,count-1], ranked[:,count]
        result["topk_boundary"] = {"score_per_head":boundary.tolist(),
            "gap_per_head":(boundary-following).tolist(),
            "relative_gap_per_head":((boundary-following)/boundary.abs().clamp_min(1e-30)).tolist()}
    fp64_scores, fp64_lse = reference_scores(q, k, positions, ds, de)
    result["qualification"]["fp32_lse_vs_fp64"] = score_error(lse, fp64_lse)
    result["qualification"]["original_scores_vs_fp64"] = score_error(original_scores, fp64_scores)
    result["qualification"]["original_vs_saved_scores"] = score_error(original_scores, saved["pooled_scores"].cuda())
    if args.budget in saved["supports"]:
        result["qualification"]["original_vs_saved_mask"] = mask_error(original_mask, saved["supports"][args.budget].cuda())
    start = time.perf_counter()
    native_topk_mask = select_support(original_scores,budget,backend)
    torch.cuda.synchronize()
    result["topk_backend"] = backend
    result["first_selected_topk_wall_ms_including_any_jit"] = (time.perf_counter()-start)*1000
    result["qualification"]["selected_topk_backend_same_scores"] = mask_error(original_mask, native_topk_mask)
    result["qualification"]["native_topk_verified"] = backend.startswith("flashinfer")
    result["topk_contract"] = {"sorted":False,"deterministic_index_order":backend=="flashinfer",
        "tie_break":1,"dsa_graph_safe":True,"consumer":"bool support mask, independent of index output order"}
    assert result["qualification"]["selected_topk_backend_same_scores"]["equal"], "Stable-top-k semantics changed"
    preparations, scorers, tunes, first_runs = {}, {}, {}, {}
    for bn in (32, 64, 128):
        start = time.perf_counter()
        scorers[bn] = FusedKnownLSEScorer(q,k,positions,ds,de,bn)
        torch.cuda.synchronize()
        preparations[bn] = (time.perf_counter()-start)*1000
        start = time.perf_counter()
        try:
            scores = scorers[bn].run(lse)
            torch.cuda.synchronize()
            first_runs[bn] = (time.perf_counter()-start)*1000
            torch.testing.assert_close(scores.double(), fp64_scores, atol=2e-8, rtol=3e-5)
            tunes[bn] = dict(qualified=True,**bench_graph(lambda: scorers[bn].run(lse), blocks=7))
        except Exception as error:
            first_runs[bn] = (time.perf_counter()-start)*1000
            tunes[bn] = {"qualified":False,"error_type":type(error).__name__,"error":str(error)}
    eligible = [b for b in tunes if tunes[b]["qualified"]]
    if not eligible:
        raise RuntimeError(f"No fused scorer configuration qualified: {tunes}")
    bn = min(eligible, key=lambda b:tunes[b]["median_us"])
    scorer = scorers[bn]
    result["fused_tuning"] = tunes
    result["fused_block_n"] = bn
    result["fused_layout"] = "BN physical columns with overlapping pool halos; emits BN-6 scores per CTA"
    result["prepare_ms"] = preparations
    result["first_fused_call_wall_ms_including_any_jit"] = first_runs
    fused_scores = scorer.run(lse).clone()
    result["qualification"]["fused_scores_vs_fp64"] = score_error(fused_scores,fp64_scores)
    result["qualification"]["fused_vs_original_mask"] = mask_error(
        select_support(fused_scores,budget,backend),original_mask)
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
        select_support(native_scores,budget,backend),original_mask)
    result["qualification"]["native_lse_vs_fp64_mask"] = mask_error(
        select_support(native_scores,budget,backend),select_support(fp64_scores,budget))
    methods = {
        "original_two_pass_argsort": lambda:select_support(qc_scores(q,k,positions,ds,de),budget),
        "two_pass_native_topk": lambda:select_support(qc_scores(q,k,positions,ds,de),budget,backend),
        "known_fp32_lse_torch_score_native_topk": lambda:select_support(scores_from_lse(q,k,positions,ds,de,lse),budget,backend),
        "known_fp32_lse_fused_score_native_topk": lambda:select_support(scorer.run(lse),budget,backend),
        "known_native_lse_fused_score_native_topk": lambda:select_support(scorer.run(native_denominator),budget,backend),
        "full_fp32_lse_fused_score_native_topk": lambda:select_support(scorer.run(streaming_lse(q,k,positions)),budget,backend),
        "lse_full_history_fp32_only": lambda:streaming_lse(q,k,positions),
        "native_prefill_attention_and_lse": native_lse,
        "known_lse_torch_score_only": lambda:scores_from_lse(q,k,positions,ds,de,lse),
        "known_lse_fused_score_only": lambda:scorer.run(lse),
        "argsort_protect_and_mask_only": lambda:select_support(original_scores,budget),
        "native_topk_protect_and_mask_only": lambda:select_support(original_scores,budget,backend),
    }
    if backend == "argsort":
        del methods["two_pass_native_topk"]
        del methods["native_topk_protect_and_mask_only"]
        methods = {name.replace("native_topk","torch_argsort"):fn for name,fn in methods.items()}
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
        "debug_assertions_and_host_device_barriers_excluded_from_all_graph_arms":True,
        "application_speedup_or_novelty_claim":False}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case",action="append",help="capture/path:layer:question")
    p.add_argument("--prepare-native-only",action="store_true",help="CPU-only ahead-of-time build with explicit FLASHINFER_CUDA_ARCH_LIST")
    p.add_argument("--summarize",nargs="+",help="Completed result JSON files; writes JSON/CSV/PNG/PDF without GPU work")
    p.add_argument("--budget",type=int,default=4096)
    p.add_argument("--blocks",type=int,default=15)
    p.add_argument("--topk-backend",choices=["argsort","flashinfer","flashinfer_set"],default="flashinfer")
    p.add_argument("--output",required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(4)
    if args.summarize:
        summarize(args.summarize,args.output)
        return
    if args.prepare_native_only:
        assert not torch.cuda.is_available(), "Preparation must not reserve a GPU"
        assert os.environ.get("FLASHINFER_CUDA_ARCH_LIST"), "Explicit target architecture required"
        start = time.time()
        output = Path(args.output)
        output.parent.mkdir(parents=True,exist_ok=True)
        record = {"status":"running","uid":os.geteuid(),"pid":os.getpid(),
                  "virtual_environment":sys.prefix,"start_epoch":start,
                  "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "target_arch":os.environ["FLASHINFER_CUDA_ARCH_LIST"],
                  "allocated_gpus":[],"gpu_hours":0,
                  "scope":"CPU-only unmodified native FlashInfer top-k preparation"}
        output.write_text(json.dumps(record,indent=2)+"\n")
        from flashinfer.topk import get_topk_module
        get_topk_module()
        record.update(status="completed",end_epoch=time.time(),elapsed_seconds=time.time()-start)
        output.write_text(json.dumps(record,indent=2)+"\n")
        return
    if not args.case:
        p.error("--case is required unless --prepare-native-only is used")
    torch.backends.cuda.matmul.allow_tf32 = False
    result = {"schema":1,"status":"running","uid":os.geteuid(),"virtual_environment":sys.prefix,
              "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "torch":torch.__version__,"flashinfer":importlib.metadata.version("flashinfer-python"),
              "gpu":torch.cuda.get_device_name(),"start_epoch":time.time(),"cases":[],
              "measurement_design":"Exploratory warm CUDA-graph medians; method order shuffled once; no paired-bootstrap or A/A confidence claim",
              "prior_art":{"lse_reuse":"https://aclanthology.org/2025.findings-acl.952/",
                           "native_topk":"https://docs.flashinfer.ai/api/topk.html"}}
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    for spec in args.case:
        try:
            result["cases"].append(case(args,spec))
        except Exception as error:
            result.update(status="failed",failed_case=spec,
                          error_type=type(error).__name__,error=str(error),end_epoch=time.time())
            output.write_text(json.dumps(result,indent=2)+"\n")
            raise
        output.write_text(json.dumps(result,indent=2)+"\n")
        gc.collect(); torch.cuda.empty_cache()
    result.update(status="completed",end_epoch=time.time())
    output.write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
