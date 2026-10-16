"""Tune then independently measure exact same-mask complete attention calls.

Randomized paired blocks and reverse order, A/A controls, and explicit cache
regimes. A cold block flushes >2x reported L2 before each measured call; flush
time is outside the event interval. These are operator results, not application
latency, quality, or concurrent-ready measurements.
"""
import argparse
import collections
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from branchtile.attention import make_plan, reference, synthetic_owners
from branchtile.attention_v2 import Attention
from branchtile.union import FixedUnion
from branchtile.resources import kernel_resources
from flashinfer_baseline import NativeSparseDecode


def load_capture(path, layer, budget, step, head_count):
    path = Path(path)
    meta = json.loads((path / "metadata.json").read_text())
    src = torch.load(path / f"source_l{layer}.pt", weights_only=True)
    masks = torch.load(path / f"masks_l{layer}_b{budget}.pt", weights_only=True)["supports"]
    active = [i for i in range(meta["B"]) if (path / f"decode_q{i}_l{layer}_s{step}.pt").exists()]
    assert active, "No naturally reached capture at this step"
    h = min(head_count, src["document_k"].shape[1])
    g = meta["gqa"]
    b = len(active)
    keys = [src["prefix_k"][0, :h], src["document_k"][0, :h]]
    values = [src["prefix_v"][0, :h], src["document_v"][0, :h]]
    owner_parts = [torch.full((h, keys[0].shape[1]), (1 << b)-1, dtype=torch.int64),
                   sum(masks[qi, :h].long() * (1 << ri) for ri, qi in enumerate(active))]
    queries = []
    for ri, qi in enumerate(active):
        dec = torch.load(path / f"decode_q{qi}_l{layer}_s{step}.pt", weights_only=True)
        queries.append(dec["query"][0, :h*g, 0].reshape(h, g, 128))
        keys.append(dec["private_suffix_k"][0, :h])
        values.append(dec["private_suffix_v"][0, :h])
        owner_parts.append(torch.full((h, keys[-1].shape[1]), 1 << ri, dtype=torch.int64))
    q = torch.stack(queries, dim=1).flatten(1, 2).contiguous().cuda()
    k, v = [torch.cat(parts, dim=1).contiguous().cuda() for parts in (keys, values)]
    owners = torch.cat(owner_parts, dim=1)
    provenance = {"capture": str(path), "layer": layer, "budget": budget,
                  "step": step, "original_active_requests": active,
                  "selected_kv_heads": list(range(h)), "model": meta["model_path"],
                  "model_revision": meta["model_revision"],
                  "document_sha256": meta["document_sha256"],
                  "includes_system_prefix_and_complete_private_suffix": True,
                  "ready_lifetime_measured": False}
    return q, k, v, owners, b, g, provenance


def graph_of(fn, repetitions=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            fn()
    warm_until = time.monotonic() + 0.03
    while time.monotonic() < warm_until:
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    return graph


def elapsed(graph, repetitions, flush=None):
    if flush is not None:
        flush.add_(1)
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    graph.replay()
    t1.record()
    t1.synchronize()
    return t0.elapsed_time(t1) * 1000 / repetitions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--pattern", default="cross")
    p.add_argument("--gqa", type=int, default=8)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--capture")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--budget", type=int, default=4096)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--splits", default="16,32,64,128")
    p.add_argument("--threads", default="128,256")
    p.add_argument("--blocks", type=int, default=30)
    p.add_argument("--include-union", action="store_true", help="Also tune fixed rows across owner-region boundaries")
    p.add_argument("--normalize-native-lse", action="store_true", help="Include native log2-to-ln conversion in timed complete call")
    p.add_argument("--include-native-raw", action="store_true", help="Also time the original log2-LSE native API separately")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    rng = random.Random(61)
    if args.capture:
        q, k, v, owners, b, g, provenance = load_capture(args.capture, args.layer, args.budget, args.step, args.heads)
    else:
        b, g, h, n = 4, args.gqa, args.heads, args.tokens
        owners = synthetic_owners(h, n, args.pattern, b, 20261016)
        q = torch.randn(h, b*g, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(h, n, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        provenance = {"kind": "synthetic", "pattern": args.pattern, "seed": 20261016}
    plan = make_plan(owners, b, g, "cuda")
    supports = [[torch.where((owners[hi] & (1 << bi)) != 0)[0] for bi in range(b)] for hi in range(k.shape[0])]
    ref, ref_lse = reference(q, k, v, owners, b, g)
    pdev = torch.cuda.get_device_properties(0)
    flush = torch.zeros(max(2*pdev.L2_cache_size, 256*1024*1024)//4, device="cuda", dtype=torch.float32)
    tuning, best = [], {"hot": {}, "cold": {}}
    tile_methods = ("compact", "fixed", "union") if args.include_union else ("compact", "fixed")
    native_methods = ("native", "native_raw") if args.include_native_raw else ("native",)
    comparison_methods = (*tile_methods, *native_methods)
    configs = [(method, bn, split, threads) for method in tile_methods
               for bn, split, threads in itertools.product((32, 64), map(int, args.splits.split(",")), map(int, args.threads.split(",")))]
    configs += [(method, 0, split, int(tc)) for method in native_methods for tc in (False, True)
                for split in ((None, 128, 256, 512, 1024) if tc else (None,))]
    rng.shuffle(configs)
    for method, bn, split, threads in configs:
        record = {"method": method, "block_n": bn, "split": split, "threads_or_tensor_cores": threads}
        try:
            start = time.monotonic()
            if method in native_methods:
                normalize_lse = args.normalize_native_lse and method != "native_raw"
                op = NativeSparseDecode(q, k, v, supports, use_tensor_cores=bool(threads), fixed_split_size=split,
                                        normalize_lse=normalize_lse)
                fn = op.run
                lse_scale = 1 if normalize_lse else math.log(2)
            else:
                op = (FixedUnion(plan, owners, split, bn, threads) if method == "union" else
                      Attention(plan, split, bn, threads, compact=(method == "compact")))
                fn = lambda op=op: op(q, k, v)
                lse_scale = 1
            out, lse = fn()
            torch.cuda.synchronize()
            record["initialization_seconds_including_jit"] = time.monotonic()-start
            record["max_abs_error_fp64"] = float((out.double()-ref).abs().max())
            record["max_relative_error_fp64_denominator_floor_1e_minus8"] = float(((out.double()-ref).abs()/ref.abs().clamp_min(1e-8)).max())
            record["relative_l2_error_fp64"] = float(torch.linalg.vector_norm(out.double()-ref)/torch.linalg.vector_norm(ref).clamp_min(1e-12))
            record["max_abs_error_vs_rounded_bf16_reference"] = float((out.double()-ref.to(torch.bfloat16).double()).abs().max())
            record["max_lse_error_fp64"] = float((lse.double()*lse_scale-ref_lse).nan_to_num().abs().max())
            torch.testing.assert_close(out.float(), ref.float(), atol=0.006, rtol=0.015)
            torch.testing.assert_close((lse*lse_scale).float(), ref_lse.float(), atol=0.0003, rtol=0.0001)
            graph = graph_of(fn)
            times = [elapsed(graph, 20) for _ in range(5)]
            record.update(status="ok", tuning_us=statistics.median(times))
            cold_graph = graph_of(fn, 1)
            cold_times = [elapsed(cold_graph, 1, flush) for _ in range(5)]
            record["tuning_us_by_cache"] = {"hot": record["tuning_us"], "cold": statistics.median(cold_times)}
            record["workspace_bytes"] = op.workspace.numel() if method in native_methods else sum(t.numel()*t.element_size() for t in (op.u,op.m,op.l,op.output,op.lse))
            for regime in best:
                if method not in best[regime] or record["tuning_us_by_cache"][regime] < best[regime][method][0]["tuning_us_by_cache"][regime]:
                    best[regime][method] = (record.copy(), op, fn)
        except Exception as exc:
            record.update(status="error", error=repr(exc))
        tuning.append(record)
        print(json.dumps(record), flush=True)
    assert all(m in best[c] for c in best for m in comparison_methods), "A comparison arm failed qualification"
    measurements = []
    for regime, repetitions in (("hot", 20), ("cold", 1)):
        graphs = {m: graph_of(x[2], repetitions) for m, x in best[regime].items()}
        graphs["fixed_AA"] = graph_of(best[regime]["fixed"][2], repetitions)
        for block in range(args.blocks):
            order = list(graphs)
            rng.shuffle(order)
            for direction, methods in (("random", order), ("reverse", list(reversed(order)))):
                row = {"cache": regime, "block": block, "direction": direction, "order": methods}
                row["us"] = {m: elapsed(graphs[m], repetitions, flush if regime=="cold" else None) for m in methods}
                measurements.append(row)
    summary = {}
    brng = np.random.default_rng(71)
    for regime in ("hot", "cold"):
        rows = [x for x in measurements if x["cache"] == regime]
        # Resample paired blocks, keeping random/reverse measurements clustered.
        ratios = {m: np.array([[next(x for x in rows if x["block"]==i and x["direction"]==d)["us"][m] /
                               next(x for x in rows if x["block"]==i and x["direction"]==d)["us"]["compact"]
                               for d in ("random", "reverse")] for i in range(args.blocks)])
                  for m in comparison_methods if m != "compact"}
        summary[regime] = {"median_us": {m: statistics.median(x["us"][m] for x in rows) for m in best[regime]},
                           "fixed_AA_median_ratio": statistics.median(x["us"]["fixed"]/x["us"]["fixed_AA"] for x in rows)}
        for m, r in ratios.items():
            boot = np.median(r[brng.integers(0, args.blocks, size=(10000,args.blocks))], axis=(1,2))
            summary[regime][m+"_over_compact"] = {"median": float(np.median(r)), "paired_block_bootstrap95": np.quantile(boot,[0.025,0.975]).tolist()}
    result = {"protocol_version": 2, "methods": comparison_methods, "scope": "same-mask operator replay; no application or quality claim", "uid": os.geteuid(),
              "packages": {p: importlib.metadata.version(p) for p in ("torch", "triton", "tilelang", "flashinfer-python", "numpy")},
              "native_lse_contract": "natural log; conversion timed" if args.normalize_native_lse else "native log2; conversion only in validation",
              "native_raw_lse_contract": "native log2; conversion only in validation" if args.include_native_raw else None,
              "python": sys.executable, "gpu_uuid": str(pdev.uuid), "q_shape": list(q.shape), "k_shape": list(k.shape),
              "provenance": provenance, "plan": plan.statistics(), "source_bytes": k.numel()*k.element_size()*2,
              "l2_bytes": pdev.L2_cache_size, "tuning": tuning,
              "selected": {c:{m:x[0] for m,x in best[c].items()} for c in best}, "measurements": measurements, "summary": summary,
              "limitations": ["No fixed-partition or hierarchy native control yet", "Same synthetic inputs repeated; cold flush is a cache control, not natural rotation", "Plan preparation reported separately, not amortized into run times"]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
