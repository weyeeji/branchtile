"""Bounded executable E1/E4 oracle search with actual planning and merge costs.

Every partition/tree is executed, not ranked only by CPU tile estimates. The
per-capture winner is a hindsight tuning oracle, not a free deployed selector.
Cold/hot selections are independent. Planning/search wall costs are retained.
"""
import argparse
import gc
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from branchtile.attention import reference, synthetic_owners
from partition_baseline import all_partitions, NativePartitionAttention, OwnPartitionAttention
from cascade_baseline import binary_hierarchies, ExactCascadeAttention
from flashinfer_baseline import NativeSparseDecode
from compare_attention import load_capture, graph_of, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--budget", type=int, default=4096)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--pattern", default="cross")
    p.add_argument("--gqa", type=int, default=8)
    p.add_argument("--blocks", type=int, default=30)
    p.add_argument("--max-tuning-seconds", type=int, default=900)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    if args.capture:
        q, k, v, owners, b, g, provenance = load_capture(args.capture, args.layer, args.budget, args.step, args.heads)
    else:
        b, g = 4, args.gqa
        owners = synthetic_owners(args.heads, args.tokens, args.pattern, b, 20261016)
        q = torch.randn(args.heads, b * g, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(args.heads, args.tokens, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        provenance = {"kind": "synthetic", "pattern": args.pattern, "seed": 20261016}
    ref, ref_lse = reference(q, k, v, owners, b, g)
    supports = [[torch.where((owners[h] & (1 << request)) != 0)[0] for request in range(b)] for h in range(k.shape[0])]
    full = (tuple(range(b)),)
    configurations = []
    for partition in all_partitions(b):
        configurations.extend({"method": "e1_native", "partition": partition, "fixed_split_size": split}
                              for split in (None, 128, 256, 512, 1024))
        configurations.extend({"method": "e1_own", "partition": partition, "splits": split, "block_n": bn}
                              for split, bn in itertools.product((32, 64, 128), (32, 64)))
    configurations.extend({"method": "e4_native", "tree": tree, "fixed_split_size": split}
                          for tree in binary_hierarchies(tuple(range(b))) for split in (None, 128, 256, 512, 1024))
    configurations.extend({"method": "e0_native", "tensor_cores": tc, "fixed_split_size": split}
                          for tc in (False, True) for split in ((None,) if not tc else (None, 128, 256, 512, 1024)))
    rng = random.Random(20261016)
    rng.shuffle(configurations)

    def build(config):
        method = config["method"]
        if method == "e0_native":
            return NativeSparseDecode(q, k, v, supports, use_tensor_cores=config["tensor_cores"],
                                      fixed_split_size=config["fixed_split_size"], normalize_lse=True)
        if method == "e1_native":
            return NativePartitionAttention(q, k, v, owners, b, config["partition"], fixed_split_size=config["fixed_split_size"])
        if method == "e1_own":
            return OwnPartitionAttention(q, k, v, owners, b, config["partition"], splits=config["splits"], block_n=config["block_n"])
        return ExactCascadeAttention(q, k, v, owners, b, config["tree"], fixed_split_size=config["fixed_split_size"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = out.with_suffix(".tuning.jsonl")
    progress.write_text("")
    pdev = torch.cuda.get_device_properties(0)
    flush = torch.zeros(max(2 * pdev.L2_cache_size, 256 * 1024 * 1024) // 4, device="cuda")
    best = {cache: {} for cache in ("hot", "cold")}
    tuning, family_search_seconds = [], {}
    search_started = time.perf_counter()
    for config in configurations:
        if time.perf_counter() - search_started > args.max_tuning_seconds:
            break
        row = dict(config)
        config_start = time.perf_counter()
        try:
            op = build(config)
            actual, lse = op.run()
            torch.cuda.synchronize()
            row.update(initialization_ms_including_jit=(time.perf_counter() - config_start) * 1000,
                       max_abs_error_fp64=float((actual.double() - ref).abs().max()),
                       max_lse_error_fp64=float((lse.double() - ref_lse).nan_to_num().abs().max()))
            torch.testing.assert_close(actual.float(), ref.float(), atol=.006, rtol=.015)
            torch.testing.assert_close(lse, ref_lse.float(), atol=.0003, rtol=.0001)
            if hasattr(op, "payload_aliases_source"):
                assert op.payload_aliases_source
            hot = graph_of(op.run, 20)
            hot_us = statistics.median(elapsed(hot, 20) for _ in range(5))
            cold = graph_of(op.run, 1)
            cold_us = statistics.median(elapsed(cold, 1, flush) for _ in range(5))
            row.update(status="ok", tuning_us_by_cache={"hot": hot_us, "cold": cold_us},
                       prepare_ms=getattr(op, "prepare_ms", getattr(op, "plan_ms", None)),
                       metadata_ms=getattr(op, "metadata_ms", None),
                       logical_kv_token_visits=getattr(op, "logical_kv_token_visits", None),
                       workspace_bytes=getattr(op, "workspace_bytes", op.workspace.numel() if hasattr(op, "workspace") and op.workspace is not None else None))
            targets = [config["method"]]
            if config.get("partition") == full:
                targets.append("full_union")
            for cache in best:
                for target in targets:
                    if target not in best[cache] or row["tuning_us_by_cache"][cache] < best[cache][target]["record"]["tuning_us_by_cache"][cache]:
                        best[cache][target] = {"record": row.copy(), "config": config, "op": op}
            del hot, cold
        except Exception as exc:
            row.update(status="error", error=repr(exc))
        row["configuration_wall_seconds"] = time.perf_counter() - config_start
        family_search_seconds[config["method"]] = family_search_seconds.get(config["method"], 0) + row["configuration_wall_seconds"]
        tuning.append(row)
        with progress.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    search_seconds = time.perf_counter() - search_started
    required = ("e0_native", "full_union", "e1_native", "e1_own", "e4_native")
    missing = {cache: [method for method in required if method not in best[cache]] for cache in best}
    measurements, summary, planning = [], {}, {}
    if not any(missing.values()):
        for cache in best:
            # E1 includes the singleton partition's optimized decode alternative.
            best[cache]["best_e1"] = min((best[cache][m] for m in ("e0_native", "e1_native", "e1_own")),
                                           key=lambda x: x["record"]["tuning_us_by_cache"][cache])
        methods = (*required, "best_e1")
        for cache, repetitions in (("hot", 20), ("cold", 1)):
            graphs = {method: graph_of(best[cache][method]["op"].run, repetitions) for method in methods}
            graphs["full_union_AA"] = graph_of(best[cache]["full_union"]["op"].run, repetitions)
            for block in range(args.blocks):
                order = list(graphs)
                rng.shuffle(order)
                for direction, sequence in (("random", order), ("reverse", list(reversed(order)))):
                    measurements.append({"cache": cache, "block": block, "direction": direction, "order": sequence,
                                         "us": {m: elapsed(graphs[m], repetitions, flush if cache == "cold" else None) for m in sequence}})
            rows = [r for r in measurements if r["cache"] == cache]
            summary[cache] = {"median_us": {m: statistics.median(r["us"][m] for r in rows) for m in graphs}, "paired_ratios": {}}
            draws = np.random.default_rng(71).integers(0, args.blocks, (10000, args.blocks))
            for numerator, denominator in (("full_union", "best_e1"), ("full_union", "e4_native"),
                                           ("best_e1", "e4_native"), ("full_union", "full_union_AA")):
                values = np.array([[next(r for r in rows if r["block"] == block and r["direction"] == direction)["us"][numerator] /
                                    next(r for r in rows if r["block"] == block and r["direction"] == direction)["us"][denominator]
                                    for direction in ("random", "reverse")] for block in range(args.blocks)])
                boot = np.median(values[draws], axis=(1, 2))
                summary[cache]["paired_ratios"][f"{numerator}_over_{denominator}"] = {
                    "median": float(np.median(values)), "paired_block_bootstrap95": np.quantile(boot, [.025, .975]).tolist()}
            del graphs
            planning[cache] = {}
            for method in ("full_union", "best_e1", "e4_native"):
                samples = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    fresh = build(best[cache][method]["config"])
                    fresh.run()
                    torch.cuda.synchronize()
                    samples.append({"warm_prepare_ms": getattr(fresh, "prepare_ms", getattr(fresh, "plan_ms", None)),
                                    "prepare_plus_one_call_host_wall_us": (time.perf_counter() - started) * 1e6})
                    del fresh
                prepare_us = statistics.median(s["warm_prepare_ms"] for s in samples) * 1000
                planning[cache][method] = {"samples": samples, "warm_prepare_median_us": prepare_us,
                    "amortized_prepare_plus_gpu_call_us": {str(reuse): prepare_us / reuse + summary[cache]["median_us"][method] for reuse in (1, 16, 64, 256)},
                    "note": "Static-plan amortization scenarios, not measured request lifetimes; oracle search cost is reported separately and is not free."}
    result = {"scope": "Executable E1 exhaustive fixed partitions and E4 binary-intersection mechanism port; not a full QUILT reproduction or application claim",
              "uid": os.geteuid(), "python": sys.executable, "gpu_uuid": str(pdev.uuid),
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "flashinfer-python", "tilelang", "triton")},
              "q_shape": list(q.shape), "k_shape": list(k.shape), "provenance": provenance,
              "numerical_contract": {"output_atol": .006, "output_rtol": .015, "lse_atol": .0003, "lse_rtol": .0001, "lse_base": "natural", "reference": "FP64"},
              "tuning_planned": len(configurations), "tuning_completed": len(tuning), "tuning_complete": len(tuning) == len(configurations),
              "partitions": all_partitions(b), "hierarchies": binary_hierarchies(tuple(range(b))),
              "search_seconds_including_jit_and_failures": search_seconds, "search_seconds_by_family": family_search_seconds,
              "oracle_scope": "Per-capture hindsight latency oracle among the measured grid; not a deployed plan selector.",
              "tuning": tuning, "missing_qualified_methods": missing,
              "selected": {cache: {method: {"config": x["config"], "record": x["record"]} for method, x in methods.items()} for cache, methods in best.items()},
              "measurements": measurements, "summary": summary, "planning": planning,
              "limitations": ["Hierarchical split partial output is native BF16 and merged with FP32 log2-LSE weights; every tested configuration is separately qualified.",
                              "E4 uses exact binary-tree intersection/residual decomposition with all nodes batched, not the original QUILT code/planner.",
                              "Only shared-source KV is zero-copy; query gathering, normalized output restoration and hierarchical merge are included in run().",
                              "Plan preparation and actual prepare+one-run wall time are measured; search is reported separately, not silently amortized away.",
                              "Same-capture hot/cold controls do not establish common-ready overlap, generation quality, full-model speedup or energy improvement."]}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tuning_complete": result["tuning_complete"], "summary": summary, "missing": missing}), flush=True)


if __name__ == "__main__":
    main()
