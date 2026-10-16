"""Measure warm planning of every selected backend and replay on a second GPU.

This avoids attributing an advantage to E4 merely because a steady-state-fast
own-union control has slower Python preparation than native full union/decode.
"""
import argparse
import hashlib
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
from partition_baseline import NativePartitionAttention, OwnPartitionAttention
from cascade_baseline import ExactCascadeAttention
from flashinfer_baseline import NativeSparseDecode
from compare_attention import load_capture, graph_of, elapsed


def measure(path):
    data = json.loads(path.read_text())
    provenance = data["provenance"]
    h, rows, _ = data["q_shape"]
    torch.manual_seed(20261016)
    if "capture" in provenance:
        q, k, v, owners, b, g, _ = load_capture(provenance["capture"], provenance["layer"], provenance["budget"], provenance["step"], h)
    else:
        b, g, n = 4, rows // 4, data["k_shape"][1]
        owners = synthetic_owners(h, n, provenance["pattern"], b, 20261016)
        q = torch.randn(h, rows, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(h, n, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
    supports = [[torch.where((owners[head] & (1 << request)) != 0)[0] for request in range(b)] for head in range(h)]
    ref, ref_lse = reference(q, k, v, owners, b, g)

    def build(config):
        if config["method"] == "e0_native":
            return NativeSparseDecode(q, k, v, supports, use_tensor_cores=config["tensor_cores"], fixed_split_size=config["fixed_split_size"], normalize_lse=True)
        if config["method"] == "e1_native":
            return NativePartitionAttention(q, k, v, owners, b, config["partition"], fixed_split_size=config["fixed_split_size"])
        if config["method"] == "e1_own":
            return OwnPartitionAttention(q, k, v, owners, b, config["partition"], splits=config["splits"], block_n=config["block_n"])
        return ExactCascadeAttention(q, k, v, owners, b, config["tree"], fixed_split_size=config["fixed_split_size"])

    pdev = torch.cuda.get_device_properties(0)
    flush = torch.zeros(max(2 * pdev.L2_cache_size, 256 * 1024 * 1024) // 4, device="cuda")
    result = {"source": str(path.relative_to(ROOT)), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "provenance": provenance, "q_shape": data["q_shape"], "k_shape": data["k_shape"],
              "original_gpu_uuid": data["gpu_uuid"], "gpu_uuid": str(pdev.uuid), "cache": {}}
    rng = random.Random(131)
    for cache, repetitions in (("hot", 20), ("cold", 1)):
        configs = {name: item["config"] for name, item in data["selected"][cache].items() if name != "best_e1"}
        full_candidates = [row for row in data["tuning"] if row["method"] == "e1_native" and row["status"] == "ok" and row["partition"] == [list(range(b))]]
        best_full = min(full_candidates, key=lambda row: row["tuning_us_by_cache"][cache])
        configs["full_native_union"] = {key: best_full[key] for key in ("method", "partition", "fixed_split_size")}
        ops, samples, numerics = {}, {name: [] for name in configs}, {}
        for name, config in configs.items():
            op = build(config)
            actual, lse = op.run()
            torch.testing.assert_close(actual.float(), ref.float(), atol=.006, rtol=.015)
            torch.testing.assert_close(lse, ref_lse.float(), atol=.0003, rtol=.0001)
            numerics[name] = {"max_abs_error_fp64": float((actual.double() - ref).abs().max()),
                              "max_lse_error_fp64": float((lse.double() - ref_lse).nan_to_num().abs().max())}
            ops[name] = op
        # Warm JIT completed above. Randomize three preparation rounds.
        for repeat in range(3):
            order = list(configs)
            rng.shuffle(order)
            for name in order:
                torch.cuda.synchronize()
                started = time.perf_counter()
                fresh = build(configs[name])
                prepared = time.perf_counter()
                fresh.run()
                torch.cuda.synchronize()
                samples[name].append({"repeat": repeat, "prepare_wall_us": (prepared - started) * 1e6,
                                      "prepare_plus_call_host_wall_us": (time.perf_counter() - started) * 1e6})
                del fresh
        graphs = {name: graph_of(op.run, repetitions) for name, op in ops.items()}
        graphs["full_native_union_AA"] = graph_of(ops["full_native_union"].run, repetitions)
        measurements = []
        for block in range(30):
            order = list(graphs)
            rng.shuffle(order)
            for direction, sequence in (("random", order), ("reverse", list(reversed(order)))):
                measurements.append({"block": block, "direction": direction, "order": sequence,
                                     "us": {name: elapsed(graphs[name], repetitions, flush if cache == "cold" else None) for name in sequence}})
        medians = {name: statistics.median(row["us"][name] for row in measurements) for name in graphs}
        ratios = {}
        draws = np.random.default_rng(137).integers(0, 30, (10000, 30))
        for numerator, denominator in (("full_native_union", "e4_native"), ("full_union", "e4_native"),
                                       ("full_native_union", "full_native_union_AA")):
            values = np.array([[next(row for row in measurements if row["block"] == block and row["direction"] == direction)["us"][numerator] /
                                next(row for row in measurements if row["block"] == block and row["direction"] == direction)["us"][denominator]
                                for direction in ("random", "reverse")] for block in range(30)])
            ratios[f"{numerator}_over_{denominator}"] = {"median": float(np.median(values)),
                "paired_block_bootstrap95": np.quantile(np.median(values[draws], axis=(1, 2)), [.025, .975]).tolist()}
        amortized = {}
        for reuse in (1, 16, 64, 256):
            costs = {name: statistics.median(sample["prepare_wall_us"] for sample in samples[name]) / reuse + medians[name] for name in configs}
            best = min(costs, key=costs.get)
            amortized[str(reuse)] = {"cost_us_by_method": costs, "best_among_selected_backend_plans": best,
                                     "e4_vs_best_cost_ratio": costs["e4_native"] / costs[best]}
        result["cache"][cache] = {"configs": configs, "numerics": numerics, "warm_preparation_samples": samples,
                                   "measurements": measurements, "median_gpu_us": medians, "paired_ratios": ratios,
                                   "amortization_scenarios": amortized}
        del graphs, ops
    result["scope"] = "Warm planning and second-card replay of selected qualified plans, including explicit native full union; not exhaustive joint optimization over every plan's preparation cost."
    result["limitations"] = ["Amortization counts are scenarios, not measured reuse lifetimes; oracle search and preparation are separate costs.",
                             "No throughput, queueing, quality, ready-window or energy claim follows.",
                             "Python metadata construction is an implementation cost; its optimization alone is not evidence of a new attention mechanism."]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", action="append", required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(4)
    for raw in args.input:
        path = ROOT / raw
        result = measure(path)
        out = path.with_name(path.stem + "_planning.json")
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"output": str(out), "second_card": result["gpu_uuid"] != result["original_gpu_uuid"],
                          "amortization_winners": {cache: {reuse: data["best_among_selected_backend_plans"] for reuse, data in record["amortization_scenarios"].items()} for cache, record in result["cache"].items()}}), flush=True)


if __name__ == "__main__":
    main()
