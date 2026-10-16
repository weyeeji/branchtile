"""Run one frozen shuffled shard of the development-only dispatch sweep."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_xqa_raw_pair import paired_ratio


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="configs/xqa_raw_dispatch_sweep_v1.json")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    spec = json.loads(Path(a.spec).read_text())
    assert 0 <= a.shard < spec["shards"]
    out = Path(a.output)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(spec["prefix_tokens"], spec["additional_private_tokens"], spec["batches"]))
    indexed = list(enumerate(grid))
    random.Random(spec["grid_shuffle_seed"]).shuffle(indexed)
    cells = indexed[a.shard::spec["shards"]]
    report = {"status": "initializing", "scope": __doc__, "spec": spec, "spec_path": a.spec,
              "spec_sha256": sha(a.spec), "shard": a.shard, "planned_cells": cells, "cases": [],
              "source_sha256": {x: sha(ROOT / x) for x in ["scripts/sweep_xqa_raw_dispatch.py", "src/xqa_raw_pair_padded_runtime.py",
                                  "src/xqa_raw_pair_general_runtime.py", "src/xqa_raw_pair_build.py", "scripts/benchmark_xqa_raw_pair.py"]}}
    def save():
        out.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_pair_build import jit_specs
        from src.xqa_raw_pair_padded_runtime import RawPairPaddedPlan
        torch.set_num_threads(2)
        assert torch.cuda.get_device_capability() == (12, 0)
        modules, receipt = jit_specs()
        modules = {g: s.build_and_load() for g, s in modules.items()}
        report["build_identity"] = receipt["identity"]
        candidate_workspace = torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        native_workspace = torch.zeros_like(candidate_workspace)
        aa_workspace = torch.zeros_like(candidate_workspace)
        raw_state = torch.empty(2 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        def bits(a, b):
            return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
        def measure(cell_id, prefix, additional, batch):
            torch.manual_seed(spec["input_seed"] + cell_id)
            private = [1 + math.ceil(additional * (.75 + .25 * i / (batch - 1))) for i in range(batch)]
            lens = [prefix + x for x in private]
            pp, tp = prefix // 16, (max(private) + 15) // 16 + 2
            pages = pp + batch * tp + 32
            cache = torch.randn((pages, 16, 2, 256), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
            perm = torch.randperm(pages, device="cuda")
            table = torch.empty((batch, pp + tp), device="cuda", dtype=torch.int32)
            table[:, :pp] = perm[:pp].int()
            for row in range(batch):
                table[row, pp:] = perm[pp + row * tp:pp + (row + 1) * tp].int()
            q = torch.randn((batch, 16, 128), device="cuda", dtype=torch.bfloat16)
            start = time.perf_counter()
            plan = RawPairPaddedPlan(modules, q, cache, table, lens, prefix, workspace=candidate_workspace, raw_state=raw_state)
            torch.cuda.synchronize()
            case = {"cell_id": cell_id, "prefix_tokens": prefix, "additional_private_tokens": additional,
                    "batch": batch, "private_lengths": private, "S": plan.splits,
                    "setup_seconds": time.perf_counter() - start, "checks": {}}
            native_o, aa_o = torch.empty_like(q), torch.empty_like(q)
            def native(output, workspace):
                xqa(q.unsqueeze(1), cache[..., :128], cache[..., 128:], table, plan.lengths,
                    output.unsqueeze(1), workspace[8 * 1024 * 1024:], workspace[:8 * 1024 * 1024],
                    2, 16, q_scale=1., kv_scale=1., kv_layout="HND", sm_count=plan.sm, enable_pdl=True)
            def native_call(): native(native_o, native_workspace)
            def aa_call(): native(aa_o, aa_workspace)
            native_call()
            aa_call()
            plan.run()
            checks = case["checks"]
            checks["initial_bits"] = bits(native_o, plan.output)
            checks["native_AA_bits"] = bits(native_o, aa_o)
            checks["FP64_rows"] = []
            for row in (0, batch - 1):
                token = torch.arange(lens[row], device="cuda")
                kv = cache[table[row, token // 16].long(), :, token % 16].double()
                scores = torch.einsum("hgd,thd->hgt", q[row].double().reshape(2, 8, 128), kv[..., :128]) / 128 ** .5
                ref = torch.einsum("hgt,thd->hgd", scores.softmax(-1), kv[..., 128:]).reshape(16, 128)
                checks["FP64_rows"].append({"row": row,
                    "native_pass": torch.allclose(native_o[row].double(), ref, atol=.006, rtol=.015),
                    "candidate_pass": torch.allclose(plan.output[row].double(), ref, atol=.006, rtol=.015)})
            del token, kv, scores, ref
            case["status"] = "qualified" if (checks["initial_bits"] and checks["native_AA_bits"] and
                all(c["native_pass"] and c["candidate_pass"] for c in checks["FP64_rows"])) else "qualification_failed"
            if case["status"] != "qualified":
                return case
            funcs = {"native": native_call, "native_AA": aa_call, "candidate": plan.run}
            graphs = {}
            for name, fn in funcs.items():
                for _ in range(3): fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): fn()
                graphs[name] = graph
            order_rng = random.Random(spec["timing_order_seed"] + cell_id)
            blocks, orders = [], []
            for _ in range(spec["blocks"]):
                order = list(graphs)
                order_rng.shuffle(order)
                orders.append(order)
                block = {}
                for name in order:
                    graphs[name].replay()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(spec["hot_graph_replays_per_sample"]): graphs[name].replay()
                    end.record()
                    end.synchronize()
                    block[name] = begin.elapsed_time(end) / spec["hot_graph_replays_per_sample"]
                blocks.append(block)
            checks["final_bits"] = bits(native_o, plan.output) and bits(native_o, aa_o)
            case["timing"] = {"blocks_ms": blocks, "orders": orders,
                              "median_ms": {name: statistics.median(x[name] for x in blocks) for name in graphs},
                              "native_over_candidate": paired_ratio(blocks, "native", "candidate"),
                              "AA": paired_ratio(blocks, "native", "native_AA")}
            case["status"] = "completed" if checks["final_bits"] else "qualification_failed_after_timing"
            return case
        for cell_id, (prefix, additional, batch) in cells:
            result = measure(cell_id, prefix, additional, batch)
            report["cases"].append(result)
            save()
            # Preserve all predeclared cells, including failed qualification; do not
            # quietly omit a bad shape while fitting the dispatch boundary.
        report["status"] = "completed" if all(c["status"] == "completed" for c in report["cases"]) else "completed_with_failed_cells"
        report["limitations"] = ["Synthetic development data only, no heldout, no natural generation or E2E.",
                                  "Hot complete-call graphs; policy cannot be claimed cold-cache robust from this sweep alone.",
                                  "Private length includes current token and varies from 75% to100% of the declared additional tail."]
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise
    save()
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "output": str(out)}))


if __name__ == "__main__":
    main()
