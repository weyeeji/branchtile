"""B8-only D64 prefix partition versus native/AA/current direct, complete CUDA graphs.

No native split, scalar recurrence, hint-control or tail changes are intended.
Every live prefix raw-state field is bit-compared before performance is admitted.
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
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_xqa_raw_pair import paired_ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture")
    p.add_argument("--batches", default="8")
    p.add_argument("--prefix", type=int, default=8192)
    p.add_argument("--extra-tail", type=int, default=0)
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert a.blocks >= 10 and a.extra_tail >= 0 and a.prefix % 16 == 0
    batches = [int(b) for b in a.batches.split(",")]
    assert batches == [8], "Bounded first experiment supports B8 only"
    out = Path(a.output)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    r = {"status": "initializing", "scope": __doc__, "args": vars(a), "cases": [],
         "source_sha256": {x: hashlib.sha256((ROOT / x).read_bytes()).hexdigest() for x in
                           ["scripts/qualify_benchmark_xqa_raw_direct_dslice.py", "src/xqa_raw_direct_dslice_build.py", "src/xqa_raw_direct_dslice_runtime.py", "src/xqa_raw_continuation_build.py", "scripts/benchmark_xqa_raw_pair.py", "src/xqa_raw_pair_padded_runtime.py", "src/xqa_raw_pair_general_runtime.py", "src/xqa_raw_pair_build.py", "src/xqa_raw_grouped_hint_build.py", "src/xqa_raw_direct_input_build.py", "src/xqa_raw_direct_input_runtime.py"]}}
    def save():
        out.write_text(json.dumps(r, indent=2) + "\n")
    save()
    try:
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_direct_dslice_build import jit_specs
        from src.xqa_raw_direct_input_build import jit_specs as copied_specs
        from src.xqa_raw_direct_dslice_runtime import RawPairDirectDSlicePlan as RawPairGeneralPlan
        from src.xqa_raw_direct_input_runtime import RawPairDirectInputPlan as CopiedPlan
        torch.set_num_threads(2)
        assert torch.cuda.get_device_capability() == (12, 0)
        torch.manual_seed(20261017)
        specs, receipt = jit_specs()
        modules = {g: s.build_and_load() for g, s in specs.items()}
        r["build_identity"] = receipt["identity"]
        copy_specs, copy_receipt = copied_specs()
        copied_modules = {g: spec.build_and_load() for g, spec in copy_specs.items()}
        r["direct_control_build_identity"] = copy_receipt["identity"]
        payload = None
        if a.capture:
            path = Path(a.capture)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            assert not payload["payload_cast_performed"]
            assert payload["q"].dtype == payload["prefix_kv"].dtype == torch.bfloat16
            assert payload["k_scale"] == payload["v_scale"] == 1.
            prefix = payload["prefix_kv"].shape[1]
            base_q = payload["q"].contiguous().cuda()
            assert max(batches) <= len(base_q), "Never duplicate captured Q to fake a larger real cohort"
            tails = [t.cuda() for t in payload["tail_kv"]]
            prefix_kv = payload["prefix_kv"].cuda()
            r["capture"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "rank": payload["rank"], "layer": payload["layer"]}
        else:
            prefix = a.prefix
            base_q = torch.randn((max(batches), 16, 128), device="cuda", dtype=torch.bfloat16)
            prefix_kv = torch.randn((2, prefix, 2, 128), device="cuda", dtype=torch.bfloat16)
            tails = [torch.randn((2, 17 + (i * 37) % 257, 2, 128), device="cuda", dtype=torch.bfloat16)
                     for i in range(len(base_q))]
        if a.extra_tail:
            tails = [torch.cat((t, torch.randn((2, a.extra_tail + i % 7, 2, 128), device="cuda", dtype=torch.bfloat16)), dim=1)
                     for i, t in enumerate(tails)]
        lens = [prefix + t.shape[1] for t in tails]
        n = len(base_q)
        if payload is not None and not a.extra_tail:
            # Preserve native physical page identity and stride for unmodified captures.
            shape, stride = payload["original_cache_shape"], payload["original_cache_stride"]
            cache = torch.empty_strided(shape, stride, device="cuda", dtype=torch.bfloat16)
            cache.zero_()
            base_table = payload["original_block_table"].to(device="cuda", dtype=torch.int32).contiguous()
            r["layout"] = "native original physical IDs and stride"
        else:
            # Explicit synthetic/stress layout: random physical page permutation, same fused stride.
            pp = prefix // 16
            tail_pages = max((t.shape[1] + 15) // 16 for t in tails) + 2
            pages = pp + n * tail_pages + 32
            cache = torch.randn((pages, 16, 2, 256), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
            perm = torch.randperm(pages, device="cuda")
            base_table = torch.empty((n, pp + tail_pages), device="cuda", dtype=torch.int32)
            base_table[:, :pp] = perm[:pp].int()
            for row in range(n):
                base_table[row, pp:] = perm[pp + row * tail_pages: pp + (row + 1) * tail_pages].int()
            r["layout"] = "random remapped shared prefix, independent private pages; synthetic extra KV" if payload is not None else "synthetic Q and KV"
        for row in range(n):
            kv = torch.cat((prefix_kv, tails[row]), dim=1)
            t = torch.arange(lens[row], device="cuda")
            cache[base_table[row, t // 16].long(), :, t % 16] = torch.cat((kv[0], kv[1]), dim=-1)
        del kv, t, prefix_kv, tails
        references = []
        for row, length in enumerate(lens):
            token = torch.arange(length, device="cuda")
            kv = cache[base_table[row, token // 16].long(), :, token % 16].double()
            scores = torch.einsum("hgd,thd->hgt", base_q[row].double().reshape(2, 8, 128), kv[..., :128]) / 128 ** .5
            references.append(torch.einsum("hgt,thd->hgd", scores.softmax(-1), kv[..., 128:]).reshape(16, 128))
        reference = torch.stack(references)
        del references, kv, scores
        workspace = torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        native_workspace = torch.zeros_like(workspace)
        shared_state = torch.empty(2 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        control_state = torch.empty_like(shared_state)
        rng = random.Random(20261017)
        permutation = list(range(n))
        rng.shuffle(permutation)
        retained = []
        def bits(x, y):
            return torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))
        for b in batches:
            rows = permutation[:b]
            index = torch.tensor(rows, device="cuda")
            q, table = base_q.index_select(0, index).contiguous(), base_table.index_select(0, index).contiguous()
            lengths = [lens[i] for i in rows]
            start = time.perf_counter()
            plan = RawPairGeneralPlan(modules, q, cache, table, lengths, prefix, workspace=workspace, raw_state=shared_state)
            copied_plan = CopiedPlan(copied_modules, q, cache, table, lengths, prefix, workspace=workspace, raw_state=control_state)
            copied_plan.lengths = plan.lengths
            torch.cuda.synchronize()
            case = {"batch": b, "rows": rows, "prefix": prefix, "lengths": lengths,
                    "S": plan.splits, "paired_b": plan.paired_b, "prefix_SM_parameter": plan.prefix_sm,
                    "tail_SM_parameter": plan.tail_sm, "orphan_SM_parameter": plan.orphan_sm,
                    "setup_wall_seconds": time.perf_counter() - start, "checks": {},
                    "grids": {"native": [plan.splits, 2, b], "direct_prefix": [plan.splits, 2, plan.groups],
                              "dslice_prefix": [plan.splits, 4, plan.groups], "tail": [plan.splits, 2, b]}}
            native_o, aa_o = torch.empty_like(q), torch.empty_like(q)
            def native(out_tensor, plan=plan, q=q, table=table):
                xqa(q.unsqueeze(1), cache[..., :128], cache[..., 128:], table, plan.lengths, out_tensor.unsqueeze(1),
                    native_workspace[8 * 1024 * 1024:], native_workspace[:8 * 1024 * 1024], 2, 16,
                    q_scale=1., kv_scale=1., kv_layout="HND", sm_count=plan.sm, enable_pdl=True)
            shared_state.fill_(float("nan"))
            control_state.fill_(float("nan"))
            native(native_o)
            plan.run()
            copied_plan.run()
            checks = case["checks"]
            checks["raw_prefix_all_fields_bits"] = torch.equal(plan.state.view(torch.int32), copied_plan.state.view(torch.int32))
            checks["raw_prefix_all_fields_finite"] = bool(torch.isfinite(plan.state).all())
            case["raw_prefix_mismatch_fields"] = {name: int((plan.state.view(-1, 35)[:, lo:hi].contiguous().view(torch.int32) != copied_plan.state.view(-1, 35)[:, lo:hi].contiguous().view(torch.int32)).sum()) for name, lo, hi in [("U", 0, 32), ("m", 32, 33), ("z", 33, 34), ("hint", 34, 35)]}
            checks["paired_bits"] = bits(native_o, plan.output)
            checks["direct_control_bits"] = bits(native_o, copied_plan.output)
            checks["AA_10_bits"] = True
            for _ in range(10):
                native(aa_o)
                checks["AA_10_bits"] &= bits(native_o, aa_o)
            ref = reference.index_select(0, index)
            checks["native_FP64"] = torch.allclose(native_o.double(), ref, atol=.006, rtol=.015)
            checks["candidate_FP64"] = torch.allclose(plan.output.double(), ref, atol=.006, rtol=.015)
            r["cases"].append(case)
            save()
            assert all(checks.values()), case
            # A final hint can reconverge after earlier hidden-state divergence.
            # Check a small declared set of native tile cuts, including inactive
            # splits, first split-stride boundaries and the earlier hint trigger.
            case["cut_checks"] = []
            cuts = sorted({x for x in [1, plan.splits - 1, plan.splits, plan.splits + 1,
                                      339, prefix // 256] if 1 <= x <= prefix // 256})
            for cut in cuts:
                plan.prefix_tokens = copied_plan.prefix_tokens = cut * 256
                shared_state.fill_(float("nan"))
                control_state.fill_(float("nan"))
                plan.run()
                copied_plan.run()
                item = {"cut_tiles": cut,
                        "raw_bits": torch.equal(plan.state.view(torch.int32), copied_plan.state.view(torch.int32)),
                        "raw_finite": bool(torch.isfinite(plan.state).all()),
                        "candidate_native_bits": bits(native_o, plan.output),
                        "control_native_bits": bits(native_o, copied_plan.output)}
                case["cut_checks"].append(item)
                save()
                assert all(v for k, v in item.items() if k != "cut_tiles"), item
            plan.prefix_tokens = copied_plan.prefix_tokens = prefix
            funcs = {"native": lambda native=native, native_o=native_o: native(native_o),
                     "native_AA": lambda native=native, aa_o=aa_o: native(aa_o), "candidate": plan.run, "direct_control": copied_plan.run}
            graphs = {}
            start = time.perf_counter()
            for name, fn in funcs.items():
                fn()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
                graphs[name] = graph
            torch.cuda.synchronize()
            case["capture_wall_seconds"] = time.perf_counter() - start
            # Same fixed graph after changed Q and tail lengths; shared state is poisoned.
            oldq = q.clone()
            q.mul_(.875)
            changed_lengths = [length - min(3, length - prefix - 1) for length in lengths]
            plan.prepare(changed_lengths)
            shared_state.fill_(float("nan"))
            control_state.fill_(float("nan"))
            graphs["native"].replay()
            graphs["candidate"].replay()
            graphs["direct_control"].replay()
            torch.cuda.synchronize()
            checks["changed_raw_all_fields_bits"] = torch.equal(plan.state.view(torch.int32), copied_plan.state.view(torch.int32))
            checks["changed_raw_all_fields_finite"] = bool(torch.isfinite(plan.state).all())
            assert checks["changed_raw_all_fields_bits"] and checks["changed_raw_all_fields_finite"]
            checks["changed_graph_bits"] = bits(native_o, plan.output)
            checks["copied_changed_graph_bits"] = bits(native_o, copied_plan.output)
            assert checks["copied_changed_graph_bits"]
            assert checks["changed_graph_bits"], case
            q.copy_(oldq)
            plan.prepare(lengths)
            blocks, orders = [], []
            for _ in range(a.blocks):
                order = list(graphs)
                rng.shuffle(order)
                orders.append(order)
                block = {}
                for name in order:
                    graphs[name].replay()
                    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    for _ in range(10):
                        graphs[name].replay()
                    end_event.record()
                    end_event.synchronize()
                    block[name] = start_event.elapsed_time(end_event) / 10
                blocks.append(block)
            case["hot"] = {"blocks_ms": blocks, "orders": orders,
                           "median_ms": {k: statistics.median(x[k] for x in blocks) for k in graphs},
                           "native_over_candidate": paired_ratio(blocks, "native", "candidate"),
                           "AA": paired_ratio(blocks, "native", "native_AA"),
                           "direct_over_dslice": paired_ratio(blocks, "direct_control", "candidate")}
            flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            cold_blocks, cold_orders = [], []
            for _ in range(a.blocks):
                order = list(graphs)
                rng.shuffle(order)
                cold_orders.append(order)
                block = {}
                for name in order:
                    flush.zero_()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    block[name] = begin.elapsed_time(end)
                cold_blocks.append(block)
            case["cold"] = {"blocks_ms": cold_blocks, "orders": cold_orders,
                "median_ms": {k: statistics.median(x[k] for x in cold_blocks) for k in graphs},
                "native_over_candidate": paired_ratio(cold_blocks, "native", "candidate"),
                "AA": paired_ratio(cold_blocks, "native", "native_AA"),
                "direct_over_dslice": paired_ratio(cold_blocks, "direct_control", "candidate")}
            if a.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for name, graph in graphs.items():
                        with torch.profiler.record_function("dslice_arm_" + name):
                            graph.replay()
                            torch.cuda.synchronize()
                trace = out.with_suffix(".trace.json")
                prof.export_chrome_trace(str(trace))
                case["trace"] = str(trace)
            checks["final_bits"] = bits(native_o, plan.output)
            assert checks["final_bits"]
            retained.append((case, plan, copied_plan, graphs, native_o))
            save()
        # Return to old graphs after every other batch shape used the same workspace/state.
        for case, plan, copied_plan, graphs, native_o in reversed(retained):
            shared_state.fill_(float("nan"))
            control_state.fill_(float("nan"))
            graphs["native"].replay()
            graphs["candidate"].replay()
            graphs["direct_control"].replay()
            torch.cuda.synchronize()
            case["checks"]["revisit_after_other_batch_shapes_bits"] = bits(native_o, plan.output)
            assert case["checks"]["revisit_after_other_batch_shapes_bits"]
            case["checks"]["revisit_raw_bits"] = torch.equal(plan.state.view(torch.int32), copied_plan.state.view(torch.int32))
            case["checks"]["revisit_raw_finite"] = bool(torch.isfinite(plan.state).all())
            assert case["checks"]["revisit_raw_bits"] and case["checks"]["revisit_raw_finite"]
        r["status"] = "qualified_and_measured"
        r["limitations"] = ["B8-only D64 experiment; grid coverage is not measured SM utilization. Doubled K/QK work may outweigh extra parallelism.", "Fixed graph per batch size, explicit setup/capture; no unlimited topology reuse.",
                            
                            "Both candidate and direct control include two full XQA launches; no input packing/copy calls. All raw prefix fields require exact agreement.",
                            "Subset Q is the original AR snapshot; appended-tail stress is synthetic and not natural output."]
    except Exception:
        r["status"] = "failed"
        r["error"] = traceback.format_exc()
        save()
        raise
    save()
    print(json.dumps({"status": r["status"], "output": str(out)}))


if __name__ == "__main__":
    main()
