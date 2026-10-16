"""Full-call native XQA versus raw paired continuation, on qualified captures.

Measures complete O-only CUDA graphs. No whole-model or request-latency claim.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_ratio(blocks, numerator, denominator):
    values = [b[numerator] / b[denominator] for b in blocks]
    rng = random.Random(20261017)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
    return {"paired_median_ratio": statistics.median(values), "bootstrap_95ci": [boot[49], boot[1949]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qualification", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--blocks", type=int, default=30)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert a.blocks >= 10 and a.replays >= 1
    out = Path(a.output)
    assert not out.exists(), out
    out.parent.mkdir(parents=True, exist_ok=True)
    r = {"status": "initializing", "scope": __doc__, "args": vars(a), "checks": {}, "timing": {},
         "source_sha256": {x: sha(ROOT / x) for x in ["scripts/benchmark_xqa_raw_pair.py", "src/xqa_raw_pair_runtime.py",
                              "src/xqa_raw_pair_build.py", "src/xqa_raw_continuation_build.py"]}}
    def save():
        out.write_text(json.dumps(r, indent=2) + "\n")
    save()
    try:
        qualification = json.loads(Path(a.qualification).read_text())
        assert qualification["status"] == "qualified_pair_only" and "capture" in qualification
        for path, digest in qualification["source_sha256"].items():
            assert sha(ROOT / path) == digest, path
        capture_path = Path(qualification["capture"]["path"])
        assert sha(capture_path) == qualification["capture"]["sha256"]
        r["qualification"] = {"path": a.qualification, "sha256": sha(a.qualification)}
        r["capture"] = qualification["capture"]
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_pair_build import jit_specs
        from src.xqa_raw_pair_runtime import RawPairPlan
        from src.xqa_raw_continuation_build import STATE_FLOATS_PER_CTA
        torch.set_num_threads(2)
        assert torch.cuda.get_device_capability() == (12, 0)
        specs, receipt = jit_specs()
        assert receipt["identity"] == qualification["build_receipt"]["identity"]
        modules = {g: spec.build_and_load() for g, spec in specs.items()}
        payload = torch.load(capture_path, map_location="cpu", weights_only=True)
        assert not payload["payload_cast_performed"] and payload["source_native_kv_dtype"] == "torch.bfloat16"
        assert payload["k_scale"] == payload["v_scale"] == 1.
        q = payload["q"].contiguous().cuda()
        prefix = payload["prefix_kv"].shape[1]
        lens = [prefix + x.shape[1] for x in payload["tail_kv"]]
        assert lens == (payload["positions"] + 1).tolist()
        cache = torch.empty_strided(payload["original_cache_shape"], payload["original_cache_stride"],
                                   device="cuda", dtype=torch.bfloat16)
        cache.zero_()
        table = payload["original_block_table"].to(device="cuda", dtype=torch.int32).contiguous()
        pkv = payload["prefix_kv"].cuda()
        for row, length in enumerate(lens):
            kv = torch.cat((pkv, payload["tail_kv"][row].cuda()), dim=1)
            t = torch.arange(length, device="cuda")
            cache[table[row, t // 16].long(), :, t % 16] = torch.cat((kv[0], kv[1]), dim=-1)
        del pkv, kv, t
        plan = RawPairPlan(modules, q, cache, table, lens, prefix)
        arms = ["native_pdl", "native_pdl_AA", "native_no_pdl", "unshared_checkpoint", "paired_complete"]
        outputs = {name: torch.empty_like(q) for name in arms[:-1]}
        outputs["paired_complete"] = plan.output
        workspaces = {name: torch.zeros_like(plan.workspace) for name in arms[:-1]}
        state = torch.empty((8 * 2 * plan.splits * STATE_FLOATS_PER_CTA,), device="cuda", dtype=torch.float32)
        k, v = cache[..., :128], cache[..., 128:]
        wrapper_k, wrapper_v = k.transpose(1, 2), v.transpose(1, 2)
        def native(name):
            work = workspaces[name]
            xqa(q.unsqueeze(1), k, v, table, plan.lengths, outputs[name].unsqueeze(1),
                work[8 * 1024 * 1024:], work[:8 * 1024 * 1024], 2, 16, q_scale=1., kv_scale=1.,
                kv_layout="HND", sm_count=plan.sm, enable_pdl=name != "native_no_pdl")
        def unshared():
            work = workspaces["unshared_checkpoint"]
            for phase in (1, 2):
                modules[8].xqa_wrapper(False, plan.sm, 2, 0, 1., None,
                                      outputs["unshared_checkpoint"].unsqueeze(1), 1., q.unsqueeze(1),
                                      None, wrapper_k, wrapper_v, None, None, table, plan.max_len, plan.lengths,
                                      8, 1., None, 1, None, None, work[:8 * 1024 * 1024],
                                      work[8 * 1024 * 1024:], False, state, prefix // 256, phase)
        funcs = {name: (lambda name=name: native(name)) for name in arms[:3]}
        funcs["unshared_checkpoint"], funcs["paired_complete"] = unshared, plan.run
        def equal_bits(x, y):
            return torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))
        for fn in funcs.values():
            fn()
        torch.cuda.synchronize()
        r["checks"]["all_initial_bit_equal"] = all(equal_bits(outputs["native_pdl"], x) for x in outputs.values())
        assert r["checks"]["all_initial_bit_equal"]
        r["checks"]["captured_O_pass"] = torch.allclose(outputs["native_pdl"], payload["o"].cuda(), atol=.006, rtol=.015)
        assert r["checks"]["captured_O_pass"]
        r["shape"] = {"batch": 8, "prefix_tokens": prefix, "private_lengths": [x-prefix for x in lens],
                      "S": plan.splits, "paired_SM_parameter": plan.paired_sm, "original_SM_parameter": plan.sm,
                      "cache_shape": list(cache.shape), "cache_stride": list(cache.stride()),
                      "paired_state_bytes": plan.state.numel() * 4, "unshared_state_bytes": state.numel() * 4,
                      "each_workspace_bytes": plan.workspace.numel()}
        graphs = {}
        r["capture_wall_seconds"] = {}
        for name, fn in funcs.items():
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            graphs[name] = graph
            r["capture_wall_seconds"][name] = time.perf_counter() - start
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        assert all(equal_bits(outputs["native_pdl"], x) for x in outputs.values())
        flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        rng = random.Random(20261017)
        for regime in ("hot", "cold"):
            blocks, orders = [], []
            replays = a.replays if regime == "hot" else 1
            for _ in range(a.blocks):
                order = arms[:]
                rng.shuffle(order)
                orders.append(order)
                block = {}
                for name in order:
                    if regime == "cold":
                        flush.zero_()
                    else:
                        graphs[name].replay()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(replays):
                        graphs[name].replay()
                    end.record()
                    end.synchronize()
                    block[name] = start.elapsed_time(end) / replays
                blocks.append(block)
            r["timing"][regime] = {"blocks_ms": blocks, "orders": orders,
                                  "median_ms": {name: statistics.median(b[name] for b in blocks) for name in arms},
                                  "native_over_pair": paired_ratio(blocks, "native_pdl", "paired_complete"),
                                  "pdl_matched_native_over_pair": paired_ratio(blocks, "native_no_pdl", "paired_complete"),
                                  "native_over_unshared": paired_ratio(blocks, "native_pdl", "unshared_checkpoint"),
                                  "AA": paired_ratio(blocks, "native_pdl", "native_pdl_AA"),
                                  "replays_per_sample": replays}
            save()
        prepare_times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            plan.prepare(lens)
            torch.cuda.synchronize()
            prepare_times.append(time.perf_counter() - start)
        r["diagnostic_prepare_wall_seconds"] = {"samples": prepare_times, "median": statistics.median(prepare_times),
            "scope": "Standalone Python physical-prefix validation and length copy, once per call; not a production device metadata implementation."}
        r["checks"]["all_final_bit_equal"] = all(equal_bits(outputs["native_pdl"], x) for x in outputs.values())
        assert r["checks"]["all_final_bit_equal"]
        if a.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                for name in arms:
                    with torch.profiler.record_function(name):
                        graphs[name].replay()
                torch.cuda.synchronize()
            trace = out.with_suffix(".trace.json")
            prof.export_chrome_trace(str(trace))
            r["profile_trace"] = str(trace)
        r["status"] = "completed"
        r["limitations"] = ["Actual one-step same-Q O-only attention, not whole-model or natural-request performance.",
                            "Paired graph includes Qpack, pair page/length copies, raw state and native tail/merge.",
                            "State checkpoint without sharing is an ablation, never the performance baseline.",
                            "Cold samples flush 256MiB outside timing and measure one replay; hot samples use repeated graph replay."]
    except Exception:
        r["status"] = "failed"
        r["error"] = traceback.format_exc()
        save()
        raise
    save()
    print(json.dumps({"status": r["status"], "output": str(out)}))


if __name__ == "__main__":
    main()
