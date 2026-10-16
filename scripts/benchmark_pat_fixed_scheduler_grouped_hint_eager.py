"""Prepared-metadata complete eager calls: local PAT compat fixed scheduler, native XQA, grouped-hint raw pair.

No CUDA graphs, no engine claim. PAT tree planning/upload is measured separately.
"""
import argparse
import hashlib
import importlib.util
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
LEGAL_PAGE16_MNWS = [[64, 128, 4], [64, 64, 4], [64, 32, 4],
                    [32, 128, 2], [32, 64, 2], [32, 32, 2], [32, 16, 2],
                    [16, 64, 1], [16, 32, 1], [16, 16, 1]]
SOURCES = ["scripts/benchmark_pat_fixed_scheduler_grouped_hint_eager.py", "scripts/build_pat_fixed_scheduler_perf_v1.py", "src/xqa_raw_pair_build.py", "src/xqa_raw_grouped_hint_build.py",
           "src/xqa_raw_pair_runtime.py", "src/xqa_raw_continuation_build.py",
           "src/dense_cascade_fp64_reference.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ratio(blocks, numerator, denominator):
    values = [b[numerator] / b[denominator] for b in blocks]
    rng = random.Random(20261017)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
    return {"paired_median": statistics.median(values), "bootstrap_95ci": [boot[49], boot[1949]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-receipt", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--caller-stream", choices=["default", "nondefault"], default="default")
    parser.add_argument("--qualification-repeats", type=int, default=30)
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert args.blocks >= 10 and args.replays >= 1 and args.qualification_repeats >= 30
    assert os.environ.get("DISABLE_STREAM", "0") in ("0", "1")
    output = Path(args.output)
    assert not output.exists(), output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "initializing", "scope": __doc__, "args": vars(args),
              "source_sha256": {p: sha(ROOT / p) for p in SOURCES},
              "build_receipt": {"path": args.build_receipt, "sha256": sha(args.build_receipt)},
              "capture": {"path": args.capture, "sha256": sha(args.capture)},
              "checks": {}, "timing": {}, "setup": {}}
    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        build = json.loads(Path(args.build_receipt).read_text())
        assert build["status"] == "built_unqualified_fixed_scheduler_perf"
        assert build["CUDA_initialized"] is False and build["flashinfer_imported"] is False
        assert build["commit"] == "8cb067f0a31dabe800958f657093f17ac5d7366d"
        assert sha(build["module"]) == build["module_sha256"]
        for name, digest in build["generated_sha256"].items():
            assert sha(ROOT / "build/pat_fixed_scheduler_perf_v1" / name) == digest
        for path, digest in build["reused_cuda_objects"].items():
            assert sha(ROOT / path) == digest
        report["build_receipt"]["admission_scope"] = "Local compat fixed scheduler plus explicit stream fork/join, unchanged CUDA arithmetic; no NaN fill or debug tensor retention. Not unchanged upstream PAT."
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_grouped_hint_build import jit_specs
        from src.xqa_raw_pair_runtime import RawPairPlan
        from src.dense_cascade_fp64_reference import full_history_fp64, compare_to_reference
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        assert torch.cuda.get_device_capability() == (12, 0)
        caller_stream = torch.cuda.default_stream() if args.caller_stream == "default" else torch.cuda.Stream()
        torch.cuda.set_stream(caller_stream)
        assert (torch.cuda.current_stream().cuda_stream == 0) == (args.caller_stream == "default")
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "device": torch.cuda.get_device_name(), "stream": torch.cuda.current_stream().cuda_stream,
                                 "DISABLE_STREAM": os.environ.get("DISABLE_STREAM"),
                                 "PAT_module_sha256": build["module_sha256"]}
        start = time.perf_counter()
        module_spec = importlib.util.spec_from_file_location("_prefix_attn", build["module"])
        pat = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(pat)
        specs, raw_receipt = jit_specs()
        modules = {g: spec.build_and_load() for g, spec in specs.items()}
        report["raw_build_receipt"] = raw_receipt
        report["setup"]["module_import_and_raw_JIT_seconds"] = time.perf_counter() - start
        payload = torch.load(args.capture, map_location="cpu", weights_only=True)
        assert payload["source_native_kv_dtype"] == "torch.bfloat16" and not payload["payload_cast_performed"]
        assert payload["k_scale"] == payload["v_scale"] == 1.
        q = payload["q"].contiguous().cuda()
        assert tuple(q.shape) == (8, 16, 128) and q.dtype == torch.bfloat16
        prefix = payload["prefix_kv"].shape[1]
        lengths = [prefix + tail.shape[1] for tail in payload["tail_kv"]]
        assert lengths == (payload["positions"] + 1).tolist()
        # Keep original full table width: this determines native XQA max_seq_len/S.
        cpu_table = payload["original_block_table"].to(dtype=torch.int32).contiguous()
        assert cpu_table.device.type == "cpu" and cpu_table.shape[0] == 8
        table = cpu_table.cuda()
        cache = torch.empty_strided(payload["original_cache_shape"], payload["original_cache_stride"],
                                   device="cuda", dtype=torch.bfloat16)
        cache.zero_()
        pkv = payload["prefix_kv"].cuda()
        for row, length in enumerate(lengths):
            kv = torch.cat((pkv, payload["tail_kv"][row].cuda()), dim=1)
            positions = torch.arange(length, device="cuda")
            cache[table[row, positions // 16].long(), :, positions % 16] = torch.cat((kv[0], kv[1]), dim=-1)
        del pkv, kv, positions
        torch.cuda.synchronize()
        start = time.perf_counter()
        raw = RawPairPlan(modules, q, cache, table, lengths, prefix)
        torch.cuda.synchronize()
        report["setup"]["raw_allocate_validate_upload_seconds"] = time.perf_counter() - start
        k, v = cache[..., :128], cache[..., 128:]
        pat_k, pat_v = k.transpose(1, 2), v.transpose(1, 2)
        assert pat_k.stride() == pat_v.stride()
        assert pat_k.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        assert pat_v.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        tree = pat.PrefixTreeCPP(16)
        start = time.perf_counter()
        tree.build_radix_tree(lengths, cpu_table)
        report["setup"]["PAT_CPU_tree_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        tree.pack_schedule(MNWs=LEGAL_PAGE16_MNWS, HRatio=8, kvHead=2, use_compute_model=False)
        report["setup"]["PAT_CPU_schedule_seconds"] = time.perf_counter() - start
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        start_event.record()
        tree.kernel_info.to_gpu(torch.device("cuda:0"))
        end_event.record()
        torch.cuda.synchronize()
        report["setup"]["PAT_metadata_upload_wall_seconds"] = time.perf_counter() - start
        report["setup"]["PAT_metadata_upload_event_ms"] = start_event.elapsed_time(end_event)
        report["PAT_contract"] = {"MNWs": LEGAL_PAGE16_MNWS, "HRatio": 8, "kvHead": 2,
                                  "use_compute_model": False, "original_multistream": False, "patched_multistream": os.environ.get("DISABLE_STREAM", "0") != "1", "official_DISABLE_STREAM": os.environ.get("DISABLE_STREAM", "0"), "scheduler": "local guarded final-box numbering repair",
                                  "K_V_alias_original_cache": True, "CPU_table_kept_alive": True,
                                  "excluded_official_page16_unsupported_tile": [16, 128, 1],
                                  "tree_metadata_fields_not_exposed_by_official_pybind": True}
        names = ["native_pdl", "native_pdl_AA", "PAT_compat_fixed_scheduler", "raw_pair_grouped_hint_complete"]
        outputs = {name: torch.empty_like(q) for name in names[:-1]}
        outputs["raw_pair_grouped_hint_complete"] = raw.output
        workspaces = {name: torch.zeros_like(raw.workspace) for name in names[:2]}
        q4 = q.unsqueeze(1)
        o4 = {name: value.unsqueeze(1) for name, value in outputs.items()}
        def native(name):
            workspace = workspaces[name]
            xqa(q4, k, v, table, raw.lengths, o4[name], workspace[8 * 1024 * 1024:],
                workspace[:8 * 1024 * 1024], 2, 16, q_scale=1., kv_scale=1.,
                kv_layout="HND", sm_count=raw.sm, enable_pdl=True)
        functions = {name: (lambda name=name: native(name)) for name in names[:2]}
        functions["PAT_compat_fixed_scheduler"] = lambda: pat.prefix_attn_with_kvcache(q4, pat_k, pat_v, tree, 128**-.5, o4["PAT_compat_fixed_scheduler"])
        functions["raw_pair_grouped_hint_complete"] = raw.run
        report["shape"] = {"B": 8, "Hq": 16, "Hkv": 2, "D": 128, "prefix": prefix, "lengths": lengths,
                           "cache_shape": list(cache.shape), "cache_stride": list(cache.stride()),
                           "PAT_K_shape": list(pat_k.shape), "PAT_K_stride": list(pat_k.stride()),
                           "table_shape": list(table.shape), "native_S": raw.splits,
                           "native_SM": raw.sm, "pair_SM": raw.paired_sm}
        save()
        for function in functions.values():
            function()
        torch.cuda.synchronize()
        reference = full_history_fp64(q, cache, table, lengths)
        def qualify():
            checks = {}
            native_output = outputs["native_pdl"]
            native_bits = native_output.contiguous().view(torch.int16)
            for name, value in outputs.items():
                checks[name] = {"FP64": compare_to_reference(value, reference),
                                "finite": bool(torch.isfinite(value).all()),
                                "native_bit_mismatch_count": int((value.contiguous().view(torch.int16) != native_bits).sum()),
                                "native_max_abs": float((value.float() - native_output.float()).abs().max())}
            checks["captured_native_O"] = {"native_bit_equal": torch.equal(native_bits, payload["o"].cuda().contiguous().view(torch.int16)),
                                           "allclose": bool(torch.allclose(native_output, payload["o"].cuda(), atol=.006, rtol=.015))}
            return checks
        def gate(checks):
            assert all(checks[name]["finite"] and checks[name]["FP64"]["passed"] for name in names), checks
            assert all(checks[name]["native_bit_mismatch_count"] == 0 for name in ("native_pdl_AA", "raw_pair_grouped_hint_complete")), checks
            assert checks["captured_native_O"]["allclose"], checks
        report["checks"]["initial"] = qualify()
        save()
        gate(report["checks"]["initial"])
        report["repeat_qualification"] = []
        for iteration in range(args.qualification_repeats):
            for function in functions.values(): function()
            torch.cuda.synchronize()
            check = qualify()
            report["repeat_qualification"].append({"iteration": iteration, "checks": check})
            save()
            gate(check)
        report["all_repeated_numerical_checks_passed"] = True
        if args.qualify_only:
            report["status"] = "qualified_fixed_scheduler_perf_artifact"
            report["limitations"] = ["No timing claim. Local PAT scheduler/stream compatibility modifications; original failures remain preserved.",
                                      "One actual state; PAT admits FP64 tolerance, grouped-hint raw and native AA require bit equality."]
            save()
            print(json.dumps({"status": report["status"], "output": str(output)}))
            return
        for _ in range(10):
            for function in functions.values():
                function()
        torch.cuda.synchronize()
        rng = random.Random(20261017)
        flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        for regime in ("hot", "cold"):
            event_blocks, wall_blocks, orders = [], [], []
            block_checks = []
            repeats = args.replays if regime == "hot" else 1
            for _ in range(args.blocks):
                order = names[:]
                rng.shuffle(order)
                orders.append(order)
                event_block, wall_block = {}, {}
                for name in order:
                    if regime == "cold":
                        flush.zero_()
                    else:
                        functions[name]()
                    torch.cuda.synchronize()
                    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start = time.perf_counter()
                    start_event.record()
                    for _ in range(repeats):
                        functions[name]()
                    end_event.record()
                    torch.cuda.synchronize()  # Includes PAT streams even if event dependencies were wrong.
                    wall_block[name] = (time.perf_counter() - start) * 1000 / repeats
                    event_block[name] = start_event.elapsed_time(end_event) / repeats
                event_blocks.append(event_block)
                wall_blocks.append(wall_block)
                # Outside timing. Each subsequent arm gets the same warmup or
                # cold flush above, restoring cache treatment after this gate.
                block_check = qualify()
                block_checks.append(block_check)
                report["latest_block_check"] = {"regime": regime, "index": len(block_checks)-1,
                                                 "checks": block_check}
                save()
                gate(block_check)
            report["timing"][regime] = {"orders": orders, "replays_per_sample": repeats,
                                         "outside_timing_block_checks": block_checks}
            for metric, blocks in (("caller_stream_event", event_blocks), ("synchronized_wall", wall_blocks)):
                report["timing"][regime][metric] = {
                    "blocks_ms": blocks, "median_ms": {name: statistics.median(b[name] for b in blocks) for name in names},
                    "AA": ratio(blocks, "native_pdl", "native_pdl_AA"),
                    "native_over_PAT": ratio(blocks, "native_pdl", "PAT_compat_fixed_scheduler"),
                    "native_over_raw": ratio(blocks, "native_pdl", "raw_pair_grouped_hint_complete"),
                    "PAT_over_raw": ratio(blocks, "PAT_compat_fixed_scheduler", "raw_pair_grouped_hint_complete")}
            save()
        for function in functions.values():
            function()
        torch.cuda.synchronize()
        report["checks"]["final"] = qualify()
        gate(report["checks"]["final"])
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                for name, function in functions.items():
                    with torch.profiler.record_function(name):
                        function()
                    torch.cuda.synchronize()
            trace = output.with_suffix(".trace.json")
            prof.export_chrome_trace(str(trace))
            report["profile_trace"] = str(trace)
        report["status"] = "completed_qualified_eager_only"
        report["limitations"] = [
            "One real BF16 B8 layer/state; no model, request, FP8 or dynamic-scheduler conclusion.",
            "All arms eager on the same declared caller stream. PAT uses a local fixed scheduler and explicit current-stream/event fork-join adaptation; per-call events, streams, temporary allocations and gather are charged. All original failures remain preserved.",
            "PAT may differ from native bits; FP64 tolerance, not native bit equality, gates its performance result.",
            "Prepared metadata operator timing excludes fixture reconstruction and tree build/upload, reported separately.",
            "CPU page table comes from saved fixture. An online GPU-to-CPU metadata readback is not measured or claimed free.",
            "Grouped-hint raw run includes Q packing, pair page/length copies, checkpoint and tail/merge; its host prepare is setup only. No per-shape oracle or old incorrect raw kernel is used.",
            "Cold regime flushes 256MiB before timing. Hot repeated eager calls include Python/host launch gaps.",
            "Every randomized block has an outside-timing FP64 gate; each next arm receives identical warmup/flush treatment. Both caller-stream events and synchronized wall are reported."]
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise
    save()
    print(json.dumps({"status": report["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
