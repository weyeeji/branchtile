"""Numerical qualification of locally fixed PAT scheduler, with NaN scratch and isolated groups.
No performance claim. Both official single-stream and multistream can be tested.
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
SOURCES = ["scripts/qualify_pat_fixed_scheduler_v1.py", "scripts/build_pat_fixed_scheduler_v1.py", "src/xqa_raw_pair_build.py",
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
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--caller-stream", choices=["default", "nondefault"], default="default")
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert 1 <= args.iterations <= 100
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
        assert build["status"] == "built_unqualified_fixed_scheduler"
        assert build["CUDA_initialized"] is False and build["flashinfer_imported"] is False
        assert sha(build["module"]) == build["module_sha256"]
        for name, digest in build["generated_sha256"].items():
            assert sha(ROOT / "build/pat_fixed_scheduler_v1" / name) == digest
        for path, digest in build["reused_cuda_objects"].items():
            assert sha(ROOT / path) == digest
        report["build_receipt"]["admission_scope"] = "Local fixed scheduler with unchanged boxes/tile/KV partition and v4 CUDA objects; not unchanged upstream PAT. NaN scratch enabled, no performance claim."
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_pair_build import jit_specs
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
                                  "use_compute_model": False, "original_multistream": os.environ.get("DISABLE_STREAM") != "1",
                                  "K_V_alias_original_cache": True, "CPU_table_kept_alive": True,
                                  "excluded_official_page16_unsupported_tile": [16, 128, 1],
                                  "tree_metadata_fields_exposed_by_diagnostic_pybind": True}
        info = tree.kernel_info
        schedule = []
        for group, tile in enumerate(info.MNWs):
            schedule.append({"group": group, "tile": tile, "queries": info.q_tables[group].cpu().tolist(),
                             "nseq": info.num_seqs_per_CTAs[group].cpu().tolist(),
                             "rank": info.CTA_ranks[group].cpu().tolist()})
        report["schedule"] = schedule
        split_counts = info.num_split_per_seq.cpu().tolist()
        def scratch_report():
            partial_o, partial_lse = pat.debug_scratch()
            torch.cuda.synchronize()
            partial_o, partial_lse = partial_o.cpu(), partial_lse.cpu()
            details = []
            for row, count in enumerate(split_counts):
                for split in range(count):
                    oo, ll = partial_o[row, :, split], partial_lse[row, :, split]
                    details.append({"row": row, "split": split,
                                    "O_nan": int(torch.isnan(oo).sum()), "LSE_nan": int(torch.isnan(ll).sum()),
                                    "O_all_finite": bool(torch.isfinite(oo).all()), "LSE_all_finite": bool(torch.isfinite(ll).all())})
            return {"shape_O": list(partial_o.shape), "shape_LSE": list(partial_lse.shape), "slots": details}
        names = ["native_pdl", "native_pdl_AA", "PAT_default", "raw_pair_complete"]
        outputs = {name: torch.empty_like(q) for name in names[:-1]}
        outputs["raw_pair_complete"] = raw.output
        workspaces = {name: torch.zeros_like(raw.workspace) for name in names[:2]}
        q4 = q.unsqueeze(1)
        o4 = {name: value.unsqueeze(1) for name, value in outputs.items()}
        def native(name):
            workspace = workspaces[name]
            xqa(q4, k, v, table, raw.lengths, o4[name], workspace[8 * 1024 * 1024:],
                workspace[:8 * 1024 * 1024], 2, 16, q_scale=1., kv_scale=1.,
                kv_layout="HND", sm_count=raw.sm, enable_pdl=True)
        functions = {name: (lambda name=name: native(name)) for name in names[:2]}
        functions["PAT_default"] = lambda: pat.prefix_attn_with_kvcache(q4, pat_k, pat_v, tree, 128**-.5, o4["PAT_default"])
        functions["raw_pair_complete"] = raw.run
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
            assert all(checks[name]["native_bit_mismatch_count"] == 0 for name in ("native_pdl_AA", "raw_pair_complete")), checks
            assert checks["captured_native_O"]["allclose"], checks
        report["checks"]["initial"] = qualify()
        report["scratch_initial"] = scratch_report()
        save()
        # Preserve any PAT failure; other arms must stay valid throughout.
        for name in ("native_pdl", "native_pdl_AA", "raw_pair_complete"):
            assert report["checks"]["initial"][name]["FP64"]["passed"]
            assert report["checks"]["initial"][name]["native_bit_mismatch_count"] == 0
        first_pat = outputs["PAT_default"].clone()
        report["iterations"] = []
        report["first_failure"] = None
        for iteration in range(args.iterations):
            # Check tensors are read only; no changed Q/KV input or changed tree.
            functions["PAT_default"]()
            torch.cuda.synchronize()
            value = outputs["PAT_default"]
            error = (value.double() - reference).abs()
            tolerance = .006 + .015 * reference.abs()
            record = {"iteration": iteration, "FP64": compare_to_reference(value, reference),
                      "finite": bool(torch.isfinite(value).all()),
                      "first_PAT_bit_mismatch_count": int((value.contiguous().view(torch.int16) != first_pat.contiguous().view(torch.int16)).sum()),
                      "failed_values_per_row_head": (error > tolerance).sum(-1).cpu().tolist()}
            record["scratch"] = scratch_report()
            report["iterations"].append(record)
            if not record["FP64"]["passed"] or not record["finite"]:
                if report["first_failure"] is None:
                    report["first_failure"] = iteration
                    snapshot = output.with_suffix(".first_failure.pt")
                    torch.save({"PAT_O": value.cpu(), "first_PAT_O": first_pat.cpu(),
                                "native_O": outputs["native_pdl"].cpu(), "reference_FP64": reference.cpu()}, snapshot)
                    report["first_failure_snapshot"] = str(snapshot)
            save()
        # Run every original MNW group alone with the SAME original slots and NaN others.
        # The gather output is expected to be NaN: this examines temporary writes only.
        full_output = outputs["PAT_default"].clone()
        report["isolated_groups"] = []
        for group in range(len(info.MNWs)):
            pat.prefix_attn_with_kvcache(q4, pat_k, pat_v, info.num_split_per_seq,
                [info.q_tables[group]], [info.block_tables[group]], [info.num_seqs_per_CTAs[group]],
                [info.CTA_ranks[group]], [info.kv_in_CTAs[group]], [info.MNWs[group]],
                info.max_split_per_seq, info.max_seqs_in_CTA, info.max_blocks_in_CTA,
                128**-.5, o4["PAT_default"])
            torch.cuda.synchronize()
            report["isolated_groups"].append({"group": group, "scratch": scratch_report()})
            po, pl = pat.debug_scratch()
            torch.save({"O_partial": po.cpu(), "LSE_partial": pl.cpu(), "schedule": schedule[group]},
                       output.with_suffix(f".group{group}.pt"))
        outputs["PAT_default"].copy_(full_output)
        for name in ("native_pdl", "native_pdl_AA", "raw_pair_complete"):
            functions[name]()
        torch.cuda.synchronize()
        report["checks"]["final"] = qualify()
        assert all(report["checks"]["final"][name]["FP64"]["passed"] and
                   report["checks"]["final"][name]["native_bit_mismatch_count"] == 0
                   for name in ("native_pdl", "native_pdl_AA", "raw_pair_complete"))
        report["PAT_numerical_pass"] = (report["checks"]["initial"]["PAT_default"]["FP64"]["passed"]
                                          and report["first_failure"] is None)
        full_scratch_checks = [report["scratch_initial"]] + [x["scratch"] for x in report["iterations"]]
        report["all_full_gather_slots_finite"] = all(s["O_all_finite"] and s["LSE_all_finite"]
            for check in full_scratch_checks for s in check["slots"])
        report["isolated_group_slot_checks"] = []
        for case in report["isolated_groups"]:
            group = schedule[case["group"]]
            writers = {(row, rank) for qs, n, rank in zip(group["queries"], group["nseq"], group["rank"])
                       for row in qs[:n]}
            failures = []
            for slot in case["scratch"]["slots"]:
                expected = (slot["row"], slot["split"]) in writers
                passed = ((slot["O_all_finite"] and slot["LSE_all_finite"]) if expected
                          else (slot["O_nan"] == 16*128 and slot["LSE_nan"] == 16))
                if not passed: failures.append(slot)
            report["isolated_group_slot_checks"].append({"group": case["group"], "passed": not failures, "failures": failures})
        assert report["PAT_numerical_pass"] and report["all_full_gather_slots_finite"]
        assert all(x["passed"] for x in report["isolated_group_slot_checks"])
        report["status"] = "qualified_fixed_scheduler"
        report["limitations"] = [
            "No timing or implementation superiority claim; failures are preserved rather than relabeled qualified.",
            "Isolated groups intentionally leave other gather slots NaN; only their temporary write coverage is assessed.",
            "Separate explicit fork/join stream-compatibility patch; mathematics/tile/tree unchanged. Default and nondefault caller streams are independently tested.",
            "Each repeat synchronizes before checking FP64. The checks themselves allocate PyTorch tensors, as did the original gate.",
            "One actual BF16 B8 state on SM120; an integration or port failure is not a general claim about PAT."]
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise
    save()
    print(json.dumps({"status": report["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
