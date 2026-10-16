"""Bounded PAT repeatability diagnostic; default versus official DISABLE_STREAM=1.

Every call synchronizes before FP64 checks. No performance or model claim.
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
SOURCES = ["scripts/qualify_pat_repeatability.py", "src/xqa_raw_pair_build.py",
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
        artifact_only_guard_failure = (
            build.get("status") == "failed" and build.get("CUDA_initialized") is True
            and build.get("module_sha256") == "046e5b5488be33dfd872dae8da8eb1d1b5d37a7a233d7f8a5665204a5fae2444"
            and build.get("error", "").splitlines()[-3:] == [
                "    assert not report['CUDA_initialized']",
                "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", "AssertionError"]
            and 'scripts/build_pat_prior_v2.py' in build.get("error", ""))
        assert build["status"] == "built_unqualified" or artifact_only_guard_failure, build["status"]
        report["build_receipt"]["original_status"] = build["status"]
        report["build_receipt"]["CPU_only_guard_failure_preserved"] = artifact_only_guard_failure
        report["build_receipt"]["original_error"] = build.get("error")
        report["build_receipt"]["admission_scope"] = (
            "Only the hash-pinned compiled artifact is admitted for independent GPU qualification. "
            "The original CPU-only build guard failed after loading: CUDA was initialized. "
            "No successful CPU-only build or clean overlap claim is made."
            if artifact_only_guard_failure else "Successful unqualified build artifact")
        assert build["commit"] == "8cb067f0a31dabe800958f657093f17ac5d7366d"
        assert sha(build["module"]) == build["module_sha256"]
        for path, digest in build["source_sha256"].items():
            assert sha(ROOT / "build/pat_prior" / path) == digest, path
        for path, digest in build["generated_sha256"].items():
            assert sha(ROOT / "build/pat_cuda13_compat" / path) == digest, path
        import torch
        from flashinfer.xqa import xqa
        from src.xqa_raw_pair_build import jit_specs
        from src.xqa_raw_pair_runtime import RawPairPlan
        from src.dense_cascade_fp64_reference import full_history_fp64, compare_to_reference
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)
        assert torch.cuda.get_device_capability() == (12, 0)
        torch.cuda.set_stream(torch.cuda.default_stream())
        assert torch.cuda.current_stream().cuda_stream == 0
        report["environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "device": torch.cuda.get_device_name(), "stream": 0,
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
                                  "tree_metadata_fields_not_exposed_by_official_pybind": True}
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
            report["iterations"].append(record)
            if not record["FP64"]["passed"] or not record["finite"]:
                if report["first_failure"] is None:
                    report["first_failure"] = iteration
                    snapshot = output.with_suffix(".first_failure.pt")
                    torch.save({"PAT_O": value.cpu(), "first_PAT_O": first_pat.cpu(),
                                "native_O": outputs["native_pdl"].cpu(), "reference_FP64": reference.cpu()}, snapshot)
                    report["first_failure_snapshot"] = str(snapshot)
            save()
        for name in ("native_pdl", "native_pdl_AA", "raw_pair_complete"):
            functions[name]()
        torch.cuda.synchronize()
        report["checks"]["final"] = qualify()
        assert all(report["checks"]["final"][name]["FP64"]["passed"] and
                   report["checks"]["final"][name]["native_bit_mismatch_count"] == 0
                   for name in ("native_pdl", "native_pdl_AA", "raw_pair_complete"))
        report["PAT_numerical_pass"] = (report["checks"]["initial"]["PAT_default"]["FP64"]["passed"]
                                          and report["first_failure"] is None)
        report["status"] = "completed_repeatability_diagnostic"
        report["limitations"] = [
            "No timing or implementation superiority claim; failures are preserved rather than relabeled qualified.",
            "Only official DISABLE_STREAM environment option changes between independent processes; mathematics/source unchanged.",
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
