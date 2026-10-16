"""Native XQA raw checkpoint/resume qualification; not a speed benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    out = Path(a.output)
    assert not out.exists(), out
    out.parent.mkdir(parents=True, exist_ok=True)
    r = {"status": "initializing", "scope": __doc__, "checks": [], "gpu_started": False,
         "source_sha256": {x: hashlib.sha256((ROOT / x).read_bytes()).hexdigest() for x in
                           ["src/xqa_raw_continuation_build.py", "scripts/qualify_xqa_raw_selfresume.py"]}}
    def save():
        out.write_text(json.dumps(r, indent=2) + "\n")
    save()
    try:
        import torch
        from src.xqa_raw_continuation_build import jit_spec, STATE_FLOATS_PER_CTA
        torch.set_num_threads(2)
        spec, receipt = jit_spec()
        r["build_receipt"] = receipt
        start = time.perf_counter()
        spec.build()
        r["build_seconds"] = time.perf_counter() - start
        r["cuda_initialized_after_build"] = torch.cuda.is_initialized()
        r["library"] = str(spec.jit_library_path)
        if a.build_only:
            assert not torch.cuda.is_initialized()
            r["status"] = "cpu_build_completed"
            save()
            print(json.dumps({"status": r["status"], "output": str(out)}))
            return
        assert torch.cuda.get_device_capability() == (12, 0)
        r["gpu_started"] = True
        module = spec.load()
        from flashinfer.xqa import xqa
        torch.manual_seed(20261017)
        b, h, d = 8, 2, 128
        # Includes empty checkpoint, mixed-boundary tile, differing private lengths,
        # and splits that have no tail work. No fake shared KV is needed for this gate.
        cases = [
            (0, [5137] * b),
            (16, [5137 + i for i in range(b)]),
            (4096, [4097 + x for x in (0, 1, 15, 127, 128, 255, 256, 511)]),
            (4352, [4353 + x for x in (0, 1, 15, 127, 128, 255, 256, 511)]),
            (4368, [4369 + x for x in (0, 1, 15, 127, 128, 255, 256, 511)]),
        ]
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        for case_id, (prefix, seqs) in enumerate(cases):
            max_pages = (max(seqs) + 15) // 16
            max_len = max_pages * 16
            splits = min(max(1, sm // (b * h)), (max_len + 255) // 256)
            # Physical page permutation + head/token-strided fused layout.
            storage = torch.randn((b * max_pages + 19, 16, h, 256), device="cuda", dtype=torch.bfloat16)
            cache = storage.permute(0, 2, 1, 3)
            k, v = cache[..., :d], cache[..., d:]
            # Public xqa converts HND -> NHD before invoking this C++ wrapper.
            wrapper_k, wrapper_v = k.transpose(1, 2), v.transpose(1, 2)
            table = torch.randperm(storage.shape[0], device="cuda")[:b * max_pages].reshape(b, max_pages).to(torch.int32)
            lengths = torch.tensor(seqs, device="cuda", dtype=torch.uint32).reshape(b, 1)
            q = torch.randn((b, 1, 16, d), device="cuda", dtype=torch.bfloat16)
            outputs = {key: torch.empty_like(q) for key in ("native", "production_pdl", "control", "resume", "aa")}
            workspace = torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            sem, scratch = workspace[:8 * 1024 * 1024], workspace[8 * 1024 * 1024:]
            state = torch.full((b * h * splits * STATE_FLOATS_PER_CTA,), float("nan"), device="cuda", dtype=torch.float32)
            def native(name, pdl=False):
                xqa(q, k, v, table, lengths, outputs[name], scratch, sem, h, 16,
                    q_scale=1., kv_scale=1., kv_layout="HND", sm_count=sm, enable_pdl=pdl)
            def private(name, phase):
                module.xqa_wrapper(False, sm, h, 0, 1., None, outputs[name], 1., q,
                                   None, wrapper_k, wrapper_v, None, None, table, max_len, lengths, b,
                                   1., None, 1, None, None, sem, scratch, False,
                                   state, prefix // 256, phase)
            native("native")
            native("production_pdl", True)
            private("control", 0)
            check = {"case": case_id, "prefix": prefix, "checkpoint_tokens": prefix // 256 * 256,
                     "lengths": seqs, "S": splits, "state_bytes": state.numel() * 4,
                     "source_cache_stride": list(cache.stride()), "table_dtype": str(table.dtype)}
            check["native_pdl_equal"] = torch.equal(outputs["native"], outputs["production_pdl"])
            check["private_unsplit_equal"] = torch.equal(outputs["native"], outputs["control"])
            check["native_AA_10"] = True
            for _ in range(10):
                native("aa")
                check["native_AA_10"] &= torch.equal(outputs["native"], outputs["aa"])
            outputs["resume"].fill_(float("nan"))
            private("resume", 1)
            check["checkpoint_did_not_write_O"] = bool(torch.isnan(outputs["resume"]).all())
            check["state_all_finite"] = bool(torch.isfinite(state).all())
            raw = state.reshape(b, h, splits, 4, 32, 35)
            active_acc = [4 * n + j for n in range(8) for j in range(2)]
            check["active_state_finite"] = bool(torch.isfinite(raw[..., active_acc]).all()) and bool(
                torch.isfinite(raw[..., :8, 32:35]).all())
            private("resume", 2)
            check["resume_bitwise_equal"] = torch.equal(outputs["native"], outputs["resume"])
            check["resume_maxabs"] = float((outputs["native"].float() - outputs["resume"].float()).abs().max())
            check["different_values"] = int((outputs["native"] != outputs["resume"]).sum())
            # Independent full-history FP64 support check.
            refs = []
            for row, length in enumerate(seqs):
                token = torch.arange(length, device="cuda")
                values = cache[table[row, token // 16].long(), :, token % 16]
                kk = values[..., :d].double().repeat_interleave(8, dim=1)
                vv = values[..., d:].double().repeat_interleave(8, dim=1)
                scores = torch.einsum("hd,thd->ht", q[row, 0].double(), kk) / d ** .5
                refs.append(torch.einsum("ht,thd->hd", scores.softmax(-1), vv))
            ref = torch.stack(refs).unsqueeze(1)
            check["native_FP64_pass"] = torch.allclose(outputs["native"].double(), ref, atol=.006, rtol=.015)
            check["resume_FP64_pass"] = torch.allclose(outputs["resume"].double(), ref, atol=.006, rtol=.015)
            r["checks"].append(check)
            save()
            required = ["native_pdl_equal", "private_unsplit_equal", "native_AA_10", "checkpoint_did_not_write_O",
                        "active_state_finite", "resume_bitwise_equal", "native_FP64_pass", "resume_FP64_pass"]
            assert all(check[key] for key in required), check
            # Capture and replay both kernels in one graph, with the same original split count.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                private("resume", 1)
                private("resume", 2)
            graph.replay()
            torch.cuda.synchronize()
            check["graph_resume_bitwise_equal"] = torch.equal(outputs["native"], outputs["resume"])
            save()
            assert check["graph_resume_bitwise_equal"]
        r["status"] = "qualified_self_resume_only"
        r["limitations"] = ["No cross-request sharing or speed claim.", "Runtime PDL disabled in candidate; both PDL native outputs explicitly checked.",
                            "No request-exit or whole-model integration; only the declared BF16 head128 GQA8 cases."]
    except Exception:
        r["status"] = "failed"
        r["error"] = traceback.format_exc()
        save()
        raise
    save()
    print(json.dumps({"status": r["status"], "output": str(out)}))


if __name__ == "__main__":
    main()
