"""Raw-state XQA pairing qualification with live graph-input changes.

No speed claim. Actual captures are full-history native states with private KV.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--capture")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    out = Path(a.output)
    assert not out.exists(), out
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "initializing", "scope": __doc__, "checks": [], "gpu_started": False,
              "source_sha256": {x: hashlib.sha256((ROOT / x).read_bytes()).hexdigest() for x in
                  ["src/xqa_raw_pair_build.py", "src/xqa_raw_pair_runtime.py", "scripts/qualify_xqa_raw_pair.py"]}}
    def save():
        out.write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        import torch
        from src.xqa_raw_pair_build import jit_specs
        torch.set_num_threads(2)
        specs, receipt = jit_specs()
        report["build_receipt"] = receipt
        for spec in specs.values():
            spec.build()
        if a.build_only:
            assert not torch.cuda.is_initialized()
            report["status"] = "cpu_build_completed"
            report["cuda_initialized"] = False
            save()
            return
        assert torch.cuda.get_device_capability() == (12, 0)
        report["gpu_started"] = True
        modules = {g: spec.load() for g, spec in specs.items()}
        from src.xqa_raw_pair_runtime import RawPairPlan
        from flashinfer.xqa import xqa
        torch.manual_seed(20261017)

        def bits(a, b):
            return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))

        def reference(q, cache, table, lens):
            values = []
            for row, length in enumerate(lens):
                t = torch.arange(length, device=q.device)
                kv = cache[table[row, t // 16].long(), :, t % 16].double()
                scores = torch.einsum("hgd,thd->hgt", q[row].double().reshape(2, 8, 128), kv[..., :128]) / 128 ** .5
                values.append(torch.einsum("hgt,thd->hgd", scores.softmax(-1), kv[..., 128:]).reshape(16, 128))
            return torch.stack(values)

        def synthetic(prefix):
            tails = [1, 1, 17, 31, 129, 255, 257, 511]
            lens = [prefix + x for x in tails]
            maxpages = (max(lens) + 15) // 16 + 2
            pages = 8 * maxpages + 32
            cache = torch.randn((pages, 16, 2, 256), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
            table = torch.randperm(pages, device="cuda")[:8 * maxpages].reshape(8, maxpages).int()
            table[1::2, :prefix // 16] = table[::2, :prefix // 16]
            q = torch.randn((8, 16, 128), device="cuda", dtype=torch.bfloat16)
            return q, cache, table, lens, None

        def captured(path):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            assert payload["q"].dtype == payload["prefix_kv"].dtype == torch.bfloat16
            assert payload["k_scale"] == payload["v_scale"] == 1.
            prefix = payload["prefix_kv"].shape[1]
            q = payload["q"].contiguous().cuda()
            shape, stride = payload["original_cache_shape"], payload["original_cache_stride"]
            cache = torch.empty_strided(shape, stride, device="cuda", dtype=torch.bfloat16)
            cache.zero_()
            table = payload["original_block_table"].to(device="cuda", dtype=torch.int32).contiguous()
            pkv = payload["prefix_kv"].cuda()
            lens = [prefix + v.shape[1] for v in payload["tail_kv"]]
            assert lens == (payload["positions"] + 1).tolist()
            for row in range(8):
                kv = torch.cat((pkv, payload["tail_kv"][row].cuda()), dim=1)
                t = torch.arange(lens[row], device="cuda")
                cache[table[row, t // 16].long(), :, t % 16] = torch.cat((kv[0], kv[1]), dim=-1)
            report["capture"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "layer": payload["layer"], "rank": payload["rank"],
                                 "cache_shape": shape, "cache_stride": stride, "payload_cast_performed": payload["payload_cast_performed"]}
            assert not payload["payload_cast_performed"]
            return prefix, (q, cache, table, lens, payload["o"].cuda())

        fixtures = []
        if a.capture:
            fixtures.append(captured(Path(a.capture)))
        else:
            for prefix in (256, 768, 4096, 4368):
                fixtures.append((prefix, synthetic(prefix)))
        for ci, (prefix, fixture) in enumerate(fixtures):
            q, cache, table, lens, captured_o = fixture
            plan = RawPairPlan(modules, q, cache, table, lens, prefix)
            workspace = torch.zeros_like(plan.workspace)
            native_o, native_pdl = torch.empty_like(q), torch.empty_like(q)
            original_q, original_cache, original_table = q.clone(), cache.clone(), table.clone()
            def native(out_tensor, pdl):
                xqa(q.unsqueeze(1), cache[..., :128], cache[..., 128:], table, plan.lengths,
                    out_tensor.unsqueeze(1), workspace[8 * 1024 * 1024:], workspace[:8 * 1024 * 1024],
                    2, 16, q_scale=1., kv_scale=1., kv_layout="HND", sm_count=plan.sm, enable_pdl=pdl)
            native(native_o, False)
            native(native_pdl, True)
            result = plan.run()
            c = {"case": ci, "prefix": prefix, "lengths": lens, "S": plan.splits,
                 "state_bytes": plan.state.numel() * 4, "native_pdl_bits": bits(native_o, native_pdl),
                 "paired_bits": bits(native_o, result), "maxabs": float((native_o.float() - result.float()).abs().max()),
                 "different_bit_values": int((native_o.view(torch.int16) != result.view(torch.int16)).sum())}
            c["native_AA_10_bits"] = True
            for _ in range(10):
                native(native_pdl, True)
                c["native_AA_10_bits"] &= bits(native_o, native_pdl)
            ref = reference(q, cache, table, lens)
            c["native_FP64_pass"] = torch.allclose(native_o.double(), ref, atol=.006, rtol=.015)
            c["paired_FP64_pass"] = torch.allclose(result.double(), ref, atol=.006, rtol=.015)
            if captured_o is not None:
                c["captured_original_O_pass"] = torch.allclose(native_o, captured_o, atol=.006, rtol=.015)
            c["inputs_not_mutated"] = bits(q, original_q) and bits(cache, original_cache) and torch.equal(table, original_table)
            report["checks"].append(c)
            save()
            required = ["native_pdl_bits", "paired_bits", "native_AA_10_bits", "native_FP64_pass", "paired_FP64_pass", "inputs_not_mutated"]
            assert all(c[k] for k in required), c
            assert c.get("captured_original_O_pass", True), c
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                plan.run()
            graph.replay()
            torch.cuda.synchronize()
            c["graph_initial_bits"] = bits(native_o, plan.output)
            # Change Q, physical pages, actual KV values and private lengths in-place.
            q.mul_(.875)
            lastpage = int(table[0, (lens[0] - 1) // 16])
            cache[lastpage, :, 0, :128].add_(.125)
            # Permute all physical pages, then update every table entry. This keeps the
            # logical KV unchanged while exercising real page indirection under one graph.
            perm = torch.arange(cache.shape[0] - 1, -1, -1, device="cuda")
            cache.copy_(cache.index_select(0, perm))
            table.copy_((cache.shape[0] - 1 - table).int())
            changed_lens = [prefix + 1 if i == 7 else n for i, n in enumerate(lens)]
            plan.prepare(changed_lens)
            plan.state.fill_(float("nan"))
            native(native_o, True)
            graph.replay()
            torch.cuda.synchronize()
            c["graph_changed_input_bits"] = bits(native_o, plan.output)
            # Swap pair members, keeping packed pair grouping unchanged.
            order = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device="cuda")
            q.copy_(q.index_select(0, order))
            table.copy_(table.index_select(0, order))
            swapped_lens = [changed_lens[i] for i in (1, 0, 3, 2, 5, 4, 7, 6)]
            plan.prepare(swapped_lens)
            native(native_o, True)
            graph.replay()
            torch.cuda.synchronize()
            c["graph_swapped_pair_bits"] = bits(native_o, plan.output)
            old_page = table[1, 0].clone()
            table[1, 0] = (table[0, 0] + 1) % cache.shape[0]
            try:
                plan.prepare(swapped_lens)
                c["invalid_prefix_rejected"] = False
            except AssertionError:
                c["invalid_prefix_rejected"] = True
            table[1, 0] = old_page
            save()
            assert all(c[k] for k in ["graph_initial_bits", "graph_changed_input_bits", "graph_swapped_pair_bits", "invalid_prefix_rejected"]), c
        report["status"] = "qualified_pair_only"
        report["limitations"] = ["B8 BF16 head128 GQA8 only; no odd B or request exit.",
                                  "No whole-model or performance result; preparation uses host checks.",
                                  "Runtime PDL off in candidate, production native PDL on explicitly checked."]
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise
    save()
    print(json.dumps({"status": report["status"], "output": str(out)}))


if __name__ == "__main__":
    main()
