"""Compare union wrapper LSE with direct native single-request FA2, untimed."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import flashinfer
import torch
from branchtile.attention import reference, synthetic_owners
from flashinfer_union import NativeUnionPrefill


def metrics(output, lse, ref, ref_lse):
    o_diff = (output.float() - ref.float()).abs()
    both_empty = torch.isneginf(lse) & torch.isneginf(ref_lse)
    lse_diff = torch.where(both_empty, 0, (lse.float() - ref_lse.float()).abs())
    return {"max_output_error": float(o_diff.max()), "max_ln_lse_error": float(lse_diff.max()),
            "output_passes_frozen_tolerance": bool(torch.isclose(output.float(), ref.float(), atol=0.006, rtol=0.015).all()),
            "lse_passes_frozen_tolerance": bool(torch.isclose(lse.float(), ref_lse.float(), atol=0.0003, rtol=0.0001).all()),
            "empty_rows_have_negative_infinite_lse": bool(torch.isneginf(lse)[torch.isneginf(ref_lse)].all())}


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(4)
    rows = []
    for pattern, b, g, n, split in (("disjoint", 4, 8, 137, 64), ("full", 1, 4, 17, None), ("empty_request", 2, 4, 5, None)):
        torch.manual_seed(603 if pattern == "empty_request" else 602)
        if pattern == "empty_request":
            owners = torch.tensor([[1, 3, 1, 2, 0], [2, 2, 2, 0, 0]], dtype=torch.int64)
        else:
            owners = synthetic_owners(2, n, pattern, b, seed=602)
            owners[:, ::11] = 0
            owners[1, ::7] = 0
            owners[:, :4] = (1 << b) - 1
            for request in range(b):
                owners[:, n - b + request] = 1 << request
        q = torch.randn(2, b * g, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, n, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        ref, ref_lse = reference(q, k, v, owners, b, g)
        union = NativeUnionPrefill(q, k, v, owners, b, fixed_split_size=split, normalize_lse=True)
        union_out, union_lse = union.run()
        direct_out = torch.zeros_like(q)
        direct_lse = torch.full(q.shape[:2], -torch.inf, device=q.device)
        for head in range(2):
            for request in range(b):
                ids = torch.where((owners[head] & (1 << request)) != 0)[0].to(q.device)
                if not len(ids):
                    continue
                # Payload gathering is diagnostic only, never a performance arm.
                o, lse = flashinfer.single_prefill_with_kv_cache(
                    q[head, request * g:(request + 1) * g][None],
                    k[head, ids][:, None], v[head, ids][:, None],
                    causal=False, backend="fa2", return_lse=True)
                direct_out[head, request * g:(request + 1) * g] = o[0]
                direct_lse[head, request * g:(request + 1) * g] = lse[0] * math.log(2)
        torch.cuda.synchronize()
        finite = torch.isfinite(ref_lse)
        rows.append({"pattern": pattern, "B": b, "GQA": g, "N": n, "union_fixed_split_size": split,
                     "union": metrics(union_out, union_lse, ref, ref_lse),
                     "direct_single_fa2": metrics(direct_out, direct_lse, ref, ref_lse),
                     "union_vs_direct_max_ln_lse_difference": float((union_lse[finite] - direct_lse[finite]).abs().max()),
                     "union_vs_direct_max_output_difference": float((union_out.float() - direct_out.float()).abs().max())})
    header = Path(flashinfer.__file__).parent / "data/include/flashinfer/attention/prefill.cuh"
    result = {"purpose": "Diagnose existing frozen-tolerance failures; no benchmark and no changed inputs/reference/tolerance",
              "pid": os.getpid(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "flashinfer": flashinfer.__version__, "torch": torch.__version__, "rows": rows,
              "source_evidence": {"path": str(header), "sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
                  "function": "compute_sfm_v", "probability_cast_line": 1868, "denominator_sum_line": 1880,
                  "observation": "Native FA2 casts FP32 probabilities to DTypeQ (BF16 here), then accumulates denominator d from those rounded probabilities using MMA rowsum. This supports an intrinsic small-support LSE precision limit rather than a union-mask layout error."},
              "qualification_policy": "Retain initial 3 pytest failures. Native union may be timed only for individual shapes/configurations passing the original FP64 output and LSE checks. No universal BF16 numerical qualification is claimed."}
    Path("results/platform/native_union_lse_diagnostic.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
