"""Small FP64 qualification of executable E1/E4; retain numerical rejections."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from branchtile.attention import reference, synthetic_owners
from partition_baseline import all_partitions, NativePartitionAttention, OwnPartitionAttention
from cascade_baseline import binary_hierarchies, exact_hierarchy_items, ExactCascadeAttention


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="native,cascade,own")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(4)
    assert [len(all_partitions(b)) for b in range(1, 5)] == [1, 2, 5, 15]
    assert len(binary_hierarchies((0, 1, 2, 3))) == 15
    # Independent exact-pair conservation, including nonlaminar overlap.
    owners = synthetic_owners(2, 257, "mixed")
    for tree in binary_hierarchies((0, 1, 2, 3)):
        items, _ = exact_hierarchy_items(owners, tree, 4)
        observed = torch.zeros(2, 4, 257, dtype=torch.int32)
        for head, requests, ids, _ in items:
            for request in requests:
                observed[head, request, ids] += 1
        expected = torch.stack([(owners & (1 << request)) != 0 for request in range(4)], dim=1)
        assert torch.equal(observed, expected.int())
    results = []
    for pattern, b, g in [("mixed", 4, 8), ("empty", 4, 8), ("cross", 3, 4), ("full", 2, 8), ("disjoint", 1, 4)]:
        torch.manual_seed(2000 + b)
        owners = synthetic_owners(2, 257, pattern, b)
        owners[1] &= (1 << max(0, b - 1)) - 1
        q = torch.randn(2, b * g, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(2, 257, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        configurations = []
        for method in a.methods.split(","):
            structures = binary_hierarchies(tuple(range(b))) if method == "cascade" else all_partitions(b)
            if pattern != "mixed":
                structures = structures[:1]
            configurations.extend((method, structure) for structure in structures)
        for method, structure in configurations:
            row = {"method": method, "pattern": pattern, "B": b, "GQA": g, "structure": structure}
            try:
                start = time.perf_counter()
                if method == "cascade":
                    op = ExactCascadeAttention(q, k, v, owners, b, structure)
                elif method == "native":
                    op = NativePartitionAttention(q, k, v, owners, b, structure)
                else:
                    op = OwnPartitionAttention(q, k, v, owners, b, structure, splits=3, block_n=32)
                errors = []
                for repeat in range(2):
                    if repeat:
                        q.normal_()
                    actual, lse = op.run()
                    ref, ref_lse = reference(q, k, v, owners, b, g)
                    errors.append({"max_abs_output_error": float((actual.double() - ref).abs().max()),
                                   "max_abs_lse_error": float((lse.double() - ref_lse).nan_to_num().abs().max())})
                    row["errors"] = errors
                    torch.testing.assert_close(actual.float(), ref.float(), atol=.006, rtol=.015)
                    torch.testing.assert_close(lse, ref_lse.float(), atol=.0003, rtol=.0001)
                assert op.payload_aliases_source
                row.update(status="ok", elapsed_seconds=time.perf_counter() - start, prepare_ms=op.prepare_ms)
            except Exception as exc:
                row.update(status="error", error=repr(exc))
            results.append(row)
            print(json.dumps(row), flush=True)
    report = {"scope": "FP64 E1/E4 qualification; native numerical rejections remain failures; tolerances unchanged",
              "uid": os.geteuid(), "python": sys.executable, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
              "pair_conservation_all_15_trees": True, "rows": results,
              "passed": sum(r["status"] == "ok" for r in results), "failed": sum(r["status"] != "ok" for r in results)}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
