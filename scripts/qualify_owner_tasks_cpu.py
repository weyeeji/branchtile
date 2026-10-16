#!/usr/bin/env python3
"""CPU coverage/packing/streamed-merge checks; no CUDA or tensor imports."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("owner_tasks_plan_cpu", ROOT / "src/branchtile/owner_tasks_plan.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(batch, name):
    n = 1031 if name == "unequal" else (8193 if name == "streaming_33" else 259)
    if name == "all_masks":
        return [[i % (1 << batch) for i in range(n)], [(i*3+1) % (1 << batch) for i in range(n)]]
    if name == "unequal":
        first = [1]*n
        first[:(1 << batch)-1] = list(range(1, 1 << batch))
        return [first, [((1 << batch)-1 if i % 3 == 0 else 1) if i < 19 else 0 for i in range(n)]]
    if name == "empty":
        return [[0]*n, [0]*n]
    if name == "one_token":
        return [[1] + [0]*(n-1), [0]*(n-1) + [1 << (batch-1)]]
    if name == "streaming_33":
        return [[1]*n, [1]*n]
    raise ValueError(name)


def cpu_plan(owners, batch):
    labels = sorted({value for head in owners for value in head} - {0}) or [1]
    offsets, maps, counts, indices = [], [], [], []
    for head in owners:
        packed, offset, row_maps, count = [], [0], [], []
        for label in labels:
            packed.extend(i for i, value in enumerate(head) if value == label)
            offset.append(len(packed))
            active = [request*8+g for request in range(batch) if label & (1 << request) for g in range(8)]
            row_maps.append(active + [0]*(32-len(active)))
            count.append(len(active))
        offsets.append(offset)
        maps.append(row_maps)
        counts.append(count)
        indices.append(packed)
    return offsets, maps, counts, labels, indices


def partial(q, keys, values, selected):
    if not selected:
        return -math.inf, 0.0, [0.0]*len(q)
    scores = [math.fsum(a*b for a, b in zip(q, keys[token]))/math.sqrt(len(q)) for token in selected]
    maximum = max(scores)
    weights = [math.exp(score-maximum) for score in scores]
    denominator = math.fsum(weights)
    numerator = [math.fsum(weight*values[token][d] for token, weight in zip(selected, weights)) for d in range(len(q))]
    return maximum, denominator, numerator


def streamed_merge(parts, dim):
    maximum, denominator, numerator = -math.inf, 0.0, [0.0]*dim
    for start in range(0, len(parts), 16):
        group = parts[start:start+16]
        new_maximum = max(maximum, max(value[0] for value in group))
        factor = math.exp(maximum-new_maximum) if denominator else 0.0
        weights = [math.exp(value[0]-new_maximum) if value[1] else 0.0 for value in group]
        numerator = [numerator[d]*factor + math.fsum(value[2][d]*weight for value, weight in zip(group, weights)) for d in range(dim)]
        denominator = denominator*factor + math.fsum(value[1]*weight for value, weight in zip(group, weights))
        maximum = new_maximum
    return ([value/denominator for value in numerator], maximum+math.log(denominator)) if denominator else ([0.0]*dim, -math.inf)


def check_case(batch, name, chunk):
    owners = fixture(batch, name)
    offsets, maps, counts, labels, indices = cpu_plan(owners, batch)
    layout = MODULE.build_task_layout(offsets, maps, counts, labels, batch, 8, len(owners[0]), chunk)
    covered, packed_extent = set(), 0
    expected = {(h, request*8+g, token) for h, head in enumerate(owners) for token, owner in enumerate(head)
                for request in range(batch) if owner & (1 << request) for g in range(8)}
    for h, region, begin, end, active, base in layout["tasks"]:
        assert begin < end and end-begin <= chunk and base == packed_extent
        packed_extent += active
        for row in maps[h][region][:active]:
            for position in range(begin, end):
                key = h, row, indices[h][position]
                assert key not in covered
                covered.add(key)
    assert covered == expected and packed_extent == layout["packed_rows"]
    dim = 3
    keys = [[[math.cos((h+1)*.7 + token*.013 + d*.19) for d in range(dim)] for token in range(len(head))] for h, head in enumerate(owners)]
    values = [[[math.sin((h+1)*.4 + token*.017 + d*.23) for d in range(dim)] for token in range(len(head))] for h, head in enumerate(owners)]
    queries = [[[math.sin((h+1)*.3 + row*.11 + d*.29) for d in range(dim)] for row in range(batch*8)] for h in range(len(owners))]
    packed = {}
    for h, region, begin, end, active, base in layout["tasks"]:
        selected = indices[h][begin:end]
        for slot, row in enumerate(maps[h][region][:active]):
            packed[base+slot] = partial(queries[h][row], keys[h], values[h], selected)
    assert len(packed) == layout["packed_rows"]
    max_output_error, max_lse_error = 0.0, 0.0
    for h, head in enumerate(owners):
        for request in range(batch):
            count = layout["contribution_counts"][h][request]
            bases = layout["merge_indices"][h][request]
            assert all(value >= 0 for value in bases[:count]) and all(value == -1 for value in bases[count:])
            assert len(set(bases[:count])) == count
            selected = [token for token, owner in enumerate(head) if owner & (1 << request)]
            for g in range(8):
                row = request*8+g
                output, lse = streamed_merge([packed[base+g] for base in bases[:count]], dim)
                m, denominator, numerator = partial(queries[h][row], keys[h], values[h], selected)
                expected_o = [value/denominator for value in numerator] if denominator else [0.0]*dim
                expected_lse = m+math.log(denominator) if denominator else -math.inf
                error = max(abs(a-b) for a, b in zip(output, expected_o))
                assert error <= 1e-12
                assert (lse == expected_lse == -math.inf) or abs(lse-expected_lse) <= 1e-12
                max_output_error = max(max_output_error, error)
                if denominator:
                    max_lse_error = max(max_lse_error, abs(lse-expected_lse))
    return {"batch": batch, "fixture": name, "chunk_tokens": chunk, "task_count": len(layout["tasks"]),
            "packed_rows": layout["packed_rows"], "covered_request_head_tokens": len(covered),
            "max_contributions": layout["max_contributions"], "math_rows_checked": len(owners)*batch*8,
            "max_output_error_fp64": max_output_error, "max_lse_error_fp64": max_lse_error,
            "structure_sha256": layout["structure_sha256"], "status": "passed"}


def version_guard_check():
    tree = ast.parse((ROOT / "src/branchtile/attention_owner_tasks.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Attention")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "validate_plan")
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "owner_task_version_guard", "exec"), scope)
    plan = SimpleNamespace(labels=[1], **{name: SimpleNamespace(_version=0) for name in ("offsets", "counts", "row_map")})
    op = SimpleNamespace(plan=plan, _labels=(1,), _metadata_versions={name: 0 for name in ("offsets", "counts", "row_map")})
    scope["validate_plan"](op)
    plan.offsets._version = 1
    try:
        scope["validate_plan"](op)
    except RuntimeError:
        return True
    raise AssertionError("Stale topology was accepted")


def invalid_metadata_checks():
    owners = fixture(3, "all_masks")
    offsets, maps, counts, labels, _ = cpu_plan(owners, 3)
    changes = []
    bad_offsets = [list(row) for row in offsets]
    bad_offsets[0][1] = -1
    changes.append((bad_offsets, maps, counts, labels, 512))
    bad_maps = [[list(row) for row in head] for head in maps]
    bad_maps[0][0][0:2] = list(reversed(bad_maps[0][0][0:2]))
    changes.append((offsets, bad_maps, counts, labels, 512))
    changes.append((offsets, maps, counts, labels+[labels[-1]], 512))
    changes.append((offsets, maps, counts, labels, 0))
    for off, mapping, count, label, chunk in changes:
        try:
            MODULE.build_task_layout(off, mapping, count, label, 3, 8, len(owners[0]), chunk)
        except ValueError:
            continue
        raise AssertionError("Malformed task metadata was accepted")
    return len(changes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/operator_owner_tasks_cpu_v1/qualification.json")
    a = p.parse_args()
    rows = [check_case(batch, name, chunk) for batch in (3, 4)
            for name in ("all_masks", "unequal", "empty", "one_token") for chunk in (256, 512, 1024)]
    rows += [check_case(batch, "streaming_33", 256) for batch in (3, 4)]
    assert all(row["max_contributions"] > 32 for row in rows if row["fixture"] == "streaming_33")
    assert version_guard_check()
    rejected = invalid_metadata_checks()
    sources = ["src/branchtile/owner_tasks_plan.py", "src/branchtile/attention_owner_tasks.py",
               "src/branchtile/triton_kernel_owner_tasks.py", "scripts/qualify_owner_tasks_cpu.py"]
    result = {"status": "cpu_exact_coverage_packing_merge_math_passed_not_gpu_qualified", "cases": rows,
              "case_count": len(rows), "host_version_guard_rejects_stale_topology": True,
              "malformed_metadata_cases_rejected": rejected,
              "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sources},
              "scope": "Python FP64 math checks task decomposition and streamed merge. No Triton compilation, GPU numerical, timing or BF16-rounding qualification.",
              "cuda_imported": False}
    output = ROOT / a.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(rows), "math_rows": sum(row["math_rows_checked"] for row in rows)}))


if __name__ == "__main__":
    main()
