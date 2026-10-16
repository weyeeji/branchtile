#!/usr/bin/env python3
"""Exercise the actual owner-task host guard on CPU Torch tensors, never CUDA."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/branchtile/attention_owner_tasks.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/operator_owner_tasks_guard_cpu_v1/qualification.json")
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    # Only the two host methods are loaded; the CUDA constructor/JIT modules are not imported.
    tree = ast.parse(SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Attention")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ("_freeze_plan_topology", "validate_plan")]
    scope = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), scope)
    import torch
    assert not torch.cuda.is_initialized()

    def fresh():
        plan = SimpleNamespace(batch=3, gqa=8, rows=24, labels=[1],
                               indices=torch.arange(65, dtype=torch.int32)[:64].reshape(2, 32),
                               offsets=torch.tensor([[0, 32], [0, 32]], dtype=torch.int32),
                               counts=torch.tensor([[8], [8]], dtype=torch.int32),
                               row_map=torch.arange(32, dtype=torch.int32).repeat(2).reshape(2, 1, 32))
        op = SimpleNamespace(plan=plan)
        scope["_freeze_plan_topology"](op)
        scope["validate_plan"](op)
        return op

    rejected = []
    def must_reject(name, change):
        op = fresh()
        change(op)
        try:
            scope["validate_plan"](op)
        except RuntimeError:
            rejected.append(name)
            return
        raise AssertionError(f"Host guard accepted {name}")

    for name in ("offsets", "counts", "row_map"):
        def replace(op, name=name):
            replacement = getattr(op.plan, name).clone()
            assert replacement._version == 0
            setattr(op.plan, name, replacement)
        must_reject(f"{name}_replacement_version0", replace)
        must_reject(f"{name}_inplace_version_change", lambda op, name=name: getattr(op.plan, name).add_(1))
    must_reject("plan_replacement_same_versions", lambda op: setattr(op, "plan", SimpleNamespace(**vars(op.plan))))
    must_reject("indices_replacement_version0", lambda op: setattr(op.plan, "indices", op.plan.indices.clone()))
    must_reject("row_map_view_same_pointer_version", lambda op: setattr(op.plan, "row_map", op.plan.row_map.view_as(op.plan.row_map)))
    must_reject("indices_shape_change", lambda op: op.plan.indices.resize_(1, 64))
    must_reject("indices_stride_change", lambda op: op.plan.indices.as_strided_((2, 32), (1, 2)))
    must_reject("indices_storage_offset_change", lambda op: op.plan.indices.as_strided_((2, 32), (32, 1), 1))
    must_reject("indices_storage_replacement_same_object", lambda op: setattr(op.plan.indices, "data", op.plan.indices.clone()))
    must_reject("indices_dtype_change", lambda op: setattr(op.plan.indices, "data", op.plan.indices.to(torch.int64)))
    must_reject("metadata_storage_replacement_same_object", lambda op: setattr(op.plan.row_map, "data", op.plan.row_map.clone()))
    for field, value in (("batch", 4), ("gqa", 4), ("rows", 32), ("labels", [2])):
        must_reject(f"static_{field}_change", lambda op, field=field, value=value: setattr(op.plan, field, value))

    op = fresh()
    original = op.plan.indices
    ptr, version = original.data_ptr(), original._version
    original.add_(1)
    assert original._version > version and original.data_ptr() == ptr
    scope["validate_plan"](op)
    assert op.plan.indices is original
    assert not torch.cuda.is_initialized()
    output = ROOT / args.output
    assert not output.exists(), "Preserve previous results; choose a fresh output path"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "cpu_real_tensor_host_guard_passed", "rejected_cases": rejected,
              "rejected_count": len(rejected), "original_indices_inplace_values_allowed": True,
              "strong_tensor_references_retained": all(tensor is getattr(op.plan, name) for name, tensor in op._plan_tensors.items()),
              "cuda_initialized": torch.cuda.is_initialized(), "gpu_work_performed": False,
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (SOURCE, Path(__file__).resolve())},
              "scope": "Actual frozen host methods and real CPU Torch tensors. This does not run CUDA or establish safety of external graph.replay; replay bypasses the host guard and topology changes require rebuild/recapture."}
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
