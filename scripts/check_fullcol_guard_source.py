"""CPU AST audit: exactly two GEMM warp-policy changes from emptyguard.

The control still gathers/scatters canonical state through shared memory. GPU
compilation and FP64 qualification remain mandatory before performance use.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from check_emptyguard_source import source_checks, without_module_docstring

ROOT = Path(__file__).resolve().parents[1]


def check():
    baseline_path = ROOT / "src/branchtile/tilelang_kernel_emptyguard.py"
    candidate_path = ROOT / "src/branchtile/tilelang_kernel_fullcol_guard.py"
    baseline = without_module_docstring(ast.parse(baseline_path.read_text()))
    candidate = without_module_docstring(ast.parse(candidate_path.read_text()))
    changes = []
    for node in ast.walk(candidate):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "gemm":
            policy = next(k.value for k in node.keywords if k.arg == "policy")
            assert ast.dump(policy.value) == ast.dump(ast.parse("T.GemmWarpPolicy", mode="eval").body)
            assert policy.attr == "FullCol"
            changes.append({"line": node.lineno, "from": "T.GemmWarpPolicy.FullRow", "to": "T.GemmWarpPolicy.FullCol"})
            policy.attr = "FullRow"
    assert len(changes) == 2
    assert ast.dump(baseline) == ast.dump(candidate), "Change beyond two GEMM warp policies"
    original_wrapper = without_module_docstring(ast.parse((ROOT / "src/branchtile/attention_emptyguard.py").read_text()))
    candidate_wrapper = without_module_docstring(ast.parse((ROOT / "src/branchtile/attention_fullcol_guard.py").read_text()))
    imports = 0
    for node in ast.walk(candidate_wrapper):
        if isinstance(node, ast.ImportFrom) and node.module == "tilelang_kernel_fullcol_guard":
            node.module = "tilelang_kernel_emptyguard"
            imports += 1
    assert imports == 1
    assert ast.dump(original_wrapper) == ast.dump(candidate_wrapper)
    return {"only_kernel_AST_changes": changes,
            "canonical_shared_CM_CL_CU_gather_scatter_preserved": True,
            "uniform_empty_shard_guard_preserved": True,
            "wrapper_unchanged_except_kernel_import": True,
            "emptyguard_original_delta_checked": source_checks()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/operator_fullcol_guard_cpu_v1/audit.json")
    args = parser.parse_args()
    names = ["src/branchtile/tilelang_kernel_emptyguard.py", "src/branchtile/tilelang_kernel_fullcol_guard.py",
             "src/branchtile/attention_emptyguard.py", "src/branchtile/attention_fullcol_guard.py",
             "scripts/check_emptyguard_source.py", "scripts/check_fullcol_guard_source.py"]
    result = {"status": "CPU_source_checks_passed", "checks": check(),
              "GPU_experiments": 0, "GPU_qualification_established": False,
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
