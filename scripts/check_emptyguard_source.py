"""CPU-only audit of the independent v2 empty-shard guard ablation.

This verifies the source delta and empty-shard algebra. It does not compile the
TileLang kernel or establish GPU numerical qualification/performance.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_module_docstring(tree):
    tree = copy.deepcopy(tree)
    assert isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Constant)
    assert isinstance(tree.body[0].value.value, str)
    tree.body.pop(0)
    return tree


def source_checks():
    original = ROOT / "src/branchtile/tilelang_kernel_v2.py"
    candidate = ROOT / "src/branchtile/tilelang_kernel_emptyguard.py"
    base = without_module_docstring(ast.parse(original.read_text()))
    guarded = without_module_docstring(ast.parse(candidate.read_text()))
    loops = [node for node in ast.walk(guarded)
             if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
             and node.target.id == "r"]
    assert len(loops) == 1
    loop = loops[0]
    assert len(loop.body) == 3
    begin, end, condition = loop.body
    assert isinstance(begin, ast.Assign) and begin.targets[0].id == "shard_begin"
    assert isinstance(end, ast.Assign) and end.targets[0].id == "shard_end"
    assert ast.dump(condition.test) == ast.dump(ast.parse("shard_begin < shard_end", mode="eval").body)
    assert not condition.orelse
    macro = next(node for node in ast.walk(guarded)
                 if isinstance(node, ast.FunctionDef) and node.name == "do_region")
    assignments = {node.targets[0].id: node.value for node in macro.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert ast.dump(begin.value) == ast.dump(assignments["begin"])
    assert ast.dump(end.value) == ast.dump(assignments["end"])
    # No lane/thread-dependent source participates in the branch. h/s/r are
    # CTA grid ids / the uniform serial region index.
    assert {n.id for n in ast.walk(condition.test) if isinstance(n, ast.Name)} == {"shard_begin", "shard_end"}
    loop.body = condition.body
    assert ast.dump(base, include_attributes=False) == ast.dump(guarded, include_attributes=False), \
        "Changes beyond a single uniform empty-shard guard"

    base_wrapper = without_module_docstring(ast.parse((ROOT / "src/branchtile/attention_v2.py").read_text()))
    wrapper = without_module_docstring(ast.parse((ROOT / "src/branchtile/attention_emptyguard.py").read_text()))
    replacements = 0
    for node in ast.walk(wrapper):
        if isinstance(node, ast.ImportFrom) and node.module == "tilelang_kernel_emptyguard":
            node.module = "tilelang_kernel_v2"
            replacements += 1
    assert replacements == 1
    assert ast.dump(base_wrapper) == ast.dump(wrapper), "Wrapper changed allocation, fixed control or merge API"
    return {"only_kernel_delta_is_CTA_uniform_empty_shard_guard": True,
            "guard_uses_identical_begin_end_formulas": True,
            "do_region_math_RowMap_gemm_state_layout_unchanged": True,
            "wrapper_unchanged_except_kernel_import": True}


def empty_state_checks():
    # Exhaustively cover uneven interval splits, including more splits than
    # tokens and nonzero region offsets. The guard must skip exactly zero work.
    tested = 0
    empty = 0
    for start in (0, 7, 129):
        for length in range(258):
            for splits in range(1, 17):
                shards = []
                for s in range(splits):
                    begin = start + length * s // splits
                    end = start + length * (s + 1) // splits
                    shards.extend(range(begin, end))
                    assert (begin < end) == (len(range(begin, end)) > 0)
                    empty += begin == end
                    tested += 1
                assert shards == list(range(start, start + length))

    # With no tile iterations, v2's state gather/scatter is exactly an identity
    # on active rows. A guard must retain every previous row and no NaNs appear
    # in the canonical empty initialization. This is algebra, not a GPU test.
    state_cases = 0
    for rows in (4, 8, 12, 16, 24, 32):
        for count in (4, 8, 12, 16, 24, 32):
            if count > rows:
                continue
            row_map = list(reversed(range(rows)))[:count]
            for initially_empty in (False, True):
                cm = [-math.inf if initially_empty else i / 7 for i in range(rows)]
                cl = [0.0 if initially_empty else i + 1.0 for i in range(rows)]
                cu = [[0.0 if initially_empty else (i + 1) * (d - 2.5)
                       for d in range(8)] for i in range(rows)]
                original = (copy.deepcopy(cm), copy.deepcopy(cl), copy.deepcopy(cu))
                # The baseline macro has no online-softmax update when its
                # loop count is zero; only these gather/scatter operations run.
                gathered = [(cm[src], cl[src], copy.deepcopy(cu[src])) for src in row_map]
                for src, (m, l, u) in zip(row_map, gathered):
                    cm[src], cl[src], cu[src] = m, l, u
                assert (cm, cl, cu) == original
                assert not any(math.isnan(x) for values in cu for x in values)
                assert not any(math.isnan(x) for x in cm + cl)
                state_cases += 1
    # Merge's stated empty-row contract: den=0 makes every weight zero, and
    # valid=False selects divisor=1; therefore O=0 and LSE=-inf, not 0/0.
    assert 0.0 / 1.0 == 0.0
    return {"uneven_split_shards_checked": tested, "empty_shards_checked": empty,
            "empty_gather_scatter_identity_cases": state_cases,
            "all_empty_contract": "CU=0, CM=-inf, CL=0; unchanged inherited merge returns O=0, LSE=-inf",
            "GPU_qualification_established": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    files = ["src/branchtile/tilelang_kernel_v2.py",
             "src/branchtile/tilelang_kernel_emptyguard.py",
             "src/branchtile/attention_v2.py",
             "src/branchtile/attention_emptyguard.py",
             "src/branchtile/attention.py",
             "scripts/check_emptyguard_source.py"]
    result = {"status": "CPU_source_and_algebra_checks_passed", "source": source_checks(),
              "algebra": empty_state_checks(), "GPU_experiments": 0,
              "source_sha256": {name: digest(ROOT / name) for name in files}}
    if args.output:
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
