"""CPU-only structure/mask/online-softmax audit for fixed M32 state residency.

No Torch/TileLang import, compilation, GPU qualification or performance claim.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def call_is(node, name):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name


def stripped(path):
    tree = ast.parse(path.read_text())
    tree.body.pop(0)
    return tree


def structural_checks():
    path = ROOT / "src/branchtile/tilelang_kernel_persistent_fixed.py"
    tree = ast.parse(path.read_text())
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    kernel = next(n for n in main.body if isinstance(n, ast.With))
    region = next(n for n in kernel.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "r")
    allocation = {n.targets[0].id: n for n in kernel.body if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and call_is(n.value, "alloc_fragment")}
    for name, shape in (("out", "(32, dim)"), ("maximum", "(32,)"), ("denominator", "(32,)")):
        assert ast.dump(allocation[name].value.args[0]) == ast.dump(ast.parse(shape, mode="eval").body)
        assert kernel.body.index(allocation[name]) < kernel.body.index(region)
    for n in ast.walk(region):
        assert not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in ("out", "maximum", "denominator")), "State replaced inside region"
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not {"RowMap", "Count", "Owner", "CU", "CM", "CL", "cu", "cm", "cl"} & names
    shared = {n.targets[0].id for n in kernel.body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name) and call_is(n.value, "alloc_shared")}
    assert shared == {"qs", "ks", "vs", "probs"}, "Canonical state added to shared memory"
    initializers = [(n.value.func.attr, ast.unparse(n.value.args[0])) for n in kernel.body
                    if isinstance(n, ast.Expr) and (call_is(n.value, "clear") or call_is(n.value, "fill"))]
    assert initializers == [("clear", "out"), ("fill", "maximum"), ("clear", "denominator")]
    assert not any(isinstance(n, ast.Expr) and (call_is(n.value, "clear") or call_is(n.value, "fill"))
                   and ast.unparse(n.value.args[0]) in ("out", "maximum", "denominator")
                   for n in ast.walk(region))
    qs_stores = [n for n in ast.walk(tree) if isinstance(n, ast.Subscript)
                 and isinstance(n.value, ast.Name) and n.value.id == "qs" and isinstance(n.ctx, ast.Store)]
    assert len(qs_stores) == 1
    assert not any(n is qs_stores[0] for n in ast.walk(region))
    policies = [n for n in ast.walk(tree) if call_is(n, "gemm")]
    assert len(policies) == 2
    assert all(ast.unparse(next(k.value for k in n.keywords if k.arg == "policy")) == "T.GemmWarpPolicy.FullCol" for n in policies)
    guard = next(n for n in region.body if isinstance(n, ast.If))
    assert ast.unparse(guard.test) == "begin < end"
    valid = next(n.value for n in ast.walk(region) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "valid")
    assert ast.dump(valid) == ast.dump(ast.parse("i < rows and (label & (1 << (i // 8))) != 0", mode="eval").body)

    # Softmax/state arithmetic matches the previously audited FullCol control.
    baseline = ast.parse((ROOT / "src/branchtile/tilelang_kernel_fullcol_guard.py").read_text())
    def updates(module, target):
        return [ast.dump(n.value) for n in ast.walk(module) if isinstance(n, (ast.Assign, ast.AugAssign))
                and isinstance(n.target if isinstance(n, ast.AugAssign) else n.targets[0], ast.Subscript)
                and isinstance((n.target if isinstance(n, ast.AugAssign) else n.targets[0]).value, ast.Name)
                and (n.target if isinstance(n, ast.AugAssign) else n.targets[0]).value.id == target]
    for target in ("factor", "denominator"):
        # Baseline denominator includes its initial canonical gather. The
        # online update is its final assignment; only that expression matches.
        assert updates(tree, target)[-1] == updates(baseline, target)[-1]
    assert updates(tree, "scores")[-1] == updates(baseline, "scores")[-1]
    for name in ("begin", "end"):
        rhs = lambda mod: next(n.value for n in ast.walk(mod) if isinstance(n, ast.Assign)
                              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)
        assert ast.dump(rhs(tree)) == ast.dump(rhs(baseline))
    wrapper = stripped(ROOT / "src/branchtile/attention_persistent_fixed.py")
    v4 = stripped(ROOT / "src/branchtile/attention_v4.py")
    for node in ast.walk(wrapper):
        if isinstance(node, ast.ImportFrom) and node.module == "tilelang_kernel_persistent_fixed":
            node.module = "tilelang_kernel_v4"
    assert ast.dump(wrapper) == ast.dump(v4), "Wrapper differs from v4 beyond producer import"
    compile_call = next(n for n in ast.walk(tree) if call_is(n, "compile"))
    assert {k.arg for k in compile_call.keywords} == {"target", "out_idx"}
    return {"all_region_M32": True, "Q_loaded_once": True, "persistent_fragment_state_initialized_once": True,
            "no_RowMap_or_canonical_shared_state": True, "two_FullCol_GEMMs": True,
            "mask_and_shard_formulas_checked": True, "online_softmax_rhs_matches_control": True,
            "wrapper_AST_matches_v4_except_import": True, "no_racecheck_or_pass_override": True}


def dense_reference(q, k, v, owners, batch):
    scores = q @ k.T / math.sqrt(q.shape[1])
    result = np.zeros_like(q)
    lse = np.full(q.shape[0], -np.inf)
    for row in range(batch * 8):
        selected = (owners & (1 << (row // 8))) != 0
        if selected.any():
            values = scores[row, selected]
            maximum = values.max()
            p = np.exp(values - maximum)
            result[row] = p @ v[selected] / p.sum()
            lse[row] = maximum + math.log(p.sum())
    return result, lse


def fixed_reference(q, k, v, owners, batch, splits, block_n):
    labels = sorted(set(owners.tolist()) - {0}) or [1]
    partials, identity_checks, skipped = [], 0, 0
    rows = batch * 8
    canonical_q = np.zeros((32, q.shape[1]))
    canonical_q[:rows] = q
    for split in range(splits):
        out = np.zeros((32, v.shape[1]))
        maximum = np.full(32, -np.inf)
        denominator = np.zeros(32)
        for label in labels:
            ids = np.flatnonzero(owners == label)
            begin, end = len(ids) * split // splits, len(ids) * (split + 1) // splits
            if begin >= end:
                skipped += 1
                continue
            valid_rows = (np.arange(32) < rows) & ((label & (1 << (np.arange(32) // 8))) != 0)
            for start in range(begin, end, block_n):
                selected = ids[start:min(end, start + block_n)]
                scores = canonical_q @ k[selected].T / math.sqrt(q.shape[1])
                scores[~valid_rows] = -np.inf
                old_m, old_l, old_o = maximum.copy(), denominator.copy(), out.copy()
                maximum = np.maximum(maximum, scores.max(axis=1))
                factor = np.zeros(32)
                finite = np.isfinite(old_m)
                factor[finite] = np.exp(old_m[finite] - maximum[finite])
                probs = np.zeros_like(scores)
                probs[valid_rows] = np.exp(scores[valid_rows] - maximum[valid_rows, None])
                denominator = denominator * factor + probs.sum(axis=1)
                out = out * factor[:, None] + probs @ v[selected]
                assert np.array_equal(maximum[~valid_rows], old_m[~valid_rows])
                assert np.array_equal(denominator[~valid_rows], old_l[~valid_rows])
                assert np.array_equal(out[~valid_rows], old_o[~valid_rows])
                identity_checks += int((~valid_rows).sum())
                assert not any(np.isnan(a).any() for a in (maximum, denominator, out))
        partials.append((maximum, denominator, out))
    merged = np.zeros((32, v.shape[1]))
    lse = np.full(32, -np.inf)
    for row in range(32):
        selected = [p for p in partials if p[1][row] > 0]
        if selected:
            maximum = max(p[0][row] for p in selected)
            weights = [math.exp(p[0][row] - maximum) for p in selected]
            den = sum(w * p[1][row] for w, p in zip(weights, selected))
            merged[row] = sum(w * p[2][row] for w, p in zip(weights, selected)) / den
            lse[row] = maximum + math.log(den)
    assert not np.isnan(merged).any() and not np.isnan(lse).any()
    assert not np.count_nonzero(merged[rows:])
    assert np.isneginf(lse[rows:]).all()
    return merged[:rows], lse[:rows], identity_checks, skipped


def numerical_checks():
    rng = np.random.default_rng(20261016)
    cases = []
    for batch in (3, 4):
        tokens, dim = 259, 128
        all_labels = list(range(1, 1 << batch))
        for pattern in ("all_masks", "unequal", "empty", "one_token"):
            owners = np.arange(tokens) % (1 << batch) if pattern == "all_masks" else np.zeros(tokens, dtype=np.int64)
            if pattern == "unequal":
                owners[:] = 1
                owners[:len(all_labels)] = all_labels
            elif pattern == "one_token":
                owners[0] = 1 << (batch - 1)
            q = rng.normal(size=(batch * 8, dim))
            k = rng.normal(size=(tokens, dim))
            v = rng.normal(size=(tokens, dim))
            expected, expected_lse = dense_reference(q, k, v, owners, batch)
            for splits in (1, 3, 8, 64):
                for block_n in (32, 64):
                    actual, lse, identities, skipped = fixed_reference(q, k, v, owners, batch, splits, block_n)
                    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
                    np.testing.assert_allclose(lse, expected_lse, atol=2e-13, rtol=2e-13)
                    assert np.array_equal(np.isneginf(lse), np.isneginf(expected_lse))
                    finite = np.isfinite(lse)
                    cases.append({"batch": batch, "pattern": pattern, "splits": splits, "block_n": block_n,
                                  "output_max_abs": float(np.max(np.abs(actual - expected))),
                                  "LSE_max_abs_finite": float(np.max(np.abs(lse[finite] - expected_lse[finite]))) if finite.any() else 0,
                                  "unowned_row_identity_checks": identities, "skipped_empty_shards": skipped})
    return {"cases": cases, "cases_passed": len(cases),
            "unowned_row_identity_checks": sum(r["unowned_row_identity_checks"] for r in cases),
            "skipped_empty_shards": sum(r["skipped_empty_shards"] for r in cases),
            "limit": "Float64 CPU mathematical simulation; does not model BF16 probability rounding, compiled layouts, races, GPU outputs or speed."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/operator_persistent_fixed_cpu_v1/audit.json")
    args = parser.parse_args()
    result = {"status": "CPU_structure_and_mathematical_checks_passed", "structure": structural_checks(),
              "mathematics": numerical_checks(), "GPU_experiments": 0,
              "torch_imported": "torch" in sys.modules, "tilelang_imported": "tilelang" in sys.modules}
    paths = ["scripts/check_persistent_fixed_cpu.py", "src/branchtile/tilelang_kernel_persistent_fixed.py",
             "src/branchtile/attention_persistent_fixed.py", "src/branchtile/attention_v4.py",
             "src/branchtile/tilelang_kernel_fullcol_guard.py", "src/branchtile/attention.py"]
    result["source_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    assert not result["torch_imported"] and not result["tilelang_imported"]
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": result["mathematics"]["cases_passed"],
                      "unowned_identity_checks": result["mathematics"]["unowned_row_identity_checks"],
                      "skipped_empty_shards": result["mathematics"]["skipped_empty_shards"],
                      "GPU_experiments": 0, "output": str(out.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
