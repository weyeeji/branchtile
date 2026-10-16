"""Historical CPU-only audit of single-buffer lane-select commit b7c7666.

No Torch/TileLang import or GPU work. Candidate sources stay unchanged.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit_operator_register_routes import bank_map, fullcol_pv_map

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REF = "b7c7666"
FROZEN_CANDIDATE_PATHS = {"src/branchtile/tilelang_kernel_lane_select.py",
                          "src/branchtile/attention_lane_select.py"}


def source_bytes(path):
    name = str(Path(path).relative_to(ROOT))
    if name in FROZEN_CANDIDATE_PATHS:
        return subprocess.run(["git", "show", f"{SOURCE_REF}:{name}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
    return Path(path).read_bytes()


class ScalarT:
    @staticmethod
    def if_then_else(condition, yes, no):
        return yes if condition else no

    @staticmethod
    def infinity(dtype):
        assert dtype == "float32"
        return float("inf")


def call(node, name):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name


def subname(node):
    return node.value.id if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) else None


def compile_expr(node):
    return compile(ast.Expression(body=copy.deepcopy(node)), "<candidate-source-expression>", "eval")


def structural_checks():
    path = ROOT / "src/branchtile/tilelang_kernel_lane_select.py"
    tree = ast.parse(source_bytes(path).decode())
    v4 = ast.parse((ROOT / "src/branchtile/tilelang_kernel_v4.py").read_text())
    functions = lambda module: {n.name: n for n in ast.walk(module) if isinstance(n, ast.FunctionDef)}
    f, old = functions(tree), functions(v4)
    for name in ("canonical_layout", "mma_layout", "row_layout"):
        assert ast.dump(f[name]) == ast.dump(old[name])
    macro = f["do_region"]
    main = f["main"]
    macro_calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "do_region"]
    assert len(macro_calls) == 2 and [n.args[-1].value for n in macro_calls] == [16, 32]
    assert len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and
                any(isinstance(d, ast.Attribute) and d.attr == "macro" for d in n.decorator_list)]) == 1
    shared = {n.targets[0].id for n in ast.walk(main) if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name) and call(n.value, "alloc_shared")}
    assert shared == {"qs", "ks", "vs", "probs", "cm", "cl"}
    cu = next(n for n in ast.walk(main) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "cu")
    assert call(cu.value, "alloc_fragment")
    assert ast.unparse(cu.value.args[0]) == "(32, dim)"
    cu_refs = [n for n in ast.walk(macro) if isinstance(n, ast.Subscript) and subname(n) == "CU"]
    assert cu_refs
    for n in cu_refs:
        row = n.slice.elts[0]
        assert {a.id for a in ast.walk(row) if isinstance(a, ast.Name)} <= {"i"}
        offsets = {eval(compile_expr(row), {}, {"i": i}) - i for i in range(8)}
        assert len(offsets) == 1 and offsets <= {0, 8, 16, 24}
        assert ast.unparse(n.slice.elts[1]) == "d"
    policies = [n for n in ast.walk(tree) if call(n, "gemm")]
    assert len(policies) == 2 and all(ast.unparse(next(k.value for k in n.keywords if k.arg == "policy")) == "T.GemmWarpPolicy.FullCol" for n in policies)
    annotated = next(n for n in ast.walk(macro) if call(n, "annotate_layout"))
    layouts = {key.id: ast.unparse(value) for key, value in zip(annotated.args[0].keys, annotated.args[0].values)}
    assert layouts == {"out": "mma_layout(M)", **{name: "row_layout(M)" for name in ("maximum", "previous", "factor", "subtotal", "denominator")}}
    region = next(n for n in ast.walk(main) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "r")
    assert call(region.body[-1].value, "sync_threads")
    compile_call = next(n for n in ast.walk(tree) if call(n, "compile"))
    assert {k.arg for k in compile_call.keywords} == {"target", "out_idx"}
    guard = next(n for n in macro.body if isinstance(n, ast.If))
    assert ast.unparse(guard.test) == "begin < end"

    # The complete token-tile arithmetic is the previously qualified FullCol
    # control, with compact=True resolved in its owner-row predicate only.
    baseline = functions(ast.parse((ROOT / "src/branchtile/tilelang_kernel_fullcol_guard.py").read_text()))["do_region"]
    get_tile = lambda module: next(n for n in ast.walk(module) if isinstance(n, ast.For)
                                   and isinstance(n.target, ast.Name) and n.target.id == "tile")
    original_tile = copy.deepcopy(get_tile(baseline))
    for n in ast.walk(original_tile):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "valid":
            n.value = ast.parse("i < Count[h, r]", mode="eval").body
    assert ast.dump(get_tile(macro)) == ast.dump(original_tile)

    # Extract selectors and metadata expressions from the actual source.
    gather = {}
    for n in ast.walk(macro):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and subname(n.test.left) == "Count":
            threshold = n.test.comparators[0].value
            request = n.body[0]
            targets = [a for a in ast.walk(n) if isinstance(a, ast.Assign) and subname(a.targets[0]) == "out"]
            assert len(targets) == 1 and request.targets[0].id == "request"
            gather[threshold // 8 - 1] = (compile_expr(request.value), compile_expr(targets[0].value))
    assert set(gather) == {0, 1, 2, 3}
    m32_guard = next(n for n in ast.walk(macro) if isinstance(n, ast.If) and ast.unparse(n.test) == "M == 32")
    m32_stores = [n for n in ast.walk(m32_guard) if isinstance(n, ast.Assign) and subname(n.targets[0]) == "out"]
    assert {eval(compile_expr(n.targets[0].slice.elts[0]), {}, {"i": 0}) for n in m32_stores} == {16, 24}
    scatter = {}
    for n in ast.walk(macro):
        if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and subname(n.test.left) == "Owner":
            request = n.test.left.slice.elts[2].value // 8
            slot = n.body[0]
            shape_if = next(a for a in ast.walk(n) if isinstance(a, ast.If) and ast.unparse(a.test) == "M == 16")
            yes, no = shape_if.body[0], shape_if.orelse[0]
            scatter[request] = (compile_expr(slot.value), compile_expr(yes.value), compile_expr(no.value))
            for a in (yes, no):
                assert eval(compile_expr(a.targets[0].slice.elts[0]), {}, {"i": 0}) == request * 8
            assert {eval(compile_expr(a.slice.elts[0]), {}, {"i": 0})
                    for a in ast.walk(yes.value) if isinstance(a, ast.Subscript) and subname(a) == "out"} <= {0, 8}
    assert set(scatter) == {0, 1, 2, 3}
    b4_guard = next(n for n in ast.walk(macro) if isinstance(n, ast.If) and ast.unparse(n.test) == "rows == 32")
    assert any(isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
               and subname(n.test.left) == "Owner" and n.test.left.slice.elts[2].value == 24 for n in ast.walk(b4_guard))
    maximum = next(n.value for n in ast.walk(macro) if isinstance(n, ast.Assign) and subname(n.targets[0]) == "maximum")
    denominator = next(n.value for n in ast.walk(macro) if isinstance(n, ast.Assign) and subname(n.targets[0]) == "denominator")
    scalar_stores = {subname(n.targets[0]): n for n in ast.walk(macro)
                     if isinstance(n, ast.Assign) and subname(n.targets[0]) in ("CM", "CL")}
    assert ast.unparse(scalar_stores["CM"].targets[0]) == "CM[src]" and ast.unparse(scalar_stores["CM"].value) == "maximum[i]"
    assert ast.unparse(scalar_stores["CL"].targets[0]) == "CL[src]" and ast.unparse(scalar_stores["CL"].value) == "denominator[i]"
    return {"layouts_AST_equal_v4": True, "macro_specializations": [16, 32],
            "canonical_CU_all_row_indices_are_i_plus_literal_bank": True,
            "no_shared_CU": True, "five_scalar_row_layouts_preserved": True,
            "full_tile_AST_equals_compact_FullCol": True,
            "M16_scatter_has_only_first_two_candidates": True,
            "B3_fourth_Owner_bank_statically_excluded": True,
            "explicit_region_barrier_and_default_racecheck_preserved": True}, gather, scatter, compile_expr(maximum), compile_expr(denominator), {name: compile_expr(node.value) for name, node in scalar_stores.items()}


def routes(gather, scatter, max_expr, den_expr, scalar_stores):
    records = []
    lane_routes = scalar_checks = scalar_state_cells = state_cells = 0
    for batch in (3, 4):
        for label in range(1, 1 << batch):
            requests = [request for request in range(batch) if label & (1 << request)]
            count = len(requests) * 8
            matrix_rows = 16 if count <= 16 else 32
            rowmap = [request * 8 + head for request in requests for head in range(8)] + [0] * (32 - count)
            cu = np.arange(1, 4097).reshape(32, 128)
            original = cu.copy()
            out = np.zeros((matrix_rows, 128), dtype=np.int64)
            env = {"T": ScalarT, "h": 0, "r": 0, "CU": cu, "out": out,
                   "RowMap": np.array([[rowmap]]), "Count": np.array([[count]]), "label": label}
            for slot, expected_request in enumerate(requests):
                request_expr, value = gather[slot]
                env["request"] = int(eval(request_expr, {}, env))
                assert env["request"] == expected_request
                for head in range(8):
                    for col in range(128):
                        env.update(i=head, d=col)
                        out[slot * 8 + head, col] = eval(value, {}, env)
                        assert out[slot * 8 + head, col] == original[expected_request * 8 + head, col]
                        assert bank_map(expected_request * 8 + head, col)[0] == fullcol_pv_map(slot * 8 + head, col)[0]
                        lane_routes += 1
            for changed in (False, True):
                cu[:] = original
                updated = out.copy() + (10000 if changed else 0)
                env["out"] = updated
                for request in requests:
                    slot_expr, m16, m32 = scatter[request]
                    env["slot"] = int(eval(slot_expr, {}, env))
                    assert env["slot"] == requests.index(request) == (label & ((1 << request) - 1)).bit_count()
                    for head in range(8):
                        for col in range(128):
                            env.update(i=head, d=col)
                            cu[request * 8 + head, col] = eval(m16 if matrix_rows == 16 else m32, {}, env)
                expected = original.copy()
                for slot, request in enumerate(requests):
                    expected[request * 8:request * 8 + 8] = updated[slot * 8:slot * 8 + 8]
                assert np.array_equal(cu, expected)
                state_cells += cu.size
            cm = np.arange(32, dtype=np.float64) / 7
            cm[::3] = -np.inf
            cl = np.where(np.isfinite(cm), np.arange(32) + 1.0, 0.0)
            env.update(CM=cm, CL=cl)
            gathered_m, gathered_l = np.full(matrix_rows, -np.inf), np.zeros(matrix_rows)
            for i in range(matrix_rows):
                env.update(i=i, src=rowmap[i], valid=i < count)
                got_m, got_l = eval(max_expr, {}, env), eval(den_expr, {}, env)
                assert got_m == (cm[rowmap[i]] if i < count else -np.inf)
                assert got_l == (cl[rowmap[i]] if i < count else 0.0)
                gathered_m[i], gathered_l[i] = got_m, got_l
                scalar_checks += 2
            for changed in (False, True):
                actual_m, actual_l = cm.copy(), cl.copy()
                new_m = gathered_m.copy() if not changed else np.arange(matrix_rows) + 0.125
                new_l = gathered_l.copy() if not changed else np.arange(matrix_rows) + 1.5
                env.update(maximum=new_m, denominator=new_l)
                for i in range(count):
                    env.update(i=i, src=rowmap[i])
                    actual_m[rowmap[i]] = eval(scalar_stores["CM"], {}, env)
                    actual_l[rowmap[i]] = eval(scalar_stores["CL"], {}, env)
                expected_m, expected_l = cm.copy(), cl.copy()
                for i in range(count):
                    expected_m[rowmap[i]], expected_l[rowmap[i]] = new_m[i], new_l[i]
                assert np.array_equal(actual_m, expected_m) and np.array_equal(actual_l, expected_l)
                scalar_state_cells += actual_m.size + actual_l.size
            records.append({"batch": batch, "label": label, "requests": requests,
                            "count": count, "M": matrix_rows, "same_lane_routes": count * 128,
                            "M16_M32_scatter_and_inactive_banks_preserved": True})
    scalar_layout_coordinates = 0
    for matrix_rows in (16, 32):
        locations = set()
        for row in range(matrix_rows):
            for rep in range(16):
                thread = (rep % 4) * 32 + (row % 8) * 4 + rep // 4
                register = row // 8
                locations.add((thread, register))
        assert locations == {(thread, register) for thread in range(128) for register in range(matrix_rows // 8)}
        scalar_layout_coordinates += len(locations) * 5
    return {"owner_cases": records, "all_B3_B4_nonempty_owner_labels": len(records),
            "active_U_coordinates_same_lane_and_source_gather_checked": lane_routes,
            "scatter_full_state_cells_checked_identity_and_changed": state_cells,
            "CM_CL_active_and_dummy_gather_checks": scalar_checks,
            "CM_CL_full_state_scatter_cells_identity_and_changed": scalar_state_cells,
            "five_replicated_scalar_layout_coordinates_checked": scalar_layout_coordinates,
            "limit": "CPU source-expression and integer/layout proof; no compilation, GPU race, spill, numerical or speed result."}


def main():
    structure, gather, scatter, maximum, denominator, scalar_stores = structural_checks()
    report = {"status": "CPU_lane_select_source_and_routes_passed", "structure": structure,
              "routing": routes(gather, scatter, maximum, denominator, scalar_stores), "GPU_experiments": 0,
              "candidate_source_git_ref": SOURCE_REF,
              "historical_limit": "Single-buffer source failed later GPU compilation at BufferUseDefCollector; CPU math/routes do not certify compiler legality.",
              "torch_imported": "torch" in sys.modules, "tilelang_imported": "tilelang" in sys.modules}
    names = ["src/branchtile/tilelang_kernel_lane_select.py", "src/branchtile/attention_lane_select.py",
             "src/branchtile/tilelang_kernel_v4.py", "src/branchtile/tilelang_kernel_fullcol_guard.py",
             "scripts/check_lane_select_cpu.py", "scripts/audit_operator_register_routes.py"]
    report["source_sha256"] = {name: hashlib.sha256(source_bytes(ROOT / name)).hexdigest() for name in names}
    assert not report["torch_imported"] and not report["tilelang_imported"]
    output = ROOT / "results/operator_lane_select_cpu_v1/audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "labels": report["routing"]["all_B3_B4_nonempty_owner_labels"],
                      "same_lane_coordinates": report["routing"]["active_U_coordinates_same_lane_and_source_gather_checked"],
                      "full_state_cells": report["routing"]["scatter_full_state_cells_checked_identity_and_changed"],
                      "GPU_experiments": 0}, indent=2))


if __name__ == "__main__":
    main()
