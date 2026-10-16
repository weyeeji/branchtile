"""Complete-call lane-bank screen retaining every register-bank pilot control.

Every partition/tree is executed, not ranked only by CPU tile estimates. The
per-capture winner is a hindsight tuning oracle, not a free deployed selector.
Cold/hot selections are independent. Planning/search wall costs are retained.
"""
import argparse
import ast
import hashlib
import gc
from functools import lru_cache
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CANDIDATE_SOURCES = {
    "e0_native": ("src/flashinfer_baseline.py",),
    "e1_native": ("src/partition_baseline.py",),
    "e1_own": ("src/partition_baseline.py", "src/branchtile/union.py"),
    "e4_native": ("src/cascade_baseline.py", "src/partition_baseline.py"),
    "dynamic_v2": ("src/dynamic_baseline.py", "src/branchtile/attention_v2.py", "src/branchtile/tilelang_kernel_v2.py"),
    "dynamic_v3": ("src/dynamic_baseline.py", "src/branchtile/attention_v3.py", "src/branchtile/tilelang_kernel_v3.py"),
    "dynamic_emptyguard": ("src/branchtile/attention_emptyguard.py", "src/branchtile/tilelang_kernel_emptyguard.py"),
    "dynamic_v4": ("src/branchtile/attention_v4.py", "src/branchtile/tilelang_kernel_v4.py"),
    "dynamic_fullcol_guard": ("src/branchtile/attention_fullcol_guard.py", "src/branchtile/tilelang_kernel_fullcol_guard.py"),
    "dynamic_persistent_fixed": ("src/branchtile/attention_persistent_fixed.py", "src/branchtile/tilelang_kernel_persistent_fixed.py"),
    "dynamic_lane_select": ("src/branchtile/attention_lane_select.py", "src/branchtile/tilelang_kernel_lane_select.py"),
    "dynamic_fullcol_wide": ("src/branchtile/attention_fullcol_guard.py", "src/branchtile/tilelang_kernel_fullcol_guard.py"),
}
COMMON_SOURCES = ("scripts/bench_operator_overhead_lane.py", "scripts/compare_attention.py",
                  "src/branchtile/attention.py", "src/branchtile/tilelang_kernel.py")


def candidate_name(config):
    """Stable per-configuration identity, independent of shuffled run order."""
    name = config["method"]
    if "partition" in config:
        name += "__p-" + "-".join("_".join(map(str, group)) for group in config["partition"])
    if "tree" in config:
        name += "__tree-" + json.dumps(config["tree"], separators=(",", ":"))
    if "fixed_split_size" in config:
        name += "__fixedsplit-" + str(config["fixed_split_size"])
    if "tensor_cores" in config:
        name += "__tc-" + str(int(config["tensor_cores"]))
    if "splits" in config:
        name += f"__splits-{config['splits']}__n-{config['block_n']}"
    if config["method"].startswith("dynamic_"):
        name += f"__threads-{config.get('threads', 128)}"
    return name


def configuration_grid(batch, include_dynamic, candidate_set, partitions, hierarchies):
    """Retain the original baseline grid; append isolated layout candidates."""
    configurations = []
    for partition in partitions:
        configurations.extend({"method": "e1_native", "partition": partition, "fixed_split_size": split}
                              for split in (None,))
        configurations.extend({"method": "e1_own", "partition": partition, "splits": split, "block_n": bn}
                              for split, bn in itertools.product((64,), (64,)))
    configurations.extend({"method": "e4_native", "tree": tree, "fixed_split_size": split}
                          for tree in hierarchies for split in (None,))
    configurations.extend({"method": "e0_native", "tensor_cores": tc, "fixed_split_size": split}
                          for tc in (False, True) for split in (None,))
    if include_dynamic:
        configurations.extend({"method": "dynamic_v2", "version": "v2", "splits": split, "block_n": bn}
                              for split, bn in itertools.product((32, 64, 128), (32, 64)))
        configurations.extend({"method": "dynamic_v3", "version": "v3", "splits": split, "block_n": 64}
                              for split in (32, 64, 128))
        configurations.extend({"method": "dynamic_emptyguard", "version": "emptyguard", "splits": split, "block_n": bn}
                              for split, bn in itertools.product((32, 64, 128), (32, 64)))
        if candidate_set in ("all", "layout"):
            configurations.extend({"method": "dynamic_v4", "version": "v4", "splits": split, "block_n": bn}
                                  for split, bn in itertools.product((32, 64, 128), (32, 64)))
        if candidate_set in ("policy", "layout", "persistent", "bankpilot", "lane"):
            configurations.extend({"method": "dynamic_fullcol_guard", "version": "fullcol_guard", "splits": split, "block_n": bn}
                                  for split, bn in itertools.product((32, 64, 128), (32, 64)))
        if candidate_set in ("persistent", "bankpilot", "lane"):
            configurations.extend({"method": "dynamic_persistent_fixed", "version": "persistent_fixed", "splits": split, "block_n": bn}
                                  for split, bn in itertools.product((32, 64, 128), (32, 64)))
        if candidate_set in ("bankpilot", "lane"):
            configurations.extend({"method": "dynamic_v4", "version": "v4", "splits": 64, "block_n": bn}
                                  for bn in (32, 64))
        if candidate_set == "lane":
            extra_static = ((128, 32), (128, 64)) if batch == 3 else ((32, 32), (128, 64))
            configurations.extend({"method": "dynamic_v4", "version": "v4", "splits": split, "block_n": bn}
                                  for split, bn in extra_static)
            configurations.extend({"method": "dynamic_lane_select", "version": "lane_select",
                                   "splits": split, "block_n": bn, "threads": 128}
                                  for split, bn in itertools.product((32, 64, 128), (32, 64)))
            configurations.extend({"method": "dynamic_fullcol_wide", "version": "fullcol_guard",
                                   "splits": split, "block_n": 64, "threads": 256}
                                  for split in (32, 64, 128))
    return configurations


def required_methods(include_dynamic, candidate_set):
    methods = ("e0_native", "full_union", "e1_native", "e1_own", "e4_native")
    if include_dynamic:
        methods += ("dynamic_v2", "dynamic_v3", "dynamic_emptyguard")
        if candidate_set in ("all", "layout"):
            methods += ("dynamic_v4",)
        if candidate_set in ("policy", "layout", "persistent", "bankpilot", "lane"):
            methods += ("dynamic_fullcol_guard",)
        if candidate_set in ("persistent", "bankpilot", "lane"):
            methods += ("dynamic_persistent_fixed",)
        if candidate_set in ("bankpilot", "lane"):
            methods += ("dynamic_v4",)
        if candidate_set == "lane":
            methods += ("dynamic_lane_select", "dynamic_fullcol_wide")
    return methods


def cpu_contract_check():
    """Load only existing pure enumeration functions via AST, without CUDA."""
    scope = {"itertools": itertools, "lru_cache": lru_cache}
    for path, name in ((ROOT / "src/partition_baseline.py", "all_partitions"),
                       (ROOT / "src/cascade_baseline.py", "binary_hierarchies")):
        tree = ast.parse(path.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    original_path = ROOT / "scripts/bench_operator_overhead.py"
    original_main = next(node for node in ast.parse(original_path.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == "main")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                                                    for target in node.targets)
    start = next(i for i, node in enumerate(original_main.body) if assigns(node, "configurations"))
    stop = next(i for i, node in enumerate(original_main.body) if i > start and assigns(node, "rng"))
    original_grid = compile(ast.Module(body=original_main.body[start:stop], type_ignores=[]), str(original_path), "exec")
    # Preserve every preexisting v3 candidate set, not just baseline counts.
    legacy_path = ROOT / "scripts/bench_operator_overhead_v3.py"
    legacy_scope = {"itertools": itertools, "json": json}
    legacy_tree = ast.parse(legacy_path.read_text())
    legacy_nodes = [node for node in legacy_tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in ("candidate_name", "configuration_grid", "required_methods")]
    exec(compile(ast.Module(body=legacy_nodes, type_ignores=[]), str(legacy_path), "exec"), legacy_scope)
    pilot_path = ROOT / "scripts/bench_operator_overhead_v4.py"
    pilot_tree = ast.parse(pilot_path.read_text())
    pilot_scope = {"itertools": itertools, "json": json}
    pilot_nodes = [node for node in pilot_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("configuration_grid", "required_methods")]
    exec(compile(ast.Module(body=pilot_nodes, type_ignores=[]), str(pilot_path), "exec"), pilot_scope)
    pilot_sources = ast.literal_eval(next(node.value for node in pilot_tree.body if isinstance(node, ast.Assign)
                                         and any(isinstance(target, ast.Name) and target.id == "CANDIDATE_SOURCES" for target in node.targets)))
    assert all(CANDIDATE_SOURCES[method] == paths for method, paths in pilot_sources.items())
    assert CANDIDATE_SOURCES["dynamic_fullcol_wide"] == CANDIDATE_SOURCES["dynamic_fullcol_guard"]
    assert COMMON_SOURCES[0] == "scripts/bench_operator_overhead_lane.py"
    assert "scripts/bench_operator_overhead_v3.py" not in COMMON_SOURCES
    rows = []
    for batch, expected_baseline, expected_policy, expected_layout in ((3, 30, 36, 42), (4, 62, 68, 74)):
        partitions = scope["all_partitions"](batch)
        hierarchies = scope["binary_hierarchies"](tuple(range(batch)))
        baseline = configuration_grid(batch, True, "baseline", partitions, hierarchies)
        policy = configuration_grid(batch, True, "policy", partitions, hierarchies)
        layout = configuration_grid(batch, True, "layout", partitions, hierarchies)
        persistent = configuration_grid(batch, True, "persistent", partitions, hierarchies)
        bankpilot = configuration_grid(batch, True, "bankpilot", partitions, hierarchies)
        lane = configuration_grid(batch, True, "lane", partitions, hierarchies)
        assert len(baseline) == expected_baseline and len(policy) == expected_policy and len(layout) == expected_layout
        assert len(persistent) == expected_policy + 6 == expected_layout
        baseline_names = {candidate_name(config) for config in baseline}
        layout_names = {candidate_name(config) for config in layout}
        policy_names = {candidate_name(config) for config in policy}
        persistent_names = {candidate_name(config) for config in persistent}
        bankpilot_names = {candidate_name(config) for config in bankpilot}
        lane_names = {candidate_name(config) for config in lane}
        original_scope = dict(scope, b=batch, args=SimpleNamespace(include_dynamic=True, candidate_set="baseline"))
        exec(original_grid, original_scope)
        assert {candidate_name(config) for config in original_scope["configurations"]} == baseline_names
        assert baseline_names <= layout_names and len(layout_names) == len(layout)
        assert baseline_names < policy_names < layout_names and len(policy_names) == len(policy)
        assert all(config["method"] != "dynamic_v4" for config in policy)
        assert sum(config["method"] == "dynamic_fullcol_guard" for config in policy) == 6
        assert sum(config["method"] == "dynamic_v4" for config in layout) == 6
        assert sum(config["method"] == "dynamic_fullcol_guard" for config in layout) == 6
        assert policy_names < persistent_names and len(persistent_names) == len(persistent)
        assert all(config["method"] != "dynamic_v4" for config in persistent)
        added = [config for config in persistent if config["method"] == "dynamic_persistent_fixed"]
        assert len(added) == 6
        assert {(config["splits"], config["block_n"]) for config in added} == set(itertools.product((32, 64, 128), (32, 64)))
        assert persistent_names - policy_names == {candidate_name(config) for config in added}
        assert required_methods(True, "persistent") == required_methods(True, "policy") + ("dynamic_persistent_fixed",)
        assert "dynamic_v4" not in required_methods(True, "persistent")
        bank_configs = [config for config in bankpilot if config["method"] == "dynamic_v4"]
        assert len(bankpilot) == len(bankpilot_names) == expected_policy + 8
        assert len(bank_configs) == 2
        assert {(config["splits"], config["block_n"]) for config in bank_configs} == {(64, 32), (64, 64)}
        assert bankpilot_names - persistent_names == {candidate_name(config) for config in bank_configs}
        assert {candidate_name(config) for config in bankpilot if config["method"] != "dynamic_v4"} == persistent_names
        assert set(required_methods(True, "bankpilot")) == set(required_methods(True, "persistent")) | {"dynamic_v4"}
        assert len(lane) == len(lane_names) == (55 if batch == 3 else 87)
        assert bankpilot_names < lane_names
        assert [config for config in lane if config["method"] not in ("dynamic_lane_select", "dynamic_fullcol_wide")
                and not (config["method"] == "dynamic_v4" and config["splits"] != 64)] == bankpilot
        lane_configs = [config for config in lane if config["method"] == "dynamic_lane_select"]
        wide_configs = [config for config in lane if config["method"] == "dynamic_fullcol_wide"]
        extra_static = [config for config in lane if config["method"] == "dynamic_v4" and config["splits"] != 64]
        expected_static = {(128, 32), (128, 64)} if batch == 3 else {(32, 32), (128, 64)}
        assert len(extra_static) == 2 and {(config["splits"], config["block_n"]) for config in extra_static} == expected_static
        assert sum(config["method"] == "dynamic_v4" for config in lane) == 4
        assert lane_names - bankpilot_names == {candidate_name(config) for config in lane_configs + wide_configs + extra_static}
        assert len(lane_configs) == 6 and len(wide_configs) == 3
        assert {(config["splits"], config["block_n"], config["threads"]) for config in lane_configs} == set(itertools.product((32, 64, 128), (32, 64), (128,)))
        assert {(config["splits"], config["block_n"], config["threads"]) for config in wide_configs} == {(split, 64, 256) for split in (32, 64, 128)}
        assert all(config["version"] == "fullcol_guard" for config in wide_configs)
        assert set(required_methods(True, "lane")) == set(required_methods(True, "bankpilot")) | {"dynamic_lane_select", "dynamic_fullcol_wide"}
        lane_control = set(required_methods(True, "lane")) - {"dynamic_lane_select"}
        assert {"dynamic_v4", "dynamic_fullcol_wide", "dynamic_fullcol_guard", "dynamic_persistent_fixed"} <= lane_control
        sample = dict(wide_configs[0])
        assert candidate_name(sample) != candidate_name(dict(sample, threads=128))
        for previous_set in ("baseline", "all", "policy", "layout", "persistent"):
            for include_dynamic in (False, True):
                original = legacy_scope["configuration_grid"](batch, include_dynamic, previous_set, partitions, hierarchies)
                current = configuration_grid(batch, include_dynamic, previous_set, partitions, hierarchies)
                assert original == current
                assert legacy_scope["required_methods"](include_dynamic, previous_set) == required_methods(include_dynamic, previous_set)
        for previous_set in ("baseline", "all", "policy", "layout", "persistent", "bankpilot"):
            for include_dynamic in (False, True):
                assert pilot_scope["configuration_grid"](batch, include_dynamic, previous_set, partitions, hierarchies) == configuration_grid(batch, include_dynamic, previous_set, partitions, hierarchies)
                assert pilot_scope["required_methods"](include_dynamic, previous_set) == required_methods(include_dynamic, previous_set)
        for config in layout + persistent + bankpilot + lane:
            assert all((ROOT / path).is_file() for path in CANDIDATE_SOURCES[config["method"]])
        rows.append({"batch": batch, "partitions": len(partitions), "hierarchies": len(hierarchies),
                     "baseline_configs": len(baseline), "layout_configs": len(layout),
                     "policy_configs": len(policy), "persistent_configs": len(persistent),
                     "bankpilot_configs": len(bankpilot), "bankpilot_v4_configs": len(bank_configs),
                     "lane_configs": len(lane), "lane_method_configs": len(lane_configs),
                     "wide_method_configs": len(wide_configs), "bankpilot_controls_retained_exactly": True,
                     "additional_v4_control_grid": sorted(expected_static), "lane_v4_control_grid_coverage": "4_of_6",
                     "lane_control_contains_v4_and_wide": True, "thread_identity_distinguishes_128_256": True,
                     "bankpilot_all_nonv4_controls_exactly_persistent_set": True,
                     "persistent_is_policy_plus_six": True, "persistent_excludes_v4": True,
                     "all_v3_candidate_sets_retained": True, "baseline_retained": True})
    assert "torch" not in sys.modules and "tilelang" not in sys.modules
    return {"status": "cpu_candidate_contract_passed", "rows": rows, "cuda_imported": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--budget", type=int, default=4096)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--pattern", default="cross")
    p.add_argument("--gqa", type=int, default=8)
    p.add_argument("--blocks", type=int, default=20)
    p.add_argument("--max-tuning-seconds", type=int, default=900)
    p.add_argument("--output", required=True)
    p.add_argument("--include-dynamic", action="store_true")
    p.add_argument("--require-ready", action="store_true")
    p.add_argument("--candidate-set", choices=("baseline", "all", "policy", "layout", "persistent", "bankpilot", "lane"), default="baseline")
    p.add_argument("--cpu-check", action="store_true", help="Validate all B3/B4 candidate grids without importing Torch")
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    if args.cpu_check:
        print(json.dumps(cpu_contract_check()))
        return
    assert args.candidate_set not in ("policy", "layout", "persistent", "bankpilot", "lane") or args.include_dynamic, "dynamic candidate sets require --include-dynamic"
    import numpy as np
    import torch
    from branchtile.attention import reference, synthetic_owners
    from partition_baseline import all_partitions, NativePartitionAttention, OwnPartitionAttention
    from cascade_baseline import binary_hierarchies, ExactCascadeAttention
    from flashinfer_baseline import NativeSparseDecode
    from dynamic_baseline import DynamicAttention
    from compare_attention import load_capture, graph_of, elapsed
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    if args.capture:
        q, k, v, owners, b, g, provenance = load_capture(args.capture, args.layer, args.budget, args.step, args.heads)
    else:
        b, g = 4, args.gqa
        owners = synthetic_owners(args.heads, args.tokens, args.pattern, b, 20261016)
        q = torch.randn(args.heads, b * g, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(args.heads, args.tokens, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        provenance = {"kind": "synthetic", "pattern": args.pattern, "seed": 20261016}
    if args.candidate_set in ("layout", "persistent", "bankpilot", "lane"):
        assert b in (3, 4) and g == 8, "Static/persistent layout scope is B3/B4 GQA8"
    if args.capture:
        meta = json.loads((Path(args.capture) / "metadata.json").read_text())
        for key in ("tensor_parallel_size", "actual_tp_rank", "document_token_range", "actual_shared_document_token_range", "private_protected_document_tail_tokens", "decode_step_definition", "mask_definition"):
            if key in meta:
                provenance[key] = meta[key]
        export_path = Path(args.capture) / "export_summary.json"
        if export_path.exists():
            export = json.loads(export_path.read_text())
            provenance["ready_group"] = next((item for item in export.get("ready_groups", []) if item["layer"] == args.layer and item["anchor"] == args.step), None)
        if args.require_ready:
            assert provenance["ready_lifetime_measured"] and b == meta["B"]
    ref, ref_lse = reference(q, k, v, owners, b, g)
    supports = [[torch.where((owners[h] & (1 << request)) != 0)[0] for request in range(b)] for h in range(k.shape[0])]
    full = (tuple(range(b)),)
    configurations = configuration_grid(b, args.include_dynamic, args.candidate_set,
                                        all_partitions(b), binary_hierarchies(tuple(range(b))))
    source_paths = set(COMMON_SOURCES)
    for config in configurations:
        source_paths.update(CANDIDATE_SOURCES[config["method"]])
    source_sha256 = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sorted(source_paths)}
    rng = random.Random(20261016)
    rng.shuffle(configurations)

    def build(config):
        method = config["method"]
        if method.startswith("dynamic_"):
            if config["version"] in ("v2","v3"):
                return DynamicAttention(q, k, v, owners, b, version=config["version"], splits=config["splits"], block_n=config["block_n"])
            from branchtile.attention import make_plan
            import importlib
            class Bound:
                def __init__(self):
                    started=time.perf_counter()
                    self.plan=make_plan(owners,b,g,q.device)
                    self.metadata_ms=self.plan.prepare_ms
                    module=importlib.import_module("branchtile.attention_"+config["version"])
                    self.op=module.Attention(self.plan,splits=config["splits"],block_n=config["block_n"],threads=config.get("threads",128))
                    self.prepare_ms=(time.perf_counter()-started)*1000
                    self.payload_aliases_source=True
                    self.logical_kv_token_visits=int((owners!=0).sum())
                    self.workspace_bytes=sum(t.numel()*t.element_size() for t in (self.op.u,self.op.m,self.op.l))
                def run(self):
                    return self.op(q,k,v)
            return Bound()
        if method == "e0_native":
            return NativeSparseDecode(q, k, v, supports, use_tensor_cores=config["tensor_cores"],
                                      fixed_split_size=config["fixed_split_size"], normalize_lse=True)
        if method == "e1_native":
            return NativePartitionAttention(q, k, v, owners, b, config["partition"], fixed_split_size=config["fixed_split_size"])
        if method == "e1_own":
            return OwnPartitionAttention(q, k, v, owners, b, config["partition"], splits=config["splits"], block_n=config["block_n"])
        return ExactCascadeAttention(q, k, v, owners, b, config["tree"], fixed_split_size=config["fixed_split_size"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = out.with_suffix(".tuning.jsonl")
    assert not out.exists() and not progress.exists(), "Retain previous results; use a new output path"
    progress.write_text("")
    pdev = torch.cuda.get_device_properties(0)
    flush = torch.zeros(max(2 * pdev.L2_cache_size, 256 * 1024 * 1024) // 4, device="cuda")
    best = {cache: {} for cache in ("hot", "cold")}
    tuning, family_search_seconds = [], {}
    search_started = time.perf_counter()
    for config in configurations:
        if time.perf_counter() - search_started > args.max_tuning_seconds:
            break
        row = dict(config, candidate_name=candidate_name(config),
                   source_paths=list(CANDIDATE_SOURCES[config["method"]]),
                   source_sha256={path: source_sha256[path] for path in CANDIDATE_SOURCES[config["method"]]})
        config_start = time.perf_counter()
        try:
            op = build(config)
            actual, lse = op.run()
            torch.cuda.synchronize()
            row.update(initialization_ms_including_jit=(time.perf_counter() - config_start) * 1000,
                       max_abs_error_fp64=float((actual.double() - ref).abs().max()),
                       max_lse_error_fp64=float((lse.double() - ref_lse).nan_to_num().abs().max()))
            torch.testing.assert_close(actual.float(), ref.float(), atol=.006, rtol=.015)
            torch.testing.assert_close(lse, ref_lse.float(), atol=.0003, rtol=.0001)
            if hasattr(op, "payload_aliases_source"):
                assert op.payload_aliases_source
            hot = graph_of(op.run, 20)
            hot_us = statistics.median(elapsed(hot, 20) for _ in range(5))
            cold = graph_of(op.run, 1)
            cold_us = statistics.median(elapsed(cold, 1, flush) for _ in range(5))
            row.update(status="ok", tuning_us_by_cache={"hot": hot_us, "cold": cold_us},
                       prepare_ms=getattr(op, "prepare_ms", getattr(op, "plan_ms", None)),
                       metadata_ms=getattr(op, "metadata_ms", None),
                       logical_kv_token_visits=getattr(op, "logical_kv_token_visits", None),
                       workspace_bytes=getattr(op, "workspace_bytes", op.workspace.numel() if hasattr(op, "workspace") and op.workspace is not None else None))
            targets = [config["method"]]
            if config.get("partition") == full:
                targets.append("full_union")
            for cache in best:
                for target in targets:
                    if target not in best[cache] or row["tuning_us_by_cache"][cache] < best[cache][target]["record"]["tuning_us_by_cache"][cache]:
                        best[cache][target] = {"record": row.copy(), "config": config, "op": op}
            del hot, cold
        except Exception as exc:
            row.update(status="error", error=repr(exc))
        row["configuration_wall_seconds"] = time.perf_counter() - config_start
        family_search_seconds[config["method"]] = family_search_seconds.get(config["method"], 0) + row["configuration_wall_seconds"]
        tuning.append(row)
        with progress.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    search_seconds = time.perf_counter() - search_started
    required = required_methods(args.include_dynamic, args.candidate_set)
    missing = {cache: [method for method in required if method not in best[cache]] for cache in best}
    measurements, summary, planning = [], {}, {}
    if not any(missing.values()):
        for cache in best:
            # E1 includes the singleton partition's optimized decode alternative.
            best[cache]["best_e1"] = min((best[cache][m] for m in ("e0_native", "e1_native", "e1_own")),
                                           key=lambda x: x["record"]["tuning_us_by_cache"][cache])
        methods = (*required, "best_e1")
        if args.include_dynamic:
            for cache in best:
                best[cache]["best_dynamic"] = min((best[cache][m] for m in required if m.startswith("dynamic_")), key=lambda x: x["record"]["tuning_us_by_cache"][cache])
                best[cache]["best_control"] = min((best[cache][m] for m in ("best_e1", "e4_native")), key=lambda x: x["record"]["tuning_us_by_cache"][cache])
            methods += ("best_dynamic", "best_control")
            if args.candidate_set == "layout":
                for cache in best:
                    best[cache]["best_layout_control"] = min(
                        (best[cache][m] for m in required if m != "dynamic_v4"),
                        key=lambda x: x["record"]["tuning_us_by_cache"][cache])
                methods += ("best_layout_control",)
            if args.candidate_set in ("bankpilot", "lane"):
                for cache in best:
                    best[cache]["best_bank_control"] = min(
                        (best[cache][m] for m in required if m not in ("dynamic_v4", "dynamic_lane_select")),
                        key=lambda x: x["record"]["tuning_us_by_cache"][cache])
                methods += ("best_bank_control",)
            if args.candidate_set == "lane":
                for cache in best:
                    best[cache]["best_lane_control"] = min(
                        (best[cache][m] for m in required if m != "dynamic_lane_select"),
                        key=lambda x: x["record"]["tuning_us_by_cache"][cache])
                methods += ("best_lane_control",)
        for cache, repetitions in (("hot", 20), ("cold", 1)):
            graphs = {method: graph_of(best[cache][method]["op"].run, repetitions) for method in methods}
            graphs["full_union_AA"] = graph_of(best[cache]["full_union"]["op"].run, repetitions)
            for block in range(args.blocks):
                order = list(graphs)
                rng.shuffle(order)
                for direction, sequence in (("random", order), ("reverse", list(reversed(order)))):
                    measurements.append({"cache": cache, "block": block, "direction": direction, "order": sequence,
                                         "us": {m: elapsed(graphs[m], repetitions, flush if cache == "cold" else None) for m in sequence}})
            rows = [r for r in measurements if r["cache"] == cache]
            summary[cache] = {"median_us": {m: statistics.median(r["us"][m] for r in rows) for m in graphs}, "paired_ratios": {}}
            draws = np.random.default_rng(71).integers(0, args.blocks, (10000, args.blocks))
            pairs = [("full_union", "best_e1"), ("full_union", "e4_native"), ("best_e1", "e4_native"), ("full_union", "full_union_AA")]
            if args.include_dynamic:
                pairs += [("best_control", "best_dynamic"), ("best_control", "dynamic_v2"), ("best_control", "dynamic_v3"), ("best_control", "dynamic_emptyguard")]
                if args.candidate_set in ("all", "layout"):
                    pairs += [("best_control", "dynamic_v4")]
                if args.candidate_set in ("policy", "layout", "persistent", "bankpilot", "lane"):
                    pairs += [("best_control", "dynamic_fullcol_guard"),
                              ("dynamic_emptyguard", "dynamic_fullcol_guard")]
                if args.candidate_set == "layout":
                    pairs += [("best_layout_control", "dynamic_v4"),
                              ("dynamic_fullcol_guard", "dynamic_v4")]
                if args.candidate_set in ("persistent", "bankpilot", "lane"):
                    pairs += [("best_control", "dynamic_persistent_fixed"),
                              ("dynamic_fullcol_guard", "dynamic_persistent_fixed")]
                if args.candidate_set in ("bankpilot", "lane"):
                    pairs += [("best_bank_control", "dynamic_v4"),
                              ("dynamic_fullcol_guard", "dynamic_v4"),
                              ("dynamic_persistent_fixed", "dynamic_v4")]
                if args.candidate_set == "lane":
                    pairs += [("best_lane_control", "dynamic_lane_select"),
                              ("best_control", "dynamic_lane_select"),
                              ("dynamic_fullcol_guard", "dynamic_lane_select"),
                              ("dynamic_persistent_fixed", "dynamic_lane_select"),
                              ("dynamic_v4", "dynamic_lane_select"),
                              ("dynamic_fullcol_wide", "dynamic_lane_select"),
                              ("dynamic_fullcol_guard", "dynamic_fullcol_wide")]
            for numerator, denominator in pairs:
                values = np.array([[next(r for r in rows if r["block"] == block and r["direction"] == direction)["us"][numerator] /
                                    next(r for r in rows if r["block"] == block and r["direction"] == direction)["us"][denominator]
                                    for direction in ("random", "reverse")] for block in range(args.blocks)])
                boot = np.median(values[draws], axis=(1, 2))
                summary[cache]["paired_ratios"][f"{numerator}_over_{denominator}"] = {
                    "median": float(np.median(values)), "paired_block_bootstrap95": np.quantile(boot, [.025, .975]).tolist()}
            del graphs
            planning[cache] = {}
            planning_methods = (("full_union", "best_e1", "e4_native", "best_dynamic") if args.include_dynamic
                                else ("full_union", "best_e1", "e4_native"))
            if args.include_dynamic and args.candidate_set == "layout":
                planning_methods += ("dynamic_v4", "dynamic_fullcol_guard", "best_layout_control")
            elif args.include_dynamic and args.candidate_set in ("policy", "persistent", "bankpilot", "lane"):
                planning_methods += ("dynamic_fullcol_guard",)
            if args.include_dynamic and args.candidate_set in ("persistent", "bankpilot", "lane"):
                planning_methods += ("dynamic_persistent_fixed",)
            if args.include_dynamic and args.candidate_set in ("bankpilot", "lane"):
                planning_methods += ("dynamic_v4", "best_bank_control")
            if args.include_dynamic and args.candidate_set == "lane":
                planning_methods += ("dynamic_lane_select", "dynamic_fullcol_wide", "best_lane_control")
            for method in planning_methods:
                samples = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    fresh = build(best[cache][method]["config"])
                    fresh.run()
                    torch.cuda.synchronize()
                    samples.append({"warm_prepare_ms": getattr(fresh, "prepare_ms", getattr(fresh, "plan_ms", None)),
                                    "prepare_plus_one_call_host_wall_us": (time.perf_counter() - started) * 1e6})
                    del fresh
                prepare_us = statistics.median(s["warm_prepare_ms"] for s in samples) * 1000
                planning[cache][method] = {"samples": samples, "warm_prepare_median_us": prepare_us,
                    "amortized_prepare_plus_gpu_call_us": {str(reuse): prepare_us / reuse + summary[cache]["median_us"][method] for reuse in (1, 16, 64, 256)},
                    "note": "Static-plan amortization scenarios, not measured request lifetimes; oracle search cost is reported separately and is not free."}
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
               for path, expected in source_sha256.items()), "Source changed during benchmark"
    result = {"scope": "Broad overhead screen: all fixed partitions/trees with a bounded split grid, not fully tuned strongest methods or an application claim",
              "protocol_version": "lane-v1",
              "candidate_set":args.candidate_set,
              "code_sha256":source_sha256,
              "candidate_sources": {method: list(paths) for method, paths in CANDIDATE_SOURCES.items()
                                    if any(config["method"] == method for config in configurations)},
              "persistent_fixed_ablation_definition": "Fixed M32 persistent rows retain extra masked MMA work versus compact dynamic rows; fullcol_guard/persistent is not a pure state-transport-only change." if args.candidate_set in ("persistent", "bankpilot", "lane") else None,
              "best_bank_control_definition": "Fastest qualified required family excluding v4 and lane_select. Original bankpilot control grid is retained." if args.candidate_set in ("bankpilot", "lane") else None,
              "bankpilot_v4_grid": [{"splits": 64, "block_n": bn} for bn in (32, 64)] if args.candidate_set in ("bankpilot", "lane") else None,
              "best_lane_control_definition": "Fastest qualified non-lane required family, including v4, wide FullCol, persistent, FullCol, v2/v3, native/E1/E4." if args.candidate_set == "lane" else None,
              "lane_grid": [config for config in configurations if config["method"] == "dynamic_lane_select"],
              "wide_control_grid": [config for config in configurations if config["method"] == "dynamic_fullcol_wide"],
              "lane_v4_control_grid": [config for config in configurations if config["method"] == "dynamic_v4"] if args.candidate_set == "lane" else None,
              "lane_v4_control_grid_coverage": "Four of six split/N points: original split64/N32,N64 plus previously qualified B3 split128/N32,N64 or B4 split32/N32 and split128/N64; not a full grid." if args.candidate_set == "lane" else None,
              "best_layout_control_definition": "Fastest qualified non-v4 method among every required family; excludes only register-bank v4." if args.candidate_set == "layout" else None,
              "uid": os.geteuid(), "python": sys.executable, "gpu_uuid": str(pdev.uuid),
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "flashinfer-python", "tilelang", "triton")},
              "q_shape": list(q.shape), "k_shape": list(k.shape), "provenance": provenance,
              "numerical_contract": {"output_atol": .006, "output_rtol": .015, "lse_atol": .0003, "lse_rtol": .0001, "lse_base": "natural", "reference": "FP64"},
              "tuning_planned": len(configurations), "tuning_completed": len(tuning), "tuning_complete": len(tuning) == len(configurations),
              "batch": b, "gqa": g, "dynamic_included": args.include_dynamic,
              "partitions": all_partitions(b), "hierarchies": binary_hierarchies(tuple(range(b))),
              "search_seconds_including_jit_and_failures": search_seconds, "search_seconds_by_family": family_search_seconds,
              "oracle_scope": "Per-capture hindsight latency oracle among the measured grid; not a deployed plan selector.",
              "tuning": tuning, "missing_qualified_methods": missing,
              "selected": {cache: {method: {"config": x["config"], "record": x["record"]} for method, x in methods.items()} for cache, methods in best.items()},
              "measurements": measurements, "summary": summary, "planning": planning,
              "limitations": ["Hierarchical split partial output is native BF16 and merged with FP32 log2-LSE weights; every tested configuration is separately qualified.",
                              "E4 uses exact binary-tree intersection/residual decomposition with all nodes batched, not the original QUILT code/planner.",
                              "Only shared-source KV is zero-copy; query gathering, normalized output restoration and hierarchical merge are included in run().",
                              "Plan preparation and actual prepare+one-run wall time are measured; search is reported separately, not silently amortized away.",
                              "Same-capture hot/cold controls do not establish common-ready overlap, generation quality, full-model speedup or energy improvement.",
                              "Persistent fixed-M32 versus compact FullCol changes masked MMA work; do not attribute the whole difference solely to state transport.",
                              "Lane uses a six-point split/N grid; lane mode strengthens v4 controls to four previously qualified points, while bankpilot mode retains its original two. A v4/lane family comparison alone does not isolate code structure from tuning-grid coverage.",
                              "Bankpilot retains the full control grid but tests v4 at only split64/N32 and split64/N64; it is not a full v4 tuning or an application result."]}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tuning_complete": result["tuning_complete"], "summary": summary, "missing": missing}), flush=True)


if __name__ == "__main__":
    main()
