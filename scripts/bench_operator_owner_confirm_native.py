"""Fixed C128/N64 confirmation against the complete native/E1-own/E4 grid.

Every native partition/tree is executed. Baseline controls are selected by a
per-capture hindsight oracle; the owner C128/N64 arm stays fixed. Hot/cold
control selections are independent. Planning/search wall costs are retained.
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
    "dynamic_owner_tasks": ("src/branchtile/attention_owner_tasks.py", "src/branchtile/owner_tasks_plan.py", "src/branchtile/triton_kernel_owner_tasks.py"),
}
COMMON_SOURCES = ("scripts/bench_operator_owner_confirm_native.py", "scripts/compare_attention.py",
                  "src/branchtile/attention.py", "src/branchtile/tilelang_kernel.py")


FROZEN_OWNER_NAME = "owner_frozen_128_64"
FROZEN_OWNER_CONFIG = {"method": "dynamic_owner_tasks", "version": "owner_tasks",
                       "splits": 64, "block_n": 64, "threads": 128, "chunk_tokens": 128}


def retain_frozen_owner(best, config, record, op):
    """Keep the one predeclared arm for both caches, without a timing comparison."""
    if config != FROZEN_OWNER_CONFIG:
        return
    for cache in best:
        assert FROZEN_OWNER_NAME not in best[cache], "Frozen arm must occur exactly once in the grid"
        best[cache][FROZEN_OWNER_NAME] = {"record": record.copy(), "config": dict(config), "op": op}


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
    if "chunk_tokens" in config:
        name += f"__chunk-{config['chunk_tokens']}"
    return name


def configuration_grid(batch, partitions, hierarchies):
    """Exact original native/E1-own/E4 grid plus one predeclared owner arm."""
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
    configurations.append(dict(FROZEN_OWNER_CONFIG))
    return configurations


def required_methods():
    return ("e0_native", "full_union", "e1_native", "e1_own", "e4_native", "dynamic_owner_tasks")


def cpu_contract_check():
    """Compare exact original native dictionaries without importing tensor libraries."""
    scope = {"itertools": itertools, "lru_cache": lru_cache, "json": json}
    for path, name in ((ROOT / "src/partition_baseline.py", "all_partitions"),
                       (ROOT / "src/cascade_baseline.py", "binary_hierarchies")):
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    previous_path = ROOT / "scripts/bench_operator_overhead_owner_grid.py"
    previous_tree = ast.parse(previous_path.read_text())
    previous_scope = {"itertools": itertools, "json": json}
    previous_nodes = [n for n in previous_tree.body if isinstance(n, ast.FunctionDef)
                      and n.name in ("candidate_name", "configuration_grid")]
    exec(compile(ast.Module(body=previous_nodes, type_ignores=[]), str(previous_path), "exec"), previous_scope)
    previous_sources = ast.literal_eval(next(n.value for n in previous_tree.body if isinstance(n, ast.Assign)
                                            and any(isinstance(t, ast.Name) and t.id == "CANDIDATE_SOURCES" for t in n.targets)))
    assert all(paths == previous_sources[method] for method, paths in CANDIDATE_SOURCES.items())
    assert COMMON_SOURCES[0] == "scripts/bench_operator_owner_confirm_native.py"
    assert FROZEN_OWNER_CONFIG == {"method": "dynamic_owner_tasks", "version": "owner_tasks", "splits": 64,
                                   "block_n": 64, "threads": 128, "chunk_tokens": 128}
    control_families = {"e0_native", "e1_native", "e1_own", "e4_native"}
    rows = []
    for batch, control_count in ((3, 15), (4, 47)):
        partitions = scope["all_partitions"](batch)
        hierarchies = scope["binary_hierarchies"](tuple(range(batch)))
        original = previous_scope["configuration_grid"](batch, True, "baseline", partitions, hierarchies)
        original_controls = [c for c in original if c["method"] in control_families]
        configurations = configuration_grid(batch, partitions, hierarchies)
        controls = [c for c in configurations if c["method"] in control_families]
        assert controls == original_controls
        assert len(controls) == control_count and len(configurations) == control_count + 1
        assert len(original) - len(controls) == 15
        removed = [c for c in original if c["method"] not in control_families]
        assert {c["method"] for c in removed} == {"dynamic_v2", "dynamic_v3", "dynamic_emptyguard"}
        assert [c for c in configurations if c["method"] == "dynamic_owner_tasks"] == [FROZEN_OWNER_CONFIG]
        assert len({candidate_name(c) for c in configurations}) == len(configurations)
        assert all(candidate_name(c) == previous_scope["candidate_name"](c) for c in configurations)
        assert sum(c["method"] == "e1_native" for c in configurations) == len(partitions)
        assert sum(c["method"] == "e1_own" for c in configurations) == len(partitions)
        assert sum(c["method"] == "e4_native" for c in configurations) == len(hierarchies)
        assert {(c["tensor_cores"], c["fixed_split_size"]) for c in configurations if c["method"] == "e0_native"} == {(False,None),(True,None)}
        assert all((c["splits"],c["block_n"]) == (64,64) for c in configurations if c["method"] == "e1_own")
        assert all(c["fixed_split_size"] is None for c in controls if c["method"] != "e1_own")
        probe = {"hot": {}, "cold": {}}
        frozen_op = object()
        for c in reversed(configurations):
            retain_frozen_owner(probe, c, {"candidate_name":candidate_name(c)}, frozen_op)
        assert all(probe[cache][FROZEN_OWNER_NAME]["config"] == FROZEN_OWNER_CONFIG
                   and probe[cache][FROZEN_OWNER_NAME]["op"] is frozen_op for cache in probe)
        assert FROZEN_OWNER_NAME not in required_methods()
        assert all((ROOT/p).is_file() for c in configurations for p in CANDIDATE_SOURCES[c["method"]])
        rows.append({"batch":batch,"partitions":len(partitions),"hierarchies":len(hierarchies),
                     "original_baseline_with_15_dynamic":len(original),"native_controls":len(controls),
                     "planned_configurations":len(configurations),"fixed_owner_configurations":1,
                     "native_E1_own_E4_grid_retained_exactly":True,"removed_old_dynamic_configurations":15,
                     "frozen_config":FROZEN_OWNER_CONFIG,"both_cache_arms_fixed_without_oracle_selection":True,
                     "source_mapping_retained_exactly":True})
    assert "torch" not in sys.modules and "tilelang" not in sys.modules
    return {"status":"cpu_native_fixed_owner_contract_passed","rows":rows,"cuda_imported":False}


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
    p.add_argument("--blocks", type=int, default=40)
    p.add_argument("--max-tuning-seconds", type=int, default=900)
    p.add_argument("--output", required=True)
    p.add_argument("--require-ready", action="store_true")
    p.add_argument("--cpu-check", action="store_true", help="Validate all B3/B4 candidate grids without importing Torch")
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    if args.cpu_check:
        print(json.dumps(cpu_contract_check()))
        return
    import numpy as np
    import torch
    from branchtile.attention import reference, synthetic_owners
    from partition_baseline import all_partitions, NativePartitionAttention, OwnPartitionAttention
    from cascade_baseline import binary_hierarchies, ExactCascadeAttention
    from flashinfer_baseline import NativeSparseDecode
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
    assert b in (3, 4) and g == 8, "Owner confirmation scope is B3/B4 GQA8"
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
    configurations = configuration_grid(b, all_partitions(b), binary_hierarchies(tuple(range(b))))
    source_paths = set(COMMON_SOURCES)
    for config in configurations:
        source_paths.update(CANDIDATE_SOURCES[config["method"]])
    source_sha256 = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in sorted(source_paths)}
    rng = random.Random(20261016)
    rng.shuffle(configurations)

    def build(config):
        method = config["method"]
        if method.startswith("dynamic_"):
            from branchtile.attention import make_plan
            import importlib
            class Bound:
                def __init__(self):
                    started=time.perf_counter()
                    self.plan=make_plan(owners,b,g,q.device)
                    self.region_plan_prepare_ms=self.plan.prepare_ms
                    module=importlib.import_module("branchtile.attention_"+config["version"])
                    options = {"splits": config["splits"], "block_n": config["block_n"], "threads": config.get("threads",128)}
                    if "chunk_tokens" in config:
                        options["chunk_tokens"] = config["chunk_tokens"]
                    self.op=module.Attention(self.plan, **options)
                    self.operator_prepare_ms=getattr(self.op,"prepare_ms",None)
                    self.operator_metadata_ms=getattr(self.op,"metadata_ms",None)
                    self.metadata_ms=self.region_plan_prepare_ms + (self.operator_metadata_ms or 0)
                    self.operator_metadata=getattr(self.op,"metadata",None)
                    self.task_count=getattr(self.op,"num_tasks",None)
                    self.packed_rows=getattr(self.op,"layout",{}).get("packed_rows")
                    self.prepare_scope="Complete RegionPlan construction, module import, task topology/metadata if applicable, all operator allocations/constructor work; first run/JIT is separately included in initialization_ms_including_jit."
                    self.graph_topology_contract=(self.operator_metadata or {}).get("graph_topology_contract")
                    self.payload_aliases_source=getattr(self.op,"payload_aliases_source",True)
                    self.logical_kv_token_visits=(self.op.logical_kv_token_visits if hasattr(self.op,"logical_kv_token_visits")
                                                  else int((owners!=0).sum()))
                    self.workspace_bytes=getattr(self.op,"workspace_bytes",sum(t.numel()*t.element_size() for t in (self.op.u,self.op.m,self.op.l)))
                    self.workspace_bytes_by_buffer=getattr(self.op,"workspace_bytes_by_buffer",None)
                    self.workspace_scope="All owner-task owned buffers including descriptors, merge tables, packed U/M/L and output/LSE; excludes upstream RegionPlan buffers, reported separately." if config["version"]=="owner_tasks" else "U/M/L partial buffers (legacy control convention)."
                    self.region_plan_bytes=sum(t.numel()*t.element_size() for t in self.plan.tensors)
                    self.prepare_ms=(time.perf_counter()-started)*1000
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
            payload_alias = getattr(op, "payload_aliases_source", None)
            if payload_alias is None and hasattr(op, "kv"):
                payload_alias = all(a.data_ptr() == b.data_ptr() for a, b in zip(op.kv, (k, v)))
            row.update(payload_aliases_source=payload_alias, prepare_ms=getattr(op, "prepare_ms", getattr(op, "plan_ms", None)),
                       metadata_ms=getattr(op, "metadata_ms", None),
                       logical_kv_token_visits=getattr(op, "logical_kv_token_visits", None),
                       workspace_bytes=getattr(op, "workspace_bytes", op.workspace.numel() if hasattr(op, "workspace") and op.workspace is not None else None),
                       workspace_bytes_by_buffer=getattr(op,"workspace_bytes_by_buffer",None),
                       workspace_scope=getattr(op,"workspace_scope",None),
                       region_plan_bytes=getattr(op,"region_plan_bytes",None),
                       region_plan_prepare_ms=getattr(op,"region_plan_prepare_ms",None),
                       operator_prepare_ms=getattr(op,"operator_prepare_ms",None),
                       operator_metadata_ms=getattr(op,"operator_metadata_ms",None),
                       operator_metadata=getattr(op,"operator_metadata",None),
                       task_count=getattr(op,"task_count",None), packed_rows=getattr(op,"packed_rows",None),
                       prepare_scope=getattr(op,"prepare_scope",None),
                       graph_topology_contract=getattr(op,"graph_topology_contract",None))
            actual, lse = op.run()
            torch.cuda.synchronize()
            row.update(initialization_ms_including_jit=(time.perf_counter() - config_start) * 1000,
                       max_abs_error_fp64=float((actual.double() - ref).abs().max()),
                       max_lse_error_fp64=float((lse.double() - ref_lse).nan_to_num().abs().max()))
            torch.testing.assert_close(actual.float(), ref.float(), atol=.006, rtol=.015)
            torch.testing.assert_close(lse, ref_lse.float(), atol=.0003, rtol=.0001)
            assert row["payload_aliases_source"] is True, "Every candidate must alias the original K/V source"
            hot = graph_of(op.run, 20)
            hot_us = statistics.median(elapsed(hot, 20) for _ in range(5))
            cold = graph_of(op.run, 1)
            cold_us = statistics.median(elapsed(cold, 1, flush) for _ in range(5))
            row.update(status="ok", tuning_us_by_cache={"hot": hot_us, "cold": cold_us})
            targets = [config["method"]]
            if config.get("partition") == full:
                targets.append("full_union")
            for cache in best:
                for target in targets:
                    if target not in best[cache] or row["tuning_us_by_cache"][cache] < best[cache][target]["record"]["tuning_us_by_cache"][cache]:
                        best[cache][target] = {"record": row.copy(), "config": config, "op": op}
            retain_frozen_owner(best, config, row, op)
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
    required = required_methods()
    missing = {cache: [method for method in (*required, FROZEN_OWNER_NAME) if method not in best[cache]] for cache in best}
    measurements, summary, planning = [], {}, {}
    if not any(missing.values()):
        for cache in best:
            best[cache]["best_e1"] = min((best[cache][m] for m in ("e0_native", "e1_native", "e1_own")),
                                         key=lambda x: x["record"]["tuning_us_by_cache"][cache])
            best[cache]["best_control"] = min((best[cache][m] for m in ("best_e1", "e4_native")),
                                              key=lambda x: x["record"]["tuning_us_by_cache"][cache])
            best[cache]["best_owner_control"] = best[cache]["best_control"]
        methods = (*required, FROZEN_OWNER_NAME, "best_e1", "best_control", "best_owner_control")
        for cache, repetitions in (("hot", 20), ("cold", 1)):
            graphs = {method: graph_of(best[cache][method]["op"].run, repetitions) for method in methods}
            graphs["full_union_AA"] = graph_of(best[cache]["full_union"]["op"].run, repetitions)
            graphs[FROZEN_OWNER_NAME + "_AA"] = graph_of(best[cache][FROZEN_OWNER_NAME]["op"].run, repetitions)
            for block in range(args.blocks):
                order = list(graphs)
                rng.shuffle(order)
                for direction, sequence in (("random", order), ("reverse", list(reversed(order)))):
                    measurements.append({"cache": cache, "block": block, "direction": direction, "order": sequence,
                                         "us": {m: elapsed(graphs[m], repetitions, flush if cache == "cold" else None) for m in sequence}})
            rows = [r for r in measurements if r["cache"] == cache]
            summary[cache] = {"median_us": {m: statistics.median(r["us"][m] for r in rows) for m in graphs}, "paired_ratios": {}}
            draws = np.random.default_rng(71).integers(0, args.blocks, (10000, args.blocks))
            pairs = [("full_union", "best_e1"), ("full_union", "e4_native"), ("best_e1", "e4_native"),
                     ("full_union", "full_union_AA"), (FROZEN_OWNER_NAME, FROZEN_OWNER_NAME + "_AA")]
            pairs += [(control, FROZEN_OWNER_NAME) for control in (
                "best_owner_control", "best_control", "e0_native", "e1_native", "e1_own", "best_e1", "e4_native", "full_union")]
            for numerator, denominator in pairs:
                values = np.array([[next(r for r in rows if r["block"] == block and r["direction"] == direction)["us"][numerator] /
                                    next(r for r in rows if r["block"] == block and r["direction"] == direction)["us"][denominator]
                                    for direction in ("random", "reverse")] for block in range(args.blocks)])
                boot = np.median(values[draws], axis=(1, 2))
                summary[cache]["paired_ratios"][f"{numerator}_over_{denominator}"] = {
                    "median": float(np.median(values)), "paired_block_bootstrap95": np.quantile(boot, [.025, .975]).tolist()}
            del graphs
            planning[cache] = {}
            # Measure preparation for every selected family and the frozen arm,
            # not merely the fastest baseline alias.
            for method in methods:
                samples = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    fresh = build(best[cache][method]["config"])
                    fresh.run()
                    torch.cuda.synchronize()
                    samples.append({"warm_prepare_ms": getattr(fresh, "prepare_ms", getattr(fresh, "plan_ms", None)),
                                    "prepare_plus_one_call_host_wall_us": (time.perf_counter() - started) * 1e6,
                                    "region_plan_prepare_ms": getattr(fresh,"region_plan_prepare_ms",None),
                                    "operator_prepare_ms": getattr(fresh,"operator_prepare_ms",None),
                                    "operator_metadata_ms": getattr(fresh,"operator_metadata_ms",None),
                                    "prepare_scope": getattr(fresh,"prepare_scope",None),
                                    "task_count": getattr(fresh,"task_count",None),
                                    "packed_rows": getattr(fresh,"packed_rows",None),
                                    "workspace_bytes": getattr(fresh,"workspace_bytes",None),
                                    "operator_metadata": getattr(fresh,"operator_metadata",None)})
                    del fresh
                prepare_us = statistics.median(s["warm_prepare_ms"] for s in samples) * 1000
                planning[cache][method] = {"samples": samples, "warm_prepare_median_us": prepare_us,
                    "amortized_prepare_plus_gpu_call_us": {str(reuse): prepare_us / reuse + summary[cache]["median_us"][method] for reuse in (1, 16, 64, 256)},
                    "note": "Static-plan amortization scenarios, not measured request lifetimes; oracle search and graph lifecycle costs are reported separately and are not free."}
    assert all(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
               for path, expected in source_sha256.items()), "Source changed during benchmark"
    result = {"scope": "Predeclared owner C128/N64 versus complete original native/E1-own/E4 bounded grid; development confirmation, not all engineering controls or an application claim",
              "protocol_version": "owner-confirm-native-v1", "arguments": vars(args),
              "candidate_set": "native_E0_E1own_E4_plus_fixed_owner", "code_sha256": source_sha256,
              "candidate_sources": {method: list(paths) for method, paths in CANDIDATE_SOURCES.items()},
              "frozen_owner_confirmation": {"alias": FROZEN_OWNER_NAME, "config": FROZEN_OWNER_CONFIG,
                  "candidate_name": candidate_name(FROZEN_OWNER_CONFIG),
                  "source_paths": list(CANDIDATE_SOURCES["dynamic_owner_tasks"]),
                  "source_sha256": {path: source_sha256[path] for path in CANDIDATE_SOURCES["dynamic_owner_tasks"]},
                  "selection": "Predeclared C128/N64 threads128; fixed across caches and captures, without hindsight owner selection.",
                  "tuning_role": "One owner candidate; its frozen alias adds a measurement, not another tuning candidate.",
                  "required_for_summary": True, "aa_controls": ["full_union_AA", FROZEN_OWNER_NAME + "_AA"],
                  "scope": "Same development document across correlated TP ranks, layers and natural steps; not formal full-dataset validation."},
              "baseline_grid_scope": "Every original E0 native SIMT/tensorcore option, every E1 native and own partition at unchanged bounded split/tile settings, every E4 binary hierarchy. Removes exactly the 15 old v2/v3/emptyguard baseline points and other later dynamic-engineering additions.",
              "best_owner_control_definition": "Fastest qualified original native/E1-own/E4 control, selected separately for hot/cold. Does not claim to cover old v4/lane/persistent engineering controls.",
              "owner_splits_semantics": "splits=64 is API compatibility only; fixed chunk_tokens=128 defines tasks.",
              "uid": os.geteuid(), "python": sys.executable, "gpu_uuid": str(pdev.uuid),
              "packages": {name: importlib.metadata.version(name) for name in ("torch", "flashinfer-python", "tilelang", "triton")},
              "q_shape": list(q.shape), "k_shape": list(k.shape), "provenance": provenance,
              "numerical_contract": {"output_atol": .006, "output_rtol": .015, "lse_atol": .0003, "lse_rtol": .0001, "lse_base": "natural", "reference": "FP64"},
              "tuning_planned": len(configurations), "tuning_completed": len(tuning), "tuning_complete": len(tuning) == len(configurations),
              "batch": b, "gqa": g, "partitions": all_partitions(b), "hierarchies": binary_hierarchies(tuple(range(b))),
              "search_seconds_including_jit_and_failures": search_seconds, "search_seconds_by_family": family_search_seconds,
              "oracle_scope": "Baseline selection is a per-capture hindsight oracle among the measured native grid; owner C128/N64 is fixed. Search is not free.",
              "tuning": tuning, "missing_qualified_methods": missing,
              "selected": {cache: {method: {"config": x["config"], "record": x["record"]} for method, x in methods.items()} for cache, methods in best.items()},
              "measurements": measurements, "summary": summary, "planning": planning,
              "limitations": ["This reduced confirmation intentionally excludes v2/v3/emptyguard/FullCol/wide/persistent/v4/lane. Earlier full-grid results are retained separately; this is not a claim that every engineering control was remeasured here.",
                              "E4 is exact binary-tree intersection/residual decomposition with native attention and included query gathering/merge, not a full QUILT reproduction.",
                              "BF16 output and natural-log FP32 LSE are measured for every complete producer/merge call; all candidates require unchanged FP64 tolerances.",
                              "Owner owned workspace includes descriptors/merge tables/output; legacy controls have different documented workspace scopes. These are not common-scope peak-memory measurements.",
                              "Owner topology is immutable within replay. Original Q/K/V/Indices values may change in place; topology changes require rebuild and recapture. graph.replay bypasses host identity/version checks.",
                              "All attempts, failures and preparation are retained. A complete successful grid requires inspecting every configuration status in addition to tuning_complete and missing-family checks.",
                              "Paired block bootstrap describes within-capture timing noise. Same-document ranks/layers/steps are correlated and do not establish cross-document, lifecycle/JCT, quality or energy benefit.",
                              "Reuse counts are amortization scenarios. Prepare-plus-one-call wall is separate; graph recapture and oracle search are not silently charged to measured GPU event latency."]}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"tuning_complete": result["tuning_complete"], "summary": summary, "missing": missing}), flush=True)


if __name__ == "__main__":
    main()
