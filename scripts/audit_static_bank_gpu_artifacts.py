#!/usr/bin/env python3
"""CPU audit of saved static-bank CUDA/SASS/launch evidence; no GPU imports."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/static_bank_v4_qualification"
KERNEL_SHA = "afa47ed8aca50923e7426096d6c17be0c0830bbfd8687b3d0fb99aee17ec5a85"
CACHE = Path("/home/qcjz/.tilelang/cache/0.1.9-x86_64/kernels")


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def serialized_metadata(binary, expected_cubin):
    """Decode FunctionInfo only when its following cubin matches exactly.

    This follows TVM FunctionInfo::Save(Stream*) and CUDA SaveToBytes, without
    loading executable.so or importing TVM. All candidate offsets are validated.
    """
    name = b"main_kernel"
    marker = struct.pack("<Q", len(name)) + name
    matches = []
    start = 0
    while (start := binary.find(marker, start)) >= 0:
        position = start + len(marker)
        start += 1
        def u64():
            nonlocal position
            value = struct.unpack_from("<Q", binary, position)[0]
            position += 8
            return value
        def string():
            nonlocal position
            length = u64()
            assert length < 1024
            value = binary[position:position+length].decode()
            position += length
            return value
        try:
            function_name = string()
            count = u64()
            assert 1 <= count <= 32
            dtypes = [struct.unpack_from("<BBH", binary, position + i*4) for i in range(count)]
            position += count * 4
            assert all(dtype == (3, 64, 1) for dtype in dtypes), "Expected pointer-only producer"
            tag_count = u64()
            assert 1 <= tag_count <= 7
            tags = [string() for _ in range(tag_count)]
            assert all(re.fullmatch(r"(?:blockIdx|threadIdx)\.[xyz]|tir\.use_dyn_shared_memory", tag) for tag in tags)
            extra_count = u64()
            assert extra_count <= count
            extras = list(struct.unpack_from("<" + "i"*extra_count, binary, position)) if extra_count else []
            position += extra_count * 4
            cubin_size = u64()
            assert cubin_size == len(expected_cubin)
            assert binary[position:position+cubin_size] == expected_cubin
            matches.append({"func_info": {"main_kernel": {"name": function_name,
                            "arg_types": ["handle"]*count, "launch_param_tags": tags,
                            "arg_extra_tags": extras}}})
        except (AssertionError, UnicodeDecodeError, struct.error, OverflowError):
            continue
    assert len(matches) == 1, f"Expected one cubin-bound metadata record, found {len(matches)}"
    return matches[0]


def source_evidence(source):
    cu_lines, unknown, gathers, scatters = [], [], [], []
    for line_number, line in enumerate(source.splitlines(), 1):
        if not re.search(r"\bcu\b", line):
            continue
        row = {"line": line_number, "text": line.strip()}
        cu_lines.append(row)
        if re.search(r"^\s*out(?:_\d+)?\[.*\]\s*=\s*cu\[", line):
            gathers.append(row)
        elif re.search(r"^\s*cu\[.*\]\s*=\s*out(?:_\d+)?\[", line):
            scatters.append(row)
        elif "float cu[32];" in line or ("make_float4" in line and "(cu +" in line):
            pass
        elif "(U +" in line and "= *(float2*)(cu +" in line:
            pass
        else:
            unknown.append(row)
    assert cu_lines and gathers and scatters and not unknown, unknown
    out_lines = []
    for line_number, line in enumerate(source.splitlines(), 1):
        if not re.search(r"\bout(?:_\d+)?(?:\[| \+)", line):
            continue
        out_lines.append({"line": line_number, "text": line.strip()})
        allowed = (re.fullmatch(r"\s*float out(?:_\d+)?\[\d+\];", line)
                   or "cu[" in line or "make_float4" in line
                   or re.search(r"^\s*(out(?:_\d+)?)\[.*\]\s*=\s*\(\1\[.*\]\s*\*\s*factor(?:_\d+)?\[", line)
                   or ("tl::mma_sync" in line and "reinterpret_cast<float*>(out" in line))
        assert allowed, {"unclassified_out_use": out_lines[-1]}
    shared_u = [line for line in source.splitlines()
                if "buf_dyn_shmem" in line and re.search(r"\b(?:cu|out(?:_\d+)?)\b", line)]
    assert not shared_u, shared_u
    assert "maximum" in source and "denominator" in source
    cm_writes = [line for line in source.splitlines() if "buf_dyn_shmem" in line
                 and re.search(r"=\s*maximum(?:_\d+)?\[", line)]
    cl_writes = [line for line in source.splitlines() if "buf_dyn_shmem" in line
                 and re.search(r"=\s*denominator(?:_\d+)?\[", line)]
    assert cm_writes and cl_writes and source.count("__syncthreads") > 0
    region_loop = re.search(r"for \(int r = 0; r < \d+; \+\+r\) \{", source)
    if region_loop:
        assert source[:region_loop.start()].rstrip().endswith("__syncthreads();")
        begin = region_loop.end()
        depth, end = 1, begin
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        assert source[begin:end-1].rstrip().endswith("__syncthreads();")
    else:
        assert "int label = Labels[0];" in source
        first_branch = source.index("if (label == 1)")
        assert source[:first_branch].rstrip().endswith("__syncthreads();")
        assert source.rfind("__syncthreads();") > max(source.rfind(cm_writes[-1]), source.rfind(cl_writes[-1]))
    return {"canonical_u_accesses_fully_classified": True,
            "active_out_accesses_fully_classified": True, "active_out_use_sites": len(out_lines),
            "canonical_u_lines": cu_lines, "direct_lane_local_gather_sites": len(gathers),
            "direct_lane_local_scatter_sites": len(scatters),
            "cu_out_shared_statement_sites": len(shared_u),
            "shared_maximum_write_sites": len(cm_writes), "shared_denominator_write_sites": len(cl_writes),
            "static_barrier_text_sites": source.count("__syncthreads"),
            "initial_shared_scalar_publish_barrier": True,
            "unconditional_region_end_barrier": True,
            "region_loop_elided_for_single_region": region_loop is None,
            "scope": "Saved generated CUDA statements and cubin/SASS, not runtime traffic counters."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="results/static_bank_artifact_audit_v1/audit.json")
    p.add_argument("--input", action="append", default=[], help="Additional qualification JSONs, e.g. later wave10 jobs")
    a = p.parse_args()
    output = ROOT / a.output
    launch_archive = output.parent / "launch_hosts"
    launch_archive.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("resources_parser_cpu", ROOT / "src/branchtile/resources_v2.py")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    paths = [BASE / "frontend_trace_v1.json", BASE / "v4_b3_n32_extended_v1.json",
             BASE / "resource_parser_probe_v2.json", *sorted((BASE / "wave8").glob("*.json"))]
    paths = list(dict.fromkeys([*paths, *(ROOT / value for value in a.input)]))
    ledger_by_output = {}
    for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines():
        event = json.loads(line)
        argv = event.get("argv", [])
        if event.get("event") == "end" and "--output" in argv:
            ledger_by_output[argv[argv.index("--output") + 1]] = event
    cache_sources = {}
    for path in CACHE.glob("*/device_kernel.cu"):
        cache_sources.setdefault(sha(path.read_bytes()), []).append(path.parent)
    jobs, artifacts = [], []
    phases = {"eager", "graph_baseline", "graph_changed_Q", "graph_changed_KV",
              "graph_changed_indices", "graph_changed_offsets_and_indices"}
    for path in paths:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        job = json.loads(raw)
        assert job["code_sha256"]["src/branchtile/tilelang_kernel_v4.py"] == KERNEL_SHA
        relative_path = str(path.relative_to(ROOT))
        terminal = ledger_by_output.get(relative_path)
        jobs.append({"input": relative_path, "sha256": sha(raw), "status": job["status"],
                     "finished_utc": job.get("finished_utc"),
                     "ledger_terminal": terminal,
                     "overall_clean_exit": terminal is not None and terminal["exit_code"] == 0 and not terminal["timed_out"],
                     "case_statuses": {case["name"]: case["status"] for case in job["cases"]}})
        for case in job["cases"]:
            if case["status"] != "passed_numerical_and_mutable_graph":
                continue
            assert {check["phase"] for check in case["checks"]} == phases
            assert all(check["status"] == "passed" for check in case["checks"])
            prefix = path.parent / (path.stem + "_compiler") / case["name"]
            files = {suffix: prefix.with_suffix(suffix).read_bytes()
                     for suffix in (".cu", ".cubin", ".sass", ".resources.txt")}
            resource = case["resources"]
            assert sha(files[".cu"]) == resource["generated_source_sha256"]
            assert sha(files[".cubin"]) == resource["cubin_sha256"]
            assert sha(files[".sass"]) == resource["sass_sha256"]
            fields = {key.lower(): int(value) for key, value in re.findall(r"\b(REG|STACK|SHARED|LOCAL):([0-9]+)", files[".resources.txt"].decode())}
            assert fields == resource["cubin_resources"]
            local_ops = re.findall(r"\b(?:LDL|STL)(?:\.|\s)", files[".sass"].decode())
            spill_free = fields["stack"] == fields["local"] == 0 and not local_ops
            source = source_evidence(files[".cu"].decode())
            candidates = cache_sources[sha(files[".cu"])]
            cache_record = None
            for directory in candidates:
                try:
                    metadata = serialized_metadata((directory / "executable.so").read_bytes(), files[".cubin"])
                    host = (directory / "host_kernel.cu").read_text()
                    launch = parser.parse_launch(host, metadata)
                    cache_record = {"directory": str(directory), "host_sha256": sha(host.encode()),
                                    "executable_sha256": sha((directory / "executable.so").read_bytes()),
                                    "metadata_decoding": "TVM FunctionInfo stream tied to identical embedded cubin bytes",
                                    "metadata": metadata}
                    host_archive = launch_archive / (sha(files[".cubin"]) + ".host.c.gz")
                    meta_archive = launch_archive / (sha(files[".cubin"]) + ".cuda_metadata.json")
                    host_archive.write_bytes(gzip.compress(host.encode(), mtime=0))
                    meta_archive.write_text(json.dumps(metadata, indent=2) + "\n")
                    cache_record.update(archived_host_gzip=str(host_archive.relative_to(ROOT)),
                                        archived_host_gzip_sha256=sha(host_archive.read_bytes()),
                                        archived_metadata=str(meta_archive.relative_to(ROOT)),
                                        archived_metadata_sha256=sha(meta_archive.read_bytes()))
                    break
                except (AssertionError, KeyError):
                    continue
            assert cache_record is not None, prefix
            assert launch["launch_grid"] == [case["q_shape"][0], job["arguments"]["splits"], 1]
            assert launch["launch_block"] == [128, 1, 1]
            assert launch["dynamic_smem_bytes"] == resource["dynamic_smem_bytes"]
            if prefix.with_suffix(".tvm_meta.json").exists():
                exported = json.loads(prefix.with_suffix(".tvm_meta.json").read_text())
                exported_launch = parser.parse_launch(prefix.with_suffix(".host.c").read_text(), exported)
                assert exported_launch == launch
            artifacts.append({"job": str(path.relative_to(ROOT)), "case": case["name"],
                              "numerical_pass": True,
                              "batch": case["batch"], "block_n": job["arguments"]["block_n"],
                              "splits": job["arguments"]["splits"], "cubin_resources": fields,
                              "total_smem_bytes": launch["dynamic_smem_bytes"] + fields["shared"],
                              "sass_local_load_store_count": len(local_ops),
                              "spill_free": spill_free,
                              "resource_observations": {
                                  "registers_per_thread": fields["reg"], "stack_bytes": fields["stack"],
                                  "local_field_bytes": fields["local"], "static_LDL_STL_sites": len(local_ops),
                                  "spill_free": spill_free,
                                  "canonical_u_shared_staging_statement_sites": source["cu_out_shared_statement_sites"],
                                  "runtime_traffic_measured": False},
                              "file_sha256": {suffix: sha(blob) for suffix, blob in files.items()},
                              "reported_launch_grid_preserved": resource["launch_grid"],
                              "cpu_reconstructed_launch": launch, "cache_evidence": cache_record,
                              "source_evidence": source})
    result = {"schema": "static-bank-artifact-audit-v1", "kernel_sha256": KERNEL_SHA,
              "script_sha256": sha(Path(__file__).read_bytes()), "jobs": jobs, "artifacts": artifacts,
              "audited_cases": len(artifacts), "audited_numerical_phases": len(artifacts)*6,
              "status": "saved_artifacts_reviewed_resource_observations_recorded",
              "all_included_cases_numerical_pass": all(row["numerical_pass"] for row in artifacts),
              "resource_observations": {"spill_case_count": sum(not row["spill_free"] for row in artifacts),
                                        "spill_case_keys": [{"job": row["job"], "case": row["case"]}
                                                            for row in artifacts if not row["spill_free"]],
                                        "runtime_traffic_measured": False},
              "limitations": ["No runtime memory traffic, cache miss, occupancy or application speed claims.",
                              "Partial cases do not convert an overall timeout into a successful job.",
                              "This audit never modifies the original resource receipts."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"audited_cases": len(artifacts), "audited_numerical_phases": len(artifacts)*6,
                      "jobs": len(jobs), "register_counts": sorted({row["cubin_resources"]["reg"] for row in artifacts})}))


if __name__ == "__main__":
    main()
