"""Extract static launch arguments using CUDA metadata and host call slots.

This module itself imports no CUDA libraries. Unsupported/dynamic host encodings
fail explicitly instead of treating unrelated integer constants as dimensions.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


def parse_launch(host, metadata, kernel_name=None):
    functions = metadata["func_info"]
    if kernel_name is None:
        assert len(functions) == 1, "Select one CUDA kernel explicitly"
        kernel_name = next(iter(functions))
    info = functions[kernel_name]
    tags = info.get("launch_param_tags", info.get("thread_axis_tags"))
    assert tags, "Missing CUDA launch_param_tags"
    tensor_arguments = len(info["arg_types"])
    symbol = re.search(r'TVMBackendGetFuncFromEnv\([^;]*?"' + re.escape(kernel_name)
                       + r'"\s*,\s*&([A-Za-z_]\w*)\)', host)
    assert symbol, "Kernel's packed-call symbol not found in host source"
    call = re.search(r'TVMFFIFunctionCall\(\s*' + re.escape(symbol.group(1))
                     + r'\s*,\s*\(TVMFFIAny\*\)\s*([A-Za-z_]\w*)\s*,\s*(\d+)\s*,', host)
    assert call, "Kernel's actual packed call not found"
    stack, argument_count = call.group(1), int(call.group(2))
    assert argument_count == tensor_arguments + len(tags), "Unsupported launch-argument count/tags"
    prefix = host[:call.start()]
    slots = {}
    types = {}
    for field, destination in (("v_int64", slots), ("type_index", types)):
        pattern = (r'\(\(\(TVMFFIAny\*\)\s*' + re.escape(stack)
                   + r'\)\[(\d+)\]\.' + field + r'\)\s*=\s*([^;]+);')
        for match in re.finditer(pattern, prefix):
            destination[int(match.group(1))] = match.group(2).strip()
    grid, block = [1, 1, 1], [1, 1, 1]
    dynamic_smem, arguments = 0, []
    seen_tags = set()
    for i, tag in enumerate(tags):
        slot = tensor_arguments + i
        assert types.get(slot) == "1", f"Launch slot {slot} is not an integer"
        expression = slots[slot]
        literal = re.sub(r'\(\s*int64_t\s*\)', '', expression)
        match = re.fullmatch(r'[()\s]*([0-9]+)[()\s]*', literal)
        assert match, f"Launch slot {slot} is not a static integer: {expression}"
        value = int(match.group(1))
        assert tag not in seen_tags, "Duplicate launch tag"
        seen_tags.add(tag)
        if tag == "tir.use_dyn_shared_memory":
            assert i == len(tags) - 1
            dynamic_smem = value
        elif re.fullmatch(r'(?:blockIdx|threadIdx)\.[xyz]', tag):
            assert value > 0
            dimension = "xyz".index(tag[-1])
            (grid if tag.startswith("blockIdx") else block)[dimension] = value
        else:
            raise AssertionError(f"Unsupported launch tag: {tag}")
        arguments.append({"slot": slot, "tag": tag, "value": value, "expression": expression})
    return {"launch_grid": grid, "launch_block": block,
            "dynamic_smem_bytes": dynamic_smem, "launch_param_tags": list(tags),
            "launch_arguments": arguments, "kernel_argument_count": tensor_arguments,
            "packed_call_argument_count": argument_count, "kernel_name": kernel_name,
            "launch_extraction": "cuda_launch_param_tags_and_actual_host_stack_slots_v2"}


def kernel_resources(kernel, artifact_prefix=None):
    executable = kernel.adapter.executable
    queue = [getattr(executable, "mod", executable)]
    cuda_modules = []
    while queue:
        module = queue.pop()
        if module.kind == "cuda":
            cuda_modules.append(module)
        queue.extend(module.imports)
    assert len(cuda_modules) == 1, "Expected exactly one producer CUDA module"
    source = kernel.get_kernel_source()
    host = kernel.adapter.host_kernel_source
    if host is None:
        host = kernel.adapter.get_host_source()
    with tempfile.TemporaryDirectory(prefix="branchtile-resources-v2-") as tmp:
        cubin = Path(tmp) / "producer.cubin"
        cuda_modules[0].write_to_file(str(cubin), "cubin")
        # TVM CUDA WriteToFile writes the authoritative function metadata too.
        metadata_path = cubin.with_suffix(".tvm_meta.json")
        assert metadata_path.is_file(), "CUDA module did not export function metadata"
        metadata_bytes = metadata_path.read_bytes()
        output = subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-resource-usage", str(cubin)],
                                check=True, capture_output=True, text=True).stdout
        cubin_bytes = cubin.read_bytes()
        if artifact_prefix:
            dest = Path(artifact_prefix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.with_suffix(".cubin").write_bytes(cubin_bytes)
            dest.with_suffix(".cu").write_text(source)
            dest.with_suffix(".host.c").write_text(host)
            dest.with_suffix(".tvm_meta.json").write_bytes(metadata_bytes)
            dest.with_suffix(".resources.txt").write_text(output)
        launch = parse_launch(host, json.loads(metadata_bytes))
    fields = {key.lower(): int(value) for key, value in re.findall(r'\b(REG|STACK|SHARED|LOCAL):([0-9]+)', output)}
    return dict(launch, resource_receipt_version=2,
                cubin_sha256=hashlib.sha256(cubin_bytes).hexdigest(), cubin_resources=fields,
                total_smem_bytes=launch["dynamic_smem_bytes"] + fields.get("shared", 0),
                generated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                generated_host_source_sha256=hashlib.sha256(host.encode()).hexdigest(),
                cuda_metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest())
