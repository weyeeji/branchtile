"""Regression tests for omitted CUDA grid dimensions and unrelated constants."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("resources_v2_cpu", ROOT / "src/branchtile/resources_v2.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def metadata(tags, arguments=9):
    return {"func_info": {"main_kernel": {"arg_types": ["handle"] * arguments,
                                           "launch_param_tags": tags}}}


def host_fixture(values, arguments=9):
    lines = ['(((TVMFFIAny*)stack)[77].v_int64) = ((int64_t)777);']
    for slot, value in enumerate(values, arguments):
        lines += [f'(((TVMFFIAny*)stack)[{slot}].type_index) = 1;',
                  f'(((TVMFFIAny*)stack)[{slot}].v_int64) = ((int64_t){value});']
    lines += ['TVMBackendGetFuncFromEnv(ctx, "main_kernel", &kernel_packed);',
              f'TVMFFIFunctionCall(kernel_packed, (TVMFFIAny*) stack, {arguments+len(values)}, &result);',
              '(((TVMFFIAny*)stack)[78].v_int64) = ((int64_t)999);']
    return "\n".join(lines)


def test_two_dimensional_grid_omits_z():
    tags = ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"]
    result = MODULE.parse_launch(host_fixture([2, 3, 128, 1, 1, 34048]), metadata(tags))
    assert result["launch_grid"] == [2, 3, 1]
    assert result["launch_block"] == [128, 1, 1]
    assert result["dynamic_smem_bytes"] == 34048
    assert [row["slot"] for row in result["launch_arguments"]] == list(range(9, 15))


def test_three_dimensional_grid_and_block():
    tags = ["blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"]
    result = MODULE.parse_launch(host_fixture([2, 64, 3, 128, 2, 1, 47360], 11), metadata(tags, 11))
    assert result["launch_grid"] == [2, 64, 3]
    assert result["launch_block"] == [128, 2, 1]
    assert result["dynamic_smem_bytes"] == 47360


def test_dynamic_launch_and_unknown_tag_rejected():
    with pytest.raises(AssertionError, match="not a static integer"):
        MODULE.parse_launch(host_fixture(["runtime_extent"]), metadata(["blockIdx.x"]))
    with pytest.raises(AssertionError, match="Unsupported launch tag"):
        MODULE.parse_launch(host_fixture([2]), metadata(["threadIdx.w"]))


def test_existing_frontend_trace_cached_host():
    directory = Path("/home/qcjz/.tilelang/cache/0.1.9-x86_64/kernels/3017940eb2a019adc26f59f38ff2aa78cdf98982391f93e16c285598ef40b5fb")
    if not directory.is_dir():
        pytest.skip("Optional historical cache fixture is unavailable")
    tags = ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"]
    # Cross-check tag presence/order against the serialized CUDA executable.
    binary = (directory / "executable.so").read_bytes()
    positions = [binary.index(tag.encode()) for tag in tags]
    assert positions == sorted(positions)
    assert b"blockIdx.z" not in binary
    result = MODULE.parse_launch((directory / "host_kernel.cu").read_text(), metadata(tags))
    assert result["launch_grid"] == [2, 3, 1]
    assert result["launch_block"] == [128, 1, 1]
    assert result["dynamic_smem_bytes"] == 34048
