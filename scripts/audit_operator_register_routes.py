#!/usr/bin/env python3
"""CPU-only ownership proof for GQA8 static-owner fragment copies.

This checks integer lane/register maps. It does not compile or run a kernel and
does not establish that TileLang honors the requested maps in generated code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/platform/operator_register_routes_20261016.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank_map(row: int, col: int) -> tuple[int, int]:
    """GQA8 request banks: eight FP32 registers per request per thread."""
    thread = (col // 32) * 32 + (row % 8) * 4 + (col % 8) // 2
    register = (row // 8) * 8 + ((col % 32) // 8) * 2 + col % 2
    return thread, register


def fullcol_pv_map(row: int, col: int) -> tuple[int, int]:
    """M16/M32 x D128, four warps, fixed 1x4 warp partition."""
    thread = (col // 32) * 32 + (row % 8) * 4 + (col % 8) // 2
    register = (
        (row // 16) * 16
        + ((col % 32) // 8) * 4
        + ((row % 16) // 8) * 2
        + col % 2
    )
    return thread, register


def fullrow_pv_map(row: int, col: int, matrix_rows: int) -> tuple[int, int]:
    """Current generated FullRow M16=1x4, M32=2x2 warp partitions."""
    if matrix_rows == 16:
        return fullcol_pv_map(row, col)
    thread = (col // 64) * 64 + (row // 16) * 32 + (row % 8) * 4 + (col % 8) // 2
    register = ((col % 64) // 8) * 4 + ((row % 16) // 8) * 2 + col % 2
    return thread, register


def assert_bijection(rows: int, mapping, registers_per_thread: int) -> None:
    actual = {mapping(row, col) for row in range(rows) for col in range(128)}
    expected = {(thread, reg) for thread in range(128) for reg in range(registers_per_thread)}
    assert actual == expected, (rows, len(actual), len(expected))


def evaluate() -> dict:
    cohorts = []
    total_routes = total_fullrow_mismatches = 0
    for batch in (3, 4):
        assert_bijection(batch * 8, bank_map, batch * 8)
        for m in (16, 32):
            assert_bijection(m, fullcol_pv_map, m)
        count = mismatches = 0
        owner_records = []
        for owner in range(1, 1 << batch):
            requests = [q for q in range(batch) if owner & (1 << q)]
            matrix_rows = 16 if len(requests) <= 2 else 32
            old_mismatches = 0
            for slot, request in enumerate(requests):
                for head in range(8):
                    for col in range(128):
                        src_thread, src_reg = bank_map(request * 8 + head, col)
                        dst_thread, dst_reg = fullcol_pv_map(slot * 8 + head, col)
                        assert src_thread == dst_thread
                        assert request * 8 <= src_reg < (request + 1) * 8
                        assert 0 <= dst_reg < matrix_rows
                        old_thread, _ = fullrow_pv_map(slot * 8 + head, col, matrix_rows)
                        old_mismatches += old_thread != src_thread
                        count += 1
            mismatches += old_mismatches
            owner_records.append({
                "owner": owner,
                "requests": requests,
                "mma_rows": matrix_rows,
                "active_coordinates": len(requests) * 8 * 128,
                "fullrow_cross_thread_coordinates": old_mismatches,
            })
        active_blocks = sum(owner.bit_count() for owner in range(1, 1 << batch))
        # v3: full canonical store + active gather + active scatter + active
        # canonical restore. Each request bank contains 8*128*4 bytes.
        shared_u_bytes = ((1 << batch) - 1) * batch * 4096 + 3 * active_blocks * 4096
        cohorts.append({
            "batch": batch,
            "gqa": 8,
            "dim": 128,
            "threads": 128,
            "request_bank_fp32_registers_per_thread": batch * 8,
            "checked_active_coordinates": count,
            "fullcol_cross_thread_coordinates": 0,
            "fullrow_cross_thread_coordinates": mismatches,
            "v3_logical_shared_u_bytes_per_cta_all_possible_regions": shared_u_bytes,
            "traffic_scope": "All 7/15 nonempty owner labels, one split CTA; excludes m/l, Q/K/V/P and compiler transformations.",
            "owners": owner_records,
        })
        total_routes += count
        total_fullrow_mismatches += mismatches
    files = [
        "src/branchtile/tilelang_kernel_v2.py",
        "src/branchtile/tilelang_kernel_v3.py",
        "src/branchtile/union.py",
        "src/branchtile/attention.py",
        "results/kernel_ablation_v3/compiler/register_u_n64.cu",
        "results/kernel_ablation_v3/compiler/register_u_n64.resources.txt",
        "results/kernel_resources.json",
    ]
    return {
        "schema": "operator-register-route-audit-v1",
        "status": "cpu_integer_routes_passed_not_compiled",
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": {name: sha256(ROOT / name) for name in files},
        "checked_active_coordinates": total_routes,
        "fullcol_cross_thread_coordinates": 0,
        "fullrow_cross_thread_coordinates": total_fullrow_mismatches,
        "cuda_initialized": False,
        "cohorts": cohorts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = evaluate()
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        assert args.output.read_text() == encoded, "Saved CPU audit differs from recomputation"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(json.dumps({key: result[key] for key in (
        "status", "checked_active_coordinates", "fullcol_cross_thread_coordinates",
        "fullrow_cross_thread_coordinates", "cuda_initialized",
    )}))


if __name__ == "__main__":
    main()
