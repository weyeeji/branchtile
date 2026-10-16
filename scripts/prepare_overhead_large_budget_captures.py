#!/usr/bin/env python3
"""Restore frozen suffix-Q supports on CPU into independent capture overlays.

Original tensors and metadata are read-only inputs by contract. Symlinks do not
change Unix permissions or make the input files immutable. No CUDA is used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from capture_model import select_support

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ("publicdev6_256", "docfinqa256", "publicdev1_256")
OUTPUT = ROOT / "results/captures/overhead_large_budget_v1"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def relative(path):
    return str(Path(path).absolute().relative_to(ROOT))


def read_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def retain_json(path, value, check):
    desired = json_bytes(value)
    if path.exists():
        assert not path.is_symlink() and path.read_bytes() == desired, path
    else:
        assert not check, f"Missing generated JSON: {path}"
        path.write_bytes(desired)


def retain_link(target, path, check):
    target = target.resolve(strict=True)
    if path.is_symlink():
        assert path.resolve(strict=True) == target, path
    else:
        assert not path.exists() and not check, path
        path.symlink_to(os.path.relpath(target, path.parent))
    return {"path": relative(path), "symlink_text": os.readlink(path),
            "source_path": relative(target), "sha256": sha(target),
            "bytes": target.stat().st_size}


def original_load_capture():
    """Execute the exact loader body without importing/JITing GPU backends."""
    path = ROOT / "scripts/compare_attention.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "load_capture")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"Path": Path, "json": json, "torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["load_capture"]


def prepare_rank(document, rank, check, loader):
    source = ROOT / f"results/captures/main30b_tp2_{document}_suffix_v2/rank{rank}"
    output = OUTPUT / document / f"rank{rank}"
    if not check:
        output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "metadata.json").read_text())
    assert original["export_semantics_version"] == 2
    assert original["gqa"] == 8 and original["kv_heads"] == 2
    assert original["actual_tp_rank"] == rank
    n = original["document_token_range"][1] - original["document_token_range"][0]
    common_n = (original["actual_shared_document_token_range"][1]
                - original["document_token_range"][0])
    tail = n - common_n
    assert tail == original["private_protected_document_tail_tokens"]
    assert 0 <= tail <= 32
    batch = original["B"]
    variants = [
        {"filename_budget_arg": 65536, "requested_budget": 65536},
        {"filename_budget_arg": 131072, "requested_budget": 131072},
        {"filename_budget_arg": "full", "requested_budget": "full"},
        {"filename_budget_arg": n, "requested_budget": "full"},
    ]
    for variant in variants:
        requested = variant["requested_budget"]
        effective = n if requested == "full" else min(requested, n)
        variant.update(effective_full_document_budget=effective,
                       effective_shared_document_budget=effective-tail,
                       full_document_tokens=n, shared_document_tokens=common_n,
                       clipped_to_full=requested != "full" and requested > n,
                       effective_tag="full" if effective == n else str(effective))
    links = [retain_link(source / "metadata.json", output / "original_metadata.json", check)]
    masks, old_checks, compatible = [], [], []
    source_inventory = []
    steps = [step for step in (16, 64, 256)
             if original["ready_lifetime_measured_by_step"].get(str(step))]
    assert steps == [16, 64, 256]
    for layer in original["layers"]:
        src_path = source / f"source_l{layer}.pt"
        links.append(retain_link(src_path, output / src_path.name, check))
        src = read_tensor(src_path)
        assert src["document_k"].shape == (1, 2, common_n, 128)
        assert src["prefix_k"].shape == src["prefix_v"].shape
        assert src["document_k"].dtype == src["document_v"].dtype == torch.bfloat16
        prefix = src["prefix_k"].shape[2]
        del src
        selectors = []
        for request in range(batch):
            path = source / f"selector_q{request}_l{layer}.pt"
            links.append(retain_link(path, output / path.name, check))
            sel = read_tensor(path)
            assert sel["observation_is_question_suffix_only"]
            assert sel["pooled_scores"].shape == (2, n)
            assert torch.isfinite(sel["pooled_scores"]).all()
            selectors.append(sel)
            for step in (1, *steps):
                path = source / f"decode_q{request}_l{layer}_s{step}.pt"
                links.append(retain_link(path, output / path.name, check))
        for budget in original["budgets"]:
            supports = torch.stack([select_support(s["pooled_scores"], budget)
                                    for s in selectors])
            path = source / f"masks_l{layer}_b{budget}.pt"
            old = read_tensor(path)
            assert torch.equal(old["supports"], supports[:, :, :common_n])
            assert all(torch.equal(s["supports"][budget], supports[q])
                       for q, s in enumerate(selectors))
            links.append(retain_link(path, output / path.name, check))
            old_checks.append({"layer": layer, "budget": budget, "bitwise_restored": True})
        for variant in variants:
            effective = variant["effective_full_document_budget"]
            supports = torch.stack([select_support(s["pooled_scores"], effective)
                                    for s in selectors])
            assert supports.sum(-1).eq(effective).all()
            assert supports[:, :, :min(4, n)].all()
            assert supports[:, :, max(0, n-32):].all()
            if tail:
                assert supports[:, :, -tail:].all()
            common = supports[:, :, :common_n].contiguous()
            assert common.sum(-1).eq(effective-tail).all()
            if effective == n:
                assert common.all()
            payload = {"supports": common, "budget": variant["filename_budget_arg"],
                       "layer": layer, "gqa": 8,
                       "private_protected_document_tail_tokens": tail,
                       "requested_budget": variant["requested_budget"],
                       "effective_full_document_budget": effective,
                       "effective_shared_document_budget": effective-tail,
                       "effective_tag": variant["effective_tag"],
                       "clipped_to_full": variant["clipped_to_full"]}
            path = output / f"masks_l{layer}_b{variant['filename_budget_arg']}.pt"
            if path.exists():
                actual = read_tensor(path)
                assert not path.is_symlink()
                assert actual.keys() == payload.keys()
                for key, value in payload.items():
                    assert (torch.equal(actual[key], value) if isinstance(value, torch.Tensor)
                            else actual[key] == value), (path, key)
            else:
                assert not check, f"Missing generated mask: {path}"
                torch.save(payload, path)
            masks.append({"path": relative(path), "sha256": sha(path),
                          "supports_shape": list(common.shape),
                          "full_support_sha256": hashlib.sha256(supports.numpy().tobytes()).hexdigest(),
                          "common_support_sha256": hashlib.sha256(common.numpy().tobytes()).hexdigest(),
                          **variant, "layer": layer})
        del selectors
        for step in steps:
            event = None
            suffixes = []
            for request in range(batch):
                dec = read_tensor(output / f"decode_q{request}_l{layer}_s{step}.pt")
                metadata = dec["metadata"]
                assert metadata["physical_full_prefix_pages_match"]
                assert metadata["hooked_kv_matches_actual_cache"]
                if event is None:
                    event = metadata["same_event_query_positions"]
                assert metadata["same_event_query_positions"] == event
                assert dec["private_suffix_k"].shape == dec["private_suffix_v"].shape
                assert dec["private_suffix_k"].shape[2] >= tail
                suffixes.append(dec["private_suffix_k"].shape[2])
            source_inventory.append({"layer": layer, "step": step,
                                     "system_prefix_tokens": prefix,
                                     "common_document_tokens": common_n,
                                     "complete_private_suffix_lengths": suffixes,
                                     "same_event_query_positions": event})
    metadata = dict(original)
    metadata.update(budgets=list(dict.fromkeys([*original["budgets"], 65536, 131072, n])),
                    derived_capture_overlay=True,
                    derived_mask_variants=variants,
                    original_metadata="original_metadata.json",
                    summary="overlay_receipt.json",
                    mask_definition=(original["mask_definition"] +
                        " Derived 64K/128K/full supports use identical frozen suffix-only scores; "
                        "requests exceeding document length are explicitly clipped to full. "
                        "The filename budget is a request, not the realized token count."))
    retain_json(output / "metadata.json", metadata, check)
    # Run the exact existing loader AST on CPU; cuda() alone is temporarily
    # replaced by identity. No GPU backend import or CUDA initialization occurs.
    original_cuda = torch.Tensor.cuda
    try:
        torch.Tensor.cuda = lambda self, *args, **kwargs: self
        for variant in variants:
            q, k, v, owners, b, g, provenance = loader(
                output, original["layers"][0], variant["filename_budget_arg"], 64, 2)
            inventory = next(x for x in source_inventory
                             if x["layer"] == original["layers"][0] and x["step"] == 64)
            expected_n = prefix + common_n + sum(inventory["complete_private_suffix_lengths"])
            assert q.shape == (2, batch*8, 128) and b == batch and g == 8
            assert k.shape == v.shape == (2, expected_n, 128)
            assert owners.shape == (2, expected_n)
            assert (owners[:, :prefix] == (1 << batch)-1).all()
            for request in range(batch):
                count = ((owners & (1 << request)) != 0).sum(-1)
                assert count.eq(prefix + variant["effective_shared_document_budget"]
                                + inventory["complete_private_suffix_lengths"][request]).all()
            assert provenance["ready_lifetime_measured"]
            assert provenance["includes_system_prefix_and_complete_private_suffix"]
            compatible.append({"budget_arg": variant["filename_budget_arg"],
                               "layer": original["layers"][0], "step": 64,
                               "exact_loader_AST_CPU_identity_cuda_passed": True})
            del q, k, v, owners
    finally:
        torch.Tensor.cuda = original_cuda
    assert not torch.cuda.is_initialized()
    receipt = {"schema": 1, "document": document, "rank": rank, "B": batch,
               "source_directory": relative(source), "output_directory": relative(output),
               "uid": os.getuid(), "python": relative(Path(sys.executable)),
               "GPU_hours": 0, "CUDA_initialized": False,
               "generator_sha256": sha(__file__),
               "selection_source_sha256": sha(ROOT / "scripts/capture_model.py"),
               "capacity_audit_source_sha256": sha(ROOT / "scripts/audit_natural_capacity_boundary.py"),
               "load_capture_source_sha256": sha(ROOT / "scripts/compare_attention.py"),
               "metadata_sha256": sha(output / "metadata.json"),
               "input_links": links, "generated_masks": masks,
               "original_mask_bitwise_restoration_checks": old_checks,
               "source_inventory": source_inventory, "load_capture_compatibility": compatible,
               "scope": "CPU-derived masks on unchanged real captures; no new model samples or GPU measurements.",
               "selection_semantics": "Frozen suffix-only pooled scores; stable descending score order; protect first4/last32 full-document tokens. Decode Q values never choose masks.",
               "symlink_policy": "Read-only use by generator; original bytes retained. Symlinks do not enforce filesystem immutability.",
               "private_tail_semantics": "Protected partial-page document tail remains in every original private suffix exactly once; omitted only from shared-document mask."}
    retain_json(output / "overlay_receipt.json", receipt, check)
    return {"document": document, "rank": rank, "B": batch,
            "directory": relative(output), "full_document_tokens": n,
            "metadata_sha256": sha(output / "metadata.json"),
            "receipt_sha256": sha(output / "overlay_receipt.json"),
            "generated_masks": len(masks), "original_mask_checks": len(old_checks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", nargs="+", choices=DEFAULT_DOCUMENTS,
                        default=list(DEFAULT_DOCUMENTS))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    loader = original_load_capture()
    rows = []
    for document in args.documents:
        for rank in (0, 1):
            row = prepare_rank(document, rank, args.check, loader)
            rows.append(row)
            print(json.dumps(row), flush=True)
    assert not torch.cuda.is_initialized()
    print(json.dumps({"status": "checked" if args.check else "prepared",
                      "ranks": len(rows), "GPU_hours": 0}), flush=True)


if __name__ == "__main__":
    main()
