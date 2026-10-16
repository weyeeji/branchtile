#!/usr/bin/env python3
"""Frozen held-out FP64 attention replay for joint support exchange masks."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
assigned_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
# Reuse the exact previously frozen CPU mask constructor, restoring the job's
# GPU visibility before any CUDA initialization. It never sees decode Q.
import probe_joint_support_exchange as selector
os.environ["CUDA_VISIBLE_DEVICES"] = assigned_devices
import numpy as np
import torch

BASE = ROOT / "results/joint_attention_fp64_v1"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def freeze_masks(source, protocol, previous_rows):
    """Construct and hash-match every mask before opening held-out queries."""
    directory = ROOT / source["directory"]
    items = []
    for item in source["selectors"]:
        path = directory / item["name"]
        assert sha(path) == item["sha256"]
        data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        assert data["observation_is_question_suffix_only"]
        items.append({"scores": data["pooled_scores"].numpy().astype(np.float64),
                      "supports": {k: data["supports"].get(k, data["supports"].get(str(k))).numpy().copy()
                                   for k in protocol["budgets"]}})
    scores = np.stack([x["scores"] for x in items])
    b, h, n = scores.shape
    protected = selector.protected_mask(n)
    result = {}
    for budget in protocol["budgets"]:
        baseline = np.stack([x["supports"][budget] for x in items])
        for head in range(h):
            variants = [("independent", None, None, baseline[:, head].copy())]
            for epsilon in protocol["epsilon"]:
                for order_name in protocol["request_orders"]:
                    order = list(range(b)) if order_name == "ascending" else list(reversed(range(b)))
                    masks, _ = selector.exchange(scores[:, head], baseline[:, head], protected,
                                                  epsilon, order, protocol["max_rounds"])
                    variants.append(("exchange", epsilon, order_name, masks))
            for method, epsilon, order, masks in variants:
                previous = previous_rows[source["id"], budget, head, method, epsilon, order]
                for q in range(b):
                    ids = np.flatnonzero(masks[q]).astype("<i8")
                    assert array_sha(ids) == previous["per_request"][q]["selected_indices_sha256"]
                result[budget, head, method, epsilon, order] = masks
    return result


def attention(logits, values, ids):
    selected = logits.index_select(1, ids)
    weights = torch.softmax(selected, dim=-1)
    return weights @ values.index_select(0, ids), torch.logsumexp(selected, dim=-1)


def errors(output, reference):
    diff = output - reference
    norm = float(np.linalg.norm(reference))
    denom = float(np.linalg.norm(output) * np.linalg.norm(reference))
    return {"relative_l2": float(np.linalg.norm(diff) / max(norm, 1e-12)),
            "max_abs": float(np.abs(diff).max()),
            "cosine": float(np.sum(output * reference) / denom) if denom else None,
            "reference_l2": norm}


def synthetic_check(device):
    logits = torch.zeros((2, 4), dtype=torch.float64, device=device)
    values = torch.tensor([[0., 0.], [1., 2.], [2., 4.], [3., 6.]], dtype=torch.float64, device=device)
    full, full_lse = attention(logits, values, torch.arange(4, device=device))
    partial, partial_lse = attention(logits, values, torch.tensor([0, 3], device=device))
    assert torch.equal(full, partial)
    assert torch.equal(full, torch.tensor([[1.5, 3.], [1.5, 3.]], dtype=torch.float64, device=device))
    repeated, repeated_lse = attention(logits, values, torch.tensor([0, 3], device=device))
    assert torch.equal(partial, repeated) and torch.equal(partial_lse, repeated_lse)
    assert torch.equal(torch.softmax(logits, dim=-1)[:, [0, 3]].sum(-1), torch.tensor([.5, .5], dtype=torch.float64, device=device))
    assert (full_lse > partial_lse).all()
    return {"status": "passed", "checks": ["uniform_dense_reference", "scattered_subset",
            "included_first_and_last_positions", "same_support_exact_repeat", "retained_mass"]}


def run_document(source, protocol, previous_rows, device):
    directory = ROOT / source["directory"]
    mask_start = time.perf_counter()
    masks = freeze_masks(source, protocol, previous_rows)
    mask_seconds = time.perf_counter() - mask_start
    assert sha(directory / "metadata.json") == source["metadata_sha256"]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["export_semantics_version"] == 2
    assert sha(directory / source["source_file"]["name"]) == source["source_file"]["sha256"]
    raw_source = torch.load(directory / source["source_file"]["name"], map_location="cpu", weights_only=True, mmap=True)
    b, h, g, d = metadata["B"], metadata["kv_heads"], metadata["gqa"], 128
    prefix_n = raw_source["prefix_k"].shape[2]
    common_n = raw_source["document_k"].shape[2]
    for key in ("prefix_k", "prefix_v", "document_k", "document_v"):
        value = raw_source[key]
        expected_n = prefix_n if key.startswith("prefix") else common_n
        assert value.shape == (1, h, expected_n, d) and value.dtype == torch.bfloat16
    doc_n = metadata["document_token_range"][1] - metadata["document_token_range"][0]
    assert metadata["document_token_range"][0] == prefix_n
    assert metadata["actual_shared_document_token_range"][0] == prefix_n
    assert prefix_n + common_n == metadata["actual_shared_document_token_range"][1]
    rows = []
    for head in range(h):
        common_k = torch.cat([raw_source["prefix_k"][0, head], raw_source["document_k"][0, head]]).to(device=device, dtype=torch.float64)
        common_v = torch.cat([raw_source["prefix_v"][0, head], raw_source["document_v"][0, head]]).to(device=device, dtype=torch.float64)
        for capture in source["decode_files"]:
            path = directory / capture["name"]
            assert sha(path) == capture["sha256"]
            dec = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            assert dec["metadata"] == capture["metadata"]
            dm = dec["metadata"]
            qidx, anchor = dm["question_index"], dm["ready_group_anchor_step"]
            private_n = dec["private_suffix_k"].shape[2]
            assert dec["query"].shape == (1, h*g, 1, d) and dec["query"].dtype == torch.bfloat16
            for key in ("private_suffix_k", "private_suffix_v"):
                assert dec[key].shape == (1, h, private_n, d) and dec[key].dtype == torch.bfloat16
            assert prefix_n + common_n + private_n == dm["query_token_position"] + 1
            assert dm["shared_prefix_end"] == prefix_n + common_n
            assert dm["hooked_kv_matches_actual_cache"] and dm["physical_full_prefix_pages_match"]
            assert dm["private_document_tail_tokens"] == doc_n - common_n
            q = dec["query"][0, head*g:(head+1)*g, 0].to(device=device, dtype=torch.float64)
            k = torch.cat([common_k, dec["private_suffix_k"][0, head].to(device=device, dtype=torch.float64)])
            v = torch.cat([common_v, dec["private_suffix_v"][0, head].to(device=device, dtype=torch.float64)])
            logits = (q @ k.T) / math.sqrt(d)
            full_ids = torch.arange(len(k), device=device)
            dense_t, dense_lse_t = attention(logits, v, full_ids)
            dense_probs = torch.softmax(logits, dim=-1)
            dense, dense_lse = dense_t.cpu().numpy(), dense_lse_t.cpu().numpy()
            dense_document_mass = dense_probs[:, prefix_n:prefix_n+common_n+dm["private_document_tail_tokens"]].sum(-1)
            for budget in protocol["budgets"]:
                independent_t = independent_lse_t = independent = independent_lse = None
                variants = [("independent", None, None)] + [("exchange", e, o) for e in protocol["epsilon"] for o in protocol["request_orders"]]
                for method, epsilon, order in variants:
                    support = masks[budget, head, method, epsilon, order][qidx]
                    assert support.sum() == budget and support[common_n:].all()
                    document_ids_cpu = np.flatnonzero(support[:common_n]) + prefix_n
                    combined = np.concatenate([np.arange(prefix_n), document_ids_cpu,
                                               np.arange(prefix_n+common_n, len(k))]).astype(np.int64)
                    assert len(np.unique(combined)) == len(combined)
                    assert len(combined) == prefix_n + budget + private_n - dm["private_document_tail_tokens"]
                    ids = torch.from_numpy(combined).to(device)
                    out_t, lse_t = attention(logits, v, ids)
                    actual_mass = dense_probs.index_select(1, ids).sum(-1)
                    # Include the request-private protected document tail in
                    # the document-only mass numerator, exactly once.
                    document_ids = np.concatenate([document_ids_cpu, np.arange(prefix_n+common_n,
                                                 prefix_n+common_n+dm["private_document_tail_tokens"])])
                    doc_mass = dense_probs.index_select(1, torch.as_tensor(document_ids, device=device)).sum(-1)
                    conditional_doc_mass = doc_mass / dense_document_mass
                    mass_lse_residual = (actual_mass - torch.exp(lse_t - dense_lse_t)).abs()
                    if method == "independent":
                        independent_t, independent_lse_t = out_t, lse_t
                        independent, independent_lse = out_t.cpu().numpy(), lse_t.cpu().numpy()
                    exact_zero = None
                    if method == "exchange" and epsilon == 0:
                        exact_zero = bool(torch.equal(out_t, independent_t) and torch.equal(lse_t, independent_lse_t))
                        assert exact_zero, "epsilon=0 must exactly reproduce independent O and LSE"
                    out, lse = out_t.cpu().numpy(), lse_t.cpu().numpy()
                    mass = actual_mass.cpu().numpy()
                    doc_cond = conditional_doc_mass.cpu().numpy()
                    mass_res = mass_lse_residual.cpu().numpy()
                    baseline_error = errors(independent, dense)
                    dense_error = errors(out, dense)
                    per_query_head = []
                    for j in range(g):
                        de, be = errors(out[j], dense[j]), errors(independent[j], dense[j])
                        per_query_head.append({"local_query_head": head*g+j,
                            "vs_dense": de, "vs_independent": errors(out[j], independent[j]),
                            "dense_relative_l2_change_from_independent": de["relative_l2"] - be["relative_l2"],
                            "dense_max_abs_change_from_independent": de["max_abs"] - be["max_abs"],
                            "lse_ln": float(lse[j]), "dense_lse_ln": float(dense_lse[j]),
                            "independent_lse_ln": float(independent_lse[j]),
                            "lse_delta_vs_dense": float(lse[j] - dense_lse[j]),
                            "lse_delta_vs_independent": float(lse[j] - independent_lse[j]),
                            "retained_total_dense_softmax_mass": float(mass[j]),
                            "retained_document_conditional_mass": float(doc_cond[j]),
                            "mass_vs_lse_identity_abs_residual": float(mass_res[j])})
                    prev = previous_rows[source["id"], budget, head, method, epsilon, order]["per_request"][qidx]
                    rows.append({"document": source["id"], "document_sha256": metadata["document_sha256"],
                        "rank": 0, "layer": 15, "kv_head": head, "question": qidx, "anchor_step": anchor,
                        "actual_prediction_step": dm["prediction_step"], "query_position": dm["query_token_position"],
                        "request_id": dm["request_id"], "budget": budget, "method": method, "epsilon": epsilon,
                        "request_order": order, "query_heads": g, "prefix_tokens": prefix_n,
                        "common_document_tokens": common_n, "full_document_tokens": doc_n,
                        "private_suffix_tokens": private_n, "private_document_tail_tokens": dm["private_document_tail_tokens"],
                        "selected_total_context_tokens": len(combined), "original_total_context_tokens": len(k),
                        "pooled_score_loss_fraction": prev["score_loss_fraction"],
                        "selected_document_indices_sha256": prev["selected_indices_sha256"],
                        "vs_dense": dense_error, "vs_independent": errors(out, independent),
                        "independent_vs_dense": baseline_error,
                        "dense_relative_l2_change_from_independent": dense_error["relative_l2"] - baseline_error["relative_l2"],
                        "dense_max_abs_change_from_independent": dense_error["max_abs"] - baseline_error["max_abs"],
                        "max_abs_lse_delta_vs_dense": float(np.max(np.abs(lse - dense_lse))),
                        "max_abs_lse_delta_vs_independent": float(np.max(np.abs(lse - independent_lse))),
                        "epsilon_zero_output_and_lse_exact": exact_zero,
                        "output_fp64_sha256": array_sha(out), "lse_fp64_sha256": array_sha(lse),
                        "dense_output_fp64_sha256": array_sha(dense), "per_query_head": per_query_head})
            del dec, q, k, v, logits, dense_probs
    assert len(rows) == b * h * 3 * 2 * 9
    return rows, {"document": source["id"], "rows": len(rows), "mask_reconstruction_cpu_wall_seconds": mask_seconds}


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert assigned_devices and "," not in assigned_devices
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", nargs="+", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    protocol_path = BASE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert sha(Path(__file__)) == protocol["script_sha256"]
    assert sha(ROOT / protocol["cpu_constructor_path"]) == protocol["cpu_constructor_sha256"]
    previous_path = ROOT / protocol["cpu_results_path"]
    assert sha(previous_path) == protocol["cpu_results_sha256"]
    previous_rows = {(r["document"], r["budget"], r["kv_head"], r["method"], r["epsilon"], r["request_order"]): r
                     for r in map(json.loads, previous_path.read_text().splitlines())}
    output = BASE / f"{args.job_id}.jsonl"
    assert not output.exists(), "Refuse to overwrite results"
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tw, tc = time.perf_counter(), time.process_time()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    qualification = synthetic_check(device)
    pdev = torch.cuda.get_device_properties(0)
    records = []
    with output.open("w") as stream:
        for document in args.documents:
            source = next(s for s in protocol["sources"] if s["id"] == document)
            rows, info = run_document(source, protocol, previous_rows, device)
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            records.append(info)
            print(json.dumps(info), flush=True)
    torch.cuda.synchronize()
    summary = {"status": "completed", "job_id": args.job_id, "documents": records,
        "protocol_sha256": sha(protocol_path), "script_sha256": sha(Path(__file__)), "result_sha256": sha(output),
        "started_utc": started, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pid": os.getpid(), "uid": os.getuid(), "venv": sys.prefix, "torch_version": torch.__version__,
        "cuda_visible_devices": assigned_devices, "gpu_name": pdev.name, "gpu_uuid": str(getattr(pdev, "uuid", "unavailable")),
        "wall_seconds_post_import": time.perf_counter()-tw, "cpu_seconds_post_import": time.process_time()-tc,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "synthetic_qualification": qualification, "scope": protocol["scope"]}
    (BASE / f"{args.job_id}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": "completed", "job_id": args.job_id, "rows": sum(r['rows'] for r in records),
                      "wall_seconds": summary['wall_seconds_post_import']}), flush=True)


if __name__ == "__main__":
    main()
