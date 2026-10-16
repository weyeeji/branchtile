"""CPU-only input-context and conditional native KV-pool capacity accounting.

Reads existing input-length metadata and precision receipts. Never reads answer
fields or formal model outputs; manifests are opened only as bytes for SHA256.
No tensor libraries, model execution, tokenizer rerun, or GPU calls.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "results/platform/native_context_capacity_20261016.json"
MODEL = "data/models/Qwen3-30B-A3B-Thinking-2507-FP8"
PAGE, ANSWER_CAP = 16, 8192


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(rows):
    return {"groups": len(rows), "questions": sum(r["B"] for r in rows)}


def build():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    inputs = {}

    def receipt(relative):
        path = ROOT / relative
        inputs[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return path

    def read(relative):
        return json.loads(receipt(relative).read_text())

    model = read(f"{MODEL}/config.json")
    assert model["max_position_embeddings"] == 262144
    assert model["rope_scaling"] is None
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        receipt(f"{MODEL}/{name}")
    protocol = read("configs/native_precision_natural_development_v1.json")
    for path in ("scripts/capture_model.py", "scripts/vllm_native_precision_worker.py",
                 ".venv-engine/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py",
                 ".venv-engine/lib/python3.12/site-packages/vllm/v1/attention/backends/flashinfer.py"):
        receipt(path)
    pools = {}
    for arm, run in (("BF16", "original_bf16"), ("FP8_E4M3_unit", "original_fp8")):
        data = read(f"results/native_precision_natural_development_v1/{run}/summary.json")
        assert data["status"] == "completed"
        assert data["engine_configuration"]["max_model_len"] == 98304
        assert data["engine_configuration"]["max_num_seqs"] == 3
        assert data["engine_configuration"]["gpu_memory_utilization"] == 0.8
        ranks = data["precision_after_warmup"]
        assert len(ranks) == 2 and {r["rank"] for r in ranks} == {0, 1}
        assert len({r["allocated_cache_token_capacity"] for r in ranks}) == 1
        assert len({r["cache_unique_storage_bytes"] for r in ranks}) == 1
        pools[arm] = {
            "allocated_tokens_per_rank": ranks[0]["allocated_cache_token_capacity"],
            "allocated_bytes_per_rank": ranks[0]["cache_unique_storage_bytes"],
            "original_max_model_len": 98304, "original_max_num_seqs": 3,
            "gpu_memory_utilization": 0.8,
            "receipt_scope": "Existing completed-result memory receipt only; this audit does not adjudicate run_jobs terminal status or quality.",
        }
    bf_pool = pools["BF16"]["allocated_tokens_per_rank"]
    fp_pool = pools["FP8_E4M3_unit"]["allocated_tokens_per_rank"]
    limits = (98304, 131072, 196608, bf_pool - PAGE, 262144)
    splits = {}
    for split in ("development8", "training167", "evaluation100"):
        metadata_path = f"results/data_qualification/docfinqa_public_{split}_v1_context.json"
        metadata = read(metadata_path)
        # Hash the manifest without parsing its questions/answers.
        manifest_path = receipt(metadata["manifest"])
        assert sha(manifest_path) == metadata["manifest_sha256"]
        assert metadata["model"] == MODEL
        rows = []
        for index, item in enumerate(metadata["rows"]):
            assert item["shared_prefix_verified"] is True
            lengths = item["prompt_lengths"]
            assert len(lengths) == item["B"] and all(isinstance(n, int) for n in lengths)
            start, end = item["document_token_range"]
            assert 0 <= start < end <= min(lengths)
            common = end // PAGE * PAGE
            rounded_private = [((n + ANSWER_CAP - common + PAGE - 1) // PAGE) * PAGE for n in lengths]
            live = common + sum(rounded_private) + PAGE  # permanent null block
            rows.append({
                "manifest_group_index": index, "document_id": item["document_id"],
                "document_sha256": item["document_sha256"], "B": item["B"],
                "prompt_lengths": lengths, "document_token_range": [start, end],
                "max_prompt_plus_answer_cap": max(lengths) + ANSWER_CAP,
                "guaranteed_shared_full_page_tokens": common,
                "private_rounded_tokens_per_request_at_cap": rounded_private,
                "conditional_group_pool_tokens_including_null": live,
                "native_context_fits": max(lengths) + ANSWER_CAP <= 262144,
            })
        assert len(rows) == metadata["groups"]
        context_counts = {str(limit): count([r for r in rows if r["max_prompt_plus_answer_cap"] <= limit])
                          for limit in limits}
        pool_counts = {}
        for name, pool in pools.items():
            cap = pool["allocated_tokens_per_rank"]
            pool_counts[name] = count([r for r in rows if r["native_context_fits"] and
                                      r["conditional_group_pool_tokens_including_null"] <= cap])
        splits[split] = {
            **count(rows), "cached_context_source": metadata_path,
            "manifest_sha256_matches": True,
            "B_histogram": {str(b): sum(r["B"] == b for r in rows) for b in sorted({r["B"] for r in rows})},
            "prompt_max_per_document_min_median_max": [min(max(r["prompt_lengths"]) for r in rows),
                statistics.median(max(r["prompt_lengths"]) for r in rows), max(max(r["prompt_lengths"]) for r in rows)],
            "context_counts_with_full_answer_reserve": context_counts,
            "conditional_pool_counts_with_native_context_and_full_group_reserve": pool_counts,
            "rows": rows,
        }
    # Existing actual short-pilot prompts cross-check the metadata, without
    # reusing or reading any generated answers from that pilot.
    by_id = {r["document_id"]: r for r in splits["development8"]["rows"]}
    for item in protocol["documents"]:
        row = by_id[item["document_id"]]
        assert row["prompt_lengths"] == item["prompt_lengths"]
        assert row["document_token_range"] == item["document_token_range"]
        assert row["document_sha256"] == item["document_sha256"]
    candidates = [r for r in splits["training167"]["rows"] if r["B"] in (2, 3) and
                  r["native_context_fits"] and max(r["prompt_lengths"]) > bf_pool - PAGE and
                  r["conditional_group_pool_tokens_including_null"] <= fp_pool]
    return {
        "schema": 1, "status": "CPU_capacity_boundary_only_no_GPU_followup_authorized",
        "date": "2026-10-16", "GPU_hours": 0, "script_sha256": sha(Path(__file__)),
        "model_native_max_position_embeddings": model["max_position_embeddings"],
        "model_rope_scaling": model["rope_scaling"], "model_revision": protocol["model_revision"],
        "page_tokens": PAGE, "answer_reserve_per_request": ANSWER_CAP,
        "observed_pools": pools,
        "formulas": {
            "context": "max_i(prompt_i) + 8192 <= configured max_model_len <= 262144",
            "common": "C=16*floor(document_end/16); includes system prefix and guaranteed identical document pages",
            "group_pool": "L=C + sum_i(16*ceil((prompt_i+8192-C)/16)) + 16 permanent-null tokens",
            "condition": "L <= observed allocated pool tokens, assuming one cold document anchor, shared pinned pages and all B requests ready",
        },
        "assumptions_and_limits": [
            "Existing exact input lengths are reused; no new tokenizer execution or formal answer/model-output inspection.",
            "Manifest bytes are hashed only. Public evaluation counts describe input coverage, not an evaluation run or permission to start one.",
            "The original pool receipts were measured at max_model_len=98304 and max_num_seqs=3. Increasing either changes buffers/profile and may reduce available pool capacity.",
            "Pool count is a conservative page-rounded reservation at all requests' full cap with guaranteed document sharing; outside the bound is not proof that a shorter natural generation cannot complete.",
            "A real common prefix may extend past the document; that extra sharing is deliberately not credited. Free evictable APC blocks can be reclaimed; unrelated live requests and transient staging are excluded.",
            "Allocator lookahead/COW/temporary requirements beyond the explicit permanent null block are not modeled; actual admission, peak usage, preemption, runtime kernels and clean exit need qualification.",
            "Most public groups have B4, exceeding the current pilot's max_num_seqs3; their group counts assume a newly qualified max_num_seqs>=B, not the existing configuration.",
            "Explicit max_model_len262144 BF16 is expected to fail vLLM startup's one-request KV-memory check at the observed .80 pool. This does not imply every shorter input requires FP8.",
            "413072 pooled FP8 slots do not extend the model's native 262144-token context. No YaRN/RoPE override, truncation, retrieval or data exclusion is introduced.",
            "Unit-scale FP8 is an established baseline and has not passed formal quality noninferiority. Capacity does not demonstrate accuracy or speed.",
        ],
        "two_development_prompt_metadata_crosschecks": len(protocol["documents"]),
        "deferred_training_only_boundary_example": {
            "selection_rule": "First manifest-order B2/B3 training group whose raw prompt exceeds observed BF16 pool minus null, but whose full-group reserve fits FP8 and native context",
            "candidate": candidates[0] if candidates else None,
            "decision": "Do not launch: no current qualified research candidate or formal quality evidence; retain only as a possible future native long-context qualification boundary",
        },
        "input_files": inputs, "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = build()
    output = ROOT / args.output
    if args.check:
        assert json.loads(output.read_text()) == result
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": args.output, "check": args.check,
                      "context_counts": {k: v["context_counts_with_full_answer_reserve"] for k, v in result["splits"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
