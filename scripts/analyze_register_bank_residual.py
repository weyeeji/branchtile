"""CPU-only logical MMA accounting for the stopped GQA8 register-bank idea.

Uses existing all-48-layer independent-mask geometry. No tensor/GPU libraries,
GPU execution, new masks, measurements, fitting, or kernel compilation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "results/platform/register_bank_residual_review_20261016.json"
# Exhaustive set partitions of three original request IDs, encoded as bitmasks.
PARTITIONS = ((1, 2, 4), (1, 6), (3, 4), (5, 2), (7,))
PROTOCOL = "configs/joint_real48_replay_protocol_v1.json"
BENCHMARK = "results/suffix_v2_replay/main30b_budget_grid_rank0_l15_s64_b32768.json"
RESOURCES = "results/kernel_resources.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator}


def row_groups(owner):
    # GQA8, M16: one/two/three requests use one/one/two M16 groups.
    return (8 * owner.bit_count() + 15) // 16


def costs(counts):
    compact = sum(row_groups(o) * n for o, n in counts.items())
    compact64 = sum(row_groups(o) * ((n + 63) // 64)
                    for o, n in counts.items())
    fixed, fixed64, token_reads = [], [], []
    for partition in PARTITIONS:
        group_tokens = [(g, sum(n for o, n in counts.items() if o & g))
                        for g in partition]
        fixed.append(sum(row_groups(g) * n for g, n in group_tokens))
        fixed64.append(sum(row_groups(g) * ((n + 63) // 64)
                           for g, n in group_tokens))
        token_reads.append(sum(n for _, n in group_tokens))
    return {"compact_M16_token_units": compact,
            "compact_M16_N64_tiles": compact64,
            "fixed_partition_M16_token_units": fixed,
            "fixed_partition_M16_N64_tiles": fixed64,
            "fixed_partition_logical_KV_token_reads": token_reads,
            "best_fixed_M16_token_units": min(fixed),
            "best_fixed_M16_N64_tiles": min(fixed64),
            "best_fixed_exact_partition_indices": [i for i, x in enumerate(fixed) if x == min(fixed)],
            "best_fixed_N64_partition_indices": [i for i, x in enumerate(fixed64) if x == min(fixed64)],
            "full_union_M16_token_units": fixed[-1],
            "full_union_M16_N64_tiles": fixed64[-1],
            "union_logical_KV_tokens": sum(counts.values()),
            "independent_logical_KV_token_reads": sum(o.bit_count() * n for o, n in counts.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="Compare against existing JSON without writing")
    args = parser.parse_args()
    inputs = {}

    def read(relative):
        path = ROOT / relative
        inputs[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return json.loads(path.read_text())

    protocol = read(PROTOCOL)
    assert protocol["budget"] == 32768
    documents = []
    captured_source_checks = []
    for doc_index, doc in enumerate(protocol["documents"]):
        folder = f"results/joint_real48_replay_v1/joint_real48_doc{doc_index}_replica0"
        summary = read(f"{folder}/summary.json")
        assert summary["status"] == "completed"
        assert summary["document_sha256"] == doc["document_sha256"]
        assert summary["document_id"] == doc["document_id"]
        assert summary["budget"] == 32768 and summary["B"] == 3
        assert summary["protocol_sha256"] == inputs[PROTOCOL]["sha256"]
        assert summary["source_sha256"] == protocol["source_sha256"]
        captured_source_checks.append({"summary": f"{folder}/summary.json",
                                       "source_and_protocol_receipts_match": True})
        cohorts = []
        for rank in range(2):
            source = f"{folder}/rank{rank}/joint_variants.json"
            variants = read(source)
            assert variants["rank"] == rank
            variant = variants["variants"]["epsilon01"]
            assert (variant["layers"], variant["requests"], variant["heads"]) == (48, 3, 2)
            assert [x["layer"] for x in variant["layer_records"]] == list(range(48))
            for layer in variant["layer_records"]:
                assert layer["shape"][:2] == [3, 2]
                assert [r["head"] for r in layer["records"]] == [0, 1]
                private_document_tail = layer["shape"][2] - layer["common_source_tokens"]
                assert 0 <= private_document_tail < 16
                for record in layer["records"]:
                    geometry = record["independent_geometry"]
                    # The container is epsilon01, but only the original geometry
                    # is read. Verify its identity against both other containers.
                    for name in ("epsilon0", "epsilon001"):
                        other = variants["variants"][name]["layer_records"][layer["layer"]]
                        assert other["layer"] == layer["layer"]
                        other_record = other["records"][record["head"]]
                        assert other_record["independent_geometry"] == geometry
                        assert [x["independent_mask_sha256"] for x in other_record["requests"]] == [
                            x["independent_mask_sha256"] for x in record["requests"]]
                    counts = {int(k): v for k, v in geometry["owner_bitmask_counts"].items()}
                    assert all(1 <= o <= 7 and isinstance(n, int) and n > 0 for o, n in counts.items())
                    assert sum(counts.values()) == geometry["common_source_union_tokens"]
                    assert [x["question"] for x in record["requests"]] == [0, 1, 2]
                    selected = [sum(n for o, n in counts.items() if o & (1 << q)) for q in range(3)]
                    assert all(r["budget"] == 32768 for r in record["requests"])
                    assert selected == [32768 - private_document_tail] * 3
                    cohorts.append({"rank": rank, "layer": layer["layer"], "head": record["head"],
                                    "private_document_tail_tokens_per_request_omitted": private_document_tail,
                                    "owner_counts_in_order_1_through_7": [counts.get(o, 0) for o in range(1, 8)],
                                    **costs(counts)})
        assert len(cohorts) == 48 * 2 * 2
        scalar_keys = [k for k, v in costs({7: 1}).items() if isinstance(v, int)]
        totals = {k: sum(c[k] for c in cohorts) for k in scalar_keys}
        ratios = {
            "best_fixed_over_compact_exact": ratio(totals["best_fixed_M16_token_units"], totals["compact_M16_token_units"]),
            "best_fixed_over_compact_N64": ratio(totals["best_fixed_M16_N64_tiles"], totals["compact_M16_N64_tiles"]),
            "full_union_over_compact_exact": ratio(totals["full_union_M16_token_units"], totals["compact_M16_token_units"]),
            "independent_over_union_logical_KV_reads": ratio(totals["independent_logical_KV_token_reads"], totals["union_logical_KV_tokens"]),
        }
        arithmetic_reduction = 1 - totals["compact_M16_token_units"] / totals["best_fixed_M16_token_units"]
        documents.append({**doc, "cohorts": len(cohorts), "totals": totals, "ratios": ratios,
                          "all_three_owner_union_fraction_median": statistics.median(
                              c["owner_counts_in_order_1_through_7"][6] / c["union_logical_KV_tokens"] for c in cohorts),
                          "conditional_pure_arithmetic_attention_latency_reduction": arithmetic_reduction,
                          "conditional_model_latency_reduction_at_attention_fractions_0p20_0p25": [
                              arithmetic_reduction * f for f in (0.20, 0.25)],
                          "per_cohort_accounting": cohorts})

    benchmark = read(BENCHMARK)
    resources = read(RESOURCES)
    current_source_hashes = {
        p: sha(ROOT / p) for p in (
            "scripts/analyze_register_bank_residual.py", "src/branchtile/tilelang_kernel_v2.py",
            "src/branchtile/tilelang_kernel_v3.py", "src/branchtile/union.py")}
    recorded_source_checks = {
        p: {"captured_sha256": h, "current_sha256": sha(ROOT / p),
            "current_matches_capture": sha(ROOT / p) == h}
        for p, h in protocol["source_sha256"].items()}
    result = {
        "schema": 1, "status": "CPU_review_complete_no_GPU_candidate_recommended",
        "uid": os.getuid(), "GPU_hours": 0, "imports": "Python standard library only",
        "scope": "Two development documents, replica0 only, 48 layers x 2 TP ranks x 2 KV heads each; no independent-document inference from layers/heads/replicas.",
        "shape": {"B": 3, "GQA": 8, "D": 128, "document_budget": 32768, "M": 16, "N": 64},
        "partition_bitmasks_in_array_order": [list(p) for p in PARTITIONS],
        "definitions": {
            "counts": "n_o is the number of selected shared-document tokens whose original owner bitmask is o, o in 1..7.",
            "row_group_count": "r(o)=ceil(8*popcount(o)/16)=ceil(popcount(o)/2)",
            "compact_exact": "C=sum_o r(o)*n_o, units M16-times-token; all D/K dimensions are common and omitted.",
            "fixed_exact": "F(P)=sum_{g in P} r(g)*sum_{o:o&g!=0} n_o; packed union may cross owner boundaries inside each fixed group.",
            "compact_N64": "C64=sum_o r(o)*ceil(n_o/64); one tail per nonempty owner region, no extra split tails.",
            "fixed_N64": "F64(P)=sum_{g in P} r(g)*ceil(sum_{o:o&g!=0} n_o/64); one tail per fixed group, no extra split tails.",
            "best_fixed": "For each rank/layer/head, independently minimize over all five B3 partitions, separately for exact and N64 counts; a hindsight logical-work oracle, not a measured latency winner or deployed chooser.",
            "aggregation": "Sum each work count over the 192 cohorts, then take ratio of sums; do not average per-cohort ratios.",
            "logical_KV": "Independent=sum_o popcount(o)*n_o; union=sum_o n_o. These count logical requests to payload, not GDDR transactions.",
        },
        "limitations": [
            "Logical MMA ratios are not complete runtime upper bounds; bandwidth, scheduling, occupancy, instructions, barriers and cache behavior can differ.",
            "System prefix, protected private document tail, question suffix and generated tail are omitted; no prepare, planning, final merge, copy or selector cost is included.",
            "The omitted private work and uncharged owner transitions favor an optimistic dynamic accounting; no split-boundary overhead is modeled.",
            "M16 arithmetic counts are not the instruction count of native SIMT/XQA or a proof of its roofline bottleneck.",
            "Fixed partition is allowed to coalesce its union across owner boundaries, as the strong existing control does.",
            "The arithmetic-only Amdahl values assume equal time per logical work unit, all attention time proportional to arithmetic, and zero added cost; they are conditional illustrations, not measured model improvements or universal bounds.",
            "Logical 2.6x KV sharing cannot establish GDDR savings; native E0 can reuse shared addresses in L2 and GPU counters are unavailable without additional permissions.",
            "No GQA16/32 shape expansion, new kernel, GPU run, mask change, quality test or novelty claim is proposed.",
        ],
        "input_files": inputs, "review_current_source_sha256": current_source_hashes,
        "original_capture_source_hash_checks": recorded_source_checks,
        "original_summary_source_receipt_checks": captured_source_checks,
        "documents": documents,
        "existing_measured_example": {"path": BENCHMARK, "provenance": benchmark["provenance"],
                                      "packages": benchmark["packages"],
                                      "tuning_completed": benchmark["tuning_completed"],
                                      "hot_median_us": benchmark["summary"]["hot"]["median_us"],
                                      "scope": "Existing corrected-suffix L15/step64 original-document rank0 operator measurement, not the hypothetical register-bank implementation and not all-48 production latency."},
        "existing_compiled_resource_receipts": resources,
        "resource_scope": "Saved shape-specific cubin receipts; no spill/DRAM/bank-conflict attribution or new compiler measurement.",
        "decision": "One unimplemented register-layout gap is technically plausible, but current GQA8 logical arithmetic headroom and unproven memory advantage do not justify restarting the stopped dynamic direction. No candidate recommended.",
    }
    path = ROOT / args.output
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        assert path.read_text() == encoded, "Stored result differs from current CPU recomputation"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded)
    print(json.dumps({"output": str(path.relative_to(ROOT)), "check": args.check,
                      "documents": [{"document_id": d["document_id"], "ratios": d["ratios"]} for d in documents]}))


if __name__ == "__main__":
    main()
