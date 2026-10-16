#!/usr/bin/env python3
"""Compare derived overlay supports against the independent capacity audit."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/captures/overhead_large_budget_v1"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    baseline_path = ROOT / "results/natural_capacity_boundary_v1/audit.json"
    baseline = json.loads(baseline_path.read_text())
    source_rows = {(r["document"], r["rank"], r["layer"], str(r["requested_budget"])): r
                   for r in baseline["geometry_rows"]}
    rows, support_checks, source_checks = [], 0, 0
    for document in ("publicdev6_256", "docfinqa256", "publicdev1_256"):
        for rank in (0, 1):
            path = DIRECTORY / document / f"rank{rank}/overlay_receipt.json"
            receipt = json.loads(path.read_text())
            assert receipt["GPU_hours"] == 0 and receipt["CUDA_initialized"] is False
            assert receipt["capacity_audit_source_sha256"] == sha(ROOT / "scripts/audit_natural_capacity_boundary.py")
            for source in receipt["input_links"]:
                expected = baseline["input_sha256"].get(source["source_path"])
                if expected is not None:
                    assert source["sha256"] == expected, source["source_path"]
                    source_checks += 1
            for mask in receipt["generated_masks"]:
                source = source_rows[(document, rank, mask["layer"], str(mask["requested_budget"]))]
                assert mask["full_support_sha256"] == source["restored_full_document_support_sha256"]
                assert mask["effective_full_document_budget"] == source["effective_full_document_budget"]
                assert mask["shared_document_tokens"] == source["common_document_tokens"]
                assert sha(ROOT / mask["path"]) == mask["sha256"]
                if mask["effective_tag"] == "full":
                    assert mask["effective_full_document_budget"] == mask["full_document_tokens"]
                support_checks += 1
            rows.append({"document": document, "rank": rank,
                         "receipt_path": str(path.relative_to(ROOT)),
                         "receipt_sha256": sha(path),
                         "support_checks": len(receipt["generated_masks"])})
    assert support_checks == 96 and source_checks > 0
    result = {"schema": 1, "status": "passed", "GPU_hours": 0,
              "scope": "Independent prior CPU-audit full-support hashes versus generated overlays; no GPU qualification.",
              "capacity_audit_path": str(baseline_path.relative_to(ROOT)),
              "capacity_audit_sha256": sha(baseline_path),
              "auditor_sha256": sha(__file__), "support_checks": support_checks,
              "shared_input_sha256_checks": source_checks, "receipts": rows}
    output = DIRECTORY / "independent_capacity_crosscheck.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "passed", "support_checks": support_checks,
                      "source_checks": source_checks, "output": str(output)}))


if __name__ == "__main__":
    main()
