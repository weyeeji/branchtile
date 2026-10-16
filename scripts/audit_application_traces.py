"""Preserve available step traces by content hash; report overwritten originals.

Older runner versions used only the output stem for auxiliary trace names, so
different model directories could collide. Never rewrite historical references
or pretend a replacement trace is the original. Primary outputs are separate.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / "results/raw/validated_application_traces"
archive.mkdir(parents=True, exist_ok=True)
records = []
for path in sorted((ROOT / "results").glob("application*/*.json")):
    try:
        report = json.loads(path.read_text())
    except json.JSONDecodeError:
        continue  # An actively written incomplete snapshot is retried later.
    for run in report.get("runs", []):
        if "step_observer_sha256" not in run:
            continue
        expected = run["step_observer_sha256"]
        original = ROOT / run["step_observer_trace"]
        saved = archive / (expected + ".jsonl")
        if saved.exists() and hashlib.sha256(saved.read_bytes()).hexdigest() == expected:
            status = "verified_content_addressed_copy"
        elif original.exists() and hashlib.sha256(original.read_bytes()).hexdigest() == expected:
            shutil.copyfile(original, saved)
            assert hashlib.sha256(saved.read_bytes()).hexdigest() == expected
            status = "verified_content_addressed_copy"
        else:
            status = "original_auxiliary_trace_unavailable_or_overwritten"
        records.append({"report": str(path.relative_to(ROOT)), "report_status": report.get("status"),
                        "group_index": run["group_index"], "arm": run["arm"], "repeat": run["repeat"],
                        "original_path": run["step_observer_trace"], "expected_sha256": expected,
                        "status": status, "verified_copy": str(saved.relative_to(ROOT)) if status.startswith("verified") else None})
counts = {status: sum(r["status"] == status for r in records) for status in sorted({r["status"] for r in records})}
output = {"cause": "Historical application step trace names omitted report directory, allowing collisions across model output directories.",
          "scope": "Auxiliary FINAL-only completion traces; raw answer text, token IDs, per-request metrics and group JCT remain in each original JSON report and are not reconstructed here.",
          "restriction": "Unavailable auxiliary traces cannot support per-step timing/ready-set claims. FINAL-only traces never established attention ready membership even when available.",
          "fix": "Future runner namespaces trace paths by report parent and stem; valid historical traces are archived by their expected content hash. No historical hash or output is silently changed.",
          "counts": counts, "records": records}
(ROOT / "results/application_trace_audit.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(counts))
