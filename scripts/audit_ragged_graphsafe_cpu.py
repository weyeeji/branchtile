"""Read-only terminal audit for the frozen graph-safe G0 experiment."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    reports = []
    for path in args.inputs:
        data = json.loads(Path(path).read_text())
        assert data["status"] == "completed" and data["all_checks_passed"]
        for check in data["checks"]:
            assert check.get("O_fp64_pass", check.get("graph_future_KV_no_leak")) is True
        for check in data["growth_checks"]:
            assert check["finite_FP64_pass"] and check["dynamic_FP64_pass"]
            assert not check["dynamic_prepare"]["signature_changed"]
        sources_match = {p: sha(p) == digest for p, digest in data["source_sha256"].items()}
        assert all(sources_match.values()), sources_match
        reports.append({
            "input": path, "sha256": sha(path), "sources_match": sources_match,
            "checks_count": len(data["checks"]),
            "ordinary_graph_capture_count": len(data["graph_captures"]),
            "workspace_bytes": data["workspace_bytes"],
            "growth_checks": data["growth_checks"],
            "hot": {k: v for k, v in data["timing"]["hot"].items() if k != "blocks"},
            "cold": {k: v for k, v in data["timing"]["cold"].items() if k != "blocks"},
        })
    report = {
        "status": "audit_passed", "gpu_started": False, "reports": reports,
        "decision": "Close complex cohort-depth scheduler integration for this evidence set.",
        "reason": "The step survives graph-safe execution, but the enhanced group cost table explains it; no acceptance or whole-model policy gain has been demonstrated.",
        "limits": [
            "Two layers from the same two-document real-AR capture pair; not actual verification forward.",
            "CPU prepare diagnostics update all three wrappers and are not production metadata overhead.",
            "Plan padded split workitems 170 imply grid [170,1,2], hence 340 blocks, not 170 CTAs.",
            "Kernel timing and analytic KV traffic do not establish measured DRAM bytes.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": len(reports), "output": str(output)}))


if __name__ == "__main__":
    main()
