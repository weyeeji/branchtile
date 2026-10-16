"""Freeze/verify complete operator waves, retaining incomplete jobs separately (CPU)."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waves", default="1,2,5,7,9,12,16,18,20,25")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    waves = [int(x) for x in args.waves.split(",")]
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    ends = {r["name"]: r for r in ledger if r["event"] == "end"}
    counts, receipts, unique, documents = {}, [], set(), set()
    for wave in waves:
        paths = sorted((ROOT / f"results/operator_overhead_wave{wave}").glob(f"overhead_w{wave}_*.json"))
        matching_ends = [r for r in ledger if r["event"] == "end" and r["name"].startswith(f"overhead_w{wave}_")]
        assert len(paths) == len(matching_ends), (wave, "Incomplete wave must be reported separately")
        counts[str(wave)] = dict(jobs=0, configurations=0, allocated_P2_gpu_hours=0.0)
        names = {r["name"] for r in matching_ends}
        for path in paths:
            data = json.loads(path.read_text())
            assert data["tuning_complete"] and all(row["status"] == "ok" for row in data["tuning"])
            name = path.stem
            assert name in names
            end = ends[name]
            assert end["category"] == "P2" and end["exit_code"] == 0 and not end["timed_out"]
            p = data["provenance"]
            key = (p["document_sha256"], p.get("actual_tp_rank"), p["layer"], p["budget"],
                   p["step"], tuple(p["selected_kv_heads"]), tuple(p["original_active_requests"]))
            unique.add(key)
            documents.add(p["document_sha256"])
            row = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), wave=wave,
                       configurations=len(data["tuning"]), batch=data["batch"],
                       allocated_P2_gpu_hours=end["gpu_hours"], real_input_key=key)
            receipts.append(row)
            count = counts[str(wave)]
            count["jobs"] += 1
            count["configurations"] += row["configurations"]
            count["allocated_P2_gpu_hours"] += row["allocated_P2_gpu_hours"]
    partial_ends = [r for r in ledger if r["event"] == "end" and r["name"].startswith("overhead_w22_")]
    abort = ROOT / "results/operator_overhead_wave22/manual_abort.json"
    fields = dict(waves=counts, complete_performance_jobs=len(receipts),
                  complete_FP64_qualified_configurations=sum(r["configurations"] for r in receipts),
                  unique_real_input_cells=len(unique), documents=len(documents),
                  allocated_P2_complete_performance_gpu_hours=sum(r["allocated_P2_gpu_hours"] for r in receipts),
                  explicit_incomplete_wave22=dict(complete_comparisons=0, terminated_jobs=len(partial_ends),
                      allocated_P2_gpu_hours=sum(r["gpu_hours"] for r in partial_ends),
                      decision_path=str(abort.relative_to(ROOT)), decision_sha256=sha(abort)),
                  receipts=receipts)
    fields = json.loads(json.dumps(fields))
    output = ROOT / args.output
    if args.verify:
        saved = json.loads(output.read_text())
        assert all(saved[key] == value for key, value in fields.items()), "Frozen evidence changed"
    else:
        assert not output.exists(), "Preserve older snapshots"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(dict(schema="operator-continuation-summary-v2",
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            producer_script_sha256=sha(Path(__file__)),
            scope="Repeated real development settings, not independent documents or application/evaluation evidence. Incomplete jobs excluded from success counts and charged separately.",
            **fields), indent=2) + "\n")
    print(json.dumps({key: fields[key] for key in (
        "complete_performance_jobs", "complete_FP64_qualified_configurations", "unique_real_input_cells",
        "documents", "allocated_P2_complete_performance_gpu_hours", "explicit_incomplete_wave22")}))


if __name__ == "__main__":
    main()
