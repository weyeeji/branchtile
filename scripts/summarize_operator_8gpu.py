"""Verify or create the frozen seven-wave operator continuation summary (CPU)."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WAVES = (1, 2, 5, 7, 9, 12, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/operator_8gpu_exploration_summary_v1/summary.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    cells, unique, counts = [], set(), {}
    for wave in WAVES:
        files = sorted((ROOT / f"results/operator_overhead_wave{wave}").glob(f"overhead_w{wave}_*.json"))
        count = 0
        for path in files:
            data = json.loads(path.read_text())
            assert data["tuning_complete"] and all(item["status"] == "ok" for item in data["tuning"])
            count += len(data["tuning"])
            p = data["provenance"]
            key = (p["document_sha256"], p.get("actual_tp_rank"), p["layer"], p["budget"],
                   p["step"], tuple(p["selected_kv_heads"]), tuple(p["original_active_requests"]))
            unique.add(key)
            cells.append(dict(wave=wave, path=str(path.relative_to(ROOT)),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest(), key=list(key),
                              batch=data["batch"], configurations=len(data["tuning"]), scope=data["scope"]))
        counts[str(wave)] = dict(cells=len(files), configurations=count)
    ledger = [json.loads(line) for line in (ROOT / "results/job_ledger.jsonl").read_text().splitlines()]
    jobs = [r for r in ledger if r["event"] == "end"
            and any(r["name"].startswith(f"overhead_w{wave}_") for wave in WAVES)]
    assert len(jobs) == len(cells) == 50
    assert all(r["exit_code"] == 0 and not r["timed_out"] for r in jobs)
    fields = dict(waves=counts, performance_job_count=len(cells),
                  configuration_count=sum(row["configurations"] for row in counts.values()),
                  unique_real_input_cells=len(unique), documents=len({row[0] for row in unique}),
                  all_configuration_checks_pass=True, all_performance_jobs_clean=True,
                  allocated_P2_gpu_hours=sum(r["gpu_hours"] for r in jobs), cell_receipts=cells)
    # Normalize tuples in the input identity to the JSON representation.
    fields = json.loads(json.dumps(fields))
    output = ROOT / args.output
    if args.verify:
        saved = json.loads(output.read_text())
        assert all(saved[key] == value for key, value in fields.items()), "Frozen summary changed"
    else:
        assert not output.exists(), "Preserve the previous snapshot"
        result = dict(schema="operator-eight-gpu-continuation-v1",
                      created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      producer_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      scope="Repeated development cells, not application quality or formal evaluation.",
                      **fields)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: fields[key] for key in ("performance_job_count", "configuration_count",
                     "unique_real_input_cells", "documents", "allocated_P2_gpu_hours")}, sort_keys=True))


if __name__ == "__main__":
    main()
