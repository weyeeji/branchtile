"""Summarize CUDA trace attribution without confusing kernel sums with latency."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def union_duration(events):
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
    total, start, stop = 0., None, None
    for left, right in intervals:
        if start is None:
            start, stop = left, right
        elif left <= stop:
            stop = max(stop, right)
        else:
            total += stop - start
            start, stop = left, right
    return total + (stop - start if start is not None else 0.)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    metadata = json.loads((ROOT / args.input).read_text())
    assert metadata["status"] == "complete"
    trace_path = ROOT / next(f["path"] for f in metadata["trace_files"] if f["path"].endswith(".json.gz"))
    events = json.load(gzip.open(trace_path, "rt"))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    groups = collections.defaultdict(list)
    names = collections.Counter()
    for e in kernels:
        name = e["name"]
        names[name] += e["dur"]
        if "flash_fwd_splitkv" in name:
            category = "attention_including_split_combine"
        elif "gemm" in name.lower() or "cublas" in name.lower():
            category = "matrix_multiply_and_reduction"
        elif "reshape_and_cache" in name:
            category = "kv_cache_write"
        else:
            category = "other"
        groups[category].append(e)
    total = sum(e["dur"] for e in kernels)
    annotations = collections.Counter(e["name"] for e in events if e.get("cat") == "user_annotation")
    report = {"input": args.input, "trace": str(trace_path.relative_to(ROOT)),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "scope": metadata["scope"], "B": metadata["B"], "prompt_lengths": metadata["prompt_lengths"],
        "iteration_annotations": dict(annotations), "kernel_count": len(kernels),
        "sum_kernel_ms": total / 1000,
        "union_kernel_busy_ms": union_duration(kernels) / 1000,
        "kernel_first_to_last_span_ms": (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) / 1000,
        "groups": {name: {"count": len(es), "sum_ms": sum(e["dur"] for e in es) / 1000,
                           "fraction_of_kernel_sum": sum(e["dur"] for e in es) / total}
                   for name, es in groups.items()},
        "top_kernels": [{"name": name, "sum_ms": value / 1000} for name, value in names.most_common(20)],
        "limitations": ["Profiler instrumentation alters runtime; this is attribution, not unprofiled JCT or speedup.",
                        "Dense attention fraction does not describe a sparse engine; remeasure after integration.",
                        "Kernel sum may overlap; union busy time and full timeline span are separately reported.",
                        "One frozen train document, one 32-iteration window, capped generation; not general performance or quality evidence."]}
    (ROOT / args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"groups": report["groups"], "iteration_annotations": report["iteration_annotations"]}, indent=2))


if __name__ == "__main__":
    main()
