"""CPU-only exact-template lengths; never treats context fit as GPU capacity."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys

TOKENIZER = None


def initialize(model):
    global TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True)


def qualify(group):
    from capture_model import make_inputs
    ids, start, end = make_inputs(TOKENIZER, group)
    lengths = list(map(len, ids))
    return {"document_id": group["document_id"], "document_sha256": group["document_sha256"],
            "B": group["B"], "prompt_lengths": lengths, "document_token_range": [start, end],
            "shared_prefix_verified": True,
            "fit_131072_with_8192_per_independent_answer": all(x+8192<=131072 for x in lengths),
            "fit_262144_with_8192_per_independent_answer": all(x+8192<=262144 for x in lengths)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", default="data/models/Qwen3-30B-A3B-Thinking-2507-FP8")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    manifest = Path(a.manifest)
    groups = json.loads(manifest.read_text())["groups"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(a.model,)) as pool:
        rows = list(pool.map(qualify, groups))
    record = {"manifest": a.manifest, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "model": a.model, "rows": rows, "groups": len(rows),
              "fit_131072_count": sum(r["fit_131072_with_8192_per_independent_answer"] for r in rows),
              "fit_262144_count": sum(r["fit_262144_with_8192_per_independent_answer"] for r in rows),
              "scope": "CPU tokenizer/template qualification only; no truncation, selection change or GPU-capacity claim. Out-of-context groups remain in denominator."}
    Path(a.output).write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    main()
