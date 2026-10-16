#!/usr/bin/env python3
"""Audit exact full-input token limits without loading weights or truncating."""
import argparse
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer
from capture_model import make_inputs, sha_bytes, dump_json


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/docfinqa_exploratory_dev8.json")
    parser.add_argument("--models", nargs="+", default=["data/models/Qwen3-4B-Thinking-2507",
                        "data/models/Qwen3-30B-A3B-Thinking-2507-FP8"])
    parser.add_argument("--output", default="results/data_context_audit.json")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for model_path in args.models:
        path = Path(model_path)
        config = json.loads((path / "config.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        for group in manifest["groups"]:
            inputs, begin, end = make_inputs(tokenizer, group)
            lengths = list(map(len, inputs))
            rows.append({"model": path.name, "document_id": group["document_id"],
                         "company": group.get("company"), "company_year": group.get("company_year"),
                         "B": len(inputs), "document_tokens_in_template": end - begin,
                         "prompt_lengths": lengths, "native_max_position_embeddings": config["max_position_embeddings"],
                         "all_questions_fit_native_context_with_8192_answer_cap": max(lengths) + 8192 <= config["max_position_embeddings"],
                         "all_questions_fit_131072_context_with_8192_answer_cap": max(lengths) + 8192 <= 131072})
    report = {"manifest": args.manifest, "manifest_sha256": sha_bytes(Path(args.manifest).read_bytes()),
              "selection_frozen_before_context_audit": True, "no_truncation_or_replacement": True,
              "scope": "token context limit only; capacity and runtime not yet qualified", "rows": rows}
    dump_json(Path(args.output), report)
    print(json.dumps({"output": args.output, "model_document_rows": len(rows)}))


if __name__ == "__main__":
    main()
