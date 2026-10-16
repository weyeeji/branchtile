"""Bounded native-engine CUDA timeline; not a task-quality or latency result."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/Qwen3-4B-Thinking-2507")
    p.add_argument("--manifest", default="data/manifests/qmsum_train_exploratory8.json")
    p.add_argument("--group", type=int, default=3)
    p.add_argument("--output", required=True)
    p.add_argument("--trace-dir", required=True)
    args = p.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    import torch
    from transformers import AutoTokenizer
    import vllm
    from vllm import LLM, SamplingParams
    from capture_model import make_inputs
    torch.set_num_threads(4)
    manifest_path = ROOT / args.manifest
    group = json.loads(manifest_path.read_text())["groups"][args.group]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    ids, start, end = make_inputs(tokenizer, group)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "initializing", "config": vars(args), "uid": os.geteuid(), "python": sys.executable,
        "torch": torch.__version__, "vllm": vllm.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "document_id": group["document_id"], "document_range": [start, end],
        "prompt_lengths": list(map(len, ids)), "B": len(ids),
        "scope": "native BF16 dense+APC CUDA-graph engine attribution; capped 256 generation for instrumentation, not quality; profiler timings carry instrumentation overhead",
        "sampling": {"temperature": .6, "top_p": .95, "top_k": 20, "max_tokens": 256, "min_tokens": 0, "ignore_eos": False},
        "profile_scope": {"warm_apc": True, "delay_iterations": 64, "max_iterations": 32}}
    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    t0 = time.monotonic()
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=65536, max_num_seqs=4, max_num_batched_tokens=4096,
              enable_prefix_caching=True, enforce_eager=False, gpu_memory_utilization=.8,
              disable_custom_all_reduce=True, seed=20261016,
              profiler_config={"profiler": "torch", "torch_profiler_dir": str(ROOT / args.trace_dir),
                               "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False,
                               "torch_profiler_activities": ["CPU", "CUDA"],
                               "delay_iterations": 64, "max_iterations": 32, "ignore_frontend": True})
    report["initialization_seconds"] = time.monotonic() - t0
    warm_start = time.monotonic()
    llm.generate([{"prompt_token_ids": x} for x in ids],
                 SamplingParams(temperature=0, max_tokens=128, min_tokens=0), use_tqdm=False)
    report["actual_shape_warmup_seconds"] = time.monotonic() - warm_start
    report["status"] = "profiling"
    save()
    llm.start_profile(profile_prefix=f"qmsum_g{args.group}_decode")
    params = [SamplingParams(temperature=.6, top_p=.95, top_k=20, max_tokens=256,
                             min_tokens=0, ignore_eos=False, seed=20261016 + args.group * 100 + i)
              for i in range(len(ids))]
    outputs = llm.generate([{"prompt_token_ids": x} for x in ids], params, use_tqdm=False)
    llm.stop_profile()
    report["requests"] = [{"request_id": x.request_id, "generated_tokens": len(x.outputs[0].token_ids),
                            "finish_reason": x.outputs[0].finish_reason,
                            "cached_tokens": x.num_cached_tokens,
                            "metrics": dataclasses.asdict(x.metrics) if dataclasses.is_dataclass(x.metrics) else repr(x.metrics)}
                           for x in outputs]
    traces = list((ROOT / args.trace_dir).rglob("*"))
    report["trace_files"] = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                             for path in traces if path.is_file()]
    report["status"] = "complete"
    report["allocated_process_seconds"] = time.monotonic() - t0
    save()


if __name__ == "__main__":
    main()
