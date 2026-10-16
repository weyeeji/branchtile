"""Real official FP8 TP2 deployment and complete-document generation qualification.

This is a deployment pilot, not an application speedup or quality comparison.
The first request is cold; the following natural group is APC-warm. Neither
is called A/A. All stages, failures, output caps, and actual outputs are kept.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_state(worker):
    """Called inside each real vLLM rank; includes the resolved MoE backend."""
    import torch
    from vllm.distributed import get_tp_group
    torch.cuda.synchronize()
    model = worker.model_runner.model
    methods = {}
    dtypes = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtypes[key] = dtypes.get(key, 0) + parameter.numel()
    for module in model.modules():
        method = getattr(module, "quant_method", None)
        if method is not None:
            key = type(method).__module__ + "." + type(method).__name__
            methods.setdefault(key, {"count": 0, "resolved_fp8_backends": set()})
            methods[key]["count"] += 1
            backend = getattr(method, "fp8_backend", None)
            if backend is not None:
                methods[key]["resolved_fp8_backends"].add(str(backend))
    for item in methods.values():
        item["resolved_fp8_backends"] = sorted(item["resolved_fp8_backends"])
    group = get_tp_group()
    free, total = torch.cuda.mem_get_info()
    cfg = worker.vllm_config
    return {"pid": os.getpid(), "tp_world_size": group.world_size,
            "rank_in_group": group.rank_in_group,
            "device_name": torch.cuda.get_device_name(),
            "device_capability": list(torch.cuda.get_device_capability()),
            "device_properties": str(torch.cuda.get_device_properties(torch.cuda.current_device())),
            "parameter_elements_by_dtype": dtypes, "quantization_methods": methods,
            "memory_free_bytes": free, "memory_total_bytes": total,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cache_dtype": cfg.cache_config.cache_dtype,
            "model_dtype": str(cfg.model_config.dtype),
            "actual_kv_tensor_dtypes": sorted({str(t.dtype) for t in getattr(worker.model_runner, "kv_caches", []) if hasattr(t, "dtype")}),
            "cache_block_size": cfg.cache_config.block_size,
            "cache_num_gpu_blocks": cfg.cache_config.num_gpu_blocks,
            "max_model_len": cfg.model_config.max_model_len}


class DeploymentWorkerExtension:
    """Named RPC avoids enabling arbitrary-function pickle serialization."""
    def deployment_state(self):
        return worker_state(self)


class Monitor:
    def __init__(self, path):
        self.path = path
        self.stop_event = threading.Event()
        self.peaks = {}
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            import pynvml as nv
            nv.nvmlInit()
            ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            handles = [(i, nv.nvmlDeviceGetHandleByIndex(int(i))) for i in ids]
            with self.path.open("w") as out:
                while not self.stop_event.is_set():
                    for index, handle in handles:
                        mem = nv.nvmlDeviceGetMemoryInfo(handle)
                        uuid = nv.nvmlDeviceGetUUID(handle)
                        if isinstance(uuid, bytes):
                            uuid = uuid.decode()
                        self.peaks[uuid] = max(self.peaks.get(uuid, 0), mem.used)
                        row = {"time_utc": now(), "monotonic": time.monotonic(),
                               "physical_index": index, "uuid": uuid,
                               "used_memory_bytes": mem.used, "total_memory_bytes": mem.total,
                               "gpu_util_percent": nv.nvmlDeviceGetUtilizationRates(handle).gpu,
                               "board_power_mw": nv.nvmlDeviceGetPowerUsage(handle)}
                        out.write(json.dumps(row) + "\n")
                    out.flush()
                    self.stop_event.wait(1)
            nv.nvmlShutdown()
        except Exception:
            self.error = traceback.format_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default="data/manifests/docfinqa_exploratory_dev8.json")
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.85)
    parser.add_argument("--moe-backend", default=None)
    parser.add_argument("--capture-dir", default=None)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    monitor = Monitor(output.with_suffix(".nvml.jsonl"))
    record = {"schema": 1, "status": "started", "started_utc": now(),
              "uid": os.geteuid(), "pid": os.getpid(), "python": sys.executable,
              "virtual_environment": sys.prefix, "arguments": vars(args),
              "visible_gpus": os.environ["CUDA_VISIBLE_DEVICES"], "stages": [],
              "scope": "Official real-weight TP2 deployment and complete-document natural-generation pilot; no application speedup or quality comparison",
              "limitations": ["The exact input manifest is recorded; this deployment pilot does not establish application quality",
                              "Cold singleton and APC-warm group have different states and are not A/A",
                              "NVML memory peak is sampled, not an exact allocation high-water mark"]}
    def save():
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    save()
    monitor.thread.start()
    try:
        import torch
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from capture_model import make_inputs
        record["packages"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                              "vllm": vllm.__version__}
        download = json.loads((ROOT / "results/Qwen3-30B-A3B-Thinking-2507-FP8_download.json").read_text())
        assert download["complete"] and download["revision"] == "60d80c83c53c3b611c642dbb8c942b3f90c5948a"
        record["checkpoint_identity"] = download
        manifest = Path(args.manifest)
        groups = json.loads(manifest.read_text())["groups"]
        group = next(g for g in groups if g["document_id"] == args.document_id)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        started = time.monotonic()
        inputs, start, end = make_inputs(tokenizer, group)
        assert all(len(ids) + args.max_new_tokens <= args.max_model_len for ids in inputs)
        record["input"] = {"manifest_sha256": digest(manifest), "document_id": group["document_id"],
                           "document_sha256": group["document_sha256"], "company_year": group.get("company_year"),
                           "B": group["B"], "document_token_range": [start, end],
                           "prompt_lengths": list(map(len, inputs)), "preparation_seconds": time.monotonic()-started,
                           "prompt_token_hashes": [hashlib.sha256(json.dumps(ids).encode()).hexdigest() for ids in inputs],
                           "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                           "tokenizer_config_sha256": digest(Path(args.model) / "tokenizer_config.json"),
                           "template_implementation_sha256": digest(ROOT / "scripts/capture_model.py"),
                           "truncated_document": False, "questions": group["questions"]}
        save()
        kwargs = dict(model=args.model, tensor_parallel_size=2, dtype="bfloat16",
                      kv_cache_dtype="auto", max_model_len=args.max_model_len,
                      max_num_seqs=4, max_num_batched_tokens=4096,
                      enable_prefix_caching=True, enforce_eager=True,
                      worker_extension_cls=("scripts.vllm_rank_capture.RankCaptureWorkerExtension" if args.capture_dir
                                            else "scripts.probe_engine_docfinqa.DeploymentWorkerExtension"),
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      disable_custom_all_reduce=True, disable_log_stats=False, seed=20261016)
        if args.moe_backend:
            kwargs["moe_backend"] = args.moe_backend
        record["engine_configuration"] = kwargs
        started = time.monotonic()
        llm = LLM(**kwargs)
        record["initialization_seconds"] = time.monotonic()-started
        record["worker_state_after_initialization"] = llm.collective_rpc("deployment_state")
        if args.capture_dir:
            capture_config = {"output": str(Path(args.capture_dir).resolve()), "layers": [0, 15, 31, 47],
                              "steps": [1, 16, 64, 256], "document_token_range": [start, end],
                              "document_sha256": group["document_sha256"],
                              "prompt_token_hashes": record["input"]["prompt_token_hashes"],
                              "model_revision": download["revision"],
                              "instrumentation_source_sha256": digest(ROOT / "scripts/vllm_rank_capture.py")}
            record["rank_capture_configuration"] = capture_config
            record["rank_capture_initialization"] = llm.collective_rpc("configure_rank_capture", args=(capture_config,))
            record["limitations"].append("Read-only rank activation instrumentation is enabled; these latencies cannot be used for application performance claims")
        save()
        for stage, indices in [("cold_singleton", [0]), ("apc_warm_natural_group", list(range(len(inputs))))]:
            if args.capture_dir:
                llm.collective_rpc("set_rank_capture_phase", args=(stage,))
            started = time.monotonic()
            entry = {"name": stage, "started_utc": now(), "question_indices": indices, "status": "running"}
            record["stages"].append(entry)
            save()
            outputs = llm.generate([{"prompt_token_ids": inputs[i]} for i in indices],
                                   SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=20261016),
                                   use_tqdm=False)
            entry.update(seconds=time.monotonic()-started, finished_utc=now(), status="completed", requests=[])
            for index, out in zip(indices, outputs):
                answer = out.outputs[0]
                metrics = dataclasses.asdict(out.metrics) if dataclasses.is_dataclass(out.metrics) else repr(out.metrics)
                entry["requests"].append({"question_index": index, "question_sha256": group["questions"][index]["question_sha256"],
                                          "input_tokens": len(out.prompt_token_ids), "generated_tokens": len(answer.token_ids),
                                          "finish_reason": answer.finish_reason, "stop_reason": answer.stop_reason,
                                          "hit_token_cap": len(answer.token_ids) >= args.max_new_tokens,
                                          "output_text": answer.text, "output_token_ids": list(answer.token_ids),
                                          "output_sha256": hashlib.sha256(answer.text.encode()).hexdigest(),
                                          "num_cached_tokens": getattr(out, "num_cached_tokens", None), "metrics": metrics})
            entry["worker_state_after_generation"] = llm.collective_rpc("deployment_state")
            save()
        if args.capture_dir:
            record["rank_capture_completed"] = llm.collective_rpc("finish_rank_capture")
        record["status"] = "completed"
    except BaseException:
        record.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        monitor.stop_event.set()
        monitor.thread.join(timeout=3)
        record.update(finished_utc=now(), sampled_peak_board_memory_bytes=monitor.peaks,
                      monitor_error=monitor.error, monitor_file=str(monitor.path))
        save()
        print(json.dumps({"status": record["status"], "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
