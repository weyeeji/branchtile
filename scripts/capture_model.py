#!/usr/bin/env python3
"""Capture actual post-RoPE Q/K/V and QC-SnapKV masks from complete documents.

This is a Transformers capture instrument, not the proposed optimized serving
engine. Natural generation uses dense attention, and captured masks are frozen
after question prefill; no sparse-quality or application-speed claim follows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]


def synchronize_devices():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def sha_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def tensor_sha(x: torch.Tensor) -> str:
    return sha_bytes(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def projected_qkv(module, hidden_states, position_embeddings):
    shape = (*hidden_states.shape[:-1], -1, module.head_dim)
    q = module.q_norm(module.q_proj(hidden_states).view(shape)).transpose(1, 2)
    k = module.k_norm(module.k_proj(hidden_states).view(shape)).transpose(1, 2)
    v = module.v_proj(hidden_states).view(shape).transpose(1, 2)
    cos, sin = (x.unsqueeze(1) for x in position_embeddings)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin, v


@torch.inference_mode()
def qc_scores(q: torch.Tensor, k: torch.Tensor, q_positions: torch.Tensor,
              doc_start: int, doc_end: int, chunk_size: int = 4096) -> torch.Tensor:
    """Full causal denominator in two streaming passes, no N x N attention."""
    assert q.shape[0] == k.shape[0] == 1
    kvh, length, dim = k.shape[1:]
    group = q.shape[1] // kvh
    q = q[0].float().reshape(kvh, group * q.shape[2], dim)
    positions = q_positions.repeat(group)
    lse = torch.full(q.shape[:2], -torch.inf, device=q.device)
    scale = dim ** -0.5
    for begin in range(0, length, chunk_size):
        stop = min(length, begin + chunk_size)
        logits = torch.bmm(q, k[0, :, begin:stop].float().transpose(-1, -2)) * scale
        causal = torch.arange(begin, stop, device=q.device)[None, None, :] <= positions[None, :, None]
        logits.masked_fill_(~causal, -torch.inf)
        lse = torch.logaddexp(lse, logits.logsumexp(-1))
    scores = torch.empty((kvh, doc_end - doc_start), device=q.device)
    for begin in range(doc_start, doc_end, chunk_size):
        stop = min(doc_end, begin + chunk_size)
        logits = torch.bmm(q, k[0, :, begin:stop].float().transpose(-1, -2)) * scale
        logits.masked_fill_(torch.arange(begin, stop, device=q.device)[None, None, :] > positions[None, :, None], -torch.inf)
        scores[:, begin - doc_start:stop - doc_start] = torch.exp(logits - lse[:, :, None]).mean(1)
    return F.avg_pool1d(scores[:, None], kernel_size=7, stride=1, padding=3,
                        count_include_pad=True)[:, 0]


def select_support(scores: torch.Tensor, budget: int) -> torch.Tensor:
    heads, n = scores.shape
    budget = min(n, budget)
    support = torch.zeros_like(scores, dtype=torch.bool)
    support[:, :min(4, n)] = True
    support[:, max(0, n - 32):] = True
    protected = int(support[0].sum())
    if budget < protected:
        raise ValueError("Budget cannot be less than protected first4/last32 union")
    ranked_scores = scores.clone()
    ranked_scores[support] = -torch.inf
    # Token columns are already in increasing original-position order.
    ranked = ranked_scores.argsort(dim=-1, descending=True, stable=True)
    support.scatter_(1, ranked[:, :budget - protected], True)
    assert (support.sum(-1) == budget).all()
    return support


def set_partitions(b: int):
    parts = [[]]
    for bit in (1 << i for i in range(b)):
        nxt = []
        for partition in parts:
            nxt.append(partition + [bit])
            for i in range(len(partition)):
                p = partition.copy()
                p[i] |= bit
                nxt.append(p)
        parts = nxt
    return parts


def opportunity(supports: torch.Tensor, gqa: int, tile_n: int = 64) -> list[dict]:
    # supports B,H,N; every cost is a logical tile opportunity, never runtime.
    b, h, n = supports.shape
    owners = sum(supports[i].to(torch.int16) * (1 << i) for i in range(b))
    rows = []
    for hi in range(h):
        lengths = torch.bincount(owners[hi].long(), minlength=1 << b).tolist()
        compact = sum(math.ceil(lengths[o] / tile_n) * math.ceil(o.bit_count() * gqa / 16)
                      for o in range(1, 1 << b))
        fixed_union = math.ceil(sum(lengths[1:]) / tile_n) * math.ceil(b * gqa / 16)
        partitions = []
        for partition in set_partitions(b):
            lens = [sum(lengths[o] for o in range(1, 1 << b) if o & group) for group in partition]
            cost = sum(math.ceil(l / tile_n) * math.ceil(group.bit_count() * gqa / 16)
                       for l, group in zip(lens, partition))
            partitions.append((cost, sum(lens), partition))
        best = min(partitions)
        rows.append({"kv_head": hi, "owner_token_counts": lengths, "document_tokens": n,
                     "union_tokens": sum(lengths[1:]), "logical_compact_m16_n64_tiles": compact,
                     "logical_fixed_union_m16_n64_tiles": fixed_union,
                     "logical_best_fixed_partition_m16_n64_tiles": best[0],
                     "best_fixed_partition": best[2], "best_fixed_kv_token_reads": best[1],
                     "compact_vs_best_fixed_tile_ratio": compact / best[0] if best[0] else None,
                     "warning": "optimistic tile-count model; does not measure runtime, cache, split, or hierarchy"})
    return rows


def make_inputs(tokenizer, group):
    text = (ROOT / group["document_path"]).read_text()
    assert sha_bytes(text.encode()) == group["document_sha256"]
    if group["dataset"] == "DocFinQA":
        instruction = "Answer the question using the annual report. Reason carefully. End with FINAL: followed by the numerical answer and its unit."
    else:
        instruction = "Answer the question using the meeting transcript. Give a concise factual summary that addresses the question."
    all_ids, doc_ranges = [], []
    for question in group["questions"]:
        content = "Document:\n" + text + "\n\nQuestion:\n" + question["question"]
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": instruction}, {"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True)
        begin = rendered.index("Document:\n") + len("Document:\n")
        end = begin + len(text)
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        doc_indices = [i for i, (s, e) in enumerate(encoded["offset_mapping"]) if e > begin and s < end]
        assert doc_indices and doc_indices == list(range(doc_indices[0], doc_indices[-1] + 1))
        all_ids.append(encoded["input_ids"])
        doc_ranges.append((doc_indices[0], doc_indices[-1] + 1))
    assert len(set(doc_ranges)) == 1, "Different document token boundaries across natural questions"
    start, end = doc_ranges[0]
    assert all(ids[:end] == all_ids[0][:end] for ids in all_ids), "Token prefix identity violated"
    return all_ids, start, end


class Capture:
    def __init__(self, args, model, out, doc_start, doc_end):
        self.args, self.model, self.out = args, model, out
        self.doc_start, self.doc_end = doc_start, doc_end
        self.phase, self.qindex, self.step = "prefix", 0, 0
        self.layers = [int(x) for x in args.layers.split(",")] if args.layers else [0, (model.config.num_hidden_layers - 1) // 3, 2 * (model.config.num_hidden_layers - 1) // 3, model.config.num_hidden_layers - 1]
        self.budgets = [int(x) for x in args.budgets.split(",")]
        self.masks, self.score_times = {}, []
        self.snapshots = []
        self.handles = [model.model.layers[i].self_attn.register_forward_pre_hook(self.hook, with_kwargs=True) for i in self.layers]

    def hook(self, module, args, kwargs):
        if self.phase == "prefix" or (self.phase == "decode" and self.step not in self.args.capture_steps):
            return
        hs = kwargs.get("hidden_states", args[0] if args else None)
        q, k, v = projected_qkv(module, hs, kwargs["position_embeddings"])
        cache = kwargs["past_key_values"]
        layer = cache.layers[module.layer_idx]
        past_len = layer.keys.shape[-2]
        all_k = torch.cat([layer.keys, k], dim=-2)
        if self.phase == "suffix":
            window = min(32, q.shape[-2])
            q_obs = q[:, :, -window:]
            positions = torch.arange(past_len + q.shape[-2] - window, past_len + q.shape[-2], device=q.device)
            synchronize_devices()
            t = time.monotonic()
            scores = qc_scores(q_obs, all_k, positions, self.doc_start, self.doc_end, self.args.score_chunk)
            masks = {budget: select_support(scores, budget).cpu() for budget in self.budgets}
            synchronize_devices()
            self.score_times.append({"question_index": self.qindex, "layer": module.layer_idx, "seconds": time.monotonic() - t})
            self.masks[(self.qindex, module.layer_idx)] = masks
            torch.save({"observation_q": q_obs.cpu(), "observation_positions": positions.cpu(),
                        "pooled_scores": scores.cpu(), "supports": masks,
                        "observation_is_question_suffix_only": True},
                       self.out / f"selector_q{self.qindex}_l{module.layer_idx}.pt")
        else:
            suffix_k = all_k[:, :, self.doc_end:].cpu()
            suffix_v = torch.cat([layer.values[:, :, self.doc_end:], v], dim=-2).cpu()
            path = self.out / f"decode_q{self.qindex}_l{module.layer_idx}_s{self.step}.pt"
            torch.save({"query": q.cpu(), "query_positions": torch.arange(past_len, past_len + q.shape[-2]),
                        "private_suffix_k": suffix_k, "private_suffix_v": suffix_v,
                        "step": self.step, "question_index": self.qindex, "layer": module.layer_idx}, path)
            self.snapshots.append(str(path.relative_to(ROOT)))


@torch.inference_mode()
def run(args):
    assert os.geteuid() != 0, "Running capture as root is forbidden"
    assert sys.prefix != sys.base_prefix, "Run capture using the experiment virtual environment"
    start_epoch = time.time()
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    manifest = json.loads(Path(args.manifest).read_text())
    group = manifest["groups"][args.group_index]
    out = Path(args.output or f'results/captures/{Path(args.model).name}/{group["document_id"]}')
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    inputs, doc_start, doc_end = make_inputs(tokenizer, group)
    metadata = {"schema": 1, "status": "running", "document_id": group["document_id"],
                "dataset": group["dataset"], "manifest": args.manifest, "manifest_sha256": sha_bytes(Path(args.manifest).read_bytes()),
                "historical_split_restored": False, "group_index": args.group_index,
                "document_sha256": group["document_sha256"], "input_ids_sha256": [tensor_sha(torch.tensor(ids, dtype=torch.int64)) for ids in inputs],
                "shared_prefix_ids_sha256": tensor_sha(torch.tensor(inputs[0][:doc_end], dtype=torch.int64)),
                "document_token_range": [doc_start, doc_end], "input_lengths": [len(x) for x in inputs],
                "model_path": args.model, "model_config_sha256": sha_bytes((Path(args.model) / "config.json").read_bytes()),
                "tokenizer_config_sha256": sha_bytes((Path(args.model) / "tokenizer_config.json").read_bytes()),
                "model_revision": args.model_revision, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "process_id": os.getpid(), "started_epoch_seconds": start_epoch,
                "virtual_environment": sys.prefix, "effective_user_id": os.geteuid(),
                "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                "gpu_uuids": [str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())],
                "allocated_gpu_count": torch.cuda.device_count(), "requested_device_map": args.device_map,
                "deployment": "single GPU" if args.device_map == "cuda" else "Transformers layer sharding, not tensor parallelism",
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "dtype": "bfloat16", "attention_implementation": "sdpa",
                "scope": "dense natural-generation capture only, not sparse quality or application benchmark",
                "scheduling": "questions captured serially using a shared prefix cache; actual joint ready overlap was not measured",
                "full_document": True, "B": len(inputs), "seed": args.seed,
                "generation": {"sampling": "greedy", "max_new_tokens": args.max_new_tokens, "natural_eos": True, "min_tokens": None},
                "selector": {"name": "QC-SnapKV variant", "window": 32, "pool": 7, "pool_count_include_pad": True,
                             "protected": "document first4/last32; counted in budget", "tie_break": "original position ascending",
                             "score_precision": "float32 matmul from BF16 post-RoPE Q,K", "causal_denominator": "entire visible prefix and suffix"},
                "questions": [{"question_sha256": q["question_sha256"], "question": q["question"]} for q in group["questions"]]}
    dump_json(out / "metadata.json", metadata)
    if max(map(len, inputs)) + args.max_new_tokens > args.max_context:
        metadata["status"] = "unservable_context_limit_no_truncation"
        metadata["ended_epoch_seconds"] = time.time()
        dump_json(out / "metadata.json", metadata)
        print(json.dumps({"status": metadata["status"], "input_lengths": metadata["input_lengths"]}), flush=True)
        return
    t0 = time.monotonic()
    load_kwargs = {}
    if args.device_map != "cuda":
        load_kwargs["max_memory"] = {i: args.max_memory_per_gpu for i in range(torch.cuda.device_count())}
        load_kwargs["max_memory"]["cpu"] = 0
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device_map,
                                               attn_implementation="sdpa", local_files_only=True, **load_kwargs).eval()
    assert not any(str(v) in ("cpu", "disk") for v in model.hf_device_map.values()), "CPU/disk parameter offload is outside this capture deployment"
    synchronize_devices()
    input_device = model.get_input_embeddings().weight.device
    metadata["actual_device_map"] = {k: str(v) for k, v in model.hf_device_map.items()}
    metadata["model_load_seconds"] = time.monotonic() - t0
    metadata["gqa"] = model.config.num_attention_heads // model.config.num_key_value_heads
    metadata["kv_heads"] = model.config.num_key_value_heads
    capture = Capture(args, model, out, doc_start, doc_end)
    cache = DynamicCache(config=model.config)
    prefix = torch.tensor(inputs[0][:doc_end], dtype=torch.long, device=input_device)[None]
    t0 = time.monotonic()
    for start in range(0, prefix.shape[1], args.prefill_chunk):
        model.model(input_ids=prefix[:, start:start + args.prefill_chunk], past_key_values=cache, use_cache=True)
        if start % (args.prefill_chunk * 8) == 0:
            print(json.dumps({"phase": "prefill", "tokens": min(start + args.prefill_chunk, doc_end), "total": doc_end}), flush=True)
    synchronize_devices()
    metadata["prefix_prefill_seconds"] = time.monotonic() - t0
    for layer in capture.layers:
        torch.save({"document_k": cache.layers[layer].keys[:, :, doc_start:doc_end].cpu(),
                    "document_v": cache.layers[layer].values[:, :, doc_start:doc_end].cpu(),
                    "prefix_k": cache.layers[layer].keys[:, :, :doc_start].cpu(),
                    "prefix_v": cache.layers[layer].values[:, :, :doc_start].cpu(),
                    "original_positions": torch.arange(doc_start, doc_end)}, out / f"source_l{layer}.pt")
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    for qi, ids in enumerate(inputs):
        capture.qindex, capture.phase, capture.step = qi, "suffix", 0
        cache.crop(doc_end)
        t0 = time.monotonic()
        result = model(input_ids=torch.tensor(ids[doc_end:], device=input_device)[None], past_key_values=cache, use_cache=True, logits_to_keep=1)
        generated = []
        token = int(result.logits[0, -1].argmax())
        capture.phase = "decode"
        natural_eos = False
        for step in range(1, args.max_new_tokens + 1):
            generated.append(token)
            if token in eos:
                natural_eos = True
                break
            capture.step = step
            result = model(input_ids=torch.tensor([[token]], device=input_device), past_key_values=cache, use_cache=True, logits_to_keep=1)
            token = int(result.logits[0, -1].argmax())
        synchronize_devices()
        metadata["questions"][qi].update({"generated_tokens": len(generated), "natural_eos": natural_eos,
            "hit_instrumentation_cap": not natural_eos, "generated_text": tokenizer.decode(generated),
            "generated_ids_sha256": tensor_sha(torch.tensor(generated, dtype=torch.int64)),
            "instrumented_suffix_and_decode_seconds": time.monotonic() - t0})
        print(json.dumps({"phase": "question_done", "q": qi, "generated": len(generated), "natural_eos": natural_eos}), flush=True)
        dump_json(out / "metadata.json", metadata)
    rows = []
    for layer in capture.layers:
        for budget in capture.budgets:
            supports = torch.stack([capture.masks[(qi, layer)][budget] for qi in range(len(inputs))])
            torch.save({"supports": supports, "budget": budget, "layer": layer, "gqa": metadata["gqa"]}, out / f"masks_l{layer}_b{budget}.pt")
            rows.extend({"layer": layer, "budget": budget, **r} for r in opportunity(supports, metadata["gqa"]))
    dump_json(out / "opportunity.json", rows)
    metadata.update({"status": "completed", "layers": capture.layers, "score_times": capture.score_times,
                     "decode_snapshots": capture.snapshots, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                     "ended_epoch_seconds": time.time(),
                     "memory_per_gpu": [{"device": i, "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
                                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(i)} for i in range(torch.cuda.device_count())],
                     "generation_limit_warning": "A capped exploratory trajectory does not establish full natural output length or common-ready lifetime."})
    dump_json(out / "metadata.json", metadata)
    print(json.dumps({"status": "completed", "output": str(out), "snapshots": len(capture.snapshots)}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/Qwen3-4B-Thinking-2507")
    p.add_argument("--model-revision", default="768f209d9ea81521153ed38c47d515654e938aea")
    p.add_argument("--device-map", default="cuda", choices=["cuda", "auto", "balanced", "sequential"])
    p.add_argument("--max-memory-per-gpu", default="22GiB")
    p.add_argument("--manifest", default="data/manifests/qmsum_train_exploratory8.json")
    p.add_argument("--group-index", type=int, default=0)
    p.add_argument("--output")
    p.add_argument("--layers")
    p.add_argument("--budgets", default="4096,8192,16384,32768")
    p.add_argument("--prefill-chunk", type=int, default=1024)
    p.add_argument("--score-chunk", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-context", type=int, default=131072)
    p.add_argument("--capture-steps", type=int, nargs="+", default=[1, 16, 64, 256])
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    try:
        run(args)
    except Exception as exc:
        group = json.loads(Path(args.manifest).read_text())["groups"][args.group_index]
        out = Path(args.output or f'results/captures/{Path(args.model).name}/{group["document_id"]}')
        path = out / "metadata.json"
        metadata = json.loads(path.read_text()) if path.exists() else {}
        metadata.update({"status": "failed", "exception_type": type(exc).__name__, "exception": str(exc),
                         "ended_epoch_seconds": time.time(), "allocated_gpu_count": torch.cuda.device_count()})
        dump_json(path, metadata)
        raise


if __name__ == "__main__":
    main()
