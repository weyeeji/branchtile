"""Read-only real TP-rank Q/K/V instrumentation for the qualified FA backend.

The shared source ends at the last complete APC page within the document. Any
document tail in a partial page stays request-private with its actual values.
This avoids pretending recomputed partial pages have bit-identical K/V.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import time
import torch
from scripts.probe_engine_docfinqa import DeploymentWorkerExtension


class RankCaptureWorkerExtension(DeploymentWorkerExtension):
    def configure_rank_capture(self, config):
        from vllm.distributed import get_tp_group
        rank = get_tp_group().rank_in_group
        self.rank_capture = {"config": config, "rank": rank, "phase": "unset",
                             "requests": {}, "files": [], "sources": {}, "positions": {}, "handles": []}
        state = self.rank_capture
        state["directory"] = Path(config["output"]) / f"rank{rank}"
        state["directory"].mkdir(parents=True, exist_ok=True)
        # vLLM 0.31's GPU runner keeps the active InputBatch as a local, not
        # as the old runner's self.input_batch. Observe its returned object.
        state["original_prepare_inputs"] = self.model_runner.prepare_inputs
        def observe_prepare_inputs(*args, **kwargs):
            batch = state["original_prepare_inputs"](*args, **kwargs)
            state["active_batch"] = batch
            return batch
        self.model_runner.prepare_inputs = observe_prepare_inputs
        for name, parent in self.model_runner.model.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn$", name)
            if not match or int(match[1]) not in config["layers"]:
                continue
            layer = int(match[1])
            assert parent.num_heads == 16 and parent.num_kv_heads == 2 and parent.head_dim == 128
            def before_parent(module, args, kwargs, layer=layer):
                positions = kwargs.get("positions")
                if positions is None:
                    positions = args[0]
                state["positions"][layer] = positions.detach()
            def after_attention(module, args, output, parent=parent, layer=layer):
                self._record_rank_attention(layer, parent, module, args)
            state["handles"].append(parent.register_forward_pre_hook(before_parent, with_kwargs=True))
            state["handles"].append(parent.attn.register_forward_hook(after_attention))
        assert len(state["handles"]) == 2*len(config["layers"])
        return {"rank": rank, "layers": config["layers"], "output": str(state["directory"])}

    def set_rank_capture_phase(self, phase):
        self.rank_capture["phase"] = phase
        return phase

    def _record_rank_attention(self, layer, parent, module, args):
        from vllm.forward_context import get_forward_context
        state = self.rank_capture
        config = state["config"]
        context = get_forward_context()
        metadata = context.attn_metadata
        assert isinstance(metadata, dict), "Capture currently qualifies eager per-layer FA metadata only"
        meta = metadata[module.layer_name]
        assert type(meta).__name__ == "FlashAttentionMetadata"
        starts = meta.query_start_loc.cpu().tolist()
        positions = state["positions"][layer][:meta.num_actual_tokens].cpu()
        q, k, v = args[:3]
        q = q[:meta.num_actual_tokens].view(-1, parent.num_heads, parent.head_dim)
        k = k[:meta.num_actual_tokens].view(-1, parent.num_kv_heads, parent.head_dim)
        v = v[:meta.num_actual_tokens].view(-1, parent.num_kv_heads, parent.head_dim)
        cache = module.kv_cache
        block_size = self.vllm_config.cache_config.block_size
        assert cache.dtype == torch.bfloat16 and cache.ndim == 4
        assert cache.shape[1:] == (parent.num_kv_heads, block_size, 2*parent.head_dim)
        key_cache, value_cache = cache.transpose(1, 2).split(parent.head_dim, dim=-1)
        doc_start, doc_end = config["document_token_range"]
        shared_end = doc_end//block_size*block_size
        batch = state["active_batch"]
        req_state = self.model_runner.req_states
        req_ids = batch.req_ids[:len(starts)-1]
        assert starts[-1] == meta.num_actual_tokens and len(req_ids) == len(starts)-1
        assert starts == batch.query_start_loc_np[:len(starts)].tolist()
        group_anchor = None
        if state["phase"] == "apc_warm_natural_group" and len(req_ids)>1 and all(b-a == 1 for a,b in zip(starts,starts[1:])):
            for i, request_id in enumerate(req_ids):
                info = state["requests"].get(request_id)
                if info is not None and info["question_index"] == 0:
                    step = int(positions[starts[i]])-info["prompt_length"]+2
                    if step in config["steps"]:
                        group_anchor = step
        for request_index, request_id in enumerate(req_ids):
            a, b = starts[request_index:request_index+2]
            if a == b:
                continue
            state_index = int(batch.idx_mapping_np[request_index])
            prompt_len = int(req_state.prompt_len.np[state_index])
            pos = positions[a:b]
            if request_id not in state["requests"]:
                prompt_ids = req_state.all_token_ids.gpu[state_index, :prompt_len].cpu().tolist()
                sha = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
                assert sha in config["prompt_token_hashes"], "Unmapped actual request"
                question_index = config["prompt_token_hashes"].index(sha)
                state["requests"][request_id] = {"request_id": request_id, "question_index": question_index,
                    "prompt_length": prompt_len, "prompt_token_sha256": sha, "phase": state["phase"]}
                self._save_rank_tensor(f"request_{request_id}_prompt.pt", {"prompt_token_ids": torch.tensor(prompt_ids),
                    "metadata": state["requests"][request_id]})
            info = state["requests"][request_id]
            block_table = meta.block_table[request_index]
            observation = (pos >= doc_end-64) & (pos < prompt_len)
            if bool(observation.any()):
                selected = torch.nonzero(observation).flatten().to(q.device) + a
                self._save_rank_tensor(f"layer{layer}_request{request_id}_observation_{int(pos[0])}.pt",
                    {"positions": pos[observation], "q": q[selected].detach().cpu(),
                     "prompt_full_page_ids": block_table[:prompt_len//block_size].cpu(), "metadata": info})
            wanted = [(step, None) for step in config["steps"] if bool((pos == prompt_len+step-2).any())]
            if group_anchor is not None:
                wanted.append((int(pos[0])-prompt_len+2, group_anchor))
            if not wanted:
                continue
            def gather(begin, end):
                ps = torch.arange(begin, end, device=cache.device)
                pages = block_table[ps//block_size].long()
                offset = ps%block_size
                return (key_cache[pages, offset].permute(1, 0, 2).contiguous().cpu(),
                        value_cache[pages, offset].permute(1, 0, 2).contiguous().cpu())
            source_pages = block_table[:shared_end//block_size].cpu()
            if layer not in state["sources"]:
                assert state["phase"] == "cold_singleton"
                source_k, source_v = gather(0, shared_end)
                state["sources"][layer] = source_pages.clone()
                self._save_rank_tensor(f"layer{layer}_shared_source.pt", {"k": source_k, "v": source_v,
                    "physical_pages": source_pages, "document_token_range": [doc_start, doc_end],
                    "shared_prefix_end": shared_end, "request_private_document_tail_tokens": doc_end-shared_end,
                    "rank": state["rank"], "kv_heads_per_rank": parent.num_kv_heads,
                    "query_heads_per_rank": parent.num_heads, "rope_applied": True})
            assert torch.equal(source_pages, state["sources"][layer]), "Shared full APC pages changed; do not invent common payload"
            for step, anchor in wanted:
                target = prompt_len+step-2
                local = int(torch.nonzero(pos == target).flatten()[0])
                row = a+local
                latest_k, latest_v = gather(target, target+1)
                assert torch.equal(latest_k[:, 0], k[row].detach().cpu()), "Actual cache K differs from hooked post-RoPE K"
                assert torch.equal(latest_v[:, 0], v[row].detach().cpu()), "Actual cache V differs from hooked V"
                private_k, private_v = gather(shared_end, target+1)
                name = (f"layer{layer}_request{request_id}_step{step}.pt" if anchor is None else
                        f"layer{layer}_ready_anchor{anchor}_request{request_id}.pt")
                self._save_rank_tensor(name,
                    {"q": q[row].detach().view(parent.num_kv_heads, -1, parent.head_dim).cpu(),
                     "private_k": private_k, "private_v": private_v,
                     "prompt_full_page_ids": block_table[:prompt_len//block_size].cpu(),
                     "metadata": dict(info, layer=layer, rank=state["rank"], prediction_step=step,
                         query_token_position=target, generated_inputs_consumed=step-1,
                         shared_prefix_end=shared_end, document_token_range=[doc_start, doc_end],
                         private_document_tail_tokens=doc_end-shared_end,
                         physical_full_prefix_pages_match=True, hooked_kv_matches_actual_cache=True,
                         actual_scheduled_request_ids=req_ids, query_start_loc=starts,
                         ready_group_anchor_step=anchor,
                         same_event_query_positions=[positions[starts[i]:starts[i+1]].tolist() for i in range(len(req_ids))],
                         sequence_lengths=meta.seq_lens.cpu().tolist(),
                         slot_mapping=meta.slot_mapping[a:b].cpu().tolist(), monotonic=time.monotonic())})

    def _save_rank_tensor(self, name, obj):
        state = self.rank_capture
        path = state["directory"] / name
        assert not path.exists(), "Do not overwrite a prior activation"
        torch.save(obj, path)
        state["files"].append({"file": name, "bytes": path.stat().st_size})

    def finish_rank_capture(self):
        state = self.rank_capture
        for handle in state["handles"]:
            handle.remove()
        self.model_runner.prepare_inputs = state["original_prepare_inputs"]
        for item in state["files"]:
            digest = hashlib.sha256()
            with (state["directory"] / item["file"]).open("rb") as stream:
                while chunk := stream.read(8*1024*1024):
                    digest.update(chunk)
            item["sha256"] = digest.hexdigest()
        record = {"rank": state["rank"], "config": state["config"], "requests": state["requests"],
                  "files": state["files"], "status": "completed",
                  "scope": "Read-only hooks on actual TP2 post-RoPE Q/K/V, cache-write identity checked; instrumentation costs invalidate speed claims",
                  "partial_page_policy": "Document tail in incomplete shared APC page is retained with request-private K/V, never silently replaced by another request's values",
                  "observation_policy": "Raw actually-computed prompt-tail Q is saved; any reuse for APC-skipped observation positions requires separate exact token-prefix and physical-page provenance validation"}
        path = state["directory"] / "metadata.json"
        path.write_text(json.dumps(record,indent=2)+'\n')
        return {"rank": state["rank"], "metadata": str(path), "files": len(state["files"])}
