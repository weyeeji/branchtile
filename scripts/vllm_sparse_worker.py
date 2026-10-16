"""Instance-scoped vLLM sparse attention bridge; original cache updates stay intact.

Pure decode bypasses original FA. Mixed prefill/decode keeps the dense prefill
call and replaces decode rows, explicitly recording the redundant work.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from scripts.probe_engine_docfinqa import DeploymentWorkerExtension
from src.vllm_sparse_executor import PagedSparseExecutor, physical_supports


class SparseAttentionWorkerExtension(DeploymentWorkerExtension):
    def configure_sparse_attention(self, config):
        from vllm.distributed import get_tp_group
        state = self.sparse_state = {'config': config, 'rank': get_tp_group().rank_in_group,
            'phase': 'unset', 'method': 'dense', 'requests': {}, 'records': [], 'logits': {},
            'positions': {}, 'handles': [], 'originals': [], 'executors': {}, 'metadata_seconds': 0.0}
        state['output'] = Path(config['output']) / f"rank{state['rank']}"
        state['output'].mkdir(parents=True, exist_ok=True)
        old_prepare = self.model_runner.prepare_inputs
        def prepare(*args, **kwargs):
            batch = old_prepare(*args, **kwargs)
            state['batch'] = batch
            state['table_cpu'] = None
            return batch
        self.model_runner.prepare_inputs = prepare
        state['old_prepare'] = old_prepare
        for name, parent in self.model_runner.model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.self_attn$', name)
            if not match:
                continue
            layer = int(match[1])
            assert parent.num_heads == 16 and parent.num_kv_heads == 2 and parent.head_dim == 128
            assert not parent.attn.attn_backend.forward_includes_kv_cache_update
            def before(module, args, kwargs, layer=layer):
                state['positions'][layer] = kwargs.get('positions', args[0] if args else None)
            state['handles'].append(parent.register_forward_pre_hook(before, with_kwargs=True))
            original = parent.attn.impl.forward
            def forward(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale=None, output_block_scale=None, layer=layer, original=original):
                return self._sparse_forward(layer, original, attn_layer, query, key, value,
                    kv_cache, attn_metadata, output, output_scale, output_block_scale)
            state['originals'].append((parent.attn.impl, original))
            parent.attn.impl.forward = forward
        assert len(state['originals']) == 48
        assert self.model_runner.batch_sharder is None, 'Qualification needs unsharded logits with explicit request mapping'
        model = self.model_runner.model
        original_logits = model.compute_logits
        def logits(*args, **kwargs):
            actual = original_logits(*args, **kwargs)
            if state['rank'] == 0 and config.get('capture_logits'):
                batch = state['batch']
                indexes = batch.logits_indices.cpu().tolist()
                starts = batch.query_start_loc_np[:batch.num_reqs + 1].tolist()
                positions = batch.positions.cpu()
                assert len(indexes) == actual.shape[0]
                for row, token_index in enumerate(indexes):
                    request_index = next(i for i in range(batch.num_reqs) if starts[i] <= token_index < starts[i + 1])
                    info = self._sparse_request_info(request_index)
                    step = int(positions[token_index]) - info['prompt_length'] + 2
                    if step >= 1:
                        key = f"q{info['question_index']}_s{step}"
                        state['logits'][key] = actual[row].detach().float().cpu()
            return actual
        model.compute_logits = logits
        state['old_logits'] = original_logits
        return {'rank': state['rank'], 'wrapped_layers': len(state['originals']), 'cache_update_preserved': True}

    def _sparse_request_info(self, request_index):
        state, batch = self.sparse_state, self.sparse_state['batch']
        request_id = batch.req_ids[request_index]
        if request_id not in state['requests']:
            slot = int(batch.idx_mapping_np[request_index])
            prompt_length = int(self.model_runner.req_states.prompt_len.np[slot])
            ids = self.model_runner.req_states.all_token_ids.gpu[slot, :prompt_length].cpu().tolist()
            digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            assert digest in state['config']['prompt_token_hashes'], 'Unmapped actual request'
            state['requests'][request_id] = {'question_index': state['config']['prompt_token_hashes'].index(digest),
                                             'prompt_length': prompt_length, 'prompt_token_sha256': digest}
        return state['requests'][request_id]

    def _sparse_forward(self, layer, original, attn_layer, query, key, value, cache, meta,
                        output, output_scale, output_block_scale):
        state = self.sparse_state
        def dense(destination):
            return original(attn_layer, query, key, value, cache, meta, output=destination,
                            output_scale=output_scale, output_block_scale=output_block_scale)
        if meta is None or state['method'] == 'dense':
            return dense(output)
        assert type(meta).__name__ == 'FlashAttentionMetadata'
        assert output_scale is None and output_block_scale is None
        started = time.perf_counter()
        starts = meta.query_start_loc.cpu().tolist()
        batch = state['batch']
        assert starts == batch.query_start_loc_np[:len(starts)].tolist()
        positions = state['positions'][layer][:meta.num_actual_tokens].cpu()
        chosen = []
        logical = []
        for request_index, (a, b) in enumerate(zip(starts, starts[1:])):
            info = self._sparse_request_info(request_index)
            if b-a == 1 and int(positions[a]) >= info['prompt_length']:
                chosen.append((request_index, a))
                # Full-budget milestone. Sparse masks will replace only document positions.
                assert state['config'].get('budget') in (None, 'full')
                logical.append([torch.arange(int(positions[a]) + 1)] * 2)
        if not chosen:
            return dense(output)
        pure = len(chosen) == len(starts)-1
        dense_reference = None
        if not pure:
            dense(output)
            if state['config'].get('check_dense'):
                dense_reference = output[[row for _, row in chosen]].clone()
        elif state['config'].get('check_dense'):
            original_output = torch.empty_like(output)
            dense(original_output)
            dense_reference = original_output[:len(chosen)]
        if state['table_cpu'] is None:
            state['table_cpu'] = meta.block_table.cpu()
        tables = state['table_cpu'][[request for request, _ in chosen]]
        ids = physical_supports(tables, logical, 2, cache.shape[2], cache.stride())
        method = state['method']
        if method not in state['executors']:
            state['executors'][method] = PagedSparseExecutor(query.device, method)
        executor = state['executors'][method]
        rows = [row for _, row in chosen]
        actual_q = query[rows].contiguous()
        executor.plan(actual_q, cache, ids)
        prepared = time.perf_counter()
        actual = executor.run(actual_q)
        output[rows] = actual
        record = {'phase': state['phase'], 'layer': layer, 'method': method, 'batch': len(chosen),
                  'pure_decode_candidate_path': pure,
                  'pure_decode_bypassed_dense': pure and not state['config'].get('check_dense'),
                  'extra_dense_for_numerical_reference': pure and bool(state['config'].get('check_dense')),
                  'mixed_redundant_dense': not pure,
                  'prepare_host_seconds': prepared-started,
                  'complete_bridge_host_seconds': time.perf_counter()-started,
                  'query_positions': [int(positions[row]) for row in rows],
                  'question_indices': [self._sparse_request_info(request)['question_index'] for request, _ in chosen],
                  'cache_shape': list(cache.shape), 'cache_stride': list(cache.stride())}
        if dense_reference is not None:
            difference = (actual.float()-dense_reference.float()).abs()
            record['max_abs_output_vs_dense'] = float(difference.max())
            record['dense_output_tolerance_passed'] = bool(torch.isclose(actual.float(), dense_reference.float(), atol=.006, rtol=.015).all())
        state['records'].append(record)
        return output

    def set_sparse_phase(self, phase, method):
        assert method in ('dense', 'e0', 'union')
        state = self.sparse_state
        state['phase'], state['method'] = phase, method
        state['records'], state['logits'] = [], {}
        return {'rank': state['rank'], 'phase': phase, 'method': method}

    def finish_sparse_phase(self):
        state = self.sparse_state
        torch.cuda.synchronize()
        path = state['output'] / f"{state['phase']}.json"
        record = {'rank': state['rank'], 'phase': state['phase'], 'method': state['method'],
                  'records': state['records'], 'logits_count': len(state['logits']),
                  'scope': 'Instrumented full-budget output/logit qualification; timings include Python metadata and optional redundant dense checks, not application speed evidence.'}
        if state['logits']:
            logits_path = path.with_suffix('.pt')
            torch.save(state['logits'], logits_path)
            record['logits_file'] = str(logits_path)
        path.write_text(json.dumps(record, indent=2)+'\n')
        return {'rank': state['rank'], 'metadata': str(path), 'records': len(state['records']), 'logits_count': len(state['logits'])}
