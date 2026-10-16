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
from src.vllm_sparse_selector import FrozenQCSelector


class SparseAttentionWorkerExtension(DeploymentWorkerExtension):
    def configure_sparse_attention(self, config):
        from vllm.distributed import get_tp_group
        state = self.sparse_state = {'config': config, 'rank': get_tp_group().rank_in_group,
            'phase': 'unset', 'method': 'dense', 'requests': {}, 'records': [], 'logits': {},
            'positions': {}, 'layer_outputs': {}, 'forward_trace': [], 'handles': [], 'originals': [], 'executors': {}, 'metadata_seconds': 0.0}
        state['selector'] = FrozenQCSelector(config['prompt_tokens'], config['document_token_range'], config['budget']) if isinstance(config.get('budget'), int) else None
        state['output'] = Path(config['output']) / f"rank{state['rank']}"
        state['output'].mkdir(parents=True, exist_ok=True)
        old_prepare = self.model_runner.prepare_inputs
        def prepare(*args, **kwargs):
            batch = old_prepare(*args, **kwargs)
            state['batch'] = batch
            state['table_cpu'] = None
            state['table_source_ptr'] = None
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
            if actual is None:
                return actual
            forced = state.get('teacher_tokens')
            if (state['rank'] == 0 and config.get('capture_logits')) or forced:
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
                        if state['rank'] == 0 and config.get('capture_logits'):
                            state['logits'][key] = actual[row].detach().float().cpu()
                        if forced:
                            tokens = forced[info['question_index']]
                            assert step <= len(tokens), 'Teacher trajectory exhausted'
                            actual[row].fill_(-torch.inf)
                            actual[row, tokens[step-1]] = 0
            return actual
        model.compute_logits = logits
        state['old_logits'] = original_logits
        return {'rank': state['rank'], 'wrapped_layers': len(state['originals']), 'cache_update_preserved': True, 'worker_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

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

    def _capture_sparse_layer_output(self, layer, meta, output):
        state = self.sparse_state
        batch = state.get('batch')
        if not state['config'].get('capture_layer_outputs') or meta is None:
            return
        starts = meta.query_start_loc.cpu().tolist()
        positions = state['positions'][layer][:meta.num_actual_tokens].cpu()
        if layer == 0:
            trace = {'phase':state['phase'],'query_start_loc':starts,'seq_lens':meta.seq_lens.cpu().tolist(),
                'num_actual_tokens':meta.num_actual_tokens,'num_computed_tokens_before_forward':batch.num_computed_tokens_np[:batch.num_reqs].tolist(),
                'requests':[]}
            for request,(a,b) in enumerate(zip(starts,starts[1:])):
                info = self._sparse_request_info(request)
                pos = positions[a:b].tolist()
                trace['requests'].append({'request_id':batch.req_ids[request],'slot':int(batch.idx_mapping_np[request]),
                    **info,'query_tokens':b-a,'post_rope_Q_positions':pos if len(pos)<=64 else {'first':pos[0],'last':pos[-1],'contiguous':pos==list(range(pos[0],pos[-1]+1))},
                    'phase_kind':'single_token_decode' if b-a==1 and pos[0]>=info['prompt_length'] else 'prefill'})
            state['forward_trace'].append(trace)
        for request, (a, b) in enumerate(zip(starts, starts[1:])):
            info = self._sparse_request_info(request)
            if b > a and int(positions[b-1]) >= info['prompt_length']-1:
                step = int(positions[b-1]) - info['prompt_length'] + 2
                key = f'l{layer}_q{info["question_index"]}_s{step}'
                state['layer_outputs'][key] = output[b-1].detach().float().cpu()

    def _sparse_forward(self, layer, original, attn_layer, query, key, value, cache, meta,
                        output, output_scale, output_block_scale):
        state = self.sparse_state
        def dense(destination):
            return original(attn_layer, query, key, value, cache, meta, output=destination,
                            output_scale=output_scale, output_block_scale=output_block_scale)
        if meta is None:
            result = dense(output)
            self._capture_sparse_layer_output(layer, meta, output)
            return result
        assert type(meta).__name__ == 'FlashAttentionMetadata'
        assert output_scale is None and output_block_scale is None
        assert abs(attn_layer.impl.scale - 128**-.5) < 1e-12
        assert getattr(attn_layer.impl, 'alibi_slopes', None) is None
        assert getattr(attn_layer.impl, 'logits_soft_cap', 0) in (None, 0)
        assert getattr(attn_layer.impl, 'sliding_window', (-1, -1)) in (None, (-1, -1))
        started = time.perf_counter()
        starts = meta.query_start_loc.cpu().tolist()
        batch = state['batch']
        assert starts == batch.query_start_loc_np[:len(starts)].tolist()
        positions = state['positions'][layer][:meta.num_actual_tokens].cpu()
        if state['table_cpu'] is None:
            state['table_cpu'] = meta.block_table.cpu()
            state['table_source_ptr'] = meta.block_table.data_ptr()
        assert state['table_source_ptr'] == meta.block_table.data_ptr(), 'Different per-layer KV groups need separate physical tables'
        if state['selector'] is not None:
            for request, (a, b) in enumerate(zip(starts, starts[1:])):
                info = self._sparse_request_info(request)
                if b > a and int(positions[a]) < info['prompt_length']:
                    state['selector'].observe(layer, info['question_index'], positions[a:b], query[a:b], state['table_cpu'][request], cache.shape[2])
                    if int(positions[b-1]) == info['prompt_length']-1:
                        state['selector'].freeze(layer, info['question_index'], cache, state['table_cpu'][request])
        if state['method'] == 'dense':
            result = dense(output)
            self._capture_sparse_layer_output(layer, meta, output)
            return result
        chosen = []
        logical = []
        for request_index, (a, b) in enumerate(zip(starts, starts[1:])):
            info = self._sparse_request_info(request_index)
            if b-a == 1 and int(positions[a]) >= info['prompt_length']:
                chosen.append((request_index, a))
                assert int(positions[a]) + 1 == int(meta.seq_lens[request_index])
                logical.append([torch.arange(int(positions[a]) + 1)] * 2 if state['selector'] is None else state['selector'].logical_supports(layer, info['question_index'], int(positions[a])))
        if not chosen:
            result = dense(output)
            self._capture_sparse_layer_output(layer, meta, output)
            return result
        pure = len(chosen) == len(starts)-1
        dense_reference = None
        if not pure:
            dense(output)
            if state['config'].get('check_dense'):
                dense_reference = output[[row for _, row in chosen]].clone()
        elif state['config'].get('check_dense'):
            original_output = torch.empty_like(output)
            dense(original_output)
            dense_reference = original_output[[row for _, row in chosen]]
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
                  'request_ids': [batch.req_ids[request] for request, _ in chosen],
                  'request_slots': [int(batch.idx_mapping_np[request]) for request, _ in chosen],
                  'prompt_hashes': [self._sparse_request_info(request)['prompt_token_sha256'] for request, _ in chosen],
                  'cache_shape': list(cache.shape), 'cache_stride': list(cache.stride())}
        if dense_reference is not None:
            difference = (actual.float()-dense_reference.float()).abs()
            record['max_abs_output_vs_dense'] = float(difference.max())
            record['dense_output_tolerance_passed'] = bool(torch.isclose(actual.float(), dense_reference.float(), atol=.006, rtol=.015).all())
        if state['config'].get('check_sparse_reference'):
            reference_method = 'e0' if method == 'e0_simt' else 'e0_simt'
            if reference_method not in state['executors']:
                state['executors'][reference_method] = PagedSparseExecutor(query.device, reference_method)
            reference_executor = state['executors'][reference_method]
            reference_executor.plan(actual_q, cache, ids)
            sparse_reference = reference_executor.run(actual_q)
            difference = (actual.float()-sparse_reference.float()).abs()
            record['same_mask_reference'] = {'method': reference_method, 'max_abs': float(difference.max()),
                'output_tolerance_passed': bool(torch.isclose(actual.float(), sparse_reference.float(), atol=.006, rtol=.015).all()),
                'same_query_cache_and_physical_ids': True, 'extra_attention_for_numerical_reference': True}
        probe_key = (state['phase'], layer)
        if (layer in state['config'].get('fp64_probe_layers', []) and probe_key not in state['fp64_probed']):
            state['fp64_probed'].add(probe_key)
            # Independent address calculation from original cache coordinates;
            # this does not use the flattened alias or physical_supports.
            reference = torch.empty_like(actual, dtype=torch.float64)
            for request, by_head in enumerate(logical):
                for head, support in enumerate(by_head):
                    qq = actual_q[request, head*8:(head+1)*8].double()
                    m = torch.full((8,), -torch.inf, dtype=torch.float64, device=query.device)
                    z = torch.zeros_like(m)
                    u = torch.zeros((8,128), dtype=torch.float64, device=query.device)
                    for offset in range(0, len(support), 4096):
                        positions64 = support[offset:offset+4096].to(query.device)
                        pages = tables[request, support[offset:offset+4096]//cache.shape[2]].to(query.device).long()
                        offsets = positions64 % cache.shape[2]
                        kk = cache[pages, head, offsets, :128].double()
                        vv = cache[pages, head, offsets, 128:].double()
                        scores = qq @ kk.T * (128**-.5)
                        next_m = torch.maximum(m, scores.max(1).values)
                        weights = torch.exp(scores-next_m[:,None])
                        rescale = torch.exp(m-next_m)
                        z = z*rescale+weights.sum(1)
                        u = u*rescale[:,None]+weights @ vv
                        m = next_m
                    reference[request, head*8:(head+1)*8] = u/z[:,None]
            def compare(reference, value):
                diff = value.double()-reference
                return {'max_abs': float(diff.abs().max()), 'rmse': float(diff.square().mean().sqrt()),
                        'frozen_output_tolerance_passed': bool(torch.isclose(value.double(), reference, atol=.006, rtol=.015).all())}
            record['independent_fp64_probe'] = {'candidate': compare(reference, actual),
                'dense': compare(reference, dense_reference) if dense_reference is not None else None,
                'method': 'streaming FP64 softmax via original cache[actual_page,head,token_offset,K_or_V], no flattened-address reuse'}
        state['records'].append(record)
        self._capture_sparse_layer_output(layer, meta, output)
        return output

    def set_sparse_phase(self, phase, method, teacher_tokens=None):
        assert method in ('dense', 'e0', 'e0_simt', 'union')
        state = self.sparse_state
        state['phase'], state['method'] = phase, method
        state['records'], state['logits'], state['layer_outputs'], state['forward_trace'] = [], {}, {}, []
        state['teacher_tokens'], state['fp64_probed'] = teacher_tokens, set()
        return {'rank': state['rank'], 'phase': phase, 'method': method}

    def finish_sparse_phase(self):
        state = self.sparse_state
        torch.cuda.synchronize()
        path = state['output'] / f"{state['phase']}.json"
        record = {'rank': state['rank'], 'phase': state['phase'], 'method': state['method'],
                  'selector': state['selector'].summary() if state['selector'] is not None else None,
                  'teacher_forced_tokens': state.get('teacher_tokens'),
                  'records': state['records'], 'logits_count': len(state['logits']),
                  'forward_trace': state['forward_trace'],
                  'budget': state['config']['budget'],
                  'scope': 'Instrumented exact-mask output/logit qualification; timings include Python metadata and optional redundant numerical references, not application speed evidence.'}
        if state['logits']:
            logits_path = path.with_suffix('.pt')
            torch.save(state['logits'], logits_path)
            record['logits_file'] = str(logits_path)
        if state['layer_outputs']:
            layers_path = path.with_name(path.stem + '_layer_outputs.pt')
            torch.save(state['layer_outputs'], layers_path)
            record['layer_outputs_file'] = str(layers_path)
            record['layer_outputs_count'] = len(state['layer_outputs'])
        if state['selector'] is not None:
            masks_path = state['output']/'frozen_logical_masks.pt'
            torch.save({f'l{layer}_q{question}':torch.stack(ids) for (layer,question),ids in state['selector'].masks.items()}, masks_path)
            record['frozen_mask_file'] = str(masks_path)
            record['frozen_mask_file_sha256'] = hashlib.sha256(masks_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(record, indent=2)+'\n')
        return {'rank': state['rank'], 'metadata': str(path), 'records': len(state['records']), 'logits_count': len(state['logits'])}
