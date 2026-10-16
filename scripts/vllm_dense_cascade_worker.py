"""Original-page full-history shared-prefix control in actual vLLM graphs."""
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from scripts.vllm_native_precision_worker import NativePrecisionWorkerExtension
from src.vllm_dense_cascade_plan import LayerSharedDenseCascadePlan


class DenseCascadeWorker(NativePrecisionWorkerExtension):
    def configure_dense_cascade(self, config):
        result = self.configure_native_precision_probe(config)
        runner = self.model_runner
        state = self.dense_cascade = {'config': config, 'capture_mode': None, 'method': 'native',
            'validation': False, 'plans': {}, 'managers': {}, 'saved_q': {}, 'saved_o': {},
            'native_layers': {}, 'prepare_records': [], 'checks': [], 'pending': None}
        caches, scales_k, scales_v = {}, {}, {}
        for name, parent in runner.model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.self_attn$', name)
            if not match:
                continue
            layer = int(match[1])
            assert not parent.attn.attn_backend.forward_includes_kv_cache_update
            assert parent.attn.attn_backend.get_name() == 'FLASHINFER'
            assert parent.num_heads == 16 and parent.num_kv_heads == 2 and parent.head_dim == 128
            caches[layer] = parent.attn.kv_cache
            scales_k[layer] = parent.attn._k_scale_float
            scales_v[layer] = parent.attn._v_scale_float
            original = parent.attn.impl.forward
            state['native_layers'][layer] = (parent.attn, original)
            def forward(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale=None, output_block_scale=None, layer=layer, original=original):
                mode = state['capture_mode']
                if mode is None or query.shape[0] == 1:
                    return original(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                                    output_scale=output_scale, output_block_scale=output_block_scale)
                assert output_scale is None and output_block_scale is None
                b = query.shape[0]
                if mode == 'checked':
                    state['saved_q'][layer, b].copy_(query)
                result = state['plans'][b].run_layer(layer, query, output=output)
                if mode == 'checked':
                    state['saved_o'][layer, b].copy_(output)
                return result
            parent.attn.impl.forward = forward
        assert sorted(caches) == list(range(48))
        state['caches'] = caches
        workspace = torch.empty(256*1024*1024, device=runner.device, dtype=torch.uint8)
        for b in range(2, config['batch']+1):
            state['plans'][b] = LayerSharedDenseCascadePlan([caches[l] for l in range(48)], b,
                config['common_prefix_tokens'], max_tail_tokens=config.get('max_tail_tokens', 16384),
                workspace=workspace, k_scales=[scales_k[l] for l in range(48)],
                v_scales=[scales_v[l] for l in range(48)])
            for layer in range(48):
                for key in ('saved_q', 'saved_o'):
                    state[key][layer, b] = torch.empty((b, 16, 128), device=runner.device, dtype=torch.bfloat16)
        previous_tables = runner.prepare_attn
        def tables(*args, **kwargs):
            value = previous_tables(*args, **kwargs)
            assert len(value[0]) == 1
            state['live_tables'] = value[0][0]
            return value
        runner.prepare_attn = tables
        previous_prepare = runner.model_state.prepare_attn
        def prepare(*args, **kwargs):
            metadata = previous_prepare(*args, **kwargs)
            state['pending'] = None
            if state['capture_mode'] is not None or state['method'] != 'cascade':
                return metadata
            from vllm.config.compilation import CUDAGraphMode
            mode = args[1] if len(args) > 1 else kwargs['cudagraph_mode']
            batch = self.native_probe['batch']
            b = batch.num_reqs
            if mode != CUDAGraphMode.FULL or b < 2 or batch.num_tokens != b:
                return metadata
            assert batch.num_tokens_after_padding == b
            positions = batch.num_computed_tokens_np[:b].tolist()
            seq_lens = [int(pos)+1 for pos in positions]
            if min(seq_lens) <= config['common_prefix_tokens']:
                return metadata
            receipt = state['plans'][b].prepare(state['live_tables'], seq_lens)
            state['pending'] = {'batch': b, 'positions': positions, 'receipt': receipt,
                                'metadata': metadata}
            state['prepare_records'].append(receipt)
            return metadata
        runner.model_state.prepare_attn = prepare
        previous_run = runner.cudagraph_manager.run_fullgraph
        def run(desc):
            pending = state['pending']
            if pending is None:
                return previous_run(desc)
            assert desc.num_tokens == desc.num_reqs == pending['batch']
            b = pending['batch']
            mode = 'checked' if state['validation'] else 'production'
            value = state['managers'][mode].run_fullgraph(desc)
            probe = self.native_probe
            batch = probe['batch']
            questions = [probe['requests'][rid] for rid in batch.req_ids[:b]]
            probe['records'].append({'phase': probe['phase'], 'method': 'dense_cascade', 'batch': b,
                'questions': questions, 'positions': pending['positions'],
                'step_prepare_started_monotonic_seconds': probe['prepare_started'],
                'step_record_finished_monotonic_seconds': time.perf_counter(),
                'complete_attention_prepare_host_seconds': probe['prepare_seconds']+pending['receipt']['complete_prepare_host_seconds'],
                'additional_cascade_prepare_host_seconds': pending['receipt']['complete_prepare_host_seconds'],
                'native_graph_replayed': False, 'candidate_graph_replayed': True,
                'dense_attention_executed_for_pure_decode': True,
                'scope': 'Exact full-history prefix+private-tail cascade; B1 and nonpure decode fall back to native.'})
            if state['validation']:
                metadata = next(iter(pending['metadata'].values()))
                checks = []
                for layer in range(48):
                    q = state['saved_q'][layer, b]
                    actual = state['saved_o'][layer, b]
                    expected = torch.empty_like(actual)
                    attn_layer, original = state['native_layers'][layer]
                    original(attn_layer, q, None, None, state['caches'][layer], metadata, output=expected)
                    diff = (actual.float()-expected.float()).abs()
                    passed = bool(torch.isclose(actual.float(), expected.float(), atol=.006, rtol=.015).all())
                    checks.append({'layer': layer, 'max_abs': float(diff.max()), 'same_Q_native_O_passed': passed})
                    assert passed, (layer, checks[-1])
                state['checks'].append({'batch': b, 'positions': pending['positions'], 'layers': checks})
            return value
        runner.cudagraph_manager.run_fullgraph = run
        result['dense_cascade_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['common_prefix_tokens'] = config['common_prefix_tokens']
        result['cache_shapes'] = {str(l): {'shape': list(c.shape), 'stride': list(c.stride()), 'dtype': str(c.dtype)} for l, c in caches.items()}
        return result

    @torch.inference_mode()
    def capture_dense_cascade_graphs(self):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        runner, state = self.model_runner, self.dense_cascade
        assert state['method'] == 'native'
        started = time.perf_counter()
        for mode in ('production', 'checked'):
            native = runner.cudagraph_manager
            manager = ModelCudaGraphManager(runner.vllm_config, runner.device, native.cudagraph_mode,
                decode_query_len=runner.decode_query_len, lora_capture_cases=runner.lora_capture_cases,
                varlen_decode=False, ubatch_runner=None)
            state['capture_mode'] = mode
            manager.capture(runner.model, runner.model_state, runner.input_buffers, runner.intermediate_tensors,
                runner.block_tables, runner.attn_groups, runner.kv_cache_config,
                pcp_manager=runner.pcp_manager, has_lora=False,
                use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                progress_bar_desc=f'Capturing dense cascade {mode}')
            state['capture_mode'] = None
            state['managers'][mode] = manager
        torch.cuda.synchronize()
        return {'rank': self.native_probe['rank'], 'capture_wall_seconds': time.perf_counter()-started,
                'scope': 'Graphs constructed before fresh real requests; numerical-check copies are absent from production graph.'}

    def set_dense_cascade_phase(self, phase, method, validation=False):
        assert method in ('native', 'cascade')
        result = self.set_native_phase(phase)
        state = self.dense_cascade
        state.update(method=method, validation=validation, prepare_records=[], checks=[], pending=None)
        return dict(result, method=method, validation=validation)

    def finish_dense_cascade_phase(self):
        state = self.dense_cascade
        probe = self.native_probe
        metadata_path = probe['output'] / f'{probe["phase"]}.json'
        graph_path = probe['output'] / f'{probe["phase"]}_graph.json'
        metadata_path.write_text(json.dumps({'rank': probe['rank'], 'phase': probe['phase'],
            'method': state['method'], 'full_history': True,
            'native_full_history': state['method'] == 'native',
            'metadata_types': list(probe['metadata'].values()),
            'scope': 'Native XQA metadata remains prepared; cascade phases replay separate graphs with exact prefix+tail arithmetic.'}, indent=2)+'\n')
        graph_path.write_text(json.dumps({'rank': probe['rank'], 'phase': probe['phase'],
            'replays': probe['records'], 'metadata_types': list(probe['metadata'].values())}, indent=2)+'\n')
        result = {'rank': probe['rank'], 'metadata': str(metadata_path),
                  'graph_metadata': str(graph_path), 'replays': len(probe['records'])}
        report = {'rank': self.native_probe['rank'], 'phase': self.native_probe['phase'],
                  'method': state['method'], 'validation': state['validation'],
                  'prepare_records': state['prepare_records'], 'checks': state['checks'],
                  'scope': 'Complete full-history arithmetic, existing cascade mechanism; natural semantics and quality still require comparison.'}
        path = self.native_probe['output'] / f'{self.native_probe["phase"]}_cascade.json'
        path.write_text(json.dumps(report, indent=2)+'\n')
        return dict(result, cascade_record=str(path))
