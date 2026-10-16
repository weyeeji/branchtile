"""Original-page full-history shared-prefix control in actual vLLM graphs."""
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from scripts.vllm_native_precision_worker import NativePrecisionWorkerExtension
from src.vllm_dense_cascade_device_plan import LayerSharedDeviceDenseCascadePlan
from src.dense_cascade_fp64_reference import full_history_fp64, compare_to_reference


class DenseCascadeDeviceWorker(NativePrecisionWorkerExtension):
    def configure_dense_cascade(self, config):
        result = self.configure_native_precision_probe(config)
        runner = self.model_runner
        state = self.dense_cascade = {'config': config, 'capture_mode': None, 'method': 'native',
            'validation': False, 'plans': {}, 'managers': {}, 'saved_q': {}, 'saved_o': {},
            'native_layers': {}, 'prepare_records': [], 'checks': [], 'pending': None,
            'capture_tables': {}, 'prefix_epochs': {}, 'cohort': None, 'used_batches': set(), 'fallbacks': []}
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
                if layer == 0:
                    state['plans'][b].prepare_device(state['capture_tables'][b], runner.input_buffers.positions[:b], verify_prefix=True)
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
            state['plans'][b] = LayerSharedDeviceDenseCascadePlan([caches[l] for l in range(48)], b,
                config['common_prefix_tokens'], max_tail_tokens=config.get('max_tail_tokens', 16384),
                workspace=workspace, k_scales=[scales_k[l] for l in range(48)],
                v_scales=[scales_v[l] for l in range(48)])
            dummy = torch.zeros((b, config['common_prefix_tokens']//16+1), device=runner.device, dtype=torch.int32)
            state['plans'][b].bind_prefix(dummy)
            def forbidden_host_plan(*args, **kwargs):
                raise AssertionError('Per-step host plan is forbidden in device-metadata control')
            state['plans'][b].prefix_wrapper.plan = forbidden_host_plan
            state['plans'][b].tail_wrapper.plan = forbidden_host_plan
            for layer in range(48):
                for key in ('saved_q', 'saved_o'):
                    state[key][layer, b] = torch.empty((b, 16, 128), device=runner.device, dtype=torch.bfloat16)
        original_append = runner.block_tables.append_block_ids
        def append(req_index, new_block_ids, overwrite):
            before = int(runner.block_tables.num_blocks.np[0, req_index])
            value = original_append(req_index, new_block_ids, overwrite)
            if overwrite or before < config['common_prefix_tokens']//16:
                state['prefix_epochs'][int(req_index)] = state['prefix_epochs'].get(int(req_index), 0)+1
            return value
        runner.block_tables.append_block_ids = append
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
            if state['capture_mode'] is not None:
                capture_batch = args[0] if args else kwargs['input_batch']
                capture_tables = args[2] if len(args)>2 else kwargs['block_tables']
                state['capture_tables'][capture_batch.num_tokens_after_padding] = capture_tables[0]
                return metadata
            if state['method'] != 'cascade':
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
            started = time.perf_counter()
            tails = [n-config['common_prefix_tokens'] for n in seq_lens]
            if max(tails)>config.get('max_tail_tokens',16384):
                state['fallbacks'].append({'reason':'tail capacity exceeded','batch':b})
                return metadata
            ids = list(batch.req_ids[:b])
            slots = [int(x) for x in batch.idx_mapping_np[:b]]
            current = {rid:(slot,state['prefix_epochs'].get(slot,0)) for rid,slot in zip(ids,slots)}
            binding = None
            if state['cohort'] is None:
                binding = state['plans'][b].bind_prefix(state['live_tables'])
                source_prefix = state['plans'][b].prefix_indices
                for other_b,other in state['plans'].items():
                    if other_b != b:
                        other.prefix_indices.copy_(source_prefix)
                        other.prefix_bound=True
                        other.bound_epoch+=1
                        other.device_flags.zero_()
                state['cohort'] = current
            elif any(rid not in state['cohort'] or state['cohort'][rid]!=epoch for rid,epoch in current.items()):
                state['fallbacks'].append({'reason':'cohort changed or prefix overwrite/preemption epoch changed','batch':b,'request_ids':ids})
                return metadata
            assert state['capture_tables'][b].data_ptr()==state['live_tables'].data_ptr(), 'Graph metadata does not alias actual gathered page table'
            state['used_batches'].add(b)
            receipt = {'batch':b,'seq_lens':seq_lens,'common_prefix_tokens':config['common_prefix_tokens'],
                'private_tail_tokens':tails,'private_tail_pages':[(t+15)//16 for t in tails],
                'actual_shared_physical_prefix_checked':True,'tail_plan_topology_changed':False,
                'host_plan_calls_this_prepare':0,'tail_plan_calls_this_prepare':0,
                'prefix_plan_calls_this_prepare':0,'layers_sharing_plan':48,
                'device_metadata_inside_model_graph':True,'device_prefix_identity_checked_each_graph':True,
                'cohort_request_subset_and_CPU_prefix_epochs_checked':True,'prefix_binding':binding,
                'complete_prepare_host_seconds':time.perf_counter()-started,'source_payload_copy':False}
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
                    check = {'layer': layer, 'max_abs': float(diff.max()), 'same_Q_native_O_passed': passed,
                             'qualification_passed': passed, 'reference_computed': False,
                             'criterion': 'Original pairwise screen; same unchanged tolerances against FP64 if screen fails.'}
                    if not passed:
                        reference_started = time.perf_counter()
                        reference = full_history_fp64(q, state['caches'][layer], state['live_tables'],
                            pending['receipt']['seq_lens'], k_scale=attn_layer._k_scale_float,
                            v_scale=attn_layer._v_scale_float)
                        candidate_check = compare_to_reference(actual, reference)
                        native_check = compare_to_reference(expected, reference)
                        qualified = candidate_check['passed'] and native_check['passed']
                        failed = (~torch.isclose(actual.float(), expected.float(), atol=.006, rtol=.015)).nonzero().cpu().tolist()
                        coordinates = [{'index': index, 'candidate': float(actual[tuple(index)]),
                            'native': float(expected[tuple(index)]), 'fp64': float(reference[tuple(index)])}
                            for index in failed[:32]]
                        check.update(reference_computed=True, qualification_passed=qualified,
                            pair_failed_but_reference_passed=qualified,
                            candidate_vs_fp64=candidate_check, native_vs_fp64=native_check,
                            pairwise_failed_values=len(failed), first_failed_coordinates=coordinates,
                            fp64_reference_wall_seconds=time.perf_counter()-reference_started)
                    checks.append(check)
                    # Both ranks finish the phase. Driver jointly checks receipts
                    # after RPC; no rank-local numeric raise that strands TP peers.
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
        capture_modes = {}
        for mode in ('production', 'checked'):
            mode_started = time.perf_counter()
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
            capture_modes[mode] = time.perf_counter()-mode_started
        torch.cuda.synchronize()
        return {'rank': self.native_probe['rank'], 'capture_wall_seconds': time.perf_counter()-started,
                'capture_wall_seconds_by_mode':capture_modes,
                'scope': 'Graphs constructed before fresh real requests; numerical-check copies are absent from production graph.'}

    def set_dense_cascade_phase(self, phase, method, validation=False):
        assert method in ('native', 'cascade')
        result = self.set_native_phase(phase)
        state = self.dense_cascade
        state.update(method=method, validation=validation, prepare_records=[], checks=[], pending=None, cohort=None, used_batches=set(), fallbacks=[])
        for plan in state['plans'].values():
            plan.device_flags.zero_()
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
        device_receipts = {str(b):state['plans'][b].device_error_receipt() for b in sorted(state['used_batches'])}
        device_valid = all(r['valid'] and r['tail_plan_info_unchanged'] for r in device_receipts.values())
        report = {'device_metadata_valid':device_valid, 'device_error_receipts':device_receipts, 'fallbacks':state['fallbacks'],
                  'rank': self.native_probe['rank'], 'phase': self.native_probe['phase'],
                  'method': state['method'], 'validation': state['validation'],
                  'prepare_records': state['prepare_records'], 'checks': state['checks'],
                  'scope': 'Complete full-history arithmetic, existing cascade mechanism; natural semantics and quality still require comparison.'}
        path = self.native_probe['output'] / f'{self.native_probe["phase"]}_cascade.json'
        path.write_text(json.dumps(report, indent=2)+'\n')
        return dict(result, cascade_record=str(path),device_metadata_valid=device_valid)
