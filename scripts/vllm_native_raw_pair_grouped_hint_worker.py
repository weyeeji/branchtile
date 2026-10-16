"""Grouped-hint revision of actual-model raw-pair exact qualification.

Native token trajectory is retained while armed. No numerical mismatch raises
on one TP rank: both ranks finish the same collective schedule, then driver gates.
"""
import hashlib
import json
from pathlib import Path
import re
import time
import torch
from scripts.vllm_native_precision_worker import NativePrecisionWorkerExtension
from src.vllm_native_raw_pair_general_plan import LayerSharedRawPairGeneralBank
from src.vllm_native_raw_pair_general_setup import capture_modes, candidate_descriptor, validate_offline_records

SOURCE_FILES = (
    'scripts/vllm_native_raw_pair_grouped_hint_worker.py',
    'src/vllm_native_raw_pair_dispatch.py', 'results/xqa_raw_dispatch_policy_v1/policy.json', 'src/vllm_native_raw_pair_general_plan.py',
    'src/vllm_native_raw_pair_general_setup.py', 'src/vllm_native_raw_pair_contract.py',
    'src/xqa_raw_pair_padded_runtime.py', 'src/xqa_raw_pair_general_runtime.py',
    'src/xqa_raw_grouped_hint_build.py', 'src/xqa_raw_pair_build.py', 'src/xqa_raw_continuation_build.py',
    'scripts/vllm_native_precision_worker.py', 'scripts/vllm_native_backend_worker.py',
)


def source_hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def execution_fingerprint(runner):
    import importlib.metadata
    model = runner.vllm_config.model_config
    config_json = model.hf_config.to_json_string()
    device = torch.cuda.get_device_properties(runner.device)
    return {'model_path': str(model.model),
        'model_config_sha256': hashlib.sha256(config_json.encode()).hexdigest(),
        'model_dtype': str(model.dtype), 'cache_dtype': runner.vllm_config.cache_config.cache_dtype,
        'TP': runner.vllm_config.parallel_config.tensor_parallel_size,
        'device_name': device.name, 'SM': device.multi_processor_count,
        'compute_capability': [device.major, device.minor],
        'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'vllm': importlib.metadata.version('vllm'),
        'flashinfer': importlib.metadata.version('flashinfer-python'),
        'native_table_columns': runner.block_tables.input_block_tables[0].shape[1],
        'attention_shape': [48, 16, 2, 128, 16],
        'prefix_policy': 'floor(physical common-prefix tokens / 256) native tiles; full cohort binding plus per-step request/slot/epoch guards'}

from src.vllm_native_raw_pair_general_setup import REQUIRED_QUALIFICATION_FIELDS
from src.vllm_native_raw_pair_dispatch import frozen_policy, eligible_from_current_state


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class NativeRawPairGroupedHintWorker(NativePrecisionWorkerExtension):
    def configure_dense_cascade(self, config):
        runner = self.model_runner
        max_b = int(config['batch'])
        enabled_batches = tuple(sorted(set(config.get('enabled_batches', range(6, max_b + 1)))))
        assert enabled_batches and min(enabled_batches) >= 6 and max(enabled_batches) <= max_b
        setup_mode = config.get('setup_mode', 'strict')
        modes = capture_modes(setup_mode)
        bare_native = runner.cudagraph_manager.run_fullgraph
        result = self.configure_native_precision_probe(config)
        state = self.raw_pair = {'config': config, 'mode': None, 'method': 'native', 'validation': False,
            'managers': {}, 'caches': {}, 'originals': {}, 'saved_q': {}, 'saved_o': {},
            'check_q': {}, 'check_o': {}, 'prepare_records': [], 'checks': [], 'fallbacks': [],
            'pending': None, 'armed': False, 'max_steps': 128, 'cohort': None, 'epochs': {},
            'native_replays': 0, 'candidate_replays': 0, 'bare_native': bare_native,
            'captured_layers': {}, 'qualification_completed': False,
            'max_batch': max_b, 'enabled_batches': enabled_batches, 'setup_mode': setup_mode,
            'capture_modes': modes, 'qualified_batches': set(), 'per_batch_steps': 8,
            'execution_fingerprint': execution_fingerprint(runner), 'dispatch_policy': frozen_policy()}
        if setup_mode == 'production_only':
            paths = config.get('offline_qualification_records', [])
            records = [json.loads(Path(path).read_text()) for path in paths]
            state['offline_qualification'] = validate_offline_records(records, source_hashes(),
                state['execution_fingerprint'], enabled_batches,
                runner.vllm_config.parallel_config.tensor_parallel_size)
            state['qualified_batches'].update(enabled_batches)
            state['qualification_completed'] = True
        assert runner.batch_sharder is None
        parallel = runner.vllm_config.parallel_config
        assert parallel.tensor_parallel_size == 2 and parallel.pipeline_parallel_size == 1
        assert getattr(parallel, 'decode_context_parallel_size', 1) == 1
        assert torch.cuda.get_device_capability(runner.device) == (12, 0)
        assert runner.vllm_config.cache_config.cache_dtype == 'auto'
        for name, parent in runner.model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.self_attn$', name)
            if not match:
                continue
            layer = int(match[1]); attn = parent.attn
            assert attn.attn_backend.get_name() == 'FLASHINFER'
            assert not attn.attn_backend.forward_includes_kv_cache_update
            assert (parent.num_heads, parent.num_kv_heads, parent.head_dim) == (16, 2, 128)
            assert attn.kv_cache.dtype == torch.bfloat16
            assert attn._k_scale_float == attn._v_scale_float == attn._q_scale_float == 1.
            assert getattr(attn.impl, 'sinks', None) is None
            assert getattr(attn.impl, 'window_left', -1) == -1
            state['caches'][layer] = attn.kv_cache
            original = attn.impl.forward
            state['originals'][layer] = original
            if setup_mode == 'strict':
                for key in ('saved_q','saved_o','check_q','check_o'):
                    state[key][layer] = torch.empty((max_b,16,128),device=runner.device,dtype=torch.bfloat16)
            def forward(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale=None, output_block_scale=None, layer=layer, original=original):
                mode = state['mode']
                b = query.shape[0]
                if mode is None or b not in enabled_batches:
                    return original(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                                    output_scale=output_scale, output_block_scale=output_block_scale)
                assert output_scale is None and output_block_scale is None
                state['captured_layers'].setdefault(mode,{}).setdefault(b,set()).add(layer)
                if mode == 'witness':
                    value = original(attn_layer, query, key, value, kv_cache, attn_metadata, output)
                    state['saved_q'][layer][:b].copy_(query); state['saved_o'][layer][:b].copy_(output)
                    return value
                if mode == 'checked': state['check_q'][layer][:b].copy_(query)
                value = state['plan'].run_layer(layer, query, output)
                if mode == 'checked': state['check_o'][layer][:b].copy_(output)
                return value
            attn.impl.forward = forward
        assert sorted(state['caches']) == list(range(48))
        from src.xqa_raw_grouped_hint_build import jit_specs
        specs, receipt = jit_specs()
        modules = {group: spec.build_and_load() for group, spec in specs.items()}
        state['build_receipt'] = receipt
        assert len(runner.block_tables.input_block_tables) == 1
        table = runner.block_tables.input_block_tables[0][:max_b]
        state['plan'] = LayerSharedRawPairGeneralBank(modules, [state['caches'][l] for l in range(48)],
                                                    table, runner.input_buffers.seq_lens, config['common_prefix_tokens'], enabled_batches)
        original_append = runner.block_tables.append_block_ids
        def append(req_index, new_block_ids, overwrite):
            before = int(runner.block_tables.num_blocks.np[0,req_index])
            value = original_append(req_index,new_block_ids,overwrite)
            if overwrite or before < config['common_prefix_tokens']//16:
                state['epochs'][int(req_index)] = state['epochs'].get(int(req_index),0)+1
            return value
        runner.block_tables.append_block_ids = append
        old_tables = runner.prepare_attn
        def tables(*args, **kwargs):
            value = old_tables(*args, **kwargs)
            state['live_tables'], state['live_slots'] = value
            return value
        runner.prepare_attn = tables
        previous_prepare = runner.model_state.prepare_attn
        def prepare(*args, **kwargs):
            metadata = previous_prepare(*args, **kwargs)
            state['pending'] = None
            if state['mode'] is not None or (state['method'] != 'cascade' and not state['armed']): return metadata
            from vllm.config.compilation import CUDAGraphMode
            mode = args[1] if len(args)>1 else kwargs['cudagraph_mode']
            batch = self.native_probe['batch']; b = batch.num_reqs
            reason = None
            if mode != CUDAGraphMode.FULL or not (b == batch.num_tokens == batch.num_tokens_after_padding and b in enabled_batches):
                reason = 'requires_actual_equal_padded_enabled_batch_pure_fullgraph_decode'
            if reason:
                state['fallbacks'].append({'batch':b,'padded_tokens':batch.num_tokens_after_padding,'reason':reason}); return metadata
            if state['armed'] and (len(state['checks']) >= state['max_steps'] or
                    sum(x['batch'] == b for x in state['checks']) >= state['per_batch_steps']):
                state['fallbacks'].append({'batch':b,'reason':'qualification_batch_quota_complete'}); return metadata
            positions = batch.num_computed_tokens_np[:b].tolist(); lengths = [int(x)+1 for x in positions]
            if min(lengths) <= config['common_prefix_tokens']:
                state['fallbacks'].append({'batch':b,'reason':'prefix_not_fully_available'}); return metadata
            eligible, policy_reason = eligible_from_current_state(b, config['common_prefix_tokens'], lengths)
            if not eligible and state['method'] == 'cascade':
                state['fallbacks'].append({'batch':b,'reason':policy_reason,
                    'max_private_tokens':max(lengths)-config['common_prefix_tokens'],
                    'policy_eligible':False,'policy_prefix_tokens':config['common_prefix_tokens'],
                    'positions':positions}); return metadata
            if state['method'] == 'cascade' and b not in state['qualified_batches']:
                state['fallbacks'].append({'batch':b,'reason':'batch_not_yet_model_qualified'}); return metadata
            started = time.perf_counter()
            ids = list(batch.req_ids[:b]); slots = [int(x) for x in batch.idx_mapping_np[:b]]
            current = {rid:(slot,state['epochs'].get(slot,0)) for rid,slot in zip(ids,slots)}
            if state['cohort'] is not None and any(state['cohort'].get(rid) != identity for rid, identity in current.items()):
                state['fallbacks'].append({'batch':b,'reason':'cohort_or_prefix_epoch_changed'}); return metadata
            meta = next(iter(metadata.values())); decode = getattr(meta,'decode',None)
            if decode is None or 'xqa' not in str(getattr(decode,'kernel','')).lower() or getattr(decode,'q_len_per_req',1) not in (None,1):
                state['fallbacks'].append({'batch':b,'reason':'not_native_single_token_XQA_metadata'}); return metadata
            if getattr(decode,'mask',None) is not None or getattr(decode,'q_cu_seq_lens',None) is not None:
                state['fallbacks'].append({'batch':b,'reason':'custom_or_ragged_decode_metadata'}); return metadata
            assert len(state['live_tables']) == len(state['live_slots']) == 1
            actual_table = state['live_tables'][0][:b]
            # The full physical check runs only at first cohort binding. Later
            # survivors retain membership/slot/epoch identity and all shared pages.
            # Native gathered table and GPU seq_lens aliases are checked each step.
            binding = None
            try:
                native_lengths = decode.seq_lens
                state['plan'].validate_aliases(actual_table, native_lengths, b)
                assert decode.block_tables.data_ptr() == actual_table.data_ptr()
                assert decode.block_tables.shape == actual_table.shape
                assert max(lengths) <= state['plan'].plans[b].max_len
                if state['cohort'] is None:
                    binding = state['plan'].bind_prefix(actual_table)
            except (AssertionError,ValueError) as exc:
                state['fallbacks'].append({'batch':b,'reason':'prefix_or_metadata_contract','detail':str(exc)}); return metadata
            if state['cohort'] is None: state['cohort'] = current
            state['live_native_lengths'] = native_lengths
            record = {'batch':b,'actual_requests':b,'padded_tokens':b,'positions':positions,'lengths':lengths,
                      'complete_prepare_host_seconds':time.perf_counter()-started,'host_physical_prefix_check':binding is not None,
                      'prefix_binding':binding,'cohort_request_slot_epoch_subset_guard':True,
                      'native_GPU_lengths_aliased':True,'host_length_copy':False,
                      'policy_eligible':eligible,'policy_reason':policy_reason,
                      'policy_prefix_tokens':config['common_prefix_tokens'],
                      'actual_shared_tile_tokens':config['common_prefix_tokens']//256*256,
                      'policy_max_private_tokens':max(lengths)-config['common_prefix_tokens'],
                      'native_splits':state['plan'].plans[b].splits,'native_table_columns':actual_table.shape[1],
                      'metadata_copy_cost_included':True,'layers_sharing_state_and_scratch':48}
            state['prepare_records'].append(record); state['pending'] = record
            return metadata
        runner.model_state.prepare_attn = prepare
        previous_run = runner.cudagraph_manager.run_fullgraph
        def run(desc):
            pending = state['pending']
            if pending is None:
                value=previous_run(desc);state['native_replays']+=1;return value
            b = pending['batch']
            assert desc.num_reqs == desc.num_tokens == b
            if state['armed'] and len(state['checks']) < state['max_steps']:
                native_hidden = previous_run(desc).clone(); state['native_replays']+=1
                self._qualify_raw_pair_state(desc,native_hidden)
                return native_hidden
            if state['method'] != 'cascade':
                value=previous_run(desc);state['native_replays']+=1;return value
            value=state['managers']['production'].run_fullgraph(desc);state['candidate_replays']+=1
            probe=self.native_probe;batch=probe['batch']
            probe['records'].append({'phase':probe['phase'],'method':'native_raw_pair_general','batch':b,
                'questions':[probe['requests'][rid] for rid in batch.req_ids[:b]],'positions':pending['positions'],
                'step_prepare_started_monotonic_seconds':probe['prepare_started'],
                'step_record_finished_monotonic_seconds':time.perf_counter(),
                'complete_attention_prepare_host_seconds':probe['prepare_seconds']+pending['complete_prepare_host_seconds'],
                'additional_raw_pair_prepare_host_seconds':pending['complete_prepare_host_seconds'],
                'native_graph_replayed':False,'candidate_graph_replayed':True,'dense_attention_executed_for_pure_decode':True})
            return value
        runner.cudagraph_manager.run_fullgraph = run
        result['dispatch_policy']=state['dispatch_policy']
        result.update(raw_pair_plan=state['plan'].receipt(),raw_pair_build_receipt=receipt)
        return result

    @torch.inference_mode()
    def capture_dense_cascade_graphs(self):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        runner,state=self.model_runner,self.raw_pair
        assert state['method']=='native'
        started=time.perf_counter(); times={}
        for mode in state['capture_modes']:
            tick=time.perf_counter(); native=runner.cudagraph_manager
            manager=ModelCudaGraphManager(runner.vllm_config,runner.device,native.cudagraph_mode,
                decode_query_len=runner.decode_query_len,lora_capture_cases=runner.lora_capture_cases,
                varlen_decode=False,ubatch_runner=None)
            from vllm.config.compilation import CUDAGraphMode
            manager._capture_descs = {key: [d for d in descs if candidate_descriptor(
                d, CUDAGraphMode.FULL, state['enabled_batches'])]
                for key, descs in manager._capture_descs.items()}
            selected = [d for ds in manager._capture_descs.values() for d in ds]
            assert {d.num_tokens for d in selected} == set(state['enabled_batches']), selected
            state['mode']=mode
            manager.capture(runner.model,runner.model_state,runner.input_buffers,runner.intermediate_tensors,
                runner.block_tables,runner.attn_groups,runner.kv_cache_config,pcp_manager=runner.pcp_manager,
                has_lora=False,use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                progress_bar_desc=f'Capturing raw pair {mode}')
            state['mode']=None;state['managers'][mode]=manager
            assert set(state['captured_layers'][mode]) == set(state['enabled_batches'])
            assert all(layers == set(range(48)) for layers in state['captured_layers'][mode].values())
            torch.cuda.synchronize();times[mode]=time.perf_counter()-tick
        return {'rank':self.native_probe['rank'],'capture_wall_seconds':time.perf_counter()-started,
                'by_mode':times,'plan':state['plan'].receipt(),'setup_mode':state['setup_mode'],'captured_modes':list(state['capture_modes']),
                'scope':'Only enabled exact-B candidate graphs captured; native fallback graphs retained. Actual batch and offline/strict qualification checked separately.'}

    def set_dense_cascade_phase(self,phase,method,validation=False):
        assert method in ('native','cascade')
        state=self.raw_pair
        # Empty qualified_batches is a valid all-native fallback phase.
        # Per-step dispatch still prohibits every unqualified candidate batch.
        result=self.set_native_phase(phase)
        state.update(method=method,validation=validation,prepare_records=[],checks=[],fallbacks=[],pending=None,
                     cohort=None,armed=False,native_replays=0,candidate_replays=0)
        return dict(result,method=method,mechanism='native_raw_pair_general')

    def arm_raw_pair_qualification(self,max_steps=128,per_batch_steps=8):
        state=self.raw_pair
        assert state['setup_mode']=='strict' and state['method']=='native' and len(state['managers'])==3
        assert max_steps>0 and per_batch_steps>0
        state.update(armed=True,max_steps=int(max_steps),per_batch_steps=int(per_batch_steps))
        return {'rank':self.native_probe['rank'],'max_steps':max_steps,'per_batch_steps':per_batch_steps,'trajectory':'Original native hidden continues to sampler.'}

    @torch.inference_mode()
    def _qualify_raw_pair_state(self,desc,native_hidden):
        state,runner=self.raw_pair,self.model_runner
        b = desc.num_tokens
        assert b in state['enabled_batches'] and desc.num_reqs == b
        assert len(state['live_slots'])==1
        slots=state['live_slots'][0][:b].clone().long()
        inputs=runner.input_buffers.input_ids[:b].clone();positions=runner.input_buffers.positions[:b].clone()
        tables=state['live_tables'][0][:b].clone()
        def current_kv():return {l:c[slots//16,:,slots%16,:].clone() for l,c in state['caches'].items()}
        def logits(hidden):
            full=runner.model.compute_logits(hidden)
            full=None if full is None else full.clone()
            local=runner.model.logits_processor(runner.model.lm_head,hidden,skip_gather=True).clone()
            return full,local
        native_kv=current_kv();native_full,native_local=logits(native_hidden)
        aa=state['bare_native'](desc).clone()
        record={k:True for k in REQUIRED_QUALIFICATION_FIELDS}
        record.update(step=len(state['checks']),positions=state['pending']['positions'],batch=b,
                      native_AA_bits_equal=bits_equal(aa,native_hidden),modes=[],layer_checks=[])
        witness=state['managers']['witness'].run_fullgraph(desc).clone()
        record['native_AA_bits_equal'] &= bits_equal(witness,native_hidden)
        for mode in ('checked','production'):
            hidden=state['managers'][mode].run_fullgraph(desc).clone();state['candidate_replays']+=1
            full,local=logits(hidden)
            after=current_kv()
            checks={'mode':mode,'hidden_bits_equal':bits_equal(hidden,native_hidden),
                    'local_vocab_logits_bits_equal':bits_equal(local,native_local),
                    'gathered_logits_bits_equal_where_present':(full is None and native_full is None) or (full is not None and native_full is not None and bits_equal(full,native_full)),
                    'all48_current_KV_slot_bits_equal':all(bits_equal(after[l],native_kv[l]) for l in range(48))}
            record['modes'].append(checks)
            for key in checks:
                if key!='mode':record[key] &= checks[key]
        for l in range(48):
            qok=bits_equal(state['saved_q'][l][:b],state['check_q'][l][:b]);ook=bits_equal(state['saved_o'][l][:b],state['check_o'][l][:b])
            record['all48_attention_Q_bits_equal'] &= qok;record['all48_attention_O_bits_equal'] &= ook
            record['layer_checks'].append({'layer':l,'Q_bits_equal':qok,'O_bits_equal':ook})
        record['input_ids_positions_tables_bits_unchanged']=bits_equal(inputs,runner.input_buffers.input_ids[:b]) and bits_equal(positions,runner.input_buffers.positions[:b]) and bits_equal(tables,state['live_tables'][0][:b])
        candidate_lengths = state['plan'].plans[b].lengths.flatten().cpu().tolist()
        native_lengths = state['live_native_lengths'].flatten().cpu().tolist()
        record['device_lengths_equal_native_and_positions'] = candidate_lengths == native_lengths == state['pending']['lengths']
        record['native_GPU_metadata_aliases_verified'] = (
            state['plan'].plans[b].lengths.data_ptr() == state['live_native_lengths'].data_ptr()
            and state['plan'].plans[b].table.data_ptr() == state['live_tables'][0].data_ptr())
        record['qualification_passed']=all(record[k] is True for k in REQUIRED_QUALIFICATION_FIELDS)
        state['checks'].append(record)
        # Restore native-written slots even after a rejected numerical intervention.
        for l,c in state['caches'].items():c[slots//16,:,slots%16,:]=native_kv[l]
        # Do not raise per rank on mismatches: TP peers complete the same calls.

    def finish_dense_cascade_phase(self):
        state,probe=self.raw_pair,self.native_probe
        metadata_path=probe['output']/f'{probe["phase"]}.json'
        graph_path=probe['output']/f'{probe["phase"]}_graph.json'
        metadata_path.write_text(json.dumps({'rank':probe['rank'],'phase':probe['phase'],
            'method':state['method'],'mechanism':'raw_pair_native_state_continuation',
            'native_full_history':state['method']=='native','full_history':True,
            'metadata_types':list(probe['metadata'].values()),
            'scope':'Native trajectory during qualification; admitted and model-qualified actual batch uses raw pair in candidate phases, all other steps use native.'},indent=2)+'\n')
        graph_path.write_text(json.dumps({'rank':probe['rank'],'phase':probe['phase'],
            'replays':probe['records'],'metadata_types':list(probe['metadata'].values())},indent=2)+'\n')
        result={'rank':probe['rank'],'metadata':str(metadata_path),'graph_metadata':str(graph_path),'replays':len(probe['records'])}
        checks=state['checks'];passed=bool(checks) and all(x['qualification_passed'] for x in checks)
        phase_valid=all(x['qualification_passed'] for x in checks)
        if checks:
            if passed:
                state['qualified_batches'].update(x['batch'] for x in checks)
            else:
                state['qualified_batches'].difference_update(x['batch'] for x in checks if not x['qualification_passed'])
            state['qualification_completed'] = bool(state['qualified_batches'])
        report={'rank':probe['rank'],'phase':probe['phase'],'mechanism':'raw_pair_native_state_continuation',
                'method':state['method'],'validation':state['validation'],'qualification_requested_steps':state['max_steps'],
                'qualification_steps':len(checks),'qualification_passed':passed,'phase_valid':phase_valid,
                'candidate_replays':state['candidate_replays'],'native_replays':state['native_replays'],
                'prepare_records':state['prepare_records'],'checks':checks,'fallbacks':state['fallbacks'],
                'plan':state['plan'].receipt(),'build_receipt':state['build_receipt'],
                'captured_layers_by_batch':{mode:{str(b):sorted(layers) for b,layers in by_b.items()}
                    for mode,by_b in state['captured_layers'].items()},
                'setup_mode':state['setup_mode'],'enabled_batches':list(state['enabled_batches']),
                'qualified_batches':sorted(state['qualified_batches']),
                'qualification_states_by_batch':{str(b):sum(x['batch']==b for x in checks) for b in state['enabled_batches']},
                'execution_fingerprint':state['execution_fingerprint'],
                'offline_qualification':state.get('offline_qualification'),
                'dispatch_policy':state['dispatch_policy'],
                'qualification_scope':'Strict natural-state checks may extend outside the performance policy; production always applies the frozen thresholds.',
                'source_sha256':source_hashes(),
                'scope':'Only observed/qualified enabled batches. Qualification retains native trajectory; all graph copies and initial synchronized cohort binding charged; later steps use native GPU lengths and CPU epoch guards in performance phases.'}
        path=probe['output']/f'{probe["phase"]}_raw_pair.json';path.write_text(json.dumps(report,indent=2)+'\n')
        return dict(result,raw_pair_record=str(path),cascade_record=str(path),qualification_passed=passed,
                    qualification_steps=len(checks),qualified_batches=sorted(state['qualified_batches']),phase_valid=phase_valid,candidate_replays=state['candidate_replays'],native_replays=state['native_replays'])
