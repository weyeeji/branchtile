"""Actual-model B8 native raw-pair intervention with byte-exact qualification.

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
from src.vllm_native_raw_pair_plan import LayerSharedNativeRawPairPlan
from src.vllm_native_raw_pair_contract import REQUIRED_QUALIFICATION_FIELDS, qualify_same_state_receipt


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class NativeRawPairWorker(NativePrecisionWorkerExtension):
    def configure_dense_cascade(self, config):
        runner = self.model_runner
        bare_native = runner.cudagraph_manager.run_fullgraph
        result = self.configure_native_precision_probe(config)
        state = self.raw_pair = {'config': config, 'mode': None, 'method': 'native', 'validation': False,
            'managers': {}, 'caches': {}, 'originals': {}, 'saved_q': {}, 'saved_o': {},
            'check_q': {}, 'check_o': {}, 'prepare_records': [], 'checks': [], 'fallbacks': [],
            'pending': None, 'armed': False, 'max_steps': 128, 'cohort': None, 'epochs': {},
            'native_replays': 0, 'candidate_replays': 0, 'bare_native': bare_native,
            'captured_layers': {}, 'qualification_completed': False}
        assert runner.batch_sharder is None
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
            for key in ('saved_q','saved_o','check_q','check_o'):
                state[key][layer] = torch.empty((8,16,128),device=runner.device,dtype=torch.bfloat16)
            def forward(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale=None, output_block_scale=None, layer=layer, original=original):
                mode = state['mode']
                if mode is None or query.shape[0] != 8:
                    return original(attn_layer, query, key, value, kv_cache, attn_metadata, output,
                                    output_scale=output_scale, output_block_scale=output_block_scale)
                assert output_scale is None and output_block_scale is None
                state['captured_layers'].setdefault(mode,set()).add(layer)
                if mode == 'witness':
                    value = original(attn_layer, query, key, value, kv_cache, attn_metadata, output)
                    state['saved_q'][layer].copy_(query); state['saved_o'][layer].copy_(output)
                    return value
                if mode == 'checked': state['check_q'][layer].copy_(query)
                value = state['plan'].run_layer(layer, query, output)
                if mode == 'checked': state['check_o'][layer].copy_(output)
                return value
            attn.impl.forward = forward
        assert sorted(state['caches']) == list(range(48))
        from src.xqa_raw_pair_build import jit_specs
        specs, receipt = jit_specs()
        modules = {group: spec.build_and_load() for group, spec in specs.items()}
        state['build_receipt'] = receipt
        assert len(runner.block_tables.input_block_tables) == 1
        table = runner.block_tables.input_block_tables[0][:8]
        state['plan'] = LayerSharedNativeRawPairPlan(modules, [state['caches'][l] for l in range(48)],
                                                    table, config['common_prefix_tokens'])
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
            if mode != CUDAGraphMode.FULL or not (b == batch.num_tokens == batch.num_tokens_after_padding == 8):
                reason = 'requires_actual_and_padded_B8_pure_fullgraph_decode'
            if reason:
                state['fallbacks'].append({'batch':b,'padded_tokens':batch.num_tokens_after_padding,'reason':reason}); return metadata
            positions = batch.num_computed_tokens_np[:8].tolist(); lengths = [int(x)+1 for x in positions]
            if min(lengths) <= config['common_prefix_tokens']:
                state['fallbacks'].append({'batch':b,'reason':'prefix_not_fully_available'}); return metadata
            started = time.perf_counter()
            ids = list(batch.req_ids[:8]); slots = [int(x) for x in batch.idx_mapping_np[:8]]
            current = {rid:(slot,state['epochs'].get(slot,0)) for rid,slot in zip(ids,slots)}
            if state['cohort'] is not None and state['cohort'] != current:
                state['fallbacks'].append({'batch':b,'reason':'cohort_or_prefix_epoch_changed'}); return metadata
            meta = next(iter(metadata.values())); decode = getattr(meta,'decode',None)
            if decode is None or 'xqa' not in str(getattr(decode,'kernel','')).lower() or getattr(decode,'q_len_per_req',1) not in (None,1):
                state['fallbacks'].append({'batch':b,'reason':'not_native_single_token_XQA_metadata'}); return metadata
            if getattr(decode,'mask',None) is not None or getattr(decode,'q_cu_seq_lens',None) is not None:
                state['fallbacks'].append({'batch':b,'reason':'custom_or_ragged_decode_metadata'}); return metadata
            assert len(state['live_tables']) == len(state['live_slots']) == 1
            actual_table = state['live_tables'][0][:8]
            # This host physical check/copy is explicitly timed once per step.
            try:
                state['plan'].prepare(actual_table,lengths)
            except (AssertionError,ValueError) as exc:
                state['fallbacks'].append({'batch':b,'reason':'prefix_or_metadata_contract','detail':str(exc)}); return metadata
            state['cohort'] = current
            record = {'batch':8,'actual_requests':8,'padded_tokens':8,'positions':positions,'lengths':lengths,
                      'complete_prepare_host_seconds':time.perf_counter()-started,'host_physical_prefix_check':True,
                      'native_splits':state['plan'].inner.splits,'native_table_columns':actual_table.shape[1],
                      'metadata_copy_cost_included':True,'layers_sharing_state_and_scratch':48}
            state['prepare_records'].append(record); state['pending'] = record
            return metadata
        runner.model_state.prepare_attn = prepare
        previous_run = runner.cudagraph_manager.run_fullgraph
        def run(desc):
            pending = state['pending']
            if pending is None:
                value=previous_run(desc);state['native_replays']+=1;return value
            assert desc.num_reqs == desc.num_tokens == 8
            if state['armed'] and len(state['checks']) < state['max_steps']:
                native_hidden = previous_run(desc).clone(); state['native_replays']+=1
                self._qualify_raw_pair_state(desc,native_hidden)
                return native_hidden
            if state['method'] != 'cascade':
                value=previous_run(desc);state['native_replays']+=1;return value
            value=state['managers']['production'].run_fullgraph(desc);state['candidate_replays']+=1
            probe=self.native_probe;batch=probe['batch']
            probe['records'].append({'phase':probe['phase'],'method':'native_raw_pair','batch':8,
                'questions':[probe['requests'][rid] for rid in batch.req_ids[:8]],'positions':pending['positions'],
                'step_prepare_started_monotonic_seconds':probe['prepare_started'],
                'step_record_finished_monotonic_seconds':time.perf_counter(),
                'complete_attention_prepare_host_seconds':probe['prepare_seconds']+pending['complete_prepare_host_seconds'],
                'additional_raw_pair_prepare_host_seconds':pending['complete_prepare_host_seconds'],
                'native_graph_replayed':False,'candidate_graph_replayed':True,'dense_attention_executed_for_pure_decode':True})
            return value
        runner.cudagraph_manager.run_fullgraph = run
        result.update(raw_pair_plan=state['plan'].receipt(),raw_pair_build_receipt=receipt)
        return result

    @torch.inference_mode()
    def capture_dense_cascade_graphs(self):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        runner,state=self.model_runner,self.raw_pair
        assert state['method']=='native'
        started=time.perf_counter(); times={}
        for mode in ('witness','checked','production'):
            tick=time.perf_counter(); native=runner.cudagraph_manager
            manager=ModelCudaGraphManager(runner.vllm_config,runner.device,native.cudagraph_mode,
                decode_query_len=runner.decode_query_len,lora_capture_cases=runner.lora_capture_cases,
                varlen_decode=False,ubatch_runner=None)
            state['mode']=mode
            manager.capture(runner.model,runner.model_state,runner.input_buffers,runner.intermediate_tensors,
                runner.block_tables,runner.attn_groups,runner.kv_cache_config,pcp_manager=runner.pcp_manager,
                has_lora=False,use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                progress_bar_desc=f'Capturing raw pair {mode}')
            state['mode']=None;state['managers'][mode]=manager
            assert state['captured_layers'][mode] == set(range(48)), (mode,state['captured_layers'])
            torch.cuda.synchronize();times[mode]=time.perf_counter()-tick
        return {'rank':self.native_probe['rank'],'capture_wall_seconds':time.perf_counter()-started,
                'by_mode':times,'plan':state['plan'].receipt(),'scope':'Captured before fresh real requests; actual B8 admitted separately.'}

    def set_dense_cascade_phase(self,phase,method,validation=False):
        assert method in ('native','cascade')
        state=self.raw_pair
        if method == 'cascade':
            assert state['qualification_completed'], 'Complete same-state model qualification before production activation'
        result=self.set_native_phase(phase)
        state.update(method=method,validation=validation,prepare_records=[],checks=[],fallbacks=[],pending=None,
                     cohort=None,armed=False,native_replays=0,candidate_replays=0)
        return dict(result,method=method,mechanism='native_raw_pair')

    def arm_raw_pair_qualification(self,max_steps=128):
        state=self.raw_pair
        assert state['method']=='native' and len(state['managers'])==3
        assert max_steps>0
        state.update(armed=True,max_steps=int(max_steps))
        return {'rank':self.native_probe['rank'],'max_steps':max_steps,'trajectory':'Original native hidden continues to sampler.'}

    @torch.inference_mode()
    def _qualify_raw_pair_state(self,desc,native_hidden):
        state,runner=self.raw_pair,self.model_runner
        assert len(state['live_slots'])==1
        slots=state['live_slots'][0][:8].clone().long()
        inputs=runner.input_buffers.input_ids[:8].clone();positions=runner.input_buffers.positions[:8].clone()
        tables=state['live_tables'][0][:8].clone()
        def current_kv():return {l:c[slots//16,:,slots%16,:].clone() for l,c in state['caches'].items()}
        def logits(hidden):
            full=runner.model.compute_logits(hidden)
            full=None if full is None else full.clone()
            local=runner.model.logits_processor(runner.model.lm_head,hidden,skip_gather=True).clone()
            return full,local
        native_kv=current_kv();native_full,native_local=logits(native_hidden)
        aa=state['bare_native'](desc).clone()
        record={k:True for k in REQUIRED_QUALIFICATION_FIELDS}
        record.update(step=len(state['checks']),positions=state['pending']['positions'],batch=8,
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
            qok=bits_equal(state['saved_q'][l],state['check_q'][l]);ook=bits_equal(state['saved_o'][l],state['check_o'][l])
            record['all48_attention_Q_bits_equal'] &= qok;record['all48_attention_O_bits_equal'] &= ook
            record['layer_checks'].append({'layer':l,'Q_bits_equal':qok,'O_bits_equal':ook})
        record['input_ids_positions_tables_bits_unchanged']=bits_equal(inputs,runner.input_buffers.input_ids[:8]) and bits_equal(positions,runner.input_buffers.positions[:8]) and bits_equal(tables,state['live_tables'][0][:8])
        record['qualification_passed']=qualify_same_state_receipt(record)
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
            'scope':'Native trajectory during qualification; admitted actual B8 uses raw pair in candidate phases, all other steps use native.'},indent=2)+'\n')
        graph_path.write_text(json.dumps({'rank':probe['rank'],'phase':probe['phase'],
            'replays':probe['records'],'metadata_types':list(probe['metadata'].values())},indent=2)+'\n')
        result={'rank':probe['rank'],'metadata':str(metadata_path),'graph_metadata':str(graph_path),'replays':len(probe['records'])}
        checks=state['checks'];passed=bool(checks) and all(x['qualification_passed'] for x in checks)
        phase_valid=all(x['qualification_passed'] for x in checks)
        if checks: state['qualification_completed'] = passed
        report={'rank':probe['rank'],'phase':probe['phase'],'mechanism':'raw_pair_native_state_continuation',
                'method':state['method'],'validation':state['validation'],'qualification_requested_steps':state['max_steps'],
                'qualification_steps':len(checks),'qualification_passed':passed,'phase_valid':phase_valid,
                'candidate_replays':state['candidate_replays'],'native_replays':state['native_replays'],
                'prepare_records':state['prepare_records'],'checks':checks,'fallbacks':state['fallbacks'],
                'plan':state['plan'].receipt(),'build_receipt':state['build_receipt'],
                'captured_B8_layers':{k:sorted(v) for k,v in state['captured_layers'].items()},
                'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ('scripts/vllm_native_raw_pair_worker.py','src/vllm_native_raw_pair_plan.py','src/xqa_raw_pair_runtime.py','src/xqa_raw_pair_build.py')},
                'scope':'B8 only. Qualification retains native trajectory; all copies and per-step synchronized physical validation charged in performance phases.'}
        path=probe['output']/f'{probe["phase"]}_raw_pair.json';path.write_text(json.dumps(report,indent=2)+'\n')
        return dict(result,raw_pair_record=str(path),cascade_record=str(path),qualification_passed=passed,
                    qualification_steps=len(checks),phase_valid=phase_valid,candidate_replays=state['candidate_replays'],native_replays=state['native_replays'])
