"""Experimental second vLLM FULL graph manager with persistent sparse metadata.

Native dense graphs remain available. Candidate graphs are explicitly captured
through the patched attention implementation, and replayed outputs are checked
against independent eager native execution during qualification. Planning runs
outside replay and is never excluded from complete costs.
"""
import hashlib
import json
from pathlib import Path
import time

import torch

from scripts.vllm_sparse_worker import SparseAttentionWorkerExtension
from src.vllm_sparse_executor import PagedSparseExecutor, physical_supports
from src.vllm_sparse_graph_executor import GraphPagedSparseExecutor


class SparseGraphWorkerExtension(SparseAttentionWorkerExtension):
    def configure_sparse_attention(self, config):
        result = super().configure_sparse_attention(config)
        state = self.sparse_state
        state.update(graph_enabled=False, graph_capturing=None, graph_caches={}, graph_ops={},
                     graph_queries={}, graph_managers={}, graph_capture_calls={}, graph_records=[],
                     graph_pending=None, graph_shadow={}, graph_phase_records=[])
        runner = self.model_runner
        manager = runner.cudagraph_manager
        assert manager is not None and manager.graphs
        old_dispatch = manager.dispatch
        def dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None, num_ubatches=1):
            if not state['graph_enabled']:
                from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
                from vllm.config.compilation import CUDAGraphMode
                return BatchExecutionDescriptor(CUDAGraphMode.NONE, num_tokens, num_reqs, num_active_loras=num_active_loras, num_ubatches=num_ubatches)
            return old_dispatch(num_reqs,num_tokens,uniform_token_count,num_active_loras,max_query_len,num_ubatches)
        manager.dispatch = dispatch
        old_prepare = runner.model_state.prepare_attn
        def prepare_attn(*args, **kwargs):
            metadata = old_prepare(*args, **kwargs)
            state['graph_pending'] = None
            if state['graph_enabled'] and not state['graph_capturing'] and state['method'] != 'dense':
                from vllm.config.compilation import CUDAGraphMode
                mode = args[1] if len(args)>1 else kwargs['cudagraph_mode']
                if mode == CUDAGraphMode.FULL:
                    self._prepare_sparse_graph(metadata)
            return metadata
        runner.model_state.prepare_attn = prepare_attn
        old_run = manager.run_fullgraph
        def run_fullgraph(desc):
            pending = state['graph_pending']
            if pending is None:
                return old_run(desc)
            assert desc.num_tokens == pending['batch']
            method = state['method']
            started = time.perf_counter()
            result = state['graph_managers'][method].run_fullgraph(desc)
            record = {'phase':state['phase'],'method':method,'batch':pending['batch'],
                'questions':pending['questions'],'positions':pending['positions'],
                'metadata_complete_host_seconds':pending['metadata_seconds'],
                'graph_submit_host_seconds':time.perf_counter()-started,
                'candidate_graph_replayed':True,'dense_attention_executed_for_pure_decode':False}
            if 'shared_prepare' in pending:
                record['shared_native_prepare']=pending['shared_prepare']
                record['extra_shadow_metadata_host_seconds']=pending['extra_shadow_metadata_seconds']
            if state['config'].get('graph_shadow_check'):
                checks=[]
                for layer in range(48):
                    op = state['graph_ops'][method,layer,pending['batch']]
                    q = state['graph_queries'][method,layer,pending['batch']]
                    if method == 'union':
                        actual=op.out.view(2,pending['batch'],8,128).permute(1,0,2,3).reshape(q.shape)
                    else:
                        actual=op.out.view(q.shape)
                    reference_method = 'e0_simt' if method != 'e0_simt' else 'e0'
                    if reference_method not in state['graph_shadow']:
                        state['graph_shadow'][reference_method]=PagedSparseExecutor(q.device,reference_method)
                    reference=state['graph_shadow'][reference_method]
                    reference.plan(q,state['graph_caches'][layer],pending['physical_ids'][layer])
                    expected=reference.run(q)
                    diff=(actual.float()-expected.float()).abs()
                    checks.append({'layer':layer,'max_abs':float(diff.max()),
                        'frozen_output_tolerance_passed':bool(torch.isclose(actual.float(),expected.float(),atol=.006,rtol=.015).all()),
                        'query_l2':float(q.float().norm()),'output_l2':float(actual.float().norm()),
                        'reference_method':reference_method})
                    if state['config'].get('graph_native_full_reference'):
                        assert state['selector'].budget == state['selector'].doc_end-state['selector'].doc_start
                        attn_layer,original=state['native_layers'][layer]
                        native=torch.empty_like(q)
                        original(attn_layer,q,None,None,state['graph_caches'][layer],
                            state['graph_native_metadata'],output=native)
                        checks[-1]['native_full_reference']={'max_abs':float((actual.float()-native.float()).abs().max()),
                            'frozen_output_tolerance_passed':bool(torch.isclose(actual.float(),native.float(),atol=.006,rtol=.015).all()),
                            'same_actual_query_cache_full_history':True,
                            'backend':state['backend_descriptions'][layer]}
                record['shadow_checks']=checks
                record['all_shadow_checks_passed']=all(x['frozen_output_tolerance_passed'] for x in checks)
            state['graph_phase_records'].append(record)
            return result
        manager.run_fullgraph = run_fullgraph
        state['original_graph_dispatch'],state['original_graph_run']=old_dispatch,old_run
        result['native_graph_descriptors']=[str(desc) for desc in manager.graphs]
        result['graph_worker_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return result

    def _sparse_forward(self, layer, original, attn_layer, query, key, value, cache, meta,
                        output, output_scale, output_block_scale):
        state=self.sparse_state
        if 'graph_caches' in state:
            state['graph_caches'][layer]=cache
        method=state.get('graph_capturing')
        if method is not None:
            batch=query.shape[0]
            op=state['graph_ops'][method,layer,batch]
            state['graph_queries'][method,layer,batch]=query
            counter=f'{method}:B{batch}:L{layer}'
            state['graph_capture_calls'][counter]=state['graph_capture_calls'].get(counter,0)+1
            return op.run(query,output=output)
        return super()._sparse_forward(layer,original,attn_layer,query,key,value,cache,meta,output,output_scale,output_block_scale)

    def _prepare_sparse_graph(self, metadata):
        state=self.sparse_state
        started=time.perf_counter()
        batch=state['batch']
        b=batch.num_reqs
        positions=batch.positions[:batch.num_tokens].cpu()
        starts=batch.query_start_loc_np[:b+1].tolist()
        info=[self._sparse_request_info(i) for i in range(b)]
        if not all(stop-start==1 and int(positions[start])>=row['prompt_length'] for start,stop,row in zip(starts,starts[1:],info)):
            return
        assert batch.num_tokens_after_padding == b, 'Qualification uses exact B1/2/3 capture sizes, no hidden request padding'
        meta=next(iter(metadata.values()))
        tables=meta.block_table[:b].cpu()
        assert starts==meta.query_start_loc[:b+1].cpu().tolist()
        assert [int(positions[i])+1 for i in range(b)]==meta.seq_lens[:b].cpu().tolist()
        physical={}
        for layer in range(48):
            logical=[state['selector'].logical_supports(layer,row['question_index'],int(positions[i])) for i,row in enumerate(info)]
            cache=state['graph_caches'][layer]
            ids=physical_supports(tables,logical,2,16,cache.stride())
            physical[layer]=ids
            op=state['graph_ops'][state['method'],layer,b]
            op.plan(state['graph_dummy_q'][b],cache,ids)
        state['graph_pending']={'batch':b,'positions':positions.tolist(),'questions':[r['question_index'] for r in info],
            'physical_ids':physical,'metadata_seconds':time.perf_counter()-started}

    @torch.inference_mode()
    def capture_sparse_graphs(self, methods=('e0','union')):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        state=self.sparse_state
        runner=self.model_runner
        assert state['selector'] is not None and len(state['selector'].masks)==48*len(state['config']['prompt_tokens'])
        assert len(state['graph_caches'])==48
        state['graph_workspace']=torch.zeros(256*1024*1024,device=runner.device,dtype=torch.uint8)
        state['graph_dummy_q']={b:torch.zeros(b,16,128,device=runner.device,dtype=torch.bfloat16) for b in (1,2,3)}
        state['graph_enabled']=False
        before=torch.cuda.memory_allocated()
        started=time.perf_counter()
        tables=state['table_cpu'][:1]
        for method in methods:
            assert method in ('e0','e0_simt','union')
            for layer,cache in state['graph_caches'].items():
                ids_one=physical_supports(tables,[[torch.arange(256)]*2],2,16,cache.stride())[0]
                for b,q in state['graph_dummy_q'].items():
                    op=GraphPagedSparseExecutor(runner.device,method,b,max_tokens=16384,workspace=state['graph_workspace'])
                    op.plan(q,cache,[ids_one]*b)
                    state['graph_ops'][method,layer,b]=op
            native=runner.cudagraph_manager
            manager=ModelCudaGraphManager(runner.vllm_config,runner.device,native.cudagraph_mode,
                decode_query_len=runner.decode_query_len,lora_capture_cases=runner.lora_capture_cases,
                varlen_decode=False,ubatch_runner=None)
            state['graph_capturing']=method
            manager.capture(runner.model,runner.model_state,runner.input_buffers,runner.intermediate_tensors,
                runner.block_tables,runner.attn_groups,runner.kv_cache_config,
                pcp_manager=runner.pcp_manager,has_lora=False,use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                progress_bar_desc=f'Capturing exact sparse {method}')
            state['graph_capturing']=None
            state['graph_managers'][method]=manager
        torch.cuda.synchronize()
        state['graph_enabled']=True
        result={'rank':state['rank'],'capture_seconds':time.perf_counter()-started,
            'additional_torch_allocated_bytes':torch.cuda.memory_allocated()-before,
            'candidate_attention_capture_calls':state['graph_capture_calls'],
            'descriptors':{m:[str(d) for d in manager.graphs] for m,manager in state['graph_managers'].items()},
            'scope':'New FULL graphs explicitly captured through candidate attention; original native graphs remain separate.'}
        (state['output']/'graph_capture.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def set_sparse_phase(self, phase, method, teacher_tokens=None):
        result=super().set_sparse_phase(phase,method,teacher_tokens)
        self.sparse_state['graph_phase_records']=[]
        return result

    def enable_sparse_graphs(self, enabled):
        self.sparse_state['graph_enabled']=bool(enabled)
        return {'rank':self.sparse_state['rank'],'graph_enabled':bool(enabled)}

    def finish_sparse_phase(self):
        result=super().finish_sparse_phase()
        state=self.sparse_state
        path=state['output']/f"{state['phase']}_graph.json"
        record={'rank':state['rank'],'phase':state['phase'],'method':state['method'],
            'capture_calls':state['graph_capture_calls'],'replays':state['graph_phase_records'],
            'scope':'Shadow checks add work and are charged; graph submission and CPU planning are not application throughput measurements.'}
        path.write_text(json.dumps(record,indent=2)+'\n')
        result['graph_metadata']=str(path)
        return result
