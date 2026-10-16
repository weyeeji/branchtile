"""No-shadow native/E0/union paths with complete per-phase wall measurement.

Minimal CPU step records remain visible and charged. No logits CPU copies,
teacher forcing, shadow attention, or layer-output snapshots are used here.
"""
import time

from scripts.vllm_shared_plan_graph_worker import SharedPlanGraphWorkerExtension


class ProductionSparseWorkerExtension(SharedPlanGraphWorkerExtension):
    def configure_sparse_attention(self,config):
        result=super().configure_sparse_attention(config)
        state=self.sparse_state;runner=self.model_runner
        prepare=runner.model_state.prepare_attn
        def timed_prepare(*args,**kwargs):
            state['step_prepare_started']=time.perf_counter()
            result=prepare(*args,**kwargs)
            state['complete_attention_prepare_host_seconds']=time.perf_counter()-state['step_prepare_started']
            return result
        runner.model_state.prepare_attn=timed_prepare
        run=runner.cudagraph_manager.run_fullgraph
        def measured_run(desc):
            before=len(state['graph_phase_records'])
            result=run(desc)
            if state.get('production_enabled'):
                batch=state['batch']
                if len(state['graph_phase_records'])==before:
                    assert state['method']=='dense'
                    state['graph_phase_records'].append({'phase':state['phase'],'method':'dense','batch':batch.num_reqs,
                        'questions':[self._sparse_request_info(i)['question_index'] for i in range(batch.num_reqs)],
                        'positions':batch.num_computed_tokens_np[:batch.num_reqs].tolist(),
                        'candidate_graph_replayed':False,'native_graph_replayed':True,
                        'dense_attention_executed_for_pure_decode':True})
                row=state['graph_phase_records'][-1]
                row['step_prepare_started_monotonic_seconds']=state['step_prepare_started']
                row['complete_attention_prepare_host_seconds']=state['complete_attention_prepare_host_seconds']
                row['step_record_finished_monotonic_seconds']=time.perf_counter()
            return result
        runner.cudagraph_manager.run_fullgraph=measured_run
        return result

    def _sparse_forward(self,layer,original,attn_layer,query,key,value,cache,meta,output,output_scale,output_block_scale):
        state=self.sparse_state
        if state.get('production_enabled') and not state.get('graph_capturing') and meta is not None:
            # Frozen histories are reused; repeated dense prefill does not need
            # to collect another observation window. Mixed batches still use
            # the explicit dense-prefill plus sparse-decode bridge and charge it.
            batch=state['batch'];method=state['method']
            if method=='dense':
                return original(attn_layer,query,key,value,cache,meta,output=output,
                    output_scale=output_scale,output_block_scale=output_block_scale)
            starts=batch.query_start_loc_np[:batch.num_reqs+1]
            positions=batch.num_computed_tokens_np[:batch.num_reqs]
            has_decode=any(int(b-a)==1 and int(pos)>=self._sparse_request_info(i)['prompt_length']
                for i,(a,b,pos) in enumerate(zip(starts,starts[1:],positions)))
            if not has_decode:
                return original(attn_layer,query,key,value,cache,meta,output=output,
                    output_scale=output_scale,output_block_scale=output_block_scale)
        return super()._sparse_forward(layer,original,attn_layer,query,key,value,cache,meta,output,output_scale,output_block_scale)

    def enable_production_sparse(self):
        state=self.sparse_state
        assert len(state['selector'].masks)==48*len(state['config']['prompt_tokens'])
        for key in ('capture_logits','capture_layer_outputs','check_dense','check_sparse_reference','graph_shadow_check'):
            state['config'][key]=False
        state['production_enabled']=True
        return {'rank':state['rank'],'teacher_forcing':False,'logit_copy':False,'layer_snapshots':False,
            'shadow_attention':False,'minimal_cpu_step_records':True,
            'allocated_bytes':__import__('torch').cuda.memory_allocated(),'reserved_bytes':__import__('torch').cuda.memory_reserved()}
