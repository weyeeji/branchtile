"""Strong E0 control: one native length plan and one GPU index update for 48 layers."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from scripts.vllm_sparse_graph_worker import SparseGraphWorkerExtension
from src.vllm_shared_decode_plan import LayerSharedDecodePlan
from src.vllm_shared_union_plan import LayerSharedUnionPlan
from src.vllm_sparse_executor import physical_supports


class SharedPlanGraphWorkerExtension(SparseGraphWorkerExtension):
    def _sparse_forward(self,layer,original,attn_layer,query,key,value,cache,meta,
                        output,output_scale,output_block_scale):
        state=self.sparse_state
        if state.get('graph_capturing') is not None and 'shared_plans' in state:
            method=state['graph_capturing']
            batch=query.shape[0]
            state['graph_queries'][method,layer,batch]=query
            counter=f'{method}:B{batch}:L{layer}'
            state['graph_capture_calls'][counter]=state['graph_capture_calls'].get(counter,0)+1
            return state['shared_plans'][method,batch].run_layer(layer,query,output=output)
        if meta is not None and 'batch' in state:
            # Compilation may bypass the late-installed parent-module hook;
            # actual runner token positions follow the same query row order.
            state['positions'][layer]=state['batch'].positions
        return super()._sparse_forward(layer,original,attn_layer,query,key,value,cache,meta,output,output_scale,output_block_scale)

    def _prepare_sparse_graph(self,metadata):
        state=self.sparse_state;started=time.perf_counter();batch=state['batch'];b=batch.num_reqs
        starts=batch.query_start_loc_np[:b+1].tolist()
        positions=batch.num_computed_tokens_np[:b].tolist()
        infos=[self._sparse_request_info(i) for i in range(b)]
        if not all(stop-start==1 and p>=row['prompt_length'] for start,stop,p,row in zip(starts,starts[1:],positions,infos)):
            return
        assert batch.num_tokens_after_padding==b
        meta=next(iter(metadata.values()))
        questions=[r['question_index'] for r in infos]
        plan=state['shared_plans'][state['method'],b]
        prepared=plan.prepare(meta.block_table[:b],questions,positions,state['shared_history_id'])
        if prepared['plan_topology_changed_requires_recapture']:
            raise RuntimeError('Native topology changed; existing engine graph must not replay until explicitly recaptured')
        metadata_seconds=time.perf_counter()-started
        physical={};shadow_started=time.perf_counter()
        if state['config'].get('graph_shadow_check'):
            assert positions==batch.positions[:b].cpu().tolist()
            assert [p+1 for p in positions]==meta.seq_lens[:b].cpu().tolist()
            tables=meta.block_table[:b].cpu()
            for layer in range(48):
                logical=[state['selector'].logical_supports(layer,q,p) for q,p in zip(questions,positions)]
                physical[layer]=physical_supports(tables,logical,2,16,state['graph_caches'][layer].stride())
        state['graph_pending']={'batch':b,'positions':positions,'questions':questions,'physical_ids':physical,
            'metadata_seconds':metadata_seconds,'extra_shadow_metadata_seconds':time.perf_counter()-shadow_started,
            'shared_prepare':prepared}

    @torch.inference_mode()
    def capture_sparse_graphs(self,methods=('e0',)):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        assert methods and all(m in ('e0','union') for m in methods)
        state=self.sparse_state;runner=self.model_runner
        assert len(state['selector'].masks)==48*len(state['config']['prompt_tokens'])
        before=torch.cuda.memory_allocated();started=time.perf_counter()
        selector=state['selector'];prompts=[len(x) for x in state['config']['prompt_tokens']]
        if selector.budget==selector.doc_end-selector.doc_start:
            expected=torch.arange(selector.doc_start,selector.doc_end)
            assert all(torch.equal(ids,expected) for by_head in selector.masks.values() for ids in by_head), 'Full document budget must select every document position'
        histories=[[[torch.cat((torch.arange(selector.doc_start),selector.masks[layer,q][head],torch.arange(selector.doc_end,prompts[q]))) for head in range(2)] for q in range(len(prompts))] for layer in range(48)]
        identity=hashlib.sha256(json.dumps({'prompt_hashes':state['config']['prompt_token_hashes'],
            'masks':[r['mask_sha256'] for r in selector.records]},sort_keys=True).encode()).hexdigest()
        state['shared_history_id']=identity
        workspace=torch.zeros(256*1024*1024,device=runner.device,dtype=torch.uint8)
        state['graph_workspace']=workspace
        state['shared_plans']={}
        tables=state['table_cpu'].to(runner.device)
        assert tables.shape[0]>=3,'Warm the complete ready B3 group before capture'
        state['graph_enabled']=False
        for method in methods:
            for b in (1,2,3):
                caches=[state['graph_caches'][l] for l in range(48)]
                if method=='e0':
                    capacity=max(len(histories[0][q][0]) for q in range(len(prompts)))+8192
                    plan=LayerSharedDecodePlan(caches,histories,prompts,identity,b,max_per_request=capacity,workspace=workspace)
                else:
                    plan=LayerSharedUnionPlan(caches,histories,prompts,identity,b,
                        common_end=(selector.doc_end//16)*16,max_generated=8192,workspace=workspace)
                plan.prepare(tables[:b],list(range(b)),[p-1 for p in prompts[:b]],identity)
                state['shared_plans'][method,b]=plan
                for layer in range(48):
                    state['graph_ops'][method,layer,b]=SimpleNamespace(out=(plan.native_outputs[layer] if method=='union' else plan.outputs[layer]))
            native=runner.cudagraph_manager
            manager=ModelCudaGraphManager(runner.vllm_config,runner.device,native.cudagraph_mode,
                decode_query_len=runner.decode_query_len,lora_capture_cases=runner.lora_capture_cases,
                varlen_decode=False,ubatch_runner=None)
            state['graph_capturing']=method
            manager.capture(runner.model,runner.model_state,runner.input_buffers,runner.intermediate_tensors,
                runner.block_tables,runner.attn_groups,runner.kv_cache_config,pcp_manager=runner.pcp_manager,
                has_lora=False,use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                progress_bar_desc=f'Capturing shared native plan {method}')
            state['graph_capturing']=None;state['graph_managers'][method]=manager
        torch.cuda.synchronize();state['graph_enabled']=True
        result={'rank':state['rank'],'capture_seconds':time.perf_counter()-started,
            'additional_torch_allocated_bytes':torch.cuda.memory_allocated()-before,
            'memory_includes_shared_workspace_and_histories':True,
            'candidate_attention_capture_calls':state['graph_capture_calls'],
            'descriptors':{m:[str(d) for d in manager.graphs] for m,manager in state['graph_managers'].items()},
            'frozen_history_identity':identity,
            'full_budget_exact_document_mask':selector.budget==selector.doc_end-selector.doc_start,
            'shared_native_plan':'Each method/B graph owns one native length plan shared by all48 layers. E0 maps indices on GPU; union also packs exact masks and validates shared complete-page physical identity. Union histories are padded to equal lengths across layers/heads.',
            'scope':'Existing native cached/incremental planning control, not a novel method.'}
        (state['output']/'graph_capture.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def set_graph_validation(self, enabled):
        state=self.sparse_state
        state['config']['graph_shadow_check']=bool(enabled)
        state['config']['check_sparse_reference']=bool(enabled)
        state['config']['capture_layer_outputs']=bool(enabled)
        return {'rank':state['rank'],'graph_shadow_check':bool(enabled),'mixed_sparse_reference':bool(enabled),'layer_output_capture':bool(enabled)}
