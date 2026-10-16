"""Real model integration of the separately qualified paged owner ABI."""
import hashlib
import json
from types import SimpleNamespace
import time
import torch
from scripts.vllm_sparse_production_worker import ProductionSparseWorkerExtension
from src.vllm_shared_decode_plan import LayerSharedDecodePlan
from src.vllm_shared_union_plan import LayerSharedUnionPlan
from src.vllm_paged_owner_plan import LayerSharedPagedOwnerPlan


class PagedOwnerProductionWorker(ProductionSparseWorkerExtension):
    def set_sparse_phase(self,phase,method,teacher_tokens=None):
        result=super().set_sparse_phase(phase,'e0' if method=='owner' else method,teacher_tokens)
        self.sparse_state['method']=method;result['method']=method
        return result

    def _sparse_forward(self,*args,**kwargs):
        state=self.sparse_state
        if state.get('method')=='owner' and not state.get('graph_capturing'):
            # Mixed prefill/decode has no pure-decode owner graph. Use the exact
            # same-support E0 bridge and explicitly charge/report this fallback.
            before=len(state['records'])
            state['method']='e0'
            try:
                return super()._sparse_forward(*args,**kwargs)
            finally:
                for record in state['records'][before:]:record['owner_e0_mixed_or_eager_fallback']=True
                state['method']='owner'
        return super()._sparse_forward(*args,**kwargs)

    def enable_owner_qualification(self):
        result=self.set_graph_validation(True)
        self.sparse_state['config']['capture_logits']=True
        result['capture_logits']=True
        return result

    @torch.inference_mode()
    def capture_sparse_graphs(self,methods=('e0',)):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        assert methods and all(m in ('e0','union','owner') for m in methods)
        state=self.sparse_state;runner=self.model_runner
        assert len(state['selector'].masks)==48*len(state['config']['prompt_tokens'])
        before=torch.cuda.memory_allocated();started=time.perf_counter()
        selector=state['selector'];prompts=[len(x) for x in state['config']['prompt_tokens']]
        selector.require_finalized_joint_supports()
        if selector.budget==selector.doc_end-selector.doc_start:
            expected=torch.arange(selector.doc_start,selector.doc_end)
            assert all(torch.equal(ids,expected) for by_head in selector.masks.values() for ids in by_head), 'Full document budget must select every document position'
        histories=[[[torch.cat((torch.arange(selector.doc_start),selector.masks[layer,q][head],torch.arange(selector.doc_end,prompts[q]))) for head in range(2)] for q in range(len(prompts))] for layer in range(48)]
        identity=hashlib.sha256(json.dumps({'prompt_hashes':state['config']['prompt_token_hashes'],
            'active_masks':selector.mask_identity()['sha256']},sort_keys=True).encode()).hexdigest()
        state['shared_history_id']=identity
        workspace=torch.zeros(256*1024*1024,device=runner.device,dtype=torch.uint8)
        state['graph_workspace']=workspace
        state['shared_plans']={}
        tables=state['table_cpu'].to(runner.device)
        assert tables.shape[0]>=len(prompts),'Warm the complete ready group before capture'
        state['graph_enabled']=False
        for method in methods:
            for b in range(1,len(prompts)+1):
                caches=[state['graph_caches'][l] for l in range(48)]
                if method=='e0':
                    capacity=max(len(histories[0][q][0]) for q in range(len(prompts)))+8192
                    plan=LayerSharedDecodePlan(caches,histories,prompts,identity,b,max_per_request=capacity,workspace=workspace)
                elif method=='owner':
                    plan=LayerSharedPagedOwnerPlan(caches,histories,prompts,identity,b,
                        common_end=(selector.doc_end//16)*16,max_generated=state['config'].get('max_generated',8192),
                        chunk_tokens=state['config'].get('owner_chunk_tokens',128),workspace=workspace)
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
            'native_backends':state['backend_descriptions'],
            'actual_cache_layouts':{str(l):{'shape':list(c.shape),'stride':list(c.stride()),
                'dtype':str(c.dtype),'storage_offset':c.storage_offset()} for l,c in state['graph_caches'].items()},
            'shared_native_plan':'Each method/B graph owns one native length plan shared by all48 layers. E0 maps indices on GPU; union also packs exact masks and validates shared complete-page physical identity. Union histories are padded to equal lengths across layers/heads.',
            'scope':'Existing native cached/incremental planning control, not a novel method.'}
        (state['output']/'graph_capture.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

