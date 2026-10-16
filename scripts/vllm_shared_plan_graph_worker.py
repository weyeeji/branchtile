"""Strong E0 control: one native length plan and one GPU index update for 48 layers."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from scripts.vllm_sparse_graph_worker import SparseGraphWorkerExtension
from src.vllm_shared_decode_plan import LayerSharedDecodePlan
from src.vllm_sparse_executor import physical_supports


class SharedPlanGraphWorkerExtension(SparseGraphWorkerExtension):
    def _sparse_forward(self,layer,original,attn_layer,query,key,value,cache,meta,
                        output,output_scale,output_block_scale):
        state=self.sparse_state
        if state.get('graph_capturing')=='e0' and 'shared_decode_plans' in state:
            batch=query.shape[0]
            state['graph_queries']['e0',layer,batch]=query
            counter=f'e0:B{batch}:L{layer}'
            state['graph_capture_calls'][counter]=state['graph_capture_calls'].get(counter,0)+1
            return state['shared_decode_plans'][batch].run_layer(layer,query,output=output)
        return super()._sparse_forward(layer,original,attn_layer,query,key,value,cache,meta,output,output_scale,output_block_scale)

    def _prepare_sparse_graph(self,metadata):
        state=self.sparse_state;started=time.perf_counter();batch=state['batch'];b=batch.num_reqs
        starts=batch.query_start_loc_np[:b+1].tolist()
        positions=batch.num_computed_tokens_np[:b].tolist()
        infos=[self._sparse_request_info(i) for i in range(b)]
        if not all(stop-start==1 and p>=row['prompt_length'] for start,stop,p,row in zip(starts,starts[1:],positions,infos)):
            return
        assert state['method']=='e0' and batch.num_tokens_after_padding==b
        meta=next(iter(metadata.values()))
        questions=[r['question_index'] for r in infos]
        plan=state['shared_decode_plans'][b]
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
        assert tuple(methods)==('e0',)
        state=self.sparse_state;runner=self.model_runner
        assert len(state['selector'].masks)==48*len(state['config']['prompt_tokens'])
        before=torch.cuda.memory_allocated();started=time.perf_counter()
        selector=state['selector'];prompts=[len(x) for x in state['config']['prompt_tokens']]
        histories=[[[torch.cat((torch.arange(selector.doc_start),selector.masks[layer,q][head],torch.arange(selector.doc_end,prompts[q]))) for head in range(2)] for q in range(len(prompts))] for layer in range(48)]
        identity=hashlib.sha256(json.dumps({'prompt_hashes':state['config']['prompt_token_hashes'],
            'masks':[r['mask_sha256'] for r in selector.records]},sort_keys=True).encode()).hexdigest()
        state['shared_history_id']=identity
        workspace=torch.zeros(256*1024*1024,device=runner.device,dtype=torch.uint8)
        state['graph_workspace']=workspace
        state['shared_decode_plans']={}
        tables=state['table_cpu'].to(runner.device)
        assert tables.shape[0]>=3,'Warm the complete ready B3 group before capture'
        state['graph_enabled']=False
        for b in (1,2,3):
            plan=LayerSharedDecodePlan([state['graph_caches'][l] for l in range(48)],histories,prompts,identity,b,
                max_per_request=16384,workspace=workspace)
            plan.prepare(tables[:b],list(range(b)),[p-1 for p in prompts[:b]],identity)
            state['shared_decode_plans'][b]=plan
            for layer in range(48):
                state['graph_ops']['e0',layer,b]=SimpleNamespace(out=plan.outputs[layer])
        native=runner.cudagraph_manager
        manager=ModelCudaGraphManager(runner.vllm_config,runner.device,native.cudagraph_mode,
            decode_query_len=runner.decode_query_len,lora_capture_cases=runner.lora_capture_cases,
            varlen_decode=False,ubatch_runner=None)
        state['graph_capturing']='e0'
        manager.capture(runner.model,runner.model_state,runner.input_buffers,runner.intermediate_tensors,
            runner.block_tables,runner.attn_groups,runner.kv_cache_config,pcp_manager=runner.pcp_manager,
            has_lora=False,use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
            progress_bar_desc='Capturing shared native plan E0')
        state['graph_capturing']=None;state['graph_managers']['e0']=manager
        torch.cuda.synchronize();state['graph_enabled']=True
        result={'rank':state['rank'],'capture_seconds':time.perf_counter()-started,
            'additional_torch_allocated_bytes':torch.cuda.memory_allocated()-before,
            'memory_includes_shared_workspace_and_histories':True,
            'candidate_attention_capture_calls':state['graph_capture_calls'],
            'descriptors':{'e0':[str(d) for d in manager.graphs]},
            'frozen_history_identity':identity,
            'shared_native_plan':'Each B1/B2/B3 graph owns one native length plan shared by all48 layers; one GPU kernel maps all layer indices through the current physical block table.',
            'scope':'Existing native cached/incremental planning control, not a novel method.'}
        (state['output']/'graph_capture.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
