"""Read-only runtime identity and CPU step records for official native backends."""
import hashlib
import json
from pathlib import Path
import time

from scripts.probe_engine_docfinqa import DeploymentWorkerExtension


def describe_metadata(meta):
    out={'type':type(meta).__module__+'.'+type(meta).__name__}
    for key in ('kv_cache_layout','q_data_type_prefill','q_data_type_decode','use_cascade','causal'):
        if hasattr(meta,key):out[key]=str(getattr(meta,key))
    for key in ('prefill','decode'):
        value=getattr(meta,key,None)
        if value is None:continue
        item={'type':type(value).__module__+'.'+type(value).__name__}
        if hasattr(value,'kernel'):item['kernel']=str(value.kernel)
        wrapper=getattr(value,'wrapper',None)
        if wrapper is not None:
            item['wrapper_type']=type(wrapper).__module__+'.'+type(wrapper).__name__
            for name in ('_backend','use_tensor_cores','is_cuda_graph_enabled','_use_tensor_cores'):
                if hasattr(wrapper,name):item[name]=str(getattr(wrapper,name))
        out[key]=item
    return out


class NativeBackendWorkerExtension(DeploymentWorkerExtension):
    def configure_native_probe(self,config):
        from vllm.distributed import get_tp_group
        runner=self.model_runner
        state=self.native_probe={'config':config,'rank':get_tp_group().rank_in_group,'phase':'warmup',
            'records':[],'metadata':{},'requests':{}}
        state['output']=Path(config['output'])/f"rank{state['rank']}";state['output'].mkdir(parents=True,exist_ok=True)
        layers=[]
        for name,module in runner.model.named_modules():
            if hasattr(module,'attn_backend') and hasattr(module,'impl'):
                layers.append({'name':name,'backend':module.attn_backend.get_name(),
                    'impl':type(module.impl).__module__+'.'+type(module.impl).__name__})
        assert len(layers)==48
        old_prepare=runner.prepare_inputs
        def prepare(*args,**kwargs):
            batch=old_prepare(*args,**kwargs);state['batch']=batch;return batch
        runner.prepare_inputs=prepare
        old_attn=runner.model_state.prepare_attn
        def attn(*args,**kwargs):
            state['prepare_started']=time.perf_counter();metadata=old_attn(*args,**kwargs)
            state['prepare_seconds']=time.perf_counter()-state['prepare_started']
            for meta in metadata.values():
                description=describe_metadata(meta);identity=json.dumps(description,sort_keys=True)
                state['metadata'].setdefault(identity,description)
            return metadata
        runner.model_state.prepare_attn=attn
        old_run=runner.cudagraph_manager.run_fullgraph
        def run(desc):
            result=old_run(desc);batch=state['batch'];questions=[]
            for i,rid in enumerate(batch.req_ids[:batch.num_reqs]):
                if rid not in state['requests']:
                    slot=int(batch.idx_mapping_np[i]);length=int(runner.req_states.prompt_len.np[slot])
                    ids=runner.req_states.all_token_ids.gpu[slot,:length].cpu().tolist()
                    digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                    assert digest in config['prompt_token_hashes']
                    state['requests'][rid]=config['prompt_token_hashes'].index(digest)
                questions.append(state['requests'][rid])
            state['records'].append({'phase':state['phase'],'method':config['backend'],'batch':batch.num_reqs,
                'questions':questions,'positions':batch.num_computed_tokens_np[:batch.num_reqs].tolist(),
                'step_prepare_started_monotonic_seconds':state['prepare_started'],
                'complete_attention_prepare_host_seconds':state['prepare_seconds'],
                'step_record_finished_monotonic_seconds':time.perf_counter(),
                'native_graph_replayed':True,'candidate_graph_replayed':False,'dense_attention_executed_for_pure_decode':True})
            return result
        runner.cudagraph_manager.run_fullgraph=run
        result={'rank':state['rank'],'layers':layers,'graph_descriptors':[str(d) for d in runner.cudagraph_manager.graphs],
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'deployment':self.deployment_state(),
            'scope':'Native model/attention/cache execution unchanged; only CPU metadata identity and step timing records.'}
        (state['output']/'native_configuration.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

    def set_native_phase(self,phase):
        state=self.native_probe;state['phase']=phase;state['records']=[];state['metadata']={}
        return {'rank':state['rank'],'phase':phase}

    def finish_native_phase(self):
        state=self.native_probe;phase=state['phase'];path=state['output']/f'{phase}.json'
        path.write_text(json.dumps({'rank':state['rank'],'phase':phase,'method':state['config']['backend'],
            'records':[],'logits_count':0,'native_full_history':True,'metadata_types':list(state['metadata'].values()),
            'scope':'Official native full-history backend, no candidate/selector/mask modification, no output/logit shadow copies.'},indent=2)+'\n')
        graph=state['output']/f'{phase}_graph.json'
        graph.write_text(json.dumps({'rank':state['rank'],'phase':phase,'replays':state['records'],
            'metadata_types':list(state['metadata'].values())},indent=2)+'\n')
        return {'rank':state['rank'],'metadata':str(path),'graph_metadata':str(graph),'replays':len(state['records'])}
