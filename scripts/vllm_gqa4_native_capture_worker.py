"""Eager native Qwen3-4B TP2 real Q/KV/O export; no attention replacement."""
import hashlib
import json
from pathlib import Path
import re
import torch
from scripts.vllm_native_backend_worker import describe_metadata


class GQA4NativeCaptureWorker:
    def configure_gqa4_capture(self, config):
        from vllm.distributed import get_tp_group
        runner=self.model_runner
        assert runner.vllm_config.model_config.enforce_eager
        assert runner.vllm_config.parallel_config.tensor_parallel_size==2
        assert runner.vllm_config.cache_config.cache_dtype=='auto'
        assert runner.vllm_config.cache_config.block_size==16
        rank=get_tp_group().rank_in_group
        directory=Path(config['output'])/f'rank{rank}'/'dense_capture';directory.mkdir(parents=True,exist_ok=False)
        state=self.gqa4_capture={'config':config,'rank':rank,'directory':directory,'armed':False,'seen':0,'files':[],'qo':{},'layers':[],'metadata':{}}
        old_prepare=runner.prepare_inputs
        def prepare(*args,**kwargs):
            batch=old_prepare(*args,**kwargs);state['batch']=batch
            if state['armed'] and batch.num_reqs==batch.num_tokens==8:state['seen']+=1
            return batch
        runner.prepare_inputs=prepare
        old_tables=runner.prepare_attn
        def tables(*args,**kwargs):
            result=old_tables(*args,**kwargs);state['tables'],state['slots']=result;return result
        runner.prepare_attn=tables
        for name,parent in runner.model.named_modules():
            match=re.search(r'(?:^|\.)layers\.(\d+)\.self_attn$',name)
            if not match:continue
            layer=int(match[1]);attn=parent.attn
            assert (parent.num_heads,parent.num_kv_heads,parent.head_dim)==(16,4,128)
            assert attn.attn_backend.get_name()=='FLASHINFER' and not attn.attn_backend.forward_includes_kv_cache_update
            assert attn.kv_cache.dtype==torch.bfloat16 and attn.kv_cache.shape[1:]==(4,16,256)
            assert attn._q_scale_float==attn._k_scale_float==attn._v_scale_float==1.
            state['layers'].append({'layer':layer,'shape':list(attn.kv_cache.shape),'stride':list(attn.kv_cache.stride()),
                                    'dtype':str(attn.kv_cache.dtype),'backend':attn.attn_backend.get_name(),
                                    'k_scale':float(attn._k_scale_float),'v_scale':float(attn._v_scale_float)})
            original=attn.impl.forward
            def forward(attn_layer,query,key,value,kv_cache,attn_metadata,output,output_scale=None,output_block_scale=None,layer=layer,original=original):
                result=original(attn_layer,query,key,value,kv_cache,attn_metadata,output,
                                output_scale=output_scale,output_block_scale=output_block_scale)
                if state['armed'] and state['seen']==3 and layer not in state['qo']:
                    assert query.shape==(8,16,128) and output.numel()==8*16*128
                    state['qo'][layer]={'q':query.detach().cpu(),'o':output.detach().reshape(8,16,128).cpu()}
                    state['metadata'][layer]=describe_metadata(attn_metadata)
                    if layer in (0,11,23,35):self._export_gqa4_layer(layer,query,output,kv_cache)
                return result
            attn.impl.forward=forward
        assert sorted(x['layer'] for x in state['layers'])==list(range(36))
        return {'rank':rank,'layers':state['layers'],'scope':'Eager native attention, third actual complete B8 state, no candidate or timed claim.'}

    def arm_gqa4_capture(self):
        self.gqa4_capture['armed']=True;self.gqa4_capture['seen']=0
        return {'armed':True}

    def _export_gqa4_layer(self,layer,query,output,cache):
        state=self.gqa4_capture;runner=self.model_runner
        assert len(state['tables'])==1
        table=state['tables'][0][:8];table_cpu=table.cpu().long()
        positions=runner.input_buffers.positions[:8].cpu().long();lengths=positions+1
        complete=int(lengths.min())//16
        same=(table_cpu[:,:complete]==table_cpu[:1,:complete]).all(0);bad=(~same).nonzero().flatten()
        common=int(bad[0]) if bad.numel() else complete
        assert common>0 and cache.shape[1:]==(4,16,256)
        prefix=cache[table[0,:common].long()].reshape(common,4,16,2,128).permute(3,0,2,1,4).reshape(2,common*16,4,128).cpu()
        tails=[]
        for row,length in enumerate(lengths.tolist()):
            logical=torch.arange(common*16,length,device=cache.device)
            tails.append(cache[table[row,logical//16].long(),:,logical%16].reshape(-1,4,2,128).permute(2,0,1,3).cpu())
        path=state['directory']/f'layer{layer}.pt'
        torch.save({'q':query.detach().cpu(),'o':output.detach().reshape(8,16,128).cpu(),'prefix_kv':prefix,'tail_kv':tails,
                    'positions':positions,'original_block_table':table_cpu,'original_cache_shape':list(cache.shape),
                    'original_cache_stride':list(cache.stride()),'original_cache_dtype':str(cache.dtype),
                    'common_complete_pages':common,'k_scale':1.,'v_scale':1.,'rank':state['rank'],'layer':layer,
                    'source_native_kv_dtype':'torch.bfloat16','payload_cast_performed':False},path)
        state['files'].append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'layer':layer,'bytes':path.stat().st_size})

    def gqa4_capture_result(self):
        state=self.gqa4_capture
        assert sorted(state['qo'])==list(range(36)) and len(state['files'])==4
        path=state['directory']/'all_layer_qo.pt';torch.save(state['qo'],path)
        result={'status':'completed','rank':state['rank'],'seen_complete_B8_states':state['seen'],'layers':state['layers'],
                'actual_metadata':state['metadata'],'layer_files':state['files'],'all_qo_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope':'Actual eager native state; native dispatch recorded rather than assumed XQA. Candidate must check original O.'}
        (state['directory']/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
