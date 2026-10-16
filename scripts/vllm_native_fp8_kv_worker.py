"""Instrumented official native FP8 KV qualification, without sparse execution.

Selected native FULL graphs copy intermediate Q/output into persistent buffers:
holding Python references alone would not defeat compiled-memory reuse. This
adds instrumentation, so none of these timings are application performance.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

import torch

from scripts.probe_engine_docfinqa import DeploymentWorkerExtension
from scripts.vllm_native_backend_worker import describe_metadata


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def package_version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None


def loaded_flashinfer_libraries():
    paths=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and '/flashinfer/' in fields[5] and fields[5].endswith('.so'):
            paths.add(fields[5])
    return [{'path':p,'bytes':Path(p).stat().st_size,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in sorted(paths)]


def fp64_from_actual_fp8_cache(query, cache, tables, positions, scale, k_scale, v_scale, chunk=4096):
    """Full causal decode against the actual stored bytes, not pre-quantized K/V.

    Cache has logical shape [physical_page, KVhead, page_tokens, 2D], with
    arbitrary actual strides. No assumed contiguous flattening or future tail.
    """
    b, qheads, dim = query.shape
    heads, page = cache.shape[1:3]
    assert cache.ndim == 4 and cache.shape[-1] == 2 * dim
    assert cache.dtype in (torch.uint8, torch.float8_e4m3fn)
    assert qheads % heads == 0 and len(positions) == b
    storage = cache.view(torch.uint8)
    outputs, lses = [], []
    for r, pos in enumerate(positions):
        q = query[r].double().reshape(heads, qheads // heads, dim)
        maximum = torch.full(q.shape[:2], -torch.inf, dtype=torch.float64, device=q.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(q)
        for first in range(0, int(pos) + 1, chunk):
            logical = torch.arange(first, min(int(pos) + 1, first + chunk), device=q.device)
            pages = tables[r, logical // page].long()
            raw = storage[pages, :, logical % page, :]
            values = raw.view(torch.float8_e4m3fn).double().permute(1, 0, 2)
            k, v = values[..., :dim] * k_scale, values[..., dim:] * v_scale
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            new_maximum = torch.maximum(maximum, scores.amax(-1))
            old_weight = (maximum - new_maximum).exp()
            weights = (scores - new_maximum[..., None]).exp()
            numerator = numerator * old_weight[..., None] + torch.matmul(weights, v)
            denominator = denominator * old_weight + weights.sum(-1)
            maximum = new_maximum
        outputs.append((numerator / denominator[..., None]).reshape(qheads, dim))
        lses.append((maximum + denominator.log()).reshape(qheads))
    return torch.stack(outputs), torch.stack(lses)


class NativeFP8KVWorkerExtension(DeploymentWorkerExtension):
    def configure_fp8_probe(self, config):
        from vllm.distributed import get_tp_group
        assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
        runner = self.model_runner
        state = self.fp8_probe = dict(config=config, rank=get_tp_group().rank_in_group,
            phase='warmup', mode='native', requests={}, steps=[], metadata={}, checks=[],
            checked=set(), refs={}, capture=False, manager=None, originals={}, layers={},
            logits={}, baseline_logits={}, teacher=None, capture_calls={})
        state['output'] = Path(config['output']) / f"rank{state['rank']}"
        state['output'].mkdir(parents=True, exist_ok=True)
        assert self.vllm_config.cache_config.cache_dtype == 'fp8_e4m3'
        assert self.vllm_config.cache_config.block_size == 16
        assert runner.batch_sharder is None

        prepare_inputs = runner.prepare_inputs
        def inputs(*args, **kwargs):
            state['batch'] = prepare_inputs(*args, **kwargs)
            return state['batch']
        runner.prepare_inputs = inputs
        prepare_tables = runner.prepare_attn
        def tables(*args, **kwargs):
            result = prepare_tables(*args, **kwargs)
            assert len(result[0]) == 1
            state['tables'] = result[0][0]
            return result
        runner.prepare_attn = tables
        prepare_attn = runner.model_state.prepare_attn
        def attn(*args, **kwargs):
            metadata = prepare_attn(*args, **kwargs)
            if not state['capture']:
                state['attn_metadata'] = metadata
                for meta in metadata.values():
                    desc = describe_metadata(meta)
                    state['metadata'].setdefault(json.dumps(desc, sort_keys=True), desc)
            return metadata
        runner.model_state.prepare_attn = attn

        for name, parent in runner.model.named_modules():
            match = re.search(r'(?:^|\.)layers\.(\d+)\.self_attn$', name)
            if not match:
                continue
            layer = int(match[1]); module = parent.attn
            assert parent.num_heads == 16 and parent.num_kv_heads == 2 and parent.head_dim == 128
            assert module.attn_backend.get_name() == 'FLASHINFER'
            assert not module.attn_backend.forward_includes_kv_cache_update
            scales = {k: float(getattr(module, k).item()) for k in ('_q_scale', '_k_scale', '_v_scale')}
            assert all(v == 1.0 for v in scales.values()), scales
            scale_floats = {k+'_float':float(getattr(module,k+'_float')) for k in scales}
            assert all(scale_floats[k+'_float']==v for k,v in scales.items())
            cache = module.kv_cache
            assert cache.dtype in (torch.uint8, torch.float8_e4m3fn)
            assert cache.shape[1:] == (2, 16, 256)
            impl = module.impl
            assert impl.window_left == -1 and not impl.logits_soft_cap and impl.sinks is None
            state['layers'][layer] = {'name':name,'backend':module.attn_backend.get_name(),
                'implementation':type(impl).__module__+'.'+type(impl).__name__,
                'scales':scales,'scale_floats':scale_floats,'scale_origin':'checkpoint has no KV scale tensors; official loader default 1.0',
                'logical_cache_dtype':str(module.kv_cache_dtype),'cache_dtype':str(cache.dtype),
                'shape':list(cache.shape),'stride':list(cache.stride()),'data_ptr':cache.data_ptr(),
                'storage_offset':cache.storage_offset(),'logical_payload_bytes':cache.numel()*cache.element_size(),
                'attention_scale':float(impl.scale)}
            original = impl.forward
            state['originals'][layer] = (module, original)
            def forward(attn_layer, query, key, value, kv_cache, meta, output,
                        output_scale=None, output_block_scale=None, layer=layer, original=original):
                result = original(attn_layer, query, key, value, kv_cache, meta, output=output,
                    output_scale=output_scale, output_block_scale=output_block_scale)
                if state['capture'] and layer in config['layers']:
                    b = query.shape[0]
                    assert b in (1,2,3) and output_scale is None and output_block_scale is None
                    ref = state['refs'].get((b,layer))
                    if ref is None:
                        ref = {'q':torch.empty_like(query),'out':torch.empty_like(output),
                            'cache':kv_cache,'k':None if key is None else torch.empty_like(key),
                            'v':None if value is None else torch.empty_like(value)}
                        state['refs'][b,layer] = ref
                    ref['q'].copy_(query); ref['out'].copy_(output)
                    if key is not None: ref['k'].copy_(key)
                    if value is not None: ref['v'].copy_(value)
                    state['capture_calls'][f'B{b}:L{layer}'] = state['capture_calls'].get(f'B{b}:L{layer}',0)+1
                elif state['mode'] == 'eager' and layer in config['layers'] and meta is not None:
                    case = self._pure_case()
                    if case is not None and (state['phase'],case['batch'],layer) not in state['checked']:
                        self._check(layer, query, output, kv_cache, key, value, case)
                return result
            impl.forward = forward
        assert len(state['layers']) == 48

        native = runner.cudagraph_manager
        original_dispatch = native.dispatch
        def dispatch(num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None, num_ubatches=1):
            if state['mode'] == 'eager':
                from vllm.config.compilation import CUDAGraphMode
                from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
                return BatchExecutionDescriptor(CUDAGraphMode.NONE,num_tokens,num_reqs,
                    num_active_loras=num_active_loras,num_ubatches=num_ubatches)
            return original_dispatch(num_reqs,num_tokens,uniform_token_count,num_active_loras,max_query_len,num_ubatches)
        native.dispatch = dispatch
        original_run = native.run_fullgraph
        def run(desc):
            result = (state['manager'].run_fullgraph(desc) if state['manager'] is not None else original_run(desc))
            case = self._pure_case()
            if case is not None:
                case['graph_descriptor'] = str(desc); case['native_FULL_graph_replayed'] = True
                state['steps'].append(case)
                if state['phase'] == 'full_A':
                    for layer in config['layers']:
                        if (state['phase'],case['batch'],layer) not in state['checked']:
                            ref=state['refs'][case['batch'],layer]
                            self._check(layer,ref['q'],ref['out'],ref['cache'],ref['k'],ref['v'],case)
            return result
        native.run_fullgraph = run

        original_logits = runner.model.compute_logits
        def logits(*args, **kwargs):
            actual = original_logits(*args, **kwargs)
            if actual is None or state['capture'] or state['phase'] not in ('full_A','full_B','eager_teacher'):
                return actual
            batch=state['batch'];starts=batch.query_start_loc_np[:batch.num_reqs+1].tolist()
            indexes=batch.logits_indices.cpu().tolist();positions=batch.positions.cpu().tolist()
            for row,index in enumerate(indexes):
                request=next(i for i in range(batch.num_reqs) if starts[i]<=index<starts[i+1])
                info=self._request_info(request);step=positions[index]-info['prompt_length']+2
                if step < 1:continue
                question=info['question'];key=f'q{question}_s{step}'
                if state['rank']==0:
                    values=actual[row].detach().float().cpu()
                    top=torch.topk(values,2)
                    item={'question':question,'step':step,'top1':int(top.indices[0]),
                        'top1_margin':float(top.values[0]-top.values[1]),'all_finite':bool(torch.isfinite(values).all())}
                    assert item['all_finite']
                    if state['phase']=='full_A':state['baseline_logits'][key]=values
                    else:
                        base=state['baseline_logits'][key];diff=values-base
                        item.update(max_abs_vs_full_A=float(diff.abs().max()),
                            relative_l2_vs_full_A=float(diff.norm()/base.norm().clamp_min(1e-30)),
                            top1_equal_full_A=bool(values.argmax()==base.argmax()))
                    if state['teacher'] is not None:
                        target=state['teacher'][question][step-1]
                        item['teacher_token_logprob_before_forcing']=float(torch.log_softmax(values,0)[target])
                    state['logits'][key]=item
                if state['teacher'] is not None:
                    actual[row].fill_(-torch.inf);actual[row,state['teacher'][question][step-1]]=0
            if state['mode']=='eager':
                case=self._pure_case()
                if case is not None:
                    case['native_FULL_graph_replayed']=False;state['steps'].append(case)
            return actual
        runner.model.compute_logits = logits
        result={'rank':state['rank'],'layers':state['layers'],'deployment':self.deployment_state(),
            'packages':{p:package_version(p) for p in ('torch','vllm','flashinfer-python','flashinfer-cubin')},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_cache_update_untouched':True,'attention_math_untouched':True,
            'no_sparse_executor_or_selector':True,'unit_scales_verified_all_48_layers':True}
        self._save('configuration',result)
        return result

    def _save(self, name, data):
        (self.fp8_probe['output']/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')

    def _request_info(self,index):
        state=self.fp8_probe;batch=state['batch'];rid=batch.req_ids[index]
        if rid not in state['requests']:
            slot=int(batch.idx_mapping_np[index]);runner=self.model_runner
            length=int(runner.req_states.prompt_len.np[slot])
            ids=runner.req_states.all_token_ids.gpu[slot,:length].cpu().tolist()
            sha=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            assert sha in state['config']['prompt_token_hashes']
            state['requests'][rid]={'question':state['config']['prompt_token_hashes'].index(sha),
                'prompt_length':length,'request_id':rid,'prompt_token_sha256':sha}
        return state['requests'][rid]

    def _pure_case(self):
        state=self.fp8_probe
        if state['capture'] or 'batch' not in state:return None
        batch=state['batch'];b=batch.num_reqs
        starts=batch.query_start_loc_np[:b+1].tolist()
        if b not in (1,2,3) or any(z-a!=1 for a,z in zip(starts,starts[1:])):return None
        infos=[self._request_info(i) for i in range(b)]
        positions=batch.num_computed_tokens_np[:b].tolist()
        if any(p<r['prompt_length'] for p,r in zip(positions,infos)):return None
        assert batch.num_tokens_after_padding==b
        return {'phase':state['phase'],'batch':b,'questions':[r['question'] for r in infos],
            'positions':positions,'prompt_lengths':[r['prompt_length'] for r in infos],
            'request_ids':[r['request_id'] for r in infos]}

    @torch.inference_mode()
    def _check(self, layer, query, output, cache, key, value, case):
        state=self.fp8_probe;started=time.perf_counter();b=case['batch']
        state['checked'].add((state['phase'],b,layer))
        assert query.shape==(b,16,128) and query.dtype==torch.bfloat16
        assert output.shape==query.shape and output.dtype==torch.bfloat16
        meta=next(iter(state['attn_metadata'].values()))
        assert meta.causal and not meta.use_cascade
        assert meta.q_data_type_prefill==meta.q_data_type_decode==torch.bfloat16
        assert 'XQA' in str(meta.decode.kernel).upper(), describe_metadata(meta)
        tables=state['tables'][:b]
        assert meta.decode.block_tables.data_ptr()==tables.data_ptr()
        assert meta.decode.seq_lens[:b].cpu().tolist()==[p+1 for p in case['positions']]
        common_pages=state['config']['common_prefix_tokens']//16
        shared=True if b==1 else all(torch.equal(tables[0,:common_pages],tables[r,:common_pages]) for r in range(1,b))
        assert shared
        if b>1:
            assert len(set(tables[:b,common_pages].cpu().tolist()))==b,'Partial prefix page must remain request-private'
        module,original=state['originals'][layer]
        scales=state['layers'][layer]['scales'];scale=state['layers'][layer]['attention_scale']
        assert all(float(getattr(module,k).item())==float(getattr(module,k+'_float'))==v==1.0 for k,v in scales.items())
        expected,lse=fp64_from_actual_fp8_cache(query,cache,tables,case['positions'],scale,scales['_k_scale'],scales['_v_scale'],state['config']['reference_chunk'])
        delta=output.double()-expected
        passed=bool(torch.isclose(output.double(),expected,atol=.006,rtol=.015).all())
        row={**case,'layer':layer,'reference':'FP64 full causal attention over actual FP8 bytes decoded with actual scalar K/V scales',
            'same_quantized_input':True,'atol':.006,'rtol':.015,'passed':passed,
            'max_abs':float(delta.abs().max()),'relative_l2':float(delta.norm()/expected.norm().clamp_min(1e-30)),
            'query_sha256':tensor_sha(query),'output_sha256':tensor_sha(output),'block_table_sha256':tensor_sha(tables),
            'query_strides':list(query.stride()),'cache_strides':list(cache.stride()),
            'actual_query_dtype':str(query.dtype),'backend_query_dtype':str(meta.q_data_type_decode),
            'Q_scale_tensor_equals_host_float_unit':True,
            'query_semantics':'BF16 hook Q equals backend requested BF16 Q; official maybe_quant_query performs no conversion when dtypes match',
            'complete_prefix_private_suffix_and_generated_tail':True,'shared_complete_prefix_pages_verified':shared,
            'full_history_lengths':[p+1 for p in case['positions']],
            'private_generated_lengths':[p-n+1 for p,n in zip(case['positions'],case['prompt_lengths'])],
            'reference_lse_finite':bool(torch.isfinite(lse).all()),'native_lse_available':False,
            'LSE_scope':'XQA O-only API has no native LSE; no LSE qualification claimed',
            'actual_backend':describe_metadata(meta)}
        if state['mode']!='eager':
            eager=torch.empty_like(output)
            original(module,query,None,None,cache,meta,output=eager)
            row['same_input_eager_vs_FULL_max_abs']=float((eager.float()-output.float()).abs().max())
            row['same_input_eager_vs_FULL_passed']=bool(torch.isclose(eager.float(),output.float(),atol=.006,rtol=.015).all())
            passed &= row['same_input_eager_vs_FULL_passed']
        if key is not None and value is not None:
            raw=cache.view(torch.uint8)
            stored=torch.stack([raw[tables[r,p//16].long(),:,p%16,:] for r,p in enumerate(case['positions'])]).view(torch.float8_e4m3fn).float()
            original_kv=torch.cat((key[:b],value[:b]),-1).float()
            dequant=stored*torch.tensor([scales['_k_scale']]*128+[scales['_v_scale']]*128,device=stored.device)
            row['current_token_quantization']={'elements':original_kv.numel(),
                'max_abs_vs_unquantized_KV':float((dequant-original_kv).abs().max()),
                'relative_l2_vs_unquantized_KV':float((dequant-original_kv).norm()/original_kv.norm().clamp_min(1e-30)),
                'unquantized_abs_gt_448_count':int((original_kv.abs()>448).sum()),
                'scope':'descriptive quantization error, not a kernel mismatch or task-quality metric'}
        row['passed']=passed
        row['complete_check_wall_seconds']=time.perf_counter()-started
        state['checks'].append(row);self._save('checks',{'rank':state['rank'],'checks':state['checks']})
        assert passed and row['reference_lse_finite'],row

    @torch.inference_mode()
    def capture_fp8_native_graphs(self):
        from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
        state=self.fp8_probe;runner=self.model_runner;native=runner.cudagraph_manager
        before=torch.cuda.memory_allocated();started=time.perf_counter()
        manager=ModelCudaGraphManager(runner.vllm_config,runner.device,native.cudagraph_mode,
            decode_query_len=runner.decode_query_len,lora_capture_cases=runner.lora_capture_cases,
            varlen_decode=False,ubatch_runner=None)
        state['capture']=True
        try:
            manager.capture(runner.model,runner.model_state,runner.input_buffers,runner.intermediate_tensors,
                runner.block_tables,runner.attn_groups,runner.kv_cache_config,pcp_manager=runner.pcp_manager,
                has_lora=False,use_aux_hidden_state_outputs=runner.use_aux_hidden_state_outputs,
                progress_bar_desc='Capturing instrumented official native FP8 KV graphs')
        finally:state['capture']=False
        state['manager']=manager;torch.cuda.synchronize()
        assert set(state['refs'])=={(b,l) for b in (1,2,3) for l in state['config']['layers']}
        result={'rank':state['rank'],'capture_wall_seconds':time.perf_counter()-started,
            'additional_allocated_bytes':torch.cuda.memory_allocated()-before,
            'persistent_instrument_buffers_bytes':sum(t.numel()*t.element_size() for ref in state['refs'].values() for k,t in ref.items() if k!='cache' and t is not None),
            'graph_descriptors':[str(x) for x in manager.graphs],'capture_calls':state['capture_calls'],
            'official_attention_forward_and_KV_update_unmodified':True,
            'instrumentation':'Extra Q/output/current-KV copies at selected native attention boundaries; original graphs retained; no speed claim'}
        self._save('graph_capture',result);return result

    def set_fp8_phase(self,phase,mode='native',teacher=None):
        assert mode in ('native','eager')
        state=self.fp8_probe;state.update(phase=phase,mode=mode,teacher=teacher,steps=[],metadata={},logits={})
        return {'rank':state['rank'],'phase':phase,'mode':mode,'teacher_forced':teacher is not None}

    def finish_fp8_phase(self):
        state=self.fp8_probe
        record={'rank':state['rank'],'phase':state['phase'],'mode':state['mode'],
            'steps':state['steps'],'metadata':list(state['metadata'].values()),'logits':state['logits'],
            'checks':[r for r in state['checks'] if r['phase']==state['phase']],
            'deployment':self.deployment_state(),'teacher_forced':state['teacher'] is not None,
            'loaded_flashinfer_library_identity':loaded_flashinfer_libraries()}
        self._save(state['phase'],record)
        return {'rank':state['rank'],'phase':state['phase'],'path':str(state['output']/f"{state['phase']}.json"),
            'pure_batches':sorted({r['batch'] for r in state['steps']}),'checks':len(record['checks'])}
