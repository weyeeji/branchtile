"""Bounded A2 FastKVzip-NVIDIA global-mask qualification and natural generation."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from capture_model import make_inputs
from sparse_reference import alias_cache,decode,tensor_hash
from fastkvzip_reference import (AUTHOR,NVIDIA,GATE_SHA,GateObserver,SharedMaskReference,
    author_threshold_case,global_support,load_gates,original_nvidia_support,protect_scores)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');temporary.replace(path)


@torch.inference_mode()
def suffix(model,token_ids,shared_cache,shared_length):
    cache=alias_cache(shared_cache,model.config)
    ids=torch.tensor(token_ids[shared_length:],device='cuda')[None]
    output=model(input_ids=ids,position_ids=torch.arange(shared_length,len(token_ids),device='cuda')[None],
                 past_key_values=cache,use_cache=True,logits_to_keep=1)
    return cache,output.logits[0,-1]


def support_stats(supports,ds,de,window):
    counts=supports.sum(-1)
    return {'retained_shared_pairs':int(counts.sum()),'total_shared_pairs':supports.numel(),
        'actual_shared_fraction':float(supports.float().mean()),'layer_head_counts':counts.cpu().tolist(),
        'minimum_head_count':int(counts.min()),'maximum_head_count':int(counts.max()),
        'unequal_head_budgets':bool((counts!=counts.flatten()[0]).any()),
        'retained_system_pairs':int(supports[...,:ds].sum()),'total_system_pairs':supports.shape[0]*supports.shape[1]*ds,
        'retained_document_pairs':int(supports[...,ds:de].sum()),'total_document_pairs':supports.shape[0]*supports.shape[1]*(de-ds),
        'first4_preserved':bool(supports[...,:4].all()),'local_window_preserved':bool(supports[...,-window:].all()),
        'support_sha256':tensor_hash(supports)}


@torch.inference_mode()
def qualification(model,inputs,shared_cache,masker,scores,raw,ds,de,eos,args):
    report={'status':'running','gate_score_comparisons':'Every shared-prefill chunk and layer against both upstream gate classes',
            'native_support_checks':[],'author_postprefill_threshold_case':author_threshold_case(raw,ds,de,args.budgets)}
    for fraction in args.budgets:
        mask=global_support(scores,fraction);native=original_nvidia_support(scores,fraction)
        assert torch.equal(mask,native),'Native allocation/mask differs'
        report['native_support_checks'].append({'fraction':fraction,'equal':True,
             'symmetric_difference':int((mask!=native).sum()),'retained_pairs':int(mask.sum())})
    # No effect at full budget: identical original cache, same backend/path,
    # identical position IDs, every layer output and logit at every sampled step.
    tokens=inputs[0];layer_runs=[];generations=[]
    for full_mask in (False,True):
        traces=[]
        handles=[layer.self_attn.register_forward_hook(
            lambda module,args,output,li=li:traces.append((li,output[0].detach().cpu())))
            for li,layer in enumerate(model.model.layers)]
        masker.active=full_mask
        if full_mask:masker.configure(global_support(scores,1.))
        cache,first=suffix(model,tokens,shared_cache,de)
        generated=decode(model,cache,first,len(tokens),args.max_new_tokens,eos,
             temperature=args.temperature,top_p=args.top_p,seed=args.seed,keep_logits=True)
        for h in handles:h.remove()
        layer_runs.append(traces);generations.append(generated)
    dense,full=generations
    dense_logits=torch.stack(dense.pop('logit_traces'));full_logits=torch.stack(full.pop('logit_traces'))
    assert dense['generated_ids']==full['generated_ids'] and torch.equal(dense_logits,full_logits)
    assert len(layer_runs[0])==len(layer_runs[1])
    layer_error=[]
    for (li,a),(lj,b) in zip(*layer_runs):
        assert li==lj and torch.equal(a,b)
        layer_error.append({'layer':li,'max_abs':float((a-b).abs().max())})
    supports=global_support(scores,.25);masker.configure(supports);masker.active=True
    cache,first=suffix(model,tokens,shared_cache,de)
    t0,t1=dense['generated_ids'][:2]
    positions=torch.tensor([[len(tokens),len(tokens)+1]],device='cuda')
    checks=[]
    for future in [t1,(t1+1)%model.config.vocab_size]:
        out=model(input_ids=torch.tensor([[t0,future]],device='cuda'),position_ids=positions,
             past_key_values=alias_cache(cache,model.config),use_cache=True,logits_to_keep=2)
        checks.append(out.logits[0,0].float().cpu())
    assert torch.equal(*checks),'Future private token leaked'
    assert all(layer.get_seq_length()==de for layer in shared_cache.layers),'Question modified shared history'
    trace=Path(args.output).with_suffix('.qualification_logits.pt')
    torch.save({'dense':dense_logits,'fullbudget':full_logits,'future_a':checks[0],'future_b':checks[1]},trace)
    report.update(status='passed',fullbudget_path='All-keep support bypasses added mask; unchanged original attention path',
        fullbudget_logits_equal=True,fullbudget_tokens_equal=True,fullbudget_layer_outputs=layer_error,
        logit_max_abs_per_step=(dense_logits-full_logits).abs().amax(-1).tolist(),
        future_causality_max_abs=float((checks[0]-checks[1]).abs().max()),
        shared_history_unchanged_after_independent_questions=True,
        trace_path=str(trace),trace_sha256=sha(trace),dense=dense,full_budget=full)
    masker.active=False
    return report


@torch.inference_mode()
def run(args,result):
    identity=json.loads(Path(args.weights_manifest).read_text())
    assert identity['complete'] and Path(identity['path']).resolve()==Path(args.model).resolve()
    for shard in identity['verified_shards']:assert (Path(args.model)/shard['file']).stat().st_size==shard['bytes']
    result.update(model_revision=identity['revision'],weights_manifest_sha256=sha(args.weights_manifest))
    start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.bfloat16,device_map='cuda',
            attn_implementation='sdpa',local_files_only=True).eval()
    torch.cuda.synchronize();result['model_initialization_seconds']=time.perf_counter()-start
    start=time.perf_counter()
    gates,authors,nvidias=load_gates(args.gate,'cuda',args.mode=='qualify')
    torch.cuda.synchronize();result['gate_initialization_seconds']=time.perf_counter()-start
    observer=GateObserver(model,gates,authors,nvidias)
    masker=SharedMaskReference(model,max_new_tokens=args.max_new_tokens)
    eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
    manifest=json.loads(Path(args.manifest).read_text())
    for gi in args.groups:
        gold=manifest['groups'][gi];inputs,ds,de=make_inputs(tokenizer,gold)
        assert all(ids[:de]==inputs[0][:de] for ids in inputs)
        assert max(map(len,inputs))+args.max_new_tokens<=model.config.max_position_embeddings
        record={'group_index':gi,'document_id':gold['document_id'],'document_sha256':gold['document_sha256'],
            'document_range':[ds,de],'shared_prefill_tokens':de,'original_natural_B':len(inputs),'questions':[]}
        result['groups'].append(record);masker.active=False
        shared_cache=DynamicCache(config=model.config);prefix=torch.tensor(inputs[0][:de],device='cuda')[None]
        observer.begin();start=time.perf_counter()
        for begin in range(0,de,args.prefill_chunk):
            end=min(de,begin+args.prefill_chunk)
            model.model(input_ids=prefix[:,begin:end],position_ids=torch.arange(begin,end,device='cuda')[None],
                        past_key_values=shared_cache,use_cache=True)
        torch.cuda.synchronize();record['shared_document_prefill_including_gate_seconds']=time.perf_counter()-start
        raw=observer.end();assert raw.shape==(36,1,8,de)
        record['gate_seconds']=sum(c['gate_seconds'] for c in observer.seconds)
        record['gate_costs']=observer.seconds;record['gate_score_qualification']=observer.checks
        start=time.perf_counter();scores,window=protect_scores(raw);torch.cuda.synchronize()
        record['protection_prepare_seconds']=time.perf_counter()-start;record['protection_window']=window
        record['score_sha256']=tensor_hash(scores);record['raw_score_sha256']=tensor_hash(raw)
        write(args.output,result)
        if args.mode=='qualify':
            record['qualification']=qualification(model,inputs,shared_cache,masker,scores,raw,ds,de,eos,args)
        else:
            for qi in range(len(inputs)):
                record['questions'].append({'question_index':qi,'question_sha256':gold['questions'][qi]['question_sha256'],
                    'prompt_tokens':len(inputs[qi]),'input_ids_sha256':tensor_hash(torch.tensor(inputs[qi],dtype=torch.int64)),
                    'status':'running','variants':[]})
            record['shared_supports']=[]
            for fraction in args.budgets:
                torch.cuda.synchronize();start=time.perf_counter();supports=global_support(scores,fraction)
                torch.cuda.synchronize();select_time=time.perf_counter()-start
                support_record={'fraction':fraction,'global_allocation_seconds':select_time,**support_stats(supports,ds,de,window)}
                assert support_record['first4_preserved'] and support_record['local_window_preserved']
                mask_path=Path(args.output).with_name(Path(args.output).stem+f'_g{gi}_f{str(fraction).replace(".","p")}.supports.pt')
                start=time.perf_counter();torch.save({'supports':supports.cpu(),'scores':scores.cpu(),'raw_scores':raw.cpu()},mask_path)
                support_record.update(mask_artifact=str(mask_path),mask_artifact_sha256=sha(mask_path),
                                      mask_export_seconds=time.perf_counter()-start)
                support_record['mask_index_preparation_seconds']=masker.configure(supports,max_private=max(map(len,inputs))-de+8)
                record['shared_supports'].append(support_record);masker.active=True
                for qi,qrecord in enumerate(record['questions']):
                    variant={'fraction':fraction,'status':'prefilling','seed':args.seed+qi,
                        'shared_support_sha256':support_record['support_sha256'],'shared_history_reused':True,
                        'physical_compression':False,'physical_gather_seconds':0.0}
                    qrecord['variants'].append(variant);write(args.output,result)
                    start=time.perf_counter();before_mask=masker.seconds
                    cache,first=suffix(model,inputs[qi],shared_cache,de)
                    torch.cuda.synchronize();variant['question_prefill_seconds']=time.perf_counter()-start
                    variant['question_mask_prepare_host_seconds']=masker.seconds-before_mask
                    def progress(tokens,positions,elapsed):
                        variant.update(generated_ids=tokens.copy(),generated_tokens=len(tokens),decode_seconds_so_far=elapsed)
                        write(args.output,result)
                    before_mask=masker.seconds
                    generated=decode(model,cache,first,len(inputs[qi]),args.max_new_tokens,eos,
                        temperature=args.temperature,top_p=args.top_p,seed=args.seed+qi,progress=progress)
                    generated.pop('logit_traces')
                    variant.update(generated,generated_text=tokenizer.decode(generated['generated_ids'],skip_special_tokens=True),
                        decode_mask_prepare_host_seconds=masker.seconds-before_mask,status='completed')
                    assert all(layer.get_seq_length()==de for layer in shared_cache.layers)
                    assert generated['final_physical_cache_length']==len(inputs[qi])+len(generated['absolute_decode_positions'])
                    write(args.output,result)
                    print(json.dumps({'group':gi,'question':qi,'fraction':fraction,'tokens':generated['generated_tokens'],'eos':generated['natural_eos']}),flush=True)
                    del cache,first;gc.collect()
            for q in record['questions']:q['status']='completed'
        record['status']='completed';write(args.output,result)
        masker.active=False;del shared_cache,raw,scores;gc.collect();torch.cuda.empty_cache()
    observer.close();masker.close()
    result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['qualify','natural'],required=True)
    p.add_argument('--model',default='data/models/Qwen3-4B-Instruct-2507')
    p.add_argument('--weights-manifest',default='results/Qwen3-4B-Instruct-2507_download.json')
    p.add_argument('--gate',default='data/models/FastKVzip-Qwen3-4B-Instruct-2507/q4_dim16_sink16.pt')
    p.add_argument('--manifest',default='data/manifests/qmsum_train_exploratory8.json')
    p.add_argument('--groups',default='3');p.add_argument('--budgets',default='0.1,0.25,0.5,1.0')
    p.add_argument('--max-new-tokens',type=int,default=8192);p.add_argument('--prefill-chunk',type=int,default=1024)
    p.add_argument('--temperature',type=float,default=.6);p.add_argument('--top-p',type=float,default=.95)
    p.add_argument('--seed',type=int,default=20261016);p.add_argument('--output',required=True)
    args=p.parse_args();assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    args.groups=list(map(int,args.groups.split(',')));args.budgets=list(map(float,args.budgets.split(',')))
    if args.mode=='qualify':assert args.max_new_tokens<=32
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(args.seed)
    def stop(signum,frame):raise TimeoutError(f'External signal {signum}; partial outputs retained')
    signal.signal(signal.SIGTERM,stop)
    result={'schema':1,'status':'running','mode':args.mode,'method':'FastKVzip-NVIDIA global mask reference',
        'uid':os.geteuid(),'pid':os.getpid(),'virtual_environment':sys.prefix,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'start_epoch':time.time(),'torch':torch.__version__,'model_path':args.model,'gate_path':args.gate,'gate_sha256':GATE_SHA,
        'manifest':args.manifest,'manifest_sha256':sha(args.manifest),'script_sha256':sha(__file__),
        'core_sha256':sha(ROOT/'src/fastkvzip_reference.py'),'original_author_gate_source_sha256':sha(AUTHOR),
        'original_nvidia_source_sha256':sha(NVIDIA),'generation':{'temperature':args.temperature,'top_p':args.top_p,
            'top_k':None,'seed':args.seed,'seed_rule':'seed + question_index','max_new_tokens':args.max_new_tokens,'natural_eos':True},
        'semantics':{'shared_scope':'Complete system prefix + document, no question or answer enters scoring',
            'allocation':'NVIDIA default global bottom-k then per-layer across-head bottom-k; unequal head/layer budgets',
            'protected':'First4 shared-prefill tokens; final max(1,floor(2%*N)) if N<32000 else4096, scores set to1.0',
            'budget_denominator':'All36 layers *8 KV heads *shared-prefill tokens, including system prefix',
            'private_history':'Every question and generated token retained; original positions/causality',
            'physical_cache':'Dense reference with exact per-head masks, no physical compression or serving speed claim',
            'chunked_prefill':'Dense 1024-token chunks; no optimized author eviction-during-prefill path'},'groups':[]}
    write(args.output,result)
    try:
        run(args,result);result.update(status='completed',end_epoch=time.time());write(args.output,result)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error),end_epoch=time.time());write(args.output,result);raise


if __name__=='__main__':main()
