"""A4 fixed-mask QC-SnapKV natural-generation quality reference.

This Transformers implementation physically gathers per-head KV and instruments
all layer selectors. Its wall times are reference costs, never vLLM application
speedups. Short qualification runs are explicitly excluded from quality claims.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.cache_utils import DynamicCache
from capture_model import make_inputs
from sparse_reference import QuestionObserver,alias_cache,compress_cache,decode,tensor_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,result):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    temporary.replace(path)


def measure_compression(cache,observer,budget,ds,de,model):
    torch.cuda.synchronize();start=time.perf_counter()
    supports=observer.supports(budget,"cuda")
    torch.cuda.synchronize();selected=time.perf_counter()
    compressed,positions=compress_cache(cache,supports,ds,de,model.config)
    torch.cuda.synchronize();end=time.perf_counter()
    audit={"budget_tokens":budget,"support_sha256":[tensor_hash(s) for s in supports],
           "position_map_sha256":[tensor_hash(p) for p in positions],
           "topk_and_support_seconds":selected-start,"gather_seconds":end-selected,
           "physical_cache_length":compressed.get_seq_length(),
           "all_layers_equal_physical_length":len({x.get_seq_length() for x in compressed.layers})==1,
           "system_prefix_retained":True,"entire_private_suffix_retained":True,
           "sorted_original_positions":all(bool((p[:,1:]>p[:,:-1]).all()) for p in positions),
           "head_supports_can_differ":any(not torch.equal(s[0],s[h]) for s in supports for h in range(1,s.shape[0]))}
    return compressed,supports,positions,audit


@torch.inference_mode()
def qualify(model,cache,observer,first_logits,ids,ds,de,eos,args):
    full,supports,positions,audit=measure_compression(cache,observer,de-ds,ds,de,model)
    identity=all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
                 for a,b in zip(cache.layers,full.layers))
    assert identity,"Full-budget gather changed source KV"
    dense=decode(model,alias_cache(cache,model.config),first_logits,len(ids),args.max_new_tokens,eos,
                 temperature=args.temperature,top_p=args.top_p,seed=args.seed,keep_logits=True)
    gathered=decode(model,full,first_logits,len(ids),args.max_new_tokens,eos,
                    temperature=args.temperature,top_p=args.top_p,seed=args.seed,keep_logits=True)
    a,b=torch.stack(dense.pop("logit_traces")),torch.stack(gathered.pop("logit_traces"))
    errors=(a-b).abs().amax(-1).tolist()
    assert dense["generated_ids"]==gathered["generated_ids"],"Full-budget tokens differ"
    assert torch.equal(a,b),f"Full-budget logits changed, maximum difference {max(errors)}"
    sparse,_,_,sparse_audit=measure_compression(cache,observer,max(36,math.ceil((de-ds)*.25)),ds,de,model)
    # In a two-token continuation, changing only the future input must leave
    # the first position's logits unchanged after physical cache compaction.
    token=dense["generated_ids"][0]
    next_token=dense["generated_ids"][1] if len(dense["generated_ids"])>1 else (token+1)%model.config.vocab_size
    position_ids=torch.tensor([[len(ids),len(ids)+1]],device="cuda")
    outputs=[]
    for future in (next_token,(next_token+1)%model.config.vocab_size):
        out=model(input_ids=torch.tensor([[token,future]],device="cuda"),position_ids=position_ids,
                  past_key_values=alias_cache(sparse,model.config),use_cache=True,logits_to_keep=2)
        outputs.append(out.logits[0,0].float().cpu())
    future_error=float((outputs[0]-outputs[1]).abs().max())
    assert torch.equal(outputs[0],outputs[1]),f"Future token leaked into prior logits: {future_error}"
    assert cache.get_seq_length()==len(ids),"Independent branch mutated the dense source cache"
    trace_path=Path(args.output).with_suffix(".qualification_logits.pt")
    torch.save({"dense":a,"full_budget":b,"future_a":outputs[0],"future_b":outputs[1]},trace_path)
    return {"status":"passed","scope":"Implementation qualification only; short capped output is not quality evidence",
            "full_budget_gather_bitwise_identity":identity,"full_budget_logit_max_abs_per_step":errors,
            "full_budget_tokens_equal":True,"future_causality_max_abs":future_error,
            "absolute_decode_positions_correct":dense["absolute_decode_positions"]==list(range(len(ids),len(ids)+len(dense["absolute_decode_positions"]))),
            "full_budget_audit":audit,"quarter_budget_causality_audit":sparse_audit,
            "dense":dense,"full_budget":gathered,
            "trace_artifact":str(trace_path),"trace_sha256":digest(trace_path)}


@torch.inference_mode()
def run(args,result):
    model_path=Path(args.model)
    identity=json.loads(Path(args.weights_manifest).read_text())
    assert identity["complete"] and Path(identity["path"]).resolve()==model_path.resolve()
    for shard in identity["verified_shards"]:
        assert (model_path/shard["file"]).stat().st_size==shard["bytes"]
    result.update(model_path=str(model_path),model_revision=identity["revision"],
                  weights_manifest=args.weights_manifest,weights_manifest_sha256=digest(args.weights_manifest),
                  model_config_sha256=digest(model_path/"config.json"))
    start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,dtype=torch.bfloat16,
                device_map="cuda",attn_implementation="sdpa",local_files_only=True).eval()
    assert all(p.device.type=="cuda" for p in model.parameters())
    torch.cuda.synchronize()
    result["model_initialization_seconds"]=time.perf_counter()-start
    result["model_config"]={"layers":model.config.num_hidden_layers,"query_heads":model.config.num_attention_heads,
                            "kv_heads":model.config.num_key_value_heads,"head_dim":model.config.head_dim,
                            "max_position_embeddings":model.config.max_position_embeddings}
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos)
    manifest=json.loads(Path(args.manifest).read_text())
    for gi in args.groups:
        group=manifest["groups"][gi]
        inputs,ds,de=make_inputs(tokenizer,group)
        assert max(map(len,inputs))+args.max_new_tokens<=model.config.max_position_embeddings
        record={"group_index":gi,"document_id":group["document_id"],"document_sha256":group["document_sha256"],
                "document_range":[ds,de],"original_natural_B":len(inputs),"questions":[]}
        result["groups"].append(record)
        observer=QuestionObserver(model,ds,de)
        prefix_cache=DynamicCache(config=model.config)
        prefix=torch.tensor(inputs[0][:de],device="cuda")[None]
        start=time.perf_counter()
        for begin in range(0,de,args.prefill_chunk):
            end=min(de,begin+args.prefill_chunk)
            model.model(input_ids=prefix[:,begin:end],position_ids=torch.arange(begin,end,device="cuda")[None],
                        past_key_values=prefix_cache,use_cache=True)
        torch.cuda.synchronize()
        record["dense_prefix_prefill_seconds"]=time.perf_counter()-start
        write(args.output,result)
        for qi in args.questions:
            assert qi<len(inputs)
            ids=inputs[qi]
            qrecord={"question_index":qi,"question_sha256":group["questions"][qi]["question_sha256"],
                     "prompt_tokens":len(ids),"input_ids_sha256":tensor_hash(torch.tensor(ids,dtype=torch.int64)),
                     "status":"prefilling","variants":[]}
            record["questions"].append(qrecord)
            cache=alias_cache(prefix_cache,model.config)
            observer.begin();start=time.perf_counter()
            out=model(input_ids=torch.tensor(ids[de:],device="cuda")[None],
                      position_ids=torch.arange(de,len(ids),device="cuda")[None],
                      past_key_values=cache,use_cache=True,logits_to_keep=1)
            observer.end();torch.cuda.synchronize()
            qrecord["question_prefill_including_instrumented_selector_seconds"]=time.perf_counter()-start
            qrecord["selector_observer_costs"]=observer.times
            qrecord["selector_observer_seconds"]=sum(x["observer_seconds"] for x in observer.times)
            first_logits=out.logits[0,-1]
            if args.mode=="qualify":
                qrecord["qualification"]=qualify(model,cache,observer,first_logits,ids,ds,de,eos,args)
            else:
                for fraction in args.budgets:
                    budget=min(de-ds,max(36,math.ceil((de-ds)*fraction)))
                    variant={"fraction":fraction,"status":"selecting","budget_tokens":budget}
                    qrecord["variants"].append(variant)
                    write(args.output,result)
                    compact,supports,positions,audit=measure_compression(cache,observer,budget,ds,de,model)
                    variant.update(audit,status="generating",seed=args.seed+qi)
                    mask_path=Path(args.output).with_suffix("").with_name(Path(args.output).stem+f"_g{gi}_q{qi}_b{budget}.masks.pt")
                    torch.save({"supports":[s.cpu() for s in supports],"scores":observer.scores,
                                "document_range":[ds,de],"budget":budget},mask_path)
                    variant.update(mask_artifact=str(mask_path),mask_artifact_sha256=digest(mask_path))
                    def progress(tokens,absolute_positions,elapsed):
                        variant.update(generated_ids=tokens.copy(),generated_tokens=len(tokens),
                                       decode_seconds_so_far=elapsed,last_absolute_position=absolute_positions[-1])
                        write(args.output,result)
                    generated=decode(model,compact,first_logits,len(ids),args.max_new_tokens,eos,
                        temperature=args.temperature,top_p=args.top_p,seed=args.seed+qi,progress=progress)
                    generated.pop("logit_traces")
                    variant.update(generated,status="completed",generated_text=tokenizer.decode(generated["generated_ids"],skip_special_tokens=True))
                    write(args.output,result)
                    print(json.dumps({"group":gi,"question":qi,"fraction":fraction,
                                      "tokens":generated["generated_tokens"],"eos":generated["natural_eos"]}),flush=True)
                    del compact,supports,positions
            qrecord["status"]="completed"
            write(args.output,result)
            del cache,out
            gc.collect()
        observer.close()
        del prefix_cache,observer
        gc.collect();torch.cuda.empty_cache()
    result["peak_cuda_allocated_bytes"]=torch.cuda.max_memory_allocated()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["qualify","natural"],required=True)
    p.add_argument("--model",default="data/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--weights-manifest",default="results/Qwen3-4B-Instruct-2507_download.json")
    p.add_argument("--manifest",default="data/manifests/qmsum_train_exploratory8.json")
    p.add_argument("--groups",default="0")
    p.add_argument("--questions",default="0")
    p.add_argument("--budgets",default="0.1,0.25,0.5,1.0")
    p.add_argument("--max-new-tokens",type=int,default=8192)
    p.add_argument("--prefill-chunk",type=int,default=1024)
    p.add_argument("--seed",type=int,default=20261016)
    p.add_argument("--temperature",type=float,default=.6)
    p.add_argument("--top-p",type=float,default=.95)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    args.groups=list(map(int,args.groups.split(",")))
    args.questions=list(map(int,args.questions.split(",")))
    args.budgets=list(map(float,args.budgets.split(",")))
    assert all(0<f<=1 for f in args.budgets)
    if args.mode=="qualify":
        assert args.max_new_tokens<=32,"Qualification must not masquerade as a quality run"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(args.seed)
    def interrupted(signum,frame):
        raise TimeoutError(f"External signal {signum}; partial variants retained")
    signal.signal(signal.SIGTERM,interrupted)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    result={"schema":1,"status":"running","mode":args.mode,"uid":os.geteuid(),"pid":os.getpid(),
            "virtual_environment":sys.prefix,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
            "start_epoch":time.time(),"torch":torch.__version__,"groups":[],
            "manifest":args.manifest,"manifest_sha256":digest(args.manifest),
            "script_sha256":digest(__file__),"core_sha256":digest(ROOT/"src/sparse_reference.py"),
            "generation":{"temperature":args.temperature,"top_p":args.top_p,"seed":args.seed,
                          "natural_eos":True,"min_tokens":None,"max_new_tokens":args.max_new_tokens},
            "selector":{"window":32,"pool":7,"normalization":"full causal original history",
                        "precision":"FP32 products/reductions from BF16 post-RoPE Q/K",
                        "protected":"document first4/last32 within budget","stable_tie":"lower original position",
                        "fixed_after_question_prefill":True,"never_refresh":True},
            "scope":"HF physical per-head gather quality reference; no production performance or candidate novelty claim",
            "qualification_is_quality_evidence":False}
    write(output,result)
    try:
        run(args,result)
        result.update(status="completed",end_epoch=time.time())
        write(output,result)
    except BaseException as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error),end_epoch=time.time())
        write(output,result)
        raise


if __name__=="__main__":
    main()
