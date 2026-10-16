"""HF QC-SnapKV quality reference, not an optimized serving implementation.

Prefill is dense. Each layer's question-suffix observation fixes a support set.
Only document KV is compressed; system prefix, full question suffix, and every
generated private token remain. Cached keys already carry their original RoPE.
Decode position_ids always use the original uncompressed absolute positions.
"""
from __future__ import annotations

import hashlib
import time

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
from selector_baseline import qc_scores, select_support


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def alias_cache(cache, config):
    """Independent append-only containers sharing immutable initial tensors."""
    result = DynamicCache(config=config)
    for source, target in zip(cache.layers,result.layers):
        if type(source) is not DynamicLayer or type(target) is not DynamicLayer:
            raise ValueError("Reference currently supports full-attention DynamicLayer only")
        target.lazy_initialization(source.keys,source.values)
        target.keys, target.values = source.keys, source.values
    return result


def compress_cache(cache, supports, doc_start, doc_end, config):
    if len(cache.layers) != len(supports):
        raise ValueError("One frozen support per layer is required")
    result = DynamicCache(config=config)
    positions = []
    before_lengths, after_lengths = [], []
    for li,(source,target,support) in enumerate(zip(cache.layers,result.layers,supports)):
        if type(source) is not DynamicLayer or type(target) is not DynamicLayer:
            raise ValueError("Full-attention DynamicLayer required")
        k,v = source.keys,source.values
        if k.shape[0] != 1 or k.shape != v.shape:
            raise ValueError("Single-request equal K/V shape required")
        h,n,d = k.shape[1:]
        support = support.to(k.device,dtype=torch.bool)
        if support.shape != (h,doc_end-doc_start) or not 0 <= doc_start < doc_end <= n:
            raise ValueError("Support shape/document range mismatch")
        counts = support.sum(-1)
        if not bool((counts == counts[0]).all()):
            raise ValueError("Every KV head must retain the same budget")
        count = int(counts[0])
        # Boolean indexing preserves original token order within every head.
        selected = torch.arange(doc_end-doc_start,device=k.device)[None].expand(h,-1)[support].reshape(h,count)+doc_start
        ids = torch.cat([torch.arange(doc_start,device=k.device)[None].expand(h,-1),selected,
                         torch.arange(doc_end,n,device=k.device)[None].expand(h,-1)],dim=1)
        index = ids[None,:,:,None].expand(1,h,ids.shape[1],d)
        new_k,new_v = torch.gather(k,2,index),torch.gather(v,2,index)
        target.lazy_initialization(new_k,new_v)
        target.keys,target.values = new_k,new_v
        positions.append(ids)
        before_lengths.append(n);after_lengths.append(ids.shape[1])
    if len(set(before_lengths)) != 1 or len(set(after_lengths)) != 1:
        raise ValueError("All layers must use the same physical cache length")
    return result,positions


def rotate_half(x):
    a,b=x.chunk(2,-1)
    return torch.cat((-b,a),dim=-1)


class QuestionObserver:
    """Recompute exact Q/K projections solely to instrument suffix selection."""
    def __init__(self,model,doc_start,doc_end):
        self.model,self.doc_start,self.doc_end=model,doc_start,doc_end
        self.active=False
        self.scores={}
        self.times=[]
        self.handles=[layer.self_attn.register_forward_pre_hook(self.hook,with_kwargs=True)
                      for layer in model.model.layers]

    def hook(self,module,args,kwargs):
        if not self.active:
            return
        hidden=kwargs.get("hidden_states",args[0] if args else None)
        torch.cuda.synchronize(hidden.device)
        start=time.perf_counter()
        shape=(*hidden.shape[:-1],-1,module.head_dim)
        q=module.q_norm(module.q_proj(hidden).view(shape)).transpose(1,2)
        k=module.k_norm(module.k_proj(hidden).view(shape)).transpose(1,2)
        cos,sin=(x.unsqueeze(1) for x in kwargs["position_embeddings"])
        q=q*cos+rotate_half(q)*sin
        k=k*cos+rotate_half(k)*sin
        layer=kwargs["past_key_values"].layers[module.layer_idx]
        past=layer.keys.shape[2]
        full_k=torch.cat([layer.keys,k],dim=2)
        window=min(32,q.shape[2])
        positions=torch.arange(past+q.shape[2]-window,past+q.shape[2],device=q.device)
        scores=qc_scores(q[:,:,-window:],full_k,positions,self.doc_start,self.doc_end)
        self.scores[module.layer_idx]=scores.cpu()
        torch.cuda.synchronize(hidden.device)
        self.times.append({"layer":module.layer_idx,"observation_tokens":window,
                           "full_causal_history_tokens":full_k.shape[2],
                           "observer_seconds":time.perf_counter()-start})

    def begin(self):
        self.scores={};self.times=[];self.active=True

    def end(self):
        self.active=False
        assert len(self.scores)==len(self.model.model.layers)

    def close(self):
        for handle in self.handles:
            handle.remove()

    def supports(self,budget,device):
        return [select_support(self.scores[i].to(device),budget)
                for i in range(len(self.model.model.layers))]


def sample_token(logits,temperature,top_p,generator):
    if temperature==0:
        return int(logits.argmax())
    scores,ids=(logits.float()/temperature).sort(descending=True)
    probs=torch.softmax(scores,-1)
    remove=probs.cumsum(-1)-probs >= top_p
    probs=probs.masked_fill(remove,0)
    slot=torch.multinomial(probs,1,generator=generator)
    return int(ids[slot])


@torch.inference_mode()
def decode(model,cache,first_logits,prompt_length,max_new_tokens,eos,
           *,temperature=0.,top_p=.95,seed=20261016,keep_logits=False,progress=None):
    device=first_logits.device
    generator=torch.Generator(device=device).manual_seed(seed)
    generated=[];traces=[];positions=[]
    logits=first_logits
    physical_start=cache.get_seq_length()
    start=time.perf_counter()
    natural_eos=False
    for step in range(max_new_tokens):
        if keep_logits:
            traces.append(logits.float().cpu())
        token=sample_token(logits,temperature,top_p,generator)
        generated.append(token)
        if token in eos:
            natural_eos=True
            break
        if step+1==max_new_tokens:
            break
        absolute_position=prompt_length+step
        position_ids=torch.tensor([[absolute_position]],device=device)
        out=model(input_ids=torch.tensor([[token]],device=device),position_ids=position_ids,
                  past_key_values=cache,use_cache=True,logits_to_keep=1)
        logits=out.logits[0,-1]
        positions.append(absolute_position)
        lengths=[layer.get_seq_length() for layer in cache.layers]
        if len(set(lengths))!=1 or lengths[0]!=physical_start+len(positions):
            raise RuntimeError("Layer cache length diverged during private-token append")
        if progress is not None and len(generated)%64==0:
            progress(generated,positions,time.perf_counter()-start)
    torch.cuda.synchronize(device)
    return {"generated_ids":generated,"generated_tokens":len(generated),
            "natural_eos":natural_eos,"hit_cap":not natural_eos and len(generated)==max_new_tokens,
            "decode_seconds":time.perf_counter()-start,
            "absolute_decode_positions":positions,"initial_physical_cache_length":physical_start,
            "final_physical_cache_length":cache.get_seq_length(),
            "logit_traces":traces}
