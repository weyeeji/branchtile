"""FastKVzip NVIDIA global-allocation mask reference, not a serving kernel.

One query-agnostic score pass over shared prefill; retain native unequal budgets
across all layers/heads. Original KV storage remains dense and is masked at
attention time. No fake-compression timing is presented as physical acceleration.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

ROOT=Path(__file__).resolve().parents[1]
GATE_SHA='5b3f75dbaada8f7252a5e365262f02b5cbca9149db1f3f4cc411aaeb3dfc1368'
AUTHOR=ROOT/'cache/fastkvzip/fastkvzip/prefill/attention/gate.py'
NVIDIA=ROOT/'cache/fastkvzip/kvpress/kvpress/presses/fastkvzip_press.py'


class Gate(nn.Module):
    """Same low-rank sink gate arithmetic as both pinned upstream classes."""
    def __init__(self,index,input_dim,nhead,ngroup,dtype,output_dim=16,sink=16):
        super().__init__()
        self.index,self.nhead,self.ngroup=index,nhead,ngroup
        self.output_dim,self.sink=output_dim,sink
        self.q_proj=nn.Linear(input_dim,nhead*ngroup*output_dim,bias=True,dtype=dtype)
        self.k_proj=nn.Linear(input_dim,nhead*output_dim,bias=False,dtype=dtype)
        self.q_norm=Qwen3RMSNorm(output_dim);self.k_norm=Qwen3RMSNorm(output_dim)
        self.k_base=nn.Parameter(torch.zeros(nhead,1,sink,output_dim))
        self.b=nn.Parameter(torch.zeros(nhead,1,ngroup,dtype=dtype))
        self.d=math.sqrt(output_dim)

    def forward(self,hidden_states):
        hidden_states=hidden_states.squeeze(0)
        shape=(hidden_states.shape[0],self.nhead,-1,self.output_dim)
        queries=self.q_norm(self.q_proj(hidden_states).view(shape))
        keys=self.k_norm(self.k_proj(hidden_states).view(shape))
        queries=queries.transpose(0,1).transpose(-1,-2);keys=keys.transpose(0,1)
        logit=torch.matmul(keys,queries)/self.d+self.b.unsqueeze(2)
        logit_base=torch.matmul(self.k_base,queries)/self.d
        score=1/(1+torch.exp(logit_base-logit).sum(2,keepdim=True))
        return score.mean(-1).squeeze(-1).unsqueeze(0)


def upstream_class(path,name,methods=None):
    """Execute only the requested class from verified downloaded source.

    This avoids importing unrelated old Transformers adapters or running network
    downloads; numerical reference method bodies themselves remain unchanged.
    """
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    if methods is not None:
        cls=ast.ClassDef(name=name,bases=[],keywords=[],decorator_list=[],
            body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods])
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    ns={'torch':torch,'nn':nn,'math':math,'Qwen3RMSNorm':Qwen3RMSNorm,'PreTrainedModel':nn.Module}
    exec(compile(module,str(path),'exec'),ns)
    return ns[name]


def load_gates(path,device,qualify=False):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==GATE_SHA
    with torch.serialization.safe_globals([(np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),
            np.ndarray,np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType]):
        states=torch.load(path,map_location='cpu',weights_only=True)['module']
    state=states[0];out=state['q_norm.weight'].numel()
    head=state['k_proj.weight'].shape[0]//out
    group=state['q_proj.weight'].shape[0]//state['k_proj.weight'].shape[0]
    hidden=state['q_proj.weight'].shape[1];dtype=state['q_proj.weight'].dtype
    assert len(states)==36 and head==8 and group==4 and out==16 and hidden==2560
    gates=[];authors=[];nvidias=[]
    author_cls=upstream_class(AUTHOR,'Weight') if qualify else None
    nvidia_cls=upstream_class(NVIDIA,'FastKVzipGate') if qualify else None
    for index,state in enumerate(states):
        gate=Gate(index,hidden,head,group,dtype,out,16).to(device)
        gate.load_state_dict(state);gates.append(gate.eval())
        if qualify:
            author=author_cls(index,hidden,out,head,group,dtype,sink=16).to(device)
            nvidia=nvidia_cls(index,hidden,head,group,dtype,out,16).to(device)
            author.load_state_dict(state);nvidia.load_state_dict(state)
            authors.append(author.eval());nvidias.append(nvidia.eval())
    return gates,authors,nvidias


class GateObserver:
    def __init__(self,model,gates,authors=(),nvidias=()):
        self.gates,self.authors,self.nvidias=gates,authors,nvidias
        self.active=False;self.parts=[[] for _ in gates];self.seconds=[];self.checks=[]
        self.handles=[layer.self_attn.register_forward_pre_hook(self.hook,with_kwargs=True)
                      for layer in model.model.layers]

    def hook(self,module,args,kwargs):
        if not self.active:return
        hidden=kwargs['hidden_states'];i=module.layer_idx
        torch.cuda.synchronize();start=time.perf_counter()
        scores=self.gates[i](hidden)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        self.parts[i].append(scores);self.seconds.append({'layer':i,'tokens':hidden.shape[1],'gate_seconds':elapsed})
        if self.authors:
            author=self.authors[i](hidden);nvidia=self.nvidias[i](hidden)
            check={'layer':i,'tokens':hidden.shape[1],'author_equal':torch.equal(scores,author),
                   'nvidia_equal':torch.equal(scores,nvidia),'author_max_abs':float((scores-author).abs().max()),
                   'nvidia_max_abs':float((scores-nvidia).abs().max())}
            self.checks.append(check)
            assert check['author_equal'] and check['nvidia_equal'],check

    def begin(self):
        self.parts=[[] for _ in self.gates];self.seconds=[];self.checks=[];self.active=True

    def end(self):
        self.active=False
        result=torch.stack([torch.cat(parts,dim=-1) for parts in self.parts])
        self.parts=[[] for _ in self.gates]
        return result

    def close(self):
        for handle in self.handles:handle.remove()


def protect_scores(raw):
    scores=raw.clone();n=raw.shape[-1]
    window=max(1,int(n*.02)) if n<32000 else 4096
    scores[...,:4]=1.;scores[...,-window:]=1.
    return scores,window


def global_support(scores,fraction):
    """Exact NVIDIA default compress_post: global allocation, then per-layer k."""
    assert scores.ndim==4 and scores.shape[1]==1
    layers,batch,heads,n=scores.shape
    keep=torch.ones((layers,heads,n),device=scores.device,dtype=torch.bool)
    if fraction<1:
        drop=int(scores.numel()*(1-fraction))
        indices=torch.topk(-scores.reshape(-1),drop).indices
        counts=torch.bincount(indices//(heads*n),minlength=layers).int()
        for layer in range(layers):
            count=counts[layer].cpu()
            gone=torch.topk(-scores[layer].reshape(1,-1),count,dim=1).indices.flatten()
            keep[layer].view(-1)[gone]=False
    return keep


def original_nvidia_support(scores,fraction):
    cls=upstream_class(NVIDIA,'FastKVzipPress',{'compress_post'})
    press=cls();press.compression_ratio=1-fraction;press.layerwise=False
    press.score_val=list(scores.unbind(0))
    modules=[SimpleNamespace(layer_idx=i,config=SimpleNamespace(_attn_implementation='sdpa')) for i in range(scores.shape[0])]
    model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=m) for m in modules]))
    press.compress_post(model)
    keep=torch.ones((scores.shape[0],scores.shape[2],scores.shape[3]),device=scores.device,dtype=torch.bool)
    for i,m in enumerate(modules):
        if hasattr(m,'masked_key_indices'):
            _,heads,positions=m.masked_key_indices;keep[i,heads.to(scores.device),positions.to(scores.device)]=False
    return keep


def author_threshold_case(raw,doc_start,doc_end,fractions=(.1,.25,.5,1.)):
    """Reproduce author post-prefill 16k default window and strict threshold."""
    n=doc_end-doc_start;window=int(.02*n) if n<16000 else 4096
    scores=raw[...,doc_start:doc_end].clone()
    if window:scores[...,-window:]=scores.max()
    rows=[]
    for fraction in fractions:
        if fraction<1:
            sorted_scores=torch.sort(scores.flatten(),descending=True).values
            threshold=sorted_scores[max(int(scores.numel()*fraction)-1,0)].item()
            keep=scores>threshold
        else:
            threshold=0.;keep=torch.ones_like(scores,dtype=torch.bool)
        rows.append({'fraction':fraction,'threshold':threshold,'kept_pairs':int(keep.sum()),
                     'actual_fraction':float(keep.float().mean()),'protected_window_pairs_kept':int(keep[...,-window:].sum()),
                     'target_pairs_below_window_pairs':fraction*n<window})
    return {'document_tokens':n,'author_prefill_chunk_default':16000,'window':window,
            'policy':'Author post-prefill global score>threshold, system prefix outside budget; not optimized chunk-prefill-evict.',
            'rows':rows}


class SharedMaskReference:
    """Mask dense cache with one fixed shared support; preserve all private KV."""
    def __init__(self,model,max_new_tokens=8192):
        self.active=False;self.supports=None;self.keep=[];self.calls=0
        self.max_new_tokens=max_new_tokens;self.seconds=0.
        self.handles=[layer.self_attn.register_forward_pre_hook(self.hook,with_kwargs=True)
                      for layer in model.model.layers]
        self.groups=model.config.num_attention_heads//model.config.num_key_value_heads

    def configure(self,supports,max_private=1024):
        start=time.perf_counter();self.supports=supports;self.calls=0;self.seconds=0.
        self.shared_length=supports.shape[-1]
        self.full=bool(supports.all())
        self.keep=[]
        if not self.full:
            for layer in supports:
                mask=torch.ones((1,layer.shape[0]*self.groups,1,self.shared_length+max_private+self.max_new_tokens),device=layer.device,dtype=torch.bool)
                mask[0,:,0,:self.shared_length]=layer.repeat_interleave(self.groups,0)
                self.keep.append(mask)
        torch.cuda.synchronize()
        return time.perf_counter()-start

    def hook(self,module,args,kwargs):
        if not self.active or self.full:return
        start=time.perf_counter()
        q=kwargs['hidden_states'].shape[1]
        past=kwargs['past_key_values'].get_seq_length(module.layer_idx)
        n=past+q
        assert past>=self.shared_length
        mask=self.keep[module.layer_idx][...,:n]
        assert mask.shape[-1]==n
        existing=kwargs['attention_mask']
        if existing is not None:
            if existing.dtype==torch.bool:mask=mask & existing
            else:mask=torch.where(mask,existing,torch.finfo(existing.dtype).min)
        elif q>1:
            causal=torch.arange(n,device=mask.device)[None,:]<=torch.arange(past,n,device=mask.device)[:,None]
            mask=mask & causal[None,None]
        kwargs['attention_mask']=mask
        self.calls+=1;self.seconds+=time.perf_counter()-start
        return args,kwargs

    def close(self):
        for handle in self.handles:handle.remove()
