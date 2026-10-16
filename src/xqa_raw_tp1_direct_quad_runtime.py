"""Independent Hq32/Hkv8 TP1 direct-quad adapter; no TP2 policy/qualification inherited."""
import torch

class RawTP1DirectQuadPlan:
    def __init__(self,modules,q,cache,table,seq_lens,prefix,workspace=None,raw_state=None):
        assert q.dtype==cache.dtype==torch.bfloat16 and q.shape[1:]==(32,128) and q.is_contiguous()
        assert cache.shape[1:]==(8,16,256) and table.dtype==torch.int32 and table.is_contiguous()
        self.b=len(q);assert 1<=self.b<=8 and table.shape[0]==self.b
        assert prefix>0 and prefix%16==0
        self.modules,self.q,self.cache,self.table,self.prefix=modules,q,cache,table,prefix
        self.groups,self.hkv,self.max_len=(self.b+3)//4,8,table.shape[1]*16
        self.sm=torch.cuda.get_device_properties(q.device).multi_processor_count
        self.splits=min(max(1,self.sm//(self.b*self.hkv)),(self.max_len+255)//256)
        self.prefix_sm,self.tail_sm=self.splits*self.groups*self.hkv,self.splits*self.b*self.hkv
        self.lengths=torch.empty((self.b,1),device=q.device,dtype=torch.uint32)
        if workspace is not None:assert workspace.dtype==torch.uint8 and workspace.device==q.device and workspace.is_contiguous() and workspace.numel()>=64*1024*1024
        self.workspace=torch.zeros(64*1024*1024,device=q.device,dtype=torch.uint8) if workspace is None else workspace
        self.sem,self.scratch=self.workspace[:8*1024*1024],self.workspace[8*1024*1024:]
        count=self.groups*self.hkv*self.splits*4*32*35
        if raw_state is not None:assert raw_state.dtype==torch.float32 and raw_state.device==q.device and raw_state.is_contiguous() and raw_state.numel()>=count
        self.state=torch.empty(count,device=q.device,dtype=torch.float32) if raw_state is None else raw_state[:count]
        self.output=torch.empty_like(q)
        # Prefix output is not written in phase1, but the host ABI checks its shape.
        self.unused=torch.empty((self.groups,1,self.hkv*16,128),device=q.device,dtype=q.dtype)
        self.k,self.v=cache[...,:128],cache[...,128:]
        self.wk,self.wv=self.k.transpose(1,2),self.v.transpose(1,2)
        self.prepare(seq_lens)

    def prepare(self,seq_lens):
        assert len(seq_lens)==self.b and all(self.prefix<x<=self.max_len for x in seq_lens)
        for start in range(0,self.b,4):assert bool((self.table[start:start+4,:self.prefix//16]==self.table[start,:self.prefix//16]).all())
        self.lengths.copy_(torch.tensor(seq_lens,device=self.q.device,dtype=torch.uint32).reshape(self.b,1))

    def call(self,group,q,o,table,lengths,batch,phase,state=None):
        self.modules[group].xqa_wrapper(False,self.splits*batch*self.hkv,self.hkv,0,1.,None,o,1.,q,
            None,self.wk,self.wv,None,None,table,self.max_len,lengths,batch,1.,None,1,None,None,
            self.sem,self.scratch,False,self.state if state is None else state,self.prefix//256,phase)

    def run(self):
        self.call(16,self.q.unsqueeze(1),self.unused,self.table,self.lengths,self.groups,1)
        self.call(4,self.q.unsqueeze(1),self.output.unsqueeze(1),self.table,self.lengths,self.b,3)
        return self.output
