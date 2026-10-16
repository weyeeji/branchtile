"""Minimal even-batch GQA4/Hkv4 plan; separate geometry, no inherited dispatch."""
import torch


class RawGQA4PairPlan:
    def __init__(self, modules, q, cache, table, seq_lens, prefix):
        assert q.dtype == cache.dtype == torch.bfloat16
        self.b = len(q)
        assert self.b % 2 == 0 and q.shape[1:] == (16,128) and q.is_contiguous()
        assert cache.shape[1:] == (4,16,256) and table.dtype == torch.int32 and table.is_contiguous()
        assert prefix % 16 == 0 and prefix > 0
        self.modules,self.q,self.cache,self.table,self.prefix = modules,q,cache,table,prefix
        self.groups,self.hkv,self.max_len = self.b//2,4,table.shape[1]*16
        self.sm = torch.cuda.get_device_properties(q.device).multi_processor_count
        self.splits = min(max(1,self.sm//(self.b*4)),(self.max_len+255)//256)
        self.lengths = torch.empty((self.b,1),device=q.device,dtype=torch.uint32)
        self.workspace = torch.zeros(64*1024*1024,device=q.device,dtype=torch.uint8)
        self.sem,self.scratch = self.workspace[:8*1024*1024],self.workspace[8*1024*1024:]
        self.state = torch.empty(self.groups*4*self.splits*4*32*35,device=q.device,dtype=torch.float32)
        self.output=torch.empty_like(q)
        self.packed=torch.empty((self.groups,1,32,128),device=q.device,dtype=q.dtype)
        self.unused=torch.empty_like(self.packed)
        self.pair_table=torch.empty_like(table[::2]);self.pair_lengths=torch.empty_like(self.lengths[::2])
        self.k,self.v=cache[...,:128],cache[...,128:]
        self.wk,self.wv=self.k.transpose(1,2),self.v.transpose(1,2)
        self.prepare(seq_lens)

    def prepare(self, seq_lens):
        assert len(seq_lens)==self.b and all(self.prefix<x<=self.max_len for x in seq_lens)
        assert bool((self.table[::2,:self.prefix//16]==self.table[1::2,:self.prefix//16]).all())
        self.lengths.copy_(torch.tensor(seq_lens,device=self.q.device,dtype=torch.uint32).reshape(self.b,1))

    def call(self,group,q,o,table,lengths,batch,phase,state=None):
        self.modules[group].xqa_wrapper(False,self.splits*batch*4,4,0,1.,None,o,1.,q,
            None,self.wk,self.wv,None,None,table,self.max_len,lengths,batch,1.,None,1,None,None,
            self.sem,self.scratch,False,self.state if state is None else state,self.prefix//256,phase)

    def run(self):
        self.packed.view(self.groups,4,2,4,128).copy_(self.q.view(self.groups,2,4,4,128).permute(0,2,1,3,4))
        self.pair_table.copy_(self.table[::2]);self.pair_lengths.copy_(self.lengths[::2])
        self.call(8,self.packed,self.unused,self.pair_table,self.pair_lengths,self.groups,1)
        self.call(4,self.q.unsqueeze(1),self.output.unsqueeze(1),self.table,self.lengths,self.b,3)
        return self.output
