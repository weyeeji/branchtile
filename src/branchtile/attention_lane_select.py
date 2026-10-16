"""Runtime lane-local selects with two region shapes (experimental)."""
import torch
import triton
from .attention import _merge
from .tilelang_kernel_lane_select import producer


class Attention:
    def __init__(self,plan,splits=64,block_n=64,threads=128,compact=True,region_partials=False):
        assert compact and not region_partials
        assert plan.gqa==8 and plan.batch in (3,4)
        self.plan,self.splits=plan,splits
        h,n=plan.indices.shape
        r=plan.rows
        self.kernel=producer(h,n,r,128,len(plan.labels),splits,block_n,threads)
        self.labels=torch.tensor(plan.labels,dtype=torch.int32,device=plan.indices.device)
        opts=dict(device=plan.indices.device,dtype=torch.float32)
        self.u=torch.empty((h,splits,r,128),**opts)
        self.m=torch.empty((h,splits,r),**opts)
        self.l=torch.empty_like(self.m)
        self.output=torch.empty((h,r,128),device=plan.indices.device,dtype=torch.bfloat16)
        self.lse=torch.empty((h,r),**opts)
        self.shape=(h,r,128)

    def __call__(self,q,k,v):
        self.kernel(q,k,v,*self.plan.tensors,self.labels,self.u,self.m,self.l)
        _merge[(self.shape[0],self.shape[1])](self.u,self.m,self.l,self.output,self.lse,
            self.shape[1],self.shape[2],self.splits,triton.next_power_of_2(self.splits),num_warps=4)
        return self.output,self.lse
