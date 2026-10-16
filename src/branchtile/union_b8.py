"""B8 fixed-row masked union control: tiles cross owner-region boundaries.

This is an existing masked-union execution mechanism, not a new BranchTile
contribution. It avoids the per-owner tail padding retained by the 2x2 control.
"""
from functools import lru_cache
import math
import torch
import triton
import tilelang
import tilelang.language as T
from .attention import _merge


@lru_cache(maxsize=128)
def union_producer(heads, tokens, rows, gqa, splits=64, block_n=64, threads=128):
    dim = 128
    M = 64
    assert rows == 64
    scale = dim**-0.5

    @T.prim_func
    def main(Q: T.Tensor((heads, rows, dim), "bfloat16"),
             K: T.Tensor((heads, tokens, dim), "bfloat16"),
             V: T.Tensor((heads, tokens, dim), "bfloat16"),
             Indices: T.Tensor((heads, tokens), "int32"),
             Lengths: T.Tensor((heads,), "int32"),
             Owners: T.Tensor((heads, tokens), "int32"),
             U: T.Tensor((heads, splits, rows, dim), "float32"),
             Mout: T.Tensor((heads, splits, rows), "float32"),
             Lout: T.Tensor((heads, splits, rows), "float32")):
        with T.Kernel(heads, splits, threads=threads) as (h, s):
            qs = T.alloc_shared((M, dim), "bfloat16")
            ks = T.alloc_shared((block_n, dim), "bfloat16")
            vs = T.alloc_shared((block_n, dim), "bfloat16")
            os = T.alloc_shared((block_n,), "int32")
            scores = T.alloc_fragment((M, block_n), "float32")
            probs = T.alloc_shared((M, block_n), "bfloat16")
            out = T.alloc_fragment((M, dim), "float32")
            maximum = T.alloc_fragment((M,), "float32")
            previous = T.alloc_fragment((M,), "float32")
            factor = T.alloc_fragment((M,), "float32")
            subtotal = T.alloc_fragment((M,), "float32")
            denominator = T.alloc_fragment((M,), "float32")
            for i, d in T.Parallel(M, dim):
                qs[i, d] = T.if_then_else(i < rows, Q[h, T.min(i, rows-1), d], 0)
            T.clear(out)
            T.fill(maximum, -T.infinity("float32"))
            T.clear(denominator)
            begin = Lengths[h]*s//splits
            end = Lengths[h]*(s+1)//splits
            for tile in T.serial(T.ceildiv(end-begin, block_n)):
                for n, d in T.Parallel(block_n, dim):
                    p = begin+tile*block_n+n
                    token = Indices[h, T.min(p, tokens-1)]
                    ks[n,d] = T.if_then_else(p < end, K[h,token,d], 0)
                    vs[n,d] = T.if_then_else(p < end, V[h,token,d], 0)
                for n in T.Parallel(block_n):
                    p = begin+tile*block_n+n
                    token = Indices[h, T.min(p, tokens-1)]
                    os[n] = T.if_then_else(p < end, Owners[h,token], 0)
                T.clear(scores)
                T.gemm(qs,ks,scores,transpose_B=True,policy=T.GemmWarpPolicy.FullRow)
                for i,n in T.Parallel(M,block_n):
                    scores[i,n] = T.if_then_else(i < rows and (os[n] & (1 << (i//gqa))) != 0,
                                                scores[i,n]*scale,-T.infinity("float32"))
                T.copy(maximum, previous)
                T.reduce_max(scores,maximum,dim=1,clear=False)
                for i in T.Parallel(M):
                    factor[i] = T.if_then_else(previous[i] == -T.infinity("float32"),0,
                                               T.exp2((previous[i]-maximum[i])*1.4426950408889634))
                for i,n in T.Parallel(M,block_n):
                    scores[i,n] = T.if_then_else(scores[i,n] == -T.infinity("float32"),0,
                                                T.exp2((scores[i,n]-maximum[i])*1.4426950408889634))
                T.reduce_sum(scores,subtotal,dim=1)
                for i in T.Parallel(M):
                    denominator[i] = denominator[i]*factor[i]+subtotal[i]
                for i,d in T.Parallel(M,dim):
                    out[i,d] *= factor[i]
                T.copy(scores,probs)
                T.gemm(probs,vs,out,policy=T.GemmWarpPolicy.FullRow)
            for i,d in T.Parallel(M,dim):
                if i < rows:
                    U[h,s,i,d] = out[i,d]
            for i in T.Parallel(M):
                if i < rows:
                    Mout[h,s,i] = maximum[i]
                    Lout[h,s,i] = denominator[i]
    return tilelang.compile(main,target="cuda -arch=sm_120",out_idx=[])


class FixedUnion:
    def __init__(self, plan, owners, splits=64, block_n=64, threads=128):
        self.plan, self.splits = plan, splits
        h, n = plan.indices.shape
        r = plan.rows
        self.lengths = plan.offsets[:, -1].contiguous()
        self.owners = owners.to(device=plan.indices.device, dtype=torch.int32).contiguous()
        self.kernel = union_producer(h,n,r,plan.gqa,splits,block_n,threads)
        kw = dict(device=plan.indices.device,dtype=torch.float32)
        self.u = torch.empty(h,splits,r,128,**kw)
        self.m = torch.empty(h,splits,r,**kw)
        self.l = torch.empty_like(self.m)
        self.output = torch.empty(h,r,128,device=plan.indices.device,dtype=torch.bfloat16)
        self.lse = torch.empty(h,r,**kw)
        self.shape = h,r,128

    def __call__(self,q,k,v):
        self.kernel(q,k,v,self.plan.indices,self.lengths,self.owners,self.u,self.m,self.l)
        _merge[(self.shape[0],self.shape[1])](self.u,self.m,self.l,self.output,self.lse,
                self.shape[1],self.shape[2],self.splits,triton.next_power_of_2(self.splits),num_warps=4)
        return self.output,self.lse
