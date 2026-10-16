"""Static-owner register-bank experiment; no canonical U shared-memory staging.

GQA8 request banks remain at fixed lane-local registers. FullCol keeps the same
thread ownership for M16 and M32 MMA accumulators. Source payload is immutable.
"""
from functools import lru_cache
import math
import tilelang
import tilelang.language as T
from tilelang.layout import Fragment


def canonical_layout():
    return Fragment((32,128),
        forward_thread_fn=lambda i,d:(d//32)*32+(i%8)*4+(d%8)//2,
        forward_index_fn=lambda i,d:(i//8)*8+((d%32)//8)*2+d%2)


def mma_layout(m):
    return Fragment((m,128),
        forward_thread_fn=lambda i,d:(d//32)*32+(i%8)*4+(d%8)//2,
        forward_index_fn=lambda i,d:(i//16)*16+((d%32)//8)*4+((i%16)//8)*2+d%2)


@lru_cache(maxsize=128)
def producer(heads,tokens,rows,dim,regions,splits=64,block_n=64,threads=128):
    assert rows in (24,32) and dim==128 and threads==128
    assert block_n in (32,64) and regions in range(1,16)
    scale=1/math.sqrt(dim)

    @T.macro
    def do_region(Q,K,V,Indices,Offsets,CU,CM,CL,qs,ks,vs,probs,h,s,r,
                  active:T.const,M:T.const,a:T.const,b:T.const,c:T.const,dq:T.const):
        scores=T.alloc_fragment((M,block_n),"float32")
        out=T.alloc_fragment((M,dim),"float32")
        maximum=T.alloc_fragment((M,),"float32")
        previous=T.alloc_fragment((M,),"float32")
        factor=T.alloc_fragment((M,),"float32")
        subtotal=T.alloc_fragment((M,),"float32")
        denominator=T.alloc_fragment((M,),"float32")
        T.annotate_layout({out:mma_layout(M)})
        begin=Offsets[h,r]+(Offsets[h,r+1]-Offsets[h,r])*s//splits
        end=Offsets[h,r]+(Offsets[h,r+1]-Offsets[h,r])*(s+1)//splits
        if begin<end:
            for i,d in T.Parallel(M,dim):
                src=T.if_then_else(i<8,a*8+i,T.if_then_else(i<16,b*8+i-8,T.if_then_else(i<24,c*8+i-16,dq*8+i-24)))
                qs[i,d]=T.if_then_else(i<active,Q[h,T.min(src,rows-1),d],0)
            T.clear(out)
            for i,d in T.Parallel(8,dim):
                out[i,d]=CU[a*8+i,d]
            if active>=16:
                for i,d in T.Parallel(8,dim):
                    out[8+i,d]=CU[b*8+i,d]
            if active>=24:
                for i,d in T.Parallel(8,dim):
                    out[16+i,d]=CU[c*8+i,d]
            if active>=32:
                for i,d in T.Parallel(8,dim):
                    out[24+i,d]=CU[dq*8+i,d]
            T.fill(maximum,-T.infinity("float32"))
            T.clear(denominator)
            for i in T.Parallel(8):
                maximum[i]=CM[a*8+i]
                denominator[i]=CL[a*8+i]
            if active>=16:
                for i in T.Parallel(8):
                    maximum[8+i]=CM[b*8+i]
                    denominator[8+i]=CL[b*8+i]
            if active>=24:
                for i in T.Parallel(8):
                    maximum[16+i]=CM[c*8+i]
                    denominator[16+i]=CL[c*8+i]
            if active>=32:
                for i in T.Parallel(8):
                    maximum[24+i]=CM[dq*8+i]
                    denominator[24+i]=CL[dq*8+i]
            for tile in T.serial(T.ceildiv(end-begin,block_n)):
                for n,d in T.Parallel(block_n,dim):
                    p=begin+tile*block_n+n
                    token=Indices[h,T.min(p,tokens-1)]
                    ks[n,d]=T.if_then_else(p<end,K[h,token,d],0)
                    vs[n,d]=T.if_then_else(p<end,V[h,token,d],0)
                T.clear(scores)
                T.gemm(qs[0:M,0:dim],ks,scores,transpose_B=True,policy=T.GemmWarpPolicy.FullCol)
                for i,n in T.Parallel(M,block_n):
                    scores[i,n]=T.if_then_else(i<active and begin+tile*block_n+n<end,scores[i,n]*scale,-T.infinity("float32"))
                T.copy(maximum,previous)
                T.reduce_max(scores,maximum,dim=1,clear=False)
                for i in T.Parallel(M):
                    factor[i]=T.if_then_else(previous[i]==-T.infinity("float32"),0,T.exp2((previous[i]-maximum[i])*1.4426950408889634))
                for i,n in T.Parallel(M,block_n):
                    scores[i,n]=T.if_then_else(scores[i,n]==-T.infinity("float32"),0,T.exp2((scores[i,n]-maximum[i])*1.4426950408889634))
                T.reduce_sum(scores,subtotal,dim=1)
                for i in T.Parallel(M):
                    denominator[i]=denominator[i]*factor[i]+subtotal[i]
                for i,d in T.Parallel(M,dim):
                    out[i,d]*=factor[i]
                T.copy(scores,probs[0:M,0:block_n])
                T.gemm(probs[0:M,0:block_n],vs,out,policy=T.GemmWarpPolicy.FullCol)
            for i,d in T.Parallel(8,dim):
                CU[a*8+i,d]=out[i,d]
            if active>=16:
                for i,d in T.Parallel(8,dim):
                    CU[b*8+i,d]=out[8+i,d]
            if active>=24:
                for i,d in T.Parallel(8,dim):
                    CU[c*8+i,d]=out[16+i,d]
            if active>=32:
                for i,d in T.Parallel(8,dim):
                    CU[dq*8+i,d]=out[24+i,d]
            for i in T.Parallel(8):
                CM[a*8+i]=maximum[i]
                CL[a*8+i]=denominator[i]
            if active>=16:
                for i in T.Parallel(8):
                    CM[b*8+i]=maximum[8+i]
                    CL[b*8+i]=denominator[8+i]
            if active>=24:
                for i in T.Parallel(8):
                    CM[c*8+i]=maximum[16+i]
                    CL[c*8+i]=denominator[16+i]
            if active>=32:
                for i in T.Parallel(8):
                    CM[dq*8+i]=maximum[24+i]
                    CL[dq*8+i]=denominator[24+i]

    @T.prim_func
    def main(Q:T.Tensor((heads,rows,dim),"bfloat16"),
             K:T.Tensor((heads,tokens,dim),"bfloat16"),
             V:T.Tensor((heads,tokens,dim),"bfloat16"),
             Indices:T.Tensor((heads,tokens),"int32"),
             Offsets:T.Tensor((heads,regions+1),"int32"),
             Labels:T.Tensor((regions,),"int32"),
             U:T.Tensor((heads,splits,rows,dim),"float32"),
             Mout:T.Tensor((heads,splits,rows),"float32"),
             Lout:T.Tensor((heads,splits,rows),"float32")):
        with T.Kernel(heads,splits,threads=threads) as (h,s):
            qs=T.alloc_shared((32,dim),"bfloat16")
            ks=T.alloc_shared((block_n,dim),"bfloat16")
            vs=T.alloc_shared((block_n,dim),"bfloat16")
            probs=T.alloc_shared((32,block_n),"bfloat16")
            cm=T.alloc_shared((32,),"float32")
            cl=T.alloc_shared((32,),"float32")
            cu=T.alloc_fragment((32,dim),"float32")
            T.annotate_layout({cu:canonical_layout()})
            T.clear(cu)
            T.fill(cm,-T.infinity("float32"))
            T.clear(cl)
            for r in T.serial(regions):
                label=Labels[r]
                if label == 1:
                    do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,8,16,0,0,0,0)
                if label == 2:
                    do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,8,16,1,0,0,0)
                if label == 3:
                    do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,16,16,0,1,0,0)
                if label == 4:
                    do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,8,16,2,0,0,0)
                if label == 5:
                    do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,16,16,0,2,0,0)
                if label == 6:
                    do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,16,16,1,2,0,0)
                if label == 7:
                    do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,24,32,0,1,2,0)
                if rows==32:
                    if label == 8:
                        do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,8,16,3,0,0,0)
                    if label == 9:
                        do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,16,16,0,3,0,0)
                    if label == 10:
                        do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,16,16,1,3,0,0)
                    if label == 11:
                        do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,24,32,0,1,3,0)
                    if label == 12:
                        do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,16,16,2,3,0,0)
                    if label == 13:
                        do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,24,32,0,2,3,0)
                    if label == 14:
                        do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,24,32,1,2,3,0)
                    if label == 15:
                        do_region(Q,K,V,Indices,Offsets,cu,cm,cl,qs,ks,vs,probs,h,s,r,32,32,0,1,2,3)
                T.sync_threads()
            for i,d in T.Parallel(rows,dim):
                U[h,s,i,d]=cu[i,d]
            for i in T.Parallel(rows):
                Mout[h,s,i]=cm[i]
                Lout[h,s,i]=cl[i]
    return tilelang.compile(main,target="cuda -arch=sm_120",out_idx=[])

