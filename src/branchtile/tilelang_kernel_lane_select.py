"""Runtime lane-local bank selects with only M16/M32 region bodies.

New experimental hypothesis: trade uniform register selects for fewer compiled
owner-specialized GEMM bodies. Canonical U has no explicit shared staging.
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


def row_layout(m):
    # Replicate each row across its four lane subgroups in all four warps.
    return Fragment((m,),replicate=16,
        forward_thread_fn=lambda i,rep:(rep%4)*32+(i%8)*4+rep//4,
        forward_index_fn=lambda i:i//8)


@lru_cache(maxsize=128)
def producer(heads, tokens, rows, dim, regions, splits=64, block_n=64, threads=128):
    assert rows in (24, 32) and dim == 128 and threads == 128
    assert block_n in (32, 64) and regions in range(1, 16) and splits > 0
    scale = 1 / math.sqrt(dim)

    @T.macro
    def do_region(Q, K, V, Indices, Offsets, RowMap, Count, Owner, Labels,
                  CU, CM, CL, qs, ks, vs, probs, h, s, r, M: T.const):
        scores = T.alloc_fragment((M, block_n), "float32")
        out = T.alloc_fragment((M, dim), "float32")
        maximum = T.alloc_fragment((M,), "float32")
        previous = T.alloc_fragment((M,), "float32")
        factor = T.alloc_fragment((M,), "float32")
        subtotal = T.alloc_fragment((M,), "float32")
        denominator = T.alloc_fragment((M,), "float32")
        T.annotate_layout({out: mma_layout(M), maximum: row_layout(M),
                           previous: row_layout(M), factor: row_layout(M),
                           subtotal: row_layout(M), denominator: row_layout(M)})
        begin = Offsets[h, r] + (Offsets[h, r + 1] - Offsets[h, r]) * s // splits
        end = Offsets[h, r] + (Offsets[h, r + 1] - Offsets[h, r]) * (s + 1) // splits
        label = Labels[r]
        if begin < end:
            for i, d in T.Parallel(M, dim):
                src = RowMap[h, r, i]
                qs[i, d] = T.if_then_else(i < Count[h, r], Q[h, T.min(src, rows - 1), d], 0)
            T.clear(out)
            if Count[h, r] >= 8:
                request = RowMap[h, r, 0] // 8
                for i, d in T.Parallel(8, dim):
                    out[i, d] = T.if_then_else(request == 0, CU[i, d],
                        T.if_then_else(request == 1, CU[8 + i, d],
                            T.if_then_else(request == 2, CU[16 + i, d], CU[24 + i, d])))
            if Count[h, r] >= 16:
                request = RowMap[h, r, 8] // 8
                for i, d in T.Parallel(8, dim):
                    out[8 + i, d] = T.if_then_else(request == 0, CU[i, d],
                        T.if_then_else(request == 1, CU[8 + i, d],
                            T.if_then_else(request == 2, CU[16 + i, d], CU[24 + i, d])))
            if M == 32:
                if Count[h, r] >= 24:
                    request = RowMap[h, r, 16] // 8
                    for i, d in T.Parallel(8, dim):
                        out[16 + i, d] = T.if_then_else(request == 0, CU[i, d],
                            T.if_then_else(request == 1, CU[8 + i, d],
                                T.if_then_else(request == 2, CU[16 + i, d], CU[24 + i, d])))
                if Count[h, r] >= 32:
                    request = RowMap[h, r, 24] // 8
                    for i, d in T.Parallel(8, dim):
                        out[24 + i, d] = T.if_then_else(request == 0, CU[i, d],
                            T.if_then_else(request == 1, CU[8 + i, d],
                                T.if_then_else(request == 2, CU[16 + i, d], CU[24 + i, d])))
            for i in T.Parallel(M):
                src = RowMap[h, r, i]
                valid = i < Count[h, r]
                maximum[i] = T.if_then_else(valid, CM[src], -T.infinity("float32"))
                denominator[i] = T.if_then_else(valid, CL[src], 0)
            for tile in T.serial(T.ceildiv(end - begin, block_n)):
                for n, d in T.Parallel(block_n, dim):
                    p = begin + tile * block_n + n
                    token = Indices[h, T.min(p, tokens - 1)]
                    ks[n, d] = T.if_then_else(p < end, K[h, token, d], 0)
                    vs[n, d] = T.if_then_else(p < end, V[h, token, d], 0)
                T.clear(scores)
                T.gemm(qs[0:M, 0:dim], ks, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for i, n in T.Parallel(M, block_n):
                    valid = i < Count[h, r]
                    scores[i, n] = T.if_then_else(valid and begin + tile * block_n + n < end,
                                                   scores[i, n] * scale, -T.infinity("float32"))
                T.copy(maximum, previous)
                T.reduce_max(scores, maximum, dim=1, clear=False)
                for i in T.Parallel(M):
                    factor[i] = T.if_then_else(previous[i] == -T.infinity("float32"), 0,
                                               T.exp2((previous[i] - maximum[i]) * 1.4426950408889634))
                for i, n in T.Parallel(M, block_n):
                    scores[i, n] = T.if_then_else(scores[i, n] == -T.infinity("float32"), 0,
                                                  T.exp2((scores[i, n] - maximum[i]) * 1.4426950408889634))
                T.reduce_sum(scores, subtotal, dim=1)
                for i in T.Parallel(M):
                    denominator[i] = denominator[i] * factor[i] + subtotal[i]
                for i, d in T.Parallel(M, dim):
                    out[i, d] *= factor[i]
                T.copy(scores, probs[0:M, 0:block_n])
                T.gemm(probs[0:M, 0:block_n], vs, out, policy=T.GemmWarpPolicy.FullCol)
            if Owner[h, r, 0] != 0:
                slot = 0
                for i, d in T.Parallel(8, dim):
                    if M == 16:
                        CU[i, d] = T.if_then_else(slot == 0, out[i, d], out[8 + i, d])
                    else:
                        CU[i, d] = T.if_then_else(slot == 0, out[i, d],
                            T.if_then_else(slot == 1, out[8 + i, d],
                                T.if_then_else(slot == 2, out[16 + i, d], out[24 + i, d])))
            if Owner[h, r, 8] != 0:
                slot = (label & 1)
                for i, d in T.Parallel(8, dim):
                    if M == 16:
                        CU[8 + i, d] = T.if_then_else(slot == 0, out[i, d], out[8 + i, d])
                    else:
                        CU[8 + i, d] = T.if_then_else(slot == 0, out[i, d],
                            T.if_then_else(slot == 1, out[8 + i, d],
                                T.if_then_else(slot == 2, out[16 + i, d], out[24 + i, d])))
            if Owner[h, r, 16] != 0:
                slot = (label & 1) + ((label >> 1) & 1)
                for i, d in T.Parallel(8, dim):
                    if M == 16:
                        CU[16 + i, d] = T.if_then_else(slot == 0, out[i, d], out[8 + i, d])
                    else:
                        CU[16 + i, d] = T.if_then_else(slot == 0, out[i, d],
                            T.if_then_else(slot == 1, out[8 + i, d],
                                T.if_then_else(slot == 2, out[16 + i, d], out[24 + i, d])))
            if rows == 32:
                if Owner[h, r, 24] != 0:
                    slot = (label & 1) + ((label >> 1) & 1) + ((label >> 2) & 1)
                    for i, d in T.Parallel(8, dim):
                        if M == 16:
                            CU[24 + i, d] = T.if_then_else(slot == 0, out[i, d], out[8 + i, d])
                        else:
                            CU[24 + i, d] = T.if_then_else(slot == 0, out[i, d],
                                T.if_then_else(slot == 1, out[8 + i, d],
                                    T.if_then_else(slot == 2, out[16 + i, d], out[24 + i, d])))
            for i in T.Parallel(M):
                src = RowMap[h, r, i]
                if i < Count[h, r]:
                    CM[src] = maximum[i]
                    CL[src] = denominator[i]

    @T.prim_func
    def main(Q: T.Tensor((heads, rows, dim), "bfloat16"),
             K: T.Tensor((heads, tokens, dim), "bfloat16"),
             V: T.Tensor((heads, tokens, dim), "bfloat16"),
             Indices: T.Tensor((heads, tokens), "int32"),
             Offsets: T.Tensor((heads, regions + 1), "int32"),
             RowMap: T.Tensor((heads, regions, 32), "int32"),
             Count: T.Tensor((heads, regions), "int32"),
             Owner: T.Tensor((heads, regions, rows), "int32"),
             Labels: T.Tensor((regions,), "int32"),
             U: T.Tensor((heads, splits, rows, dim), "float32"),
             Mout: T.Tensor((heads, splits, rows), "float32"),
             Lout: T.Tensor((heads, splits, rows), "float32")):
        with T.Kernel(heads, splits, threads=threads) as (h, s):
            qs = T.alloc_shared((32, dim), "bfloat16")
            ks = T.alloc_shared((block_n, dim), "bfloat16")
            vs = T.alloc_shared((block_n, dim), "bfloat16")
            probs = T.alloc_shared((32, block_n), "bfloat16")
            cm = T.alloc_shared((32,), "float32")
            cl = T.alloc_shared((32,), "float32")
            cu = T.alloc_fragment((32, dim), "float32")
            T.annotate_layout({cu: canonical_layout()})
            T.clear(cu)
            T.fill(cm, -T.infinity("float32"))
            T.clear(cl)
            for r in T.serial(regions):
                if Count[h, r] <= 16:
                    do_region(Q, K, V, Indices, Offsets, RowMap, Count, Owner, Labels,
                              cu, cm, cl, qs, ks, vs, probs, h, s, r, 16)
                else:
                    do_region(Q, K, V, Indices, Offsets, RowMap, Count, Owner, Labels,
                              cu, cm, cl, qs, ks, vs, probs, h, s, r, 32)
                T.sync_threads()
            for i, d in T.Parallel(rows, dim):
                U[h, s, i, d] = cu[i, d]
            for i in T.Parallel(rows):
                Mout[h, s, i] = cm[i]
                Lout[h, s, i] = cl[i]
    return tilelang.compile(main, target="cuda -arch=sm_120", out_idx=[])
