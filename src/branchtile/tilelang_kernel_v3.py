"""BranchTile v3: register canonical u with bit-preserving KV scratch staging.

Token indices address an immutable physical KV source. Each region is a disjoint
owner set. Only metadata is packed; K and V are never copied by this operator.
"""
from functools import lru_cache
import math
import tilelang
import tilelang.language as T


@lru_cache(maxsize=128)
def producer(heads, tokens, rows, dim, regions, splits=1, block_n=64,
             threads=128, compact=True, region_partials=False):
    assert rows == 32 and dim == 128 and block_n == 64
    assert compact and not region_partials, "v3 qualification is limited to fused B4/GQA8"
    output_splits = splits * regions if region_partials else splits
    fixed_m = 16 if rows <= 16 else 32
    scale = 1 / math.sqrt(dim)

    @T.macro
    def do_region(Q, K, V, Indices, Offsets, RowMap, Count, Owner,
                  CM, CL, CU, qs, ks, vs, probs, h, s, r, M: T.const):
        scores = T.alloc_fragment((M, block_n), "float32")
        out = T.alloc_fragment((M, dim), "float32")
        maximum = T.alloc_fragment((M,), "float32")
        previous = T.alloc_fragment((M,), "float32")
        factor = T.alloc_fragment((M,), "float32")
        subtotal = T.alloc_fragment((M,), "float32")
        denominator = T.alloc_fragment((M,), "float32")
        begin = Offsets[h, r] + (Offsets[h, r + 1] - Offsets[h, r]) * s // splits
        end = Offsets[h, r] + (Offsets[h, r + 1] - Offsets[h, r]) * (s + 1) // splits
        for i, d in T.Parallel(M, dim):
            src = RowMap[h, r, i] if compact else i
            valid = i < Count[h, r] if compact else i < rows
            qs[i, d] = T.if_then_else(valid, Q[h, src, d], 0)
        for i, d in T.Parallel(M, dim):
            src = RowMap[h, r, i] if compact else i
            valid = i < Count[h, r] if compact else i < rows
            out[i, d] = T.if_then_else(valid, T.reinterpret(T.Cast("uint32", T.reinterpret(CU[src * 2 + d // 64, (d % 64) * 2], "uint16")) | (T.Cast("uint32", T.reinterpret(CU[src * 2 + d // 64, (d % 64) * 2 + 1], "uint16")) << 16), "float32"), 0)
        for i in T.Parallel(M):
            src = RowMap[h, r, i] if compact else i
            valid = i < Count[h, r] if compact else i < rows
            maximum[i] = T.if_then_else(valid, CM[src], -T.infinity("float32"))
            denominator[i] = T.if_then_else(valid, CL[src], 0)
        for tile in T.serial(T.ceildiv(end - begin, block_n)):
            for n, d in T.Parallel(block_n, dim):
                p = begin + tile * block_n + n
                token = Indices[h, T.min(p, tokens - 1)]
                ks[n, d] = T.if_then_else(p < end, K[h, token, d], 0)
                vs[n, d] = T.if_then_else(p < end, V[h, token, d], 0)
            T.clear(scores)
            T.gemm(qs[0:M, 0:dim], ks, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
            for i, n in T.Parallel(M, block_n):
                valid = i < Count[h, r] if compact else (i < rows and Owner[h, r, i] != 0)
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
            T.gemm(probs[0:M, 0:block_n], vs, out, policy=T.GemmWarpPolicy.FullRow)
        for i, d in T.Parallel(M, dim):
            src = RowMap[h, r, i] if compact else i
            valid = i < Count[h, r] if compact else i < rows
            if valid:
                bits = T.reinterpret(out[i, d], "uint32")
                CU[src * 2 + d // 64, (d % 64) * 2] = T.reinterpret(T.Cast("uint16", bits & 65535), "bfloat16")
                CU[src * 2 + d // 64, (d % 64) * 2 + 1] = T.reinterpret(T.Cast("uint16", bits >> 16), "bfloat16")
        for i in T.Parallel(M):
            src = RowMap[h, r, i] if compact else i
            valid = i < Count[h, r] if compact else i < rows
            if valid:
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
             U: T.Tensor((heads, output_splits, rows, dim), "float32"),
             Mout: T.Tensor((heads, output_splits, rows), "float32"),
             Lout: T.Tensor((heads, output_splits, rows), "float32")):
        with T.Kernel(heads, splits, regions if region_partials else 1, threads=threads) as (h, s, rg):
            qs = T.alloc_shared((32, dim), "bfloat16")
            ks = T.alloc_shared((block_n, dim), "bfloat16")
            vs = T.alloc_shared((block_n, dim), "bfloat16")
            probs = T.alloc_shared((32, block_n), "bfloat16")
            cm = T.alloc_shared((rows,), "float32")
            cl = T.alloc_shared((rows,), "float32")
            canonical_u = T.alloc_fragment((rows, dim), "float32")
            T.fill(cm, -T.infinity("float32"))
            T.clear(cl)
            T.clear(canonical_u)
            output_s = rg * splits + s if region_partials else s
            for r in T.serial(rg if region_partials else 0, rg + 1 if region_partials else regions):
                # Two BF16-typed cells carry the exact 32 FP bits; no quantization.
                for i, d in T.Parallel(rows, dim):
                    bits = T.reinterpret(canonical_u[i, d], "uint32")
                    ks[i * 2 + d // 64, (d % 64) * 2] = T.reinterpret(T.Cast("uint16", bits & 65535), "bfloat16")
                    ks[i * 2 + d // 64, (d % 64) * 2 + 1] = T.reinterpret(T.Cast("uint16", bits >> 16), "bfloat16")
                T.sync_threads()
                if compact:
                    if Count[h, r] <= 16:
                        do_region(Q, K, V, Indices, Offsets, RowMap, Count, Owner,
                                  cm, cl, ks, qs, ks, vs, probs, h, s, r, 16)
                    else:
                        do_region(Q, K, V, Indices, Offsets, RowMap, Count, Owner,
                                  cm, cl, ks, qs, ks, vs, probs, h, s, r, 32)
                else:
                    do_region(Q, K, V, Indices, Offsets, RowMap, Count, Owner,
                              cm, cl, ks, qs, ks, vs, probs, h, s, r, fixed_m)
                T.sync_threads()
                for i, d in T.Parallel(rows, dim):
                    if Owner[h, r, i] != 0:
                        canonical_u[i, d] = T.reinterpret(T.Cast("uint32", T.reinterpret(ks[i * 2 + d // 64, (d % 64) * 2], "uint16")) | (T.Cast("uint32", T.reinterpret(ks[i * 2 + d // 64, (d % 64) * 2 + 1], "uint16")) << 16), "float32")
                T.sync_threads()
            for i, d in T.Parallel(rows, dim):
                U[h, output_s, i, d] = canonical_u[i, d]
            T.copy(cm, Mout[h, output_s, :])
            T.copy(cl, Lout[h, output_s, :])
    return tilelang.compile(main, target="cuda -arch=sm_120", out_idx=[])
