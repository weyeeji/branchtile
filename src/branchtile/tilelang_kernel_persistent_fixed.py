"""Fixed M32 rows with persistent output/softmax fragments across owner regions.

Strong ablation control: extra masked MMA work in exchange for no canonical-state
gather/scatter. Immutable source KV, runtime owner labels, no RowMap transport.
"""
from functools import lru_cache
import math
import tilelang
import tilelang.language as T


@lru_cache(maxsize=128)
def producer(heads, tokens, rows, dim, regions, splits=64, block_n=64, threads=128):
    assert rows in (24, 32) and dim == 128 and threads == 128
    assert block_n in (32, 64) and regions in range(1, 16) and splits > 0
    scale = 1 / math.sqrt(dim)

    @T.prim_func
    def main(Q: T.Tensor((heads, rows, dim), "bfloat16"),
             K: T.Tensor((heads, tokens, dim), "bfloat16"),
             V: T.Tensor((heads, tokens, dim), "bfloat16"),
             Indices: T.Tensor((heads, tokens), "int32"),
             Offsets: T.Tensor((heads, regions + 1), "int32"),
             Labels: T.Tensor((regions,), "int32"),
             U: T.Tensor((heads, splits, rows, dim), "float32"),
             Mout: T.Tensor((heads, splits, rows), "float32"),
             Lout: T.Tensor((heads, splits, rows), "float32")):
        with T.Kernel(heads, splits, threads=threads) as (h, s):
            qs = T.alloc_shared((32, dim), "bfloat16")
            ks = T.alloc_shared((block_n, dim), "bfloat16")
            vs = T.alloc_shared((block_n, dim), "bfloat16")
            probs = T.alloc_shared((32, block_n), "bfloat16")
            scores = T.alloc_fragment((32, block_n), "float32")
            out = T.alloc_fragment((32, dim), "float32")
            maximum = T.alloc_fragment((32,), "float32")
            previous = T.alloc_fragment((32,), "float32")
            factor = T.alloc_fragment((32,), "float32")
            subtotal = T.alloc_fragment((32,), "float32")
            denominator = T.alloc_fragment((32,), "float32")
            for i, d in T.Parallel(32, dim):
                qs[i, d] = T.if_then_else(i < rows, Q[h, T.min(i, rows - 1), d], 0)
            T.clear(out)
            T.fill(maximum, -T.infinity("float32"))
            T.clear(denominator)
            for r in T.serial(regions):
                begin = Offsets[h, r] + (Offsets[h, r + 1] - Offsets[h, r]) * s // splits
                end = Offsets[h, r] + (Offsets[h, r + 1] - Offsets[h, r]) * (s + 1) // splits
                label = Labels[r]
                if begin < end:
                    for tile in T.serial(T.ceildiv(end - begin, block_n)):
                        for n, d in T.Parallel(block_n, dim):
                            p = begin + tile * block_n + n
                            token = Indices[h, T.min(p, tokens - 1)]
                            ks[n, d] = T.if_then_else(p < end, K[h, token, d], 0)
                            vs[n, d] = T.if_then_else(p < end, V[h, token, d], 0)
                        T.clear(scores)
                        T.gemm(qs, ks, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                        for i, n in T.Parallel(32, block_n):
                            valid = i < rows and (label & (1 << (i // 8))) != 0
                            scores[i, n] = T.if_then_else(valid and begin + tile * block_n + n < end,
                                                         scores[i, n] * scale, -T.infinity("float32"))
                        T.copy(maximum, previous)
                        T.reduce_max(scores, maximum, dim=1, clear=False)
                        for i in T.Parallel(32):
                            factor[i] = T.if_then_else(previous[i] == -T.infinity("float32"), 0,
                                                       T.exp2((previous[i] - maximum[i]) * 1.4426950408889634))
                        for i, n in T.Parallel(32, block_n):
                            scores[i, n] = T.if_then_else(scores[i, n] == -T.infinity("float32"), 0,
                                                          T.exp2((scores[i, n] - maximum[i]) * 1.4426950408889634))
                        T.reduce_sum(scores, subtotal, dim=1)
                        for i in T.Parallel(32):
                            denominator[i] = denominator[i] * factor[i] + subtotal[i]
                        for i, d in T.Parallel(32, dim):
                            out[i, d] *= factor[i]
                        T.copy(scores, probs)
                        T.gemm(probs, vs, out, policy=T.GemmWarpPolicy.FullCol)
            for i, d in T.Parallel(rows, dim):
                U[h, s, i, d] = out[i, d]
            for i in T.Parallel(rows):
                Mout[h, s, i] = maximum[i]
                Lout[h, s, i] = denominator[i]
    return tilelang.compile(main, target="cuda -arch=sm_120", out_idx=[])
