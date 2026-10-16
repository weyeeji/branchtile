"""Experimental shared-prefix QK dataflow, unqualified for model execution.

Unit-scale E4M3 KV, BF16 Q, identical BF16 PV in both arms. Short Q is
represented by two independently row-scaled FP8 components. This is approximate
for some BF16 values, and changes QK reduction even when reconstruction is exact.
Multi-component arithmetic and shared-prefix packing are established prior art.
"""
import triton
import triton.language as tl


@triton.jit
def _components(q):
    q32 = q.to(tl.float32)
    qmax = tl.maximum(tl.max(tl.abs(q32), 1), 1.0e-30)
    sh = tl.exp2(tl.ceil(tl.log2(qmax / 448.0)))
    qh = (q32 / sh[:, None]).to(tl.float8e4nv)
    residual = q32 - sh[:, None] * qh.to(tl.float32)
    rmax = tl.maximum(tl.max(tl.abs(residual), 1), 1.0e-30)
    sl = tl.exp2(tl.ceil(tl.log2(rmax / 448.0)))
    ql = (residual / sl[:, None]).to(tl.float8e4nv)
    return qh, ql, sh, sl


@triton.jit
def qk_probe(Q, K, Scores, ROWS: tl.constexpr, TOKENS: tl.constexpr,
             M: tl.constexpr, N: tl.constexpr, D: tl.constexpr, COMPONENTS: tl.constexpr):
    head, rm, nt = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = rm * M + tl.arange(0, M)
    tokens = nt * N + tl.arange(0, N)
    dims = tl.arange(0, D)
    q = tl.load(Q + (head * ROWS + rows[:, None]) * D + dims[None, :], rows[:, None] < ROWS, other=0)
    k = tl.load(K + (head * TOKENS + tokens[:, None]) * D + dims[None, :], tokens[:, None] < TOKENS, other=0.0)
    if COMPONENTS:
        qh, ql, sh, sl = _components(q)
        hi = tl.dot(qh, tl.trans(k), out_dtype=tl.float32, max_num_imprecise_acc=0)
        lo = tl.dot(ql, tl.trans(k), out_dtype=tl.float32, max_num_imprecise_acc=0)
        scores = hi * sh[:, None] + lo * sl[:, None]
    else:
        scores = tl.dot(q, tl.trans(k.to(tl.bfloat16)), out_dtype=tl.float32)
    scores *= D ** -0.5
    tl.store(Scores + (head * ROWS + rows[:, None]) * TOKENS + tokens[None, :], scores,
             (rows[:, None] < ROWS) & (tokens[None, :] < TOKENS))


@triton.jit
def producer(Q, K, V, U, Maximum, Denominator,
             ROWS: tl.constexpr, TOKENS: tl.constexpr, SPLITS: tl.constexpr,
             CHUNK: tl.constexpr, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
             COMPONENTS: tl.constexpr):
    head, row_tile, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = row_tile * M + tl.arange(0, M)
    dims = tl.arange(0, D)
    ns = tl.arange(0, N)
    q = tl.load(Q + (head * ROWS + rows[:, None]) * D + dims[None, :], rows[:, None] < ROWS, other=0)
    if COMPONENTS:
        qh, ql, sh, sl = _components(q)
    maximum = tl.full((M,), -float('inf'), tl.float32)
    denominator = tl.full((M,), 0., tl.float32)
    accumulator = tl.full((M, D), 0., tl.float32)
    for tile in range(tl.cdiv(CHUNK, N)):
        tokens = split * CHUNK + tile * N + ns
        live = (tokens < TOKENS) & (tile * N + ns < CHUNK)
        offsets = (head.to(tl.int64) * TOKENS + tokens[:, None]) * D + dims[None, :]
        k = tl.load(K + offsets, live[:, None], other=0.0)
        v = tl.load(V + offsets, live[:, None], other=0.0).to(tl.bfloat16)
        if COMPONENTS:
            hi_scores = tl.dot(qh, tl.trans(k), out_dtype=tl.float32, max_num_imprecise_acc=0)
            lo_scores = tl.dot(ql, tl.trans(k), out_dtype=tl.float32, max_num_imprecise_acc=0)
            scores = hi_scores * sh[:, None] + lo_scores * sl[:, None]
        else:
            scores = tl.dot(q, tl.trans(k.to(tl.bfloat16)), out_dtype=tl.float32)
        scores *= D ** -0.5
        scores = tl.where((rows[:, None] < ROWS) & live[None, :], scores, -float('inf'))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_maximum == -float('inf'), 0., next_maximum)
        factor = tl.where(maximum == -float('inf'), 0., tl.exp(maximum - safe_max))
        p = tl.exp(scores - safe_max[:, None])
        denominator = denominator * factor + tl.sum(p, 1)
        accumulator *= factor[:, None]
        accumulator = tl.dot(p.to(tl.bfloat16), v, accumulator, out_dtype=tl.float32)
        maximum = next_maximum
    partial_rows = (head * SPLITS + split) * ROWS + rows
    tl.store(U + partial_rows[:, None] * D + dims[None, :], accumulator, rows[:, None] < ROWS)
    tl.store(Maximum + partial_rows, maximum, rows < ROWS)
    tl.store(Denominator + partial_rows, denominator, rows < ROWS)


@triton.jit
def merge(U, Maximum, Denominator, Output,
          ROWS: tl.constexpr, SPLITS: tl.constexpr, D: tl.constexpr, GROUP: tl.constexpr = 16):
    head, row = tl.program_id(0), tl.program_id(1)
    ns = tl.arange(0, GROUP)
    dims = tl.arange(0, D)
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0., tl.float32)
    accumulator = tl.full((D,), 0., tl.float32)
    for chunk in range(tl.cdiv(SPLITS, GROUP)):
        splits = chunk * GROUP + ns
        offsets = (head * SPLITS + splits) * ROWS + row
        mm = tl.load(Maximum + offsets, splits < SPLITS, other=-float('inf'))
        ll = tl.load(Denominator + offsets, splits < SPLITS, other=0.)
        uu = tl.load(U + offsets[:, None] * D + dims[None, :], splits[:, None] < SPLITS, other=0.)
        next_max = tl.maximum(maximum, tl.max(mm, 0))
        safe_max = tl.where(next_max == -float('inf'), 0., next_max)
        factor = tl.where(denominator > 0., tl.exp(maximum - safe_max), 0.)
        weights = tl.where(ll > 0., tl.exp(mm - safe_max), 0.)
        accumulator = accumulator * factor + tl.sum(uu * weights[:, None], 0)
        denominator = denominator * factor + tl.sum(ll * weights, 0)
        maximum = next_max
    tl.store(Output + (head * ROWS + row) * D + dims,
             accumulator / tl.where(denominator > 0., denominator, 1.))


class SharedPrefixAttention:
    """Preallocated two-kernel experiment. Q [H,B*G,D], KV [H,T,D].

    A measured call includes producer (including all repeated Q conversions) and
    O-only merge. No private suffix, paged mapping, engine hook, or scale != 1.
    """
    def __init__(self, q, k, v, components, chunk=1024, block_m=64, block_n=64):
        import torch
        assert q.ndim == 3 and k.ndim == 3 and v.shape == k.shape
        assert q.shape[0] == k.shape[0] and q.shape[-1] == k.shape[-1] == 128
        assert q.dtype == torch.bfloat16 and k.dtype == v.dtype == torch.float8_e4m3fn
        assert q.is_cuda and k.device == v.device == q.device
        assert all(t.is_contiguous() for t in (q, k, v))
        assert chunk > 0 and block_m in (16, 32, 64) and block_n in (32, 64)
        self.q, self.k, self.v = q, k, v
        self.heads, self.rows, self.d = q.shape
        self.tokens = k.shape[1]
        assert self.rows > 0 and self.tokens > 0
        self.splits = triton.cdiv(self.tokens, chunk)
        self.components, self.chunk, self.m, self.n = bool(components), chunk, block_m, block_n
        self.u = torch.empty((self.heads, self.splits, self.rows, self.d), device=q.device, dtype=torch.float32)
        self.maximum = torch.empty(self.u.shape[:-1], device=q.device, dtype=torch.float32)
        self.denominator = torch.empty_like(self.maximum)
        self.output = torch.empty_like(q)
        self.workspace_bytes = sum(t.numel() * t.element_size() for t in (self.u, self.maximum, self.denominator))
        self.last_producer = self.last_merge = None

    def __call__(self):
        self.last_producer = producer[(self.heads, triton.cdiv(self.rows, self.m), self.splits)](
            self.q, self.k, self.v, self.u, self.maximum, self.denominator,
            ROWS=self.rows, TOKENS=self.tokens, SPLITS=self.splits,
            CHUNK=self.chunk, M=self.m, N=self.n, D=self.d, COMPONENTS=self.components,
            num_warps=4, num_stages=1)
        self.last_merge = merge[(self.heads, self.rows)](
            self.u, self.maximum, self.denominator, self.output,
            ROWS=self.rows, SPLITS=self.splits, D=self.d, num_warps=4, num_stages=1)
        return self.output
