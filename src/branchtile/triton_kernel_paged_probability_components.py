"""Independent paged-alias QK/PV component experiment; fixed power2 P scales.

Both arms use two-component QK; the switch changes only BF16 versus two-component PV.
Unit-scale E4M3 KV only. P is first rounded to BF16, matching the control.
"""
import triton
import triton.language as tl
from .triton_kernel_query_components import _components


@triton.jit
def producer(Q, Cache, Pages, U, Maximum, Denominator,
             BATCH: tl.constexpr, HEADS: tl.constexpr, GQA: tl.constexpr,
             TOKENS: tl.constexpr, SPLITS: tl.constexpr, PAGE: tl.constexpr,
             QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
             CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr, CS3: tl.constexpr,
             CHUNK: tl.constexpr, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
             COMPONENTS: tl.constexpr, PV_COMPONENTS: tl.constexpr):
    head, row_tile, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = row_tile * M + tl.arange(0, M)
    dims = tl.arange(0, D)
    ns = tl.arange(0, N)
    qoff = (rows[:, None] // GQA) * QS0 + (head * GQA + rows[:, None] % GQA) * QS1 + dims[None, :] * QS2
    q = tl.load(Q + qoff, rows[:, None] < BATCH * GQA, other=0)
    if COMPONENTS:
        qh, ql, sh, sl = _components(q)
    maximum = tl.full((M,), -float('inf'), tl.float32)
    denominator = tl.full((M,), 0., tl.float32)
    accumulator = tl.full((M, D), 0., tl.float32)
    for tile in range(tl.cdiv(CHUNK, N)):
        tokens = split * CHUNK + tile * N + ns
        live = (tokens < TOKENS) & (tile * N + ns < CHUNK)
        physical = tl.load(Pages + tokens // PAGE, live, other=0).to(tl.int64)
        offsets = physical[:, None] * CS0 + head * CS1 + (tokens[:, None] % PAGE) * CS2 + dims[None, :] * CS3
        k = tl.load(Cache + offsets, live[:, None], other=0.0)
        v = tl.load(Cache + offsets + D * CS3, live[:, None], other=0.0)
        if COMPONENTS:
            hi = tl.dot(qh, tl.trans(k), out_dtype=tl.float32, max_num_imprecise_acc=0)
            lo = tl.dot(ql, tl.trans(k), out_dtype=tl.float32, max_num_imprecise_acc=0)
            scores = hi * sh[:, None] + lo * sl[:, None]
        else:
            scores = tl.dot(q, tl.trans(k.to(tl.bfloat16)), out_dtype=tl.float32)
        scores *= D ** -0.5
        scores = tl.where((rows[:, None] < BATCH * GQA) & live[None, :], scores, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float('inf'), 0., next_max)
        factor = tl.where(maximum == -float('inf'), 0., tl.exp(maximum - safe_max))
        p = tl.exp(scores - safe_max[:, None])
        denominator = denominator * factor + tl.sum(p, 1)
        accumulator *= factor[:, None]
        if PV_COMPONENTS:
            # Accumulator stays in units 2^-8 until its final store.
            # BF16 rounding boundary matches the Q-only control.
            pb = p.to(tl.bfloat16).to(tl.float32)
            ph = (pb * 256.).to(tl.float8e4nv)
            pl = ((pb - ph.to(tl.float32) * (1. / 256.)) * 8192.).to(tl.float8e4nv)
            accumulator = tl.dot(ph, v, accumulator, out_dtype=tl.float32, max_num_imprecise_acc=0)
            accumulator *= 32.
            accumulator = tl.dot(pl, v, accumulator, out_dtype=tl.float32, max_num_imprecise_acc=0)
            accumulator *= (1. / 32.)
        else:
            accumulator = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), accumulator, out_dtype=tl.float32)
        maximum = next_max
    if PV_COMPONENTS:
        accumulator *= (1. / 256.)
    packed = (head * SPLITS + split) * (BATCH * GQA) + rows
    tl.store(U + packed[:, None] * D + dims[None, :], accumulator, rows[:, None] < BATCH * GQA)
    tl.store(Maximum + packed, maximum, rows < BATCH * GQA)
    tl.store(Denominator + packed, denominator, rows < BATCH * GQA)


@triton.jit
def merge(U, Maximum, Denominator, Output, LSE,
          BATCH: tl.constexpr, HEADS: tl.constexpr, GQA: tl.constexpr,
          SPLITS: tl.constexpr, D: tl.constexpr, OS0: tl.constexpr, OS1: tl.constexpr,
          OS2: tl.constexpr, WRITE_LSE: tl.constexpr, GROUP: tl.constexpr = 16):
    head, row = tl.program_id(0), tl.program_id(1)
    ns = tl.arange(0, GROUP)
    dims = tl.arange(0, D)
    maximum = tl.full((), -float('inf'), tl.float32)
    denominator = tl.full((), 0., tl.float32)
    accumulator = tl.full((D,), 0., tl.float32)
    for chunk in range(tl.cdiv(SPLITS, GROUP)):
        splits = chunk * GROUP + ns
        offsets = (head * SPLITS + splits) * (BATCH * GQA) + row
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
    request, qhead = row // GQA, head * GQA + row % GQA
    tl.store(Output + request * OS0 + qhead * OS1 + dims * OS2,
             accumulator / tl.where(denominator > 0., denominator, 1.))
    if WRITE_LSE:
        tl.store(LSE + request * (HEADS * GQA) + qhead,
                 tl.where(denominator > 0., maximum * 1.4426950408889634 + tl.log2(denominator), -float('inf')))


class PagedProbabilityComponentsAttention:
    """Preallocated graph-safe call on original Q and cache/page-ID aliases.

    Caller owns page identity/lifetime and value updates. No CPU D2H validation,
    Q packing, cache compaction, metadata kernel, or library patch in a call.
    Replacing any tensor/shape/stride requires rebuilding and recapturing.
    """
    def __init__(self, q, cache, prefix_page_ids, prefix_tokens, components=True, pv_components=False,
                 chunk=1024, block_m=64, block_n=64, write_lse=True, k_scale=1., v_scale=1.):
        import torch
        assert q.ndim == 3 and q.dtype == torch.bfloat16 and q.is_cuda
        assert cache.ndim == 4 and cache.dtype in (torch.uint8, torch.float8_e4m3fn)
        assert cache.device == prefix_page_ids.device == q.device
        assert prefix_page_ids.ndim == 1 and prefix_page_ids.is_contiguous()
        assert prefix_page_ids.dtype in (torch.int32, torch.int64)
        assert prefix_tokens > 0 and prefix_tokens % cache.shape[2] == 0
        assert len(prefix_page_ids) >= prefix_tokens // cache.shape[2]
        assert k_scale == v_scale == 1., 'Only unit FP8 KV scales are qualified for this prototype'
        self.batch, qheads, self.d = q.shape
        self.heads, self.page = cache.shape[1:3]
        assert self.d == 128 and cache.shape[-1] == 2 * self.d
        assert qheads % self.heads == 0
        self.gqa = qheads // self.heads
        assert self.gqa == 8 and self.page == 16
        assert block_m in (32, 64) and block_n == 64 and chunk == 1024
        self.q, self.cache, self.pages = q, cache.view(torch.float8_e4m3fn), prefix_page_ids
        self.tokens, self.splits = prefix_tokens, triton.cdiv(prefix_tokens, chunk)
        self.components, self.chunk, self.m, self.n = bool(components), chunk, block_m, block_n
        self.write_lse = bool(write_lse)
        self.pv_components = bool(pv_components)
        self.u = torch.empty((self.heads, self.splits, self.batch * self.gqa, self.d), device=q.device, dtype=torch.float32)
        self.maximum = torch.empty(self.u.shape[:-1], device=q.device, dtype=torch.float32)
        self.denominator = torch.empty_like(self.maximum)
        self.output = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
        self.lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        self.workspace_bytes = sum(t.numel() * t.element_size() for t in (self.u, self.maximum, self.denominator, self.lse))
        self.last_producer = self.last_merge = None

    def __call__(self):
        qs, cs, os = self.q.stride(), self.cache.stride(), self.output.stride()
        self.last_producer = producer[(self.heads, triton.cdiv(self.batch * self.gqa, self.m), self.splits)](
            self.q, self.cache, self.pages, self.u, self.maximum, self.denominator,
            BATCH=self.batch, HEADS=self.heads, GQA=self.gqa, TOKENS=self.tokens,
            SPLITS=self.splits, PAGE=self.page, QS0=qs[0], QS1=qs[1], QS2=qs[2],
            CS0=cs[0], CS1=cs[1], CS2=cs[2], CS3=cs[3],
            CHUNK=self.chunk, M=self.m, N=self.n, D=self.d, COMPONENTS=self.components, PV_COMPONENTS=self.pv_components,
            num_warps=4, num_stages=1)
        self.last_merge = merge[(self.heads, self.batch * self.gqa)](
            self.u, self.maximum, self.denominator, self.output, self.lse,
            BATCH=self.batch, HEADS=self.heads, GQA=self.gqa, SPLITS=self.splits, D=self.d,
            OS0=os[0], OS1=os[1], OS2=os[2], WRITE_LSE=self.write_lse, num_warps=4, num_stages=1)
        return self.output
