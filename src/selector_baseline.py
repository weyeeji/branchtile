"""Exact QC-SnapKV semantics and known-LSE selector baselines.

LSE reuse is prior work (MiniKV, ACL Findings 2025, section 3.2,
https://aclanthology.org/2025.findings-acl.952/), not a novelty claim.
Native stable selection follows https://docs.flashinfer.ai/api/topk.html.
The fused score kernel
uses BF16 Tensor Core products and FP32 reductions, so selected-mask equality
must be measured rather than inferred from mathematical equivalence.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def flatten_observations(q, k, positions):
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("Expected Q[1,H*G,W,D] and K[1,H,N,D]")
    h, _, d = k.shape[1:]
    if q.shape[-1] != d or q.shape[1] % h or positions.numel() != q.shape[2]:
        raise ValueError("Incompatible GQA or observation positions")
    g = q.shape[1] // h
    return q[0].reshape(h, g * q.shape[2], d), positions.repeat(g)


def streaming_lse(q, k, positions, chunk_size=4096, dtype=torch.float32):
    """Full-history causal denominator; distinct, charged pass when not reused."""
    qf, pos = flatten_observations(q, k, positions)
    qf = qf.to(dtype)
    lse = torch.full(qf.shape[:2], -torch.inf, dtype=dtype, device=q.device)
    for begin in range(0, k.shape[2], chunk_size):
        end = min(k.shape[2], begin + chunk_size)
        logits = torch.bmm(qf, k[0, :, begin:end].to(dtype).transpose(-1, -2)) * q.shape[-1] ** -0.5
        logits.masked_fill_(torch.arange(begin, end, device=q.device)[None, None] > pos[None, :, None], -torch.inf)
        lse = torch.logaddexp(lse, logits.logsumexp(-1))
    return lse


def scores_from_lse(q, k, positions, doc_start, doc_end, lse, chunk_size=4096):
    """Original PyTorch second pass, including pool7 with zero padding."""
    qf, pos = flatten_observations(q, k, positions)
    qf = qf.to(lse.dtype)
    scores = torch.empty((k.shape[1], doc_end - doc_start), dtype=lse.dtype, device=q.device)
    for begin in range(doc_start, doc_end, chunk_size):
        end = min(doc_end, begin + chunk_size)
        logits = torch.bmm(qf, k[0, :, begin:end].to(lse.dtype).transpose(-1, -2)) * q.shape[-1] ** -0.5
        logits.masked_fill_(torch.arange(begin, end, device=q.device)[None, None] > pos[None, :, None], -torch.inf)
        scores[:, begin-doc_start:end-doc_start] = torch.exp(logits - lse[:, :, None]).mean(1)
    return F.avg_pool1d(scores[:, None], 7, stride=1, padding=3, count_include_pad=True)[:, 0]


def qc_scores(q, k, positions, doc_start, doc_end, chunk_size=4096):
    return scores_from_lse(q, k, positions, doc_start, doc_end,
                           streaming_lse(q, k, positions, chunk_size), chunk_size)


def reference_scores(q, k, positions, doc_start, doc_end, chunk_size=4096):
    lse = streaming_lse(q, k, positions, chunk_size, torch.float64)
    return scores_from_lse(q, k, positions, doc_start, doc_end, lse, chunk_size), lse


def protected_count(n):
    return min(n, 4 + 32)


def select_support(scores, budget, backend="argsort"):
    """Stable lower-original-position boundary tie, after pooling/protection."""
    _, n = scores.shape
    budget = min(n, budget)
    protected = protected_count(n)
    if budget < protected:
        raise ValueError("Budget cannot be less than protected first4/last32 union")
    support = torch.zeros_like(scores, dtype=torch.bool)
    support[:, :min(4, n)] = True
    support[:, max(0, n-32):] = True
    if budget == protected:
        return support
    ranked_scores = scores.clone()
    ranked_scores.masked_fill_(support, -torch.inf)
    count = budget - protected
    if backend == "argsort":
        indices = ranked_scores.argsort(dim=-1, descending=True, stable=True)[:, :count]
    elif backend == "flashinfer":
        import flashinfer
        _, indices = flashinfer.top_k(ranked_scores, count, sorted=False,
                                      deterministic=True, tie_break=1,
                                      dsa_graph_safe=True)
    else:
        raise ValueError(backend)
    support.scatter_(1, indices, True)
    return support


@triton.jit
def _fused_scores(Q, K, POS, LSE, OUT,
                  QH: tl.constexpr, QR: tl.constexpr, QD: tl.constexpr,
                  KH: tl.constexpr, KN: tl.constexpr, KD: tl.constexpr,
                  ROWS: tl.constexpr, D: tl.constexpr, DS: tl.constexpr,
                  DN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                  BH: tl.constexpr):
    head, block = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, BM)
    dims = tl.arange(0, D)
    # Halo is recomputed by neighboring tiles; no intermediate score array.
    cols = block * BN + tl.arange(0, BH) - 3
    q = tl.load(Q + head*QH + rows[:, None]*QR + dims[None, :]*QD,
                rows[:, None] < ROWS, other=0)
    key = tl.load(K + head*KH + (DS+cols)[None, :]*KN + dims[:, None]*KD,
                  (cols[None, :] >= 0) & (cols[None, :] < DN), other=0)
    logits = tl.dot(q, key).to(tl.float32) * (D ** -0.5)
    positions = tl.load(POS + rows, rows < ROWS, other=-1)
    lse = tl.load(LSE + head*ROWS + rows, rows < ROWS, other=0)
    valid = ((rows[:, None] < ROWS) & (cols[None, :] >= 0) &
             (cols[None, :] < DN) & (DS+cols[None, :] <= positions[:, None]))
    probs = tl.where(valid, tl.exp(logits - lse[:, None]), 0.)
    reduced = tl.sum(probs, 0) / ROWS
    outcols = tl.arange(0, BN)
    pooled = tl.full((BN,), 0, tl.float32)
    for offset in tl.static_range(7):
        pooled += tl.gather(reduced, outcols + offset, 0)
    pooled /= 7.
    tl.store(OUT + head*DN + block*BN + outcols, pooled, block*BN+outcols < DN)


class FusedKnownLSEScorer:
    """Prepares views/position repetition; no payload gather or document copy.

    Construction cost is separate and must be reported. run() writes a stable
    output allocation suitable for CUDA graph replay. LSE is natural logarithm
    shaped [H,G*W], in head-major/GQA-major/observation-major order.
    """
    def __init__(self, q, k, positions, doc_start, doc_end, block_n=64):
        qf, pos = flatten_observations(q, k, positions)
        if not q.is_cuda or q.dtype != torch.bfloat16 or k.dtype != q.dtype:
            raise ValueError("CUDA BF16 inputs are required")
        if q.shape[-1] != 128 or block_n not in (32, 64, 128):
            raise ValueError("D=128 and block_n in {32,64,128} are supported")
        if not 0 <= doc_start < doc_end <= k.shape[2]:
            raise ValueError("Invalid document range")
        self.q, self.k, self.positions = qf, k[0], pos.contiguous()
        self.start, self.n, self.block_n = doc_start, doc_end-doc_start, block_n
        self.out = torch.empty((k.shape[1], self.n), device=k.device, dtype=torch.float32)

    def run(self, lse):
        if lse.shape != self.q.shape[:2] or lse.dtype != torch.float32 or not lse.is_contiguous():
            raise ValueError("LSE must be contiguous FP32 [H,G*W]")
        h, rows, d = self.q.shape
        _fused_scores[(h, triton.cdiv(self.n, self.block_n))](
            self.q, self.k, self.positions, lse, self.out,
            *self.q.stride(), *self.k.stride(), rows, d, self.start, self.n,
            max(16, triton.next_power_of_2(rows)), self.block_n,
            triton.next_power_of_2(self.block_n+6), num_warps=8)
        return self.out
