"""Diagnostic paged owner: FP32 probability/value PV with IEEE dot; original BF16 QK."""
import triton
import triton.language as tl


@triton.jit
def _task_body(Q, K, V, Indices, RowMap, U, Maximum, Denominator,
               head, region, begin, end, active, packed_base,
               ROWS: tl.constexpr, TOKENS: tl.constexpr, REGIONS: tl.constexpr, HEADS: tl.constexpr, QS0: tl.constexpr, QS1: tl.constexpr,
               M: tl.constexpr, N: tl.constexpr, D: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, D)
    ns = tl.arange(0, N)
    source_row = tl.load(RowMap + (head*REGIONS+region)*32 + rows, rows < active, other=0)
    q = tl.load(Q + (source_row[:, None]//8)*QS0 + (head.to(tl.int64)*8 + source_row[:, None]%8)*QS1 + cols[None, :],
                rows[:, None] < active, other=0)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.full((M,), 0, tl.float32)
    accumulator = tl.full((M, D), 0, tl.float32)
    for tile in range(tl.cdiv(end-begin, N)):
        positions = begin + tile*N + ns
        live = positions < end
        token = tl.load(Indices + head.to(tl.int64)*TOKENS + positions, live, other=0)
        offsets = token[:, None].to(tl.int64)*(2*D) + cols[None, :]
        k = tl.load(K + offsets, live[:, None], other=0)
        v = tl.load(V + offsets, live[:, None], other=0)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * (D ** -0.5)
        scores = tl.where((rows[:, None] < active) & live[None, :], scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        factor = tl.where(maximum == -float("inf"), 0.0, tl.exp(maximum-safe_maximum))
        probabilities = tl.exp(scores-safe_maximum[:, None])
        denominator = denominator*factor + tl.sum(probabilities, 1)
        accumulator *= factor[:, None]
        accumulator = tl.dot(probabilities, v.to(tl.float32), accumulator, out_dtype=tl.float32, input_precision="ieee")
        maximum = next_maximum
    packed_rows = packed_base.to(tl.int64) + rows
    tl.store(U + packed_rows[:, None]*D + cols[None, :], accumulator, rows[:, None] < active)
    tl.store(Maximum + packed_rows, maximum, rows < active)
    tl.store(Denominator + packed_rows, denominator, rows < active)


@triton.jit
def producer(Q, K, V, Indices, RowMap, Tasks, U, Maximum, Denominator,
             ROWS: tl.constexpr, TOKENS: tl.constexpr, REGIONS: tl.constexpr, HEADS: tl.constexpr, QS0: tl.constexpr, QS1: tl.constexpr,
             N: tl.constexpr, D: tl.constexpr):
    task = tl.program_id(0)
    head = tl.load(Tasks + task*6)
    region = tl.load(Tasks + task*6+1)
    begin = tl.load(Tasks + task*6+2)
    end = tl.load(Tasks + task*6+3)
    active = tl.load(Tasks + task*6+4)
    packed_base = tl.load(Tasks + task*6+5)
    if active > 0 and active <= 16:
        _task_body(Q, K, V, Indices, RowMap, U, Maximum, Denominator,
                   head, region, begin, end, active, packed_base,
                   ROWS, TOKENS, REGIONS, HEADS, QS0, QS1, 16, N, D)
    elif active > 16:
        _task_body(Q, K, V, Indices, RowMap, U, Maximum, Denominator,
                   head, region, begin, end, active, packed_base,
                   ROWS, TOKENS, REGIONS, HEADS, QS0, QS1, 32, N, D)


@triton.jit
def merge(U, Maximum, Denominator, MergeIndices, ContributionCounts, Output, LSE,
          BATCH: tl.constexpr, GQA: tl.constexpr, ROWS: tl.constexpr,
          MAX_CONTRIB: tl.constexpr, D: tl.constexpr, HEADS: tl.constexpr, GROUP: tl.constexpr = 16):
    head, row = tl.program_id(0), tl.program_id(1)
    request, group_head = row // GQA, row % GQA
    count = tl.load(ContributionCounts + head*BATCH + request)
    contribution = tl.arange(0, GROUP)
    cols = tl.arange(0, D)
    running_maximum = tl.full((), -float("inf"), tl.float32)
    running_denominator = tl.full((), 0, tl.float32)
    accumulator = tl.full((D,), 0, tl.float32)
    for group in range(tl.cdiv(count, GROUP)):
        positions = group*GROUP + contribution
        base = tl.load(MergeIndices + (head*BATCH+request)*MAX_CONTRIB + positions,
                       positions < count, other=-1)
        packed_row = base.to(tl.int64) + group_head
        live = (positions < count) & (base >= 0)
        maximum = tl.load(Maximum + packed_row, live, other=-float("inf"))
        denominator = tl.load(Denominator + packed_row, live, other=0.0)
        next_maximum = tl.maximum(running_maximum, tl.max(maximum, 0))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        factor = tl.where(running_denominator > 0, tl.exp(running_maximum-safe_maximum), 0.0)
        weights = tl.where(live & (denominator > 0), tl.exp(maximum-safe_maximum), 0.0)
        values = tl.load(U + packed_row[:, None]*D + cols[None, :], live[:, None], other=0.0)
        accumulator = accumulator*factor + tl.sum(values*weights[:, None], 0)
        running_denominator = running_denominator*factor + tl.sum(denominator*weights, 0)
        running_maximum = next_maximum
    valid = running_denominator > 0
    tl.store(Output + (request*HEADS*GQA+head*GQA+group_head)*D + cols, accumulator/tl.where(valid, running_denominator, 1.0))
    tl.store(LSE + head*ROWS+row,
             tl.where(valid, running_maximum+tl.log(running_denominator), -float("inf")))
