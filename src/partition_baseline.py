"""Executable exact fixed-query partitions sharing an immutable physical KV source.

E1 enumerates all Bell(B) partitions (15 for B=4). Native execution batches all
(head, group) items into one paged-prefill call. Noncanonical query packing and
output restoration are part of run(), never hidden in preparation. An alternative
backend runs the existing fixed-union kernel per group, also including packing.
"""
import math
import time

import flashinfer
import torch
import triton
import triton.language as tl


def all_partitions(batch):
    if not 1 <= batch <= 4:
        raise ValueError("Supported request count is 1..4")
    result = [()]
    for request in range(batch):
        following = []
        for partition in result:
            following.append(partition + ((request,),))
            for i in range(len(partition)):
                following.append(partition[:i] + (partition[i] + (request,),) + partition[i + 1:])
        result = following
    return sorted(result)


def validate_partition(partition, batch):
    partition = tuple(tuple(group) for group in partition)
    if not all(partition) or sorted(x for group in partition for x in group) != list(range(batch)):
        raise ValueError("Partition must contain each request exactly once")
    return partition


@triton.jit
def _gather_q(Q, Map, Packed, G: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    item, g = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, BD)
    source = tl.load(Map + item)
    value = tl.load(Q + (source * G + g) * D + d, d < D, other=0)
    tl.store(Packed + (item * G + g) * D + d, value, d < D)


@triton.jit
def _restore_merge(O, S, Slots, Output, LSE, G: tl.constexpr, D: tl.constexpr,
                   PARTS: tl.constexpr, BP: tl.constexpr, BD: tl.constexpr):
    item, g = tl.program_id(0), tl.program_id(1)
    p, d = tl.arange(0, BP), tl.arange(0, BD)
    slots = tl.load(Slots + item * PARTS + p, p < PARTS, other=-1)
    score = tl.load(S + slots * G + g, slots >= 0, other=-float('inf'))
    valid = (slots >= 0) & (score != -float('inf'))
    maximum = tl.max(score, 0)
    weights = tl.where(valid, tl.exp2(score - maximum), 0)
    denominator = tl.sum(weights, 0)
    values = tl.load(O + (slots[:, None] * G + g) * D + d[None, :],
                     valid[:, None] & (d[None, :] < D), other=0)
    value = tl.sum(values.to(tl.float32) * weights[:, None], 0) / tl.maximum(denominator, 1e-30)
    tl.store(Output + (item * G + g) * D + d, value, d < D)
    ln_lse = (maximum + tl.log2(denominator)) * 0.6931471805599453
    tl.store(LSE + item * G + g, tl.where(denominator > 0, ln_lse, -float('inf')))


@triton.jit
def _scatter_group(O, S, Map, Output, LSE, G: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    item, g = tl.program_id(0), tl.program_id(1)
    target = tl.load(Map + item)
    d = tl.arange(0, BD)
    value = tl.load(O + (item * G + g) * D + d, d < D, other=0)
    tl.store(Output + (target * G + g) * D + d, value, d < D)
    score = tl.load(S + item * G + g)
    tl.store(LSE + target * G + g, score)


class _NativeGroupedAttention:
    """Internal batched native executor; items=(head,requests,ids,mask_or_None).

The same request may appear in several items (hierarchical partials), in which
case run() performs a stable LSE-weighted merge. Empty items are omitted. No K/V
payload gather occurs. Native partial O is BF16 and LSE is log2; final LSE is ln.
"""
    def _initialize(self, q, k, v, batch, items, *, fixed_split_size, started, metadata_ms):
        h, n, d = k.shape
        g = q.shape[1] // batch
        if d != 128 or q.shape != (h, batch * g, d) or v.shape != k.shape:
            raise ValueError("Expected contiguous q[H,B*G,128], k/v[H,N,128]")
        if not all(t.is_cuda and t.is_contiguous() and t.device == q.device for t in (q, k, v)):
            raise ValueError("Q/K/V must be contiguous on the same CUDA device")
        self.q_source, self.batch, self.gqa = q, batch, g
        self.shape, self.metadata_ms = tuple(q.shape), metadata_ms
        self.kv = (k.view(h * n, 1, 1, d), v.view(h * n, 1, 1, d))
        self.payload_aliases_source = all(a.data_ptr() == b.data_ptr() for a, b in zip(self.kv, (k, v)))
        pointers, qptr, indices, maps, masks = [0], [0], [], [], []
        slots = [[] for _ in range(h * batch)]
        use_mask = any(item[3] is not None for item in items)
        self.logical_kv_token_visits = 0
        for head, requests, ids, mask in items:
            if not len(ids):
                continue
            pointers.append(pointers[-1] + len(ids))
            indices.append(ids.to(torch.int32) + head * n)
            self.logical_kv_token_visits += len(ids)
            for request in requests:
                slots[head * batch + request].append(len(maps))
                maps.append(head * batch + request)
            qptr.append(len(maps))
            if use_mask:
                masks.append((mask if mask is not None else torch.ones(len(requests), len(ids), dtype=torch.bool)).flatten())
        self.item_count, self.query_positions = len(pointers) - 1, len(maps)
        self.max_contributions = max(1, max(map(len, slots)))
        slot_cpu = torch.full((h * batch, self.max_contributions), -1, dtype=torch.int32)
        for row, contributions in enumerate(slots):
            slot_cpu[row, :len(contributions)] = torch.tensor(contributions, dtype=torch.int32)
        self.slots = slot_cpu.to(q.device)
        self.query_map = torch.tensor(maps, dtype=torch.int32, device=q.device)
        self.identity = maps == list(range(h * batch))
        self.direct_output = self.identity and self.max_contributions == 1 and all(
            mask is None or bool(mask.any(dim=1).all()) for _, _, ids, mask in items if len(ids))
        self.q = q.view(h * batch, g, d) if self.identity else torch.empty((max(1, len(maps)), g, d), dtype=q.dtype, device=q.device)
        self.partial_output = torch.empty((max(1, len(maps)), g, d), dtype=q.dtype, device=q.device)
        self.partial_lse = torch.empty((max(1, len(maps)), g), dtype=torch.float32, device=q.device)
        self.output = torch.empty_like(q)
        self.lse = torch.empty((h, batch * g), dtype=torch.float32, device=q.device)
        self.workspace = None
        self.wrapper = None
        if self.item_count:
            self.indptr = torch.tensor(pointers, dtype=torch.int32, device=q.device)
            self.qo_indptr = torch.tensor(qptr, dtype=torch.int32, device=q.device)
            self.indices = torch.cat(indices).to(q.device)
            self.last = torch.ones(self.item_count, dtype=torch.int32, device=q.device)
            self.custom_mask = torch.cat(masks).to(q.device) if use_mask else None
            self.workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=q.device)
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, kv_layout="NHD", backend="fa2")
            self.wrapper.plan(self.qo_indptr, self.indptr, self.indices, self.last, g, 1, d, 1,
                              causal=False, custom_mask=self.custom_mask, pos_encoding_mode="NONE",
                              q_data_type=q.dtype, kv_data_type=k.dtype, fixed_split_size=fixed_split_size)
        torch.cuda.synchronize(q.device)
        self.prepare_ms = (time.perf_counter() - started) * 1000
        self.workspace_bytes = 0 if self.workspace is None else self.workspace.numel()

    def run(self):
        if self.query_positions and not self.identity:
            _gather_q[(self.query_positions, self.gqa)](self.q_source, self.query_map, self.q,
                                                       self.gqa, 128, 128, num_warps=4)
        if self.wrapper is not None:
            self.wrapper.run(self.q, self.kv, out=self.partial_output, lse=self.partial_lse, return_lse=True)
        if self.direct_output:
            torch.mul(self.partial_lse, math.log(2), out=self.lse.view(-1, self.gqa))
            return self.partial_output.view(self.shape), self.lse
        # Restoration and exact-partition/cascade normalization are always timed.
        _restore_merge[(self.shape[0] * self.batch, self.gqa)](
            self.partial_output, self.partial_lse, self.slots, self.output, self.lse,
            self.gqa, 128, self.max_contributions, triton.next_power_of_2(self.max_contributions), 128, num_warps=4)
        return self.output, self.lse


class NativePartitionAttention(_NativeGroupedAttention):
    def __init__(self, q, k, v, owners, batch, partition, *, fixed_split_size=None):
        started = time.perf_counter()
        partition = validate_partition(partition, batch)
        owners = owners.to(device="cpu", dtype=torch.int64)
        if owners.shape != k.shape[:2] or int(owners.min()) < 0 or int(owners.max()) >= 1 << batch:
            raise ValueError("Invalid request owner mask")
        items = []
        for head in range(k.shape[0]):
            for group in partition:
                bits = sum(1 << request for request in group)
                ids = torch.where((owners[head] & bits) != 0)[0]
                mask = torch.stack([(owners[head, ids] & (1 << request)) != 0 for request in group])
                items.append((head, group, ids, mask))
        self.partition = partition
        self._initialize(q, k, v, batch, items, fixed_split_size=fixed_split_size,
                         started=started, metadata_ms=(time.perf_counter() - started) * 1000)


class OwnPartitionAttention:
    def __init__(self, q, k, v, owners, batch, partition, *, splits=64, block_n=64, threads=128):
        from branchtile.attention import make_plan
        from branchtile.union import FixedUnion
        started = time.perf_counter()
        self.partition = validate_partition(partition, batch)
        self.q_source, self.batch = q, batch
        self.gqa = q.shape[1] // batch
        h, _, d = k.shape
        self.shape = tuple(q.shape)
        self.output, self.lse = torch.empty_like(q), torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        self.parts = []
        self.payload_aliases_source = True
        self.metadata_ms = 0
        self.workspace_bytes = 0
        self.logical_kv_token_visits = 0
        for group in self.partition:
            meta_start = time.perf_counter()
            mapped = sum(((owners & (1 << request)) != 0).long() << i for i, request in enumerate(group))
            plan = make_plan(mapped, len(group), self.gqa, q.device)
            self.logical_kv_token_visits += int((mapped != 0).sum())
            mapping = torch.tensor([head * batch + b for head in range(h) for b in group], device=q.device, dtype=torch.int32)
            identity = len(group) == batch and list(group) == list(range(batch))
            packed = q if identity else torch.empty((h, len(group) * self.gqa, d), device=q.device, dtype=q.dtype)
            self.metadata_ms += (time.perf_counter() - meta_start) * 1000
            op = FixedUnion(plan, mapped, splits, block_n, threads)
            self.parts.append((op, mapping, packed, identity, k, v))
            self.workspace_bytes += sum(t.numel() * t.element_size() for t in (op.u, op.m, op.l, op.output, op.lse))
        torch.cuda.synchronize(q.device)
        self.prepare_ms = (time.perf_counter() - started) * 1000

    def run(self):
        for op, mapping, packed, identity, k, v in self.parts:
            if not identity:
                _gather_q[(mapping.numel(), self.gqa)](self.q_source, mapping, packed, self.gqa, 128, 128, num_warps=4)
            out, lse = op(packed, k, v)
            if identity:
                return out, lse
            _scatter_group[(mapping.numel(), self.gqa)](out, lse, mapping, self.output, self.lse, self.gqa, 128, 128, num_warps=4)
        return self.output, self.lse
