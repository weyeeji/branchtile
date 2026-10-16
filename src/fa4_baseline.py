"""Unmodified official FA4 b35 kernels on exact supports (SM120).

FullSourceMasked uses the original physical KV tensors with a token mask and
therefore also visits unselected source columns. GatheredDecode is E0 with
independent request supports; its complete call includes payload gathering.
Neither pretends that SM120 supports FA4's paged or block-sparse interfaces.
"""
import math
import torch
import cutlass
import cutlass.cute as cute
from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute import utils


@cute.jit
def owner_mask(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    # The SM120 path calls mask_mod with length-one TensorSSA coordinates.
    b = utils.ssa_to_scalar(batch_idx)
    q = utils.ssa_to_scalar(q_idx)
    k = utils.ssa_to_scalar(kv_idx)
    keep = (aux_tensors[0][b, k] & (1 << q)) != 0
    return utils.scalar_to_ssa(keep, cutlass.Boolean)


class FullSourceMasked:
    lse_log_base = math.e

    def __init__(self, q, k, v, owners, b, *, pack_gqa=True, tile_mn=None):
        h, n, d = k.shape
        self.original_shape = q.shape
        g = q.shape[1] // b
        self.q = q.view(h, b, g, d)
        self.k = k.view(h, n, 1, d)
        self.v = v.view(h, n, 1, d)
        self.owners = owners.to(device=q.device, dtype=torch.int32).contiguous()
        self.out = torch.empty_like(self.q)
        self.lse = torch.empty((h, g, b), dtype=torch.float32, device=q.device)
        self.pack_gqa = pack_gqa
        self.tile_mn = tile_mn
        assert self.k.data_ptr() == k.data_ptr() and self.v.data_ptr() == v.data_ptr()

    def run(self):
        out, lse, *_ = _flash_attn_fwd(self.q, self.k, self.v, causal=False,
            mask_mod=owner_mask, aux_tensors=[self.owners], num_splits=0,
            pack_gqa=self.pack_gqa, tile_mn=self.tile_mn, return_lse=True,
            out=self.out, lse=self.lse)
        return out.view(self.original_shape), lse.transpose(1, 2).reshape(self.original_shape[:2])


class GatheredDecode:
    lse_log_base = math.e

    def __init__(self, q, k, v, owners, b, *, pack_gqa=True, tile_mn=None):
        h, n, d = k.shape
        self.original_shape = q.shape
        g = q.shape[1] // b
        indices, pointers = [], [0]
        for head in range(h):
            for request in range(b):
                ids = torch.where((owners[head] & (1 << request)) != 0)[0]
                indices.append(ids + head*n)
                pointers.append(pointers[-1] + len(ids))
        self.indices = torch.cat(indices).to(q.device)
        self.source_k, self.source_v = k.view(h*n, d), v.view(h*n, d)
        self.q = q.view(h*b, g, d)
        self.k = torch.empty((pointers[-1], d), dtype=k.dtype, device=q.device)
        self.v = torch.empty_like(self.k)
        self.cuq = torch.arange(h*b+1, dtype=torch.int32, device=q.device)
        self.cuk = torch.tensor(pointers, dtype=torch.int32, device=q.device)
        self.max_k = max(b-a for a,b in zip(pointers,pointers[1:]))
        self.out = torch.empty_like(self.q)
        self.lse = torch.empty((g,h*b), dtype=torch.float32, device=q.device)
        self.pack_gqa, self.tile_mn = pack_gqa, tile_mn
        self.payload_copy_bytes_per_call = 2*self.k.numel()*self.k.element_size()

    def gather(self):
        torch.index_select(self.source_k, 0, self.indices, out=self.k)
        torch.index_select(self.source_v, 0, self.indices, out=self.v)

    def run_attention_only(self):
        out, lse, *_ = _flash_attn_fwd(self.q, self.k[:,None], self.v[:,None],
            cu_seqlens_q=self.cuq, cu_seqlens_k=self.cuk,
            max_seqlen_q=1, max_seqlen_k=self.max_k, causal=False,
            num_splits=0, pack_gqa=self.pack_gqa, tile_mn=self.tile_mn,
            return_lse=True, out=self.out, lse=self.lse)
        return out.view(self.original_shape), lse.transpose(0,1).reshape(self.original_shape[:2])

    def run(self):
        self.gather()
        return self.run_attention_only()
