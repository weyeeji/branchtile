"""Exact-support qualification for the native joint-query union baseline."""
import math

import pytest
import torch

from branchtile.attention import reference, synthetic_owners
from flashinfer_union import NativeUnionPrefill


@pytest.mark.parametrize("pattern,b,g,n,split,normalize", [
    ("cross", 4, 4, 197, None, False),
    ("mixed", 4, 8, 301, 128, True),
    ("tree", 3, 4, 127, None, True),
    ("disjoint", 4, 8, 137, 64, False),
    ("full", 1, 4, 17, None, True),
])
def test_native_union_same_supports(pattern, b, g, n, split, normalize):
    torch.manual_seed(602)
    h, d = 2, 128
    owners = synthetic_owners(h, n, pattern, b, seed=602)
    owners[:, ::11] = 0  # Different union cardinalities and noncontiguous pages.
    owners[1, ::7] = 0
    owners[:, :4] = (1 << b) - 1  # Common system prefix.
    for request in range(b):
        owners[:, n - b + request] = 1 << request  # Complete private suffix.
    q = torch.randn(h, b * g, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(h, n, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    op = NativeUnionPrefill(q, k, v, owners, b, fixed_split_size=split, normalize_lse=normalize)
    # The native pages alias the original immutable payload.
    assert op.kv[0].data_ptr() == k.data_ptr()
    assert op.kv[1].data_ptr() == v.data_ptr()
    out, lse = op.run()
    ref, ref_lse = reference(q, k, v, owners, b, g)
    scale = 1 if normalize else math.log(2)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.006, rtol=0.015)
    torch.testing.assert_close((lse * scale).float(), ref_lse.float(), atol=0.0003, rtol=0.0001)
    assert out.data_ptr() == op.out.data_ptr()
    assert lse.data_ptr() == (op.lse_ln if normalize else op.lse).data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.run()
    # Same graph must consume the live Q view, not a stale packed query copy.
    q.mul_(0.75)
    graph.replay()
    torch.cuda.synchronize()
    ref, ref_lse = reference(q, k, v, owners, b, g)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.006, rtol=0.015)
    torch.testing.assert_close((lse * scale).float(), ref_lse.float(), atol=0.0003, rtol=0.0001)


def test_native_union_empty_request_with_nonempty_union():
    torch.manual_seed(603)
    owners = torch.tensor([[1, 3, 1, 2, 0], [2, 2, 2, 0, 0]], dtype=torch.int64)
    q = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 5, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out, lse = NativeUnionPrefill(q, k, v, owners, 2, normalize_lse=True).run()
    ref, ref_lse = reference(q, k, v, owners, 2, 4)
    torch.testing.assert_close(out.float(), ref.float(), atol=0.006, rtol=0.015)
    torch.testing.assert_close(lse.float(), ref_lse.float(), atol=0.0003, rtol=0.0001)


def test_native_union_rejects_zero_union_head():
    q = torch.zeros(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(2, 5, 128, device="cuda", dtype=torch.bfloat16)
    owners = torch.ones(2, 5, dtype=torch.int64)
    owners[1] = 0
    with pytest.raises(ValueError, match="nonempty union"):
        NativeUnionPrefill(q, k, k, owners, 4)
