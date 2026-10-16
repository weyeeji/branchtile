import pytest
import torch
from branchtile.attention import make_plan, reference, synthetic_owners
from branchtile.attention_v2 import Attention


@pytest.mark.parametrize("block_n", [32, 64])
@pytest.mark.parametrize("gqa,pattern", [(8, "mixed"), (4, "cross"), (8, "empty")])
@pytest.mark.parametrize("region_partials", [False, True])
def test_explicit_shared_reuse(block_n, gqa, pattern, region_partials):
    torch.manual_seed(1234)
    owners = synthetic_owners(2, 259, pattern)
    owners[1] &= 7
    plan = make_plan(owners, 4, gqa, "cuda")
    q = torch.randn(2, 4*gqa, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 259, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    op = Attention(plan, splits=3, block_n=block_n, region_partials=region_partials)
    actual, lse = op(q, k, v)
    expected, expected_lse = reference(q, k, v, owners, 4, gqa)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.015)
    torch.testing.assert_close(lse, expected_lse.float(), atol=0.00005, rtol=0.00005)


def test_runtime_m16_to_m32_with_shared_reuse():
    torch.manual_seed(5678)
    owners = synthetic_owners(2, 259, "mixed")
    plan = make_plan(owners, device="cuda")
    q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 259, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    op = Attention(plan, splits=3, block_n=64)
    for change in (False, True):
        if change:
            q.normal_()
            owners[owners == 1] = 15
            plan.counts[:, 0] = 32
            plan.row_map[:, 0] = torch.arange(32, device="cuda", dtype=torch.int32)
            plan.owner_rows[:, 0] = 1
        actual, lse = op(q, k, v)
        expected, expected_lse = reference(q, k, v, owners)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.015)
        torch.testing.assert_close(lse, expected_lse.float(), atol=0.00005, rtol=0.00005)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("compact", [True, False])
def test_gqa4_small_batch(batch, compact):
    torch.manual_seed(9876)
    owners = synthetic_owners(2, 79, "mixed", batch=batch)
    plan = make_plan(owners, batch, 4, "cuda")
    q = torch.randn(2, batch*4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 79, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    op = Attention(plan, splits=3, block_n=32, compact=compact)
    actual, lse = op(q, k, v)
    expected, expected_lse = reference(q, k, v, owners, batch, 4)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.015)
    torch.testing.assert_close(lse, expected_lse.float(), atol=0.00005, rtol=0.00005)
