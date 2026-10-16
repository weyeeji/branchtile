import pytest
import torch
from branchtile.attention import make_plan, reference, synthetic_owners
from branchtile.attention_v3 import Attention


@pytest.mark.parametrize('pattern', ['cross', 'mixed', 'full', 'empty'])
def test_register_canonical_u(pattern):
    torch.manual_seed(106)
    owners = synthetic_owners(2, 259, pattern)
    owners[1] &= 7
    plan = make_plan(owners, device='cuda')
    q = torch.randn(2, 32, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, 259, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    op = Attention(plan, splits=3, block_n=64)
    for change in (False, True):
        if change:
            q.normal_()
        actual, lse = op(q, k, v)
        expected, expected_lse = reference(q, k, v, owners)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.015)
        torch.testing.assert_close(lse, expected_lse.float(), atol=0.00005, rtol=0.00005)


def test_runtime_change():
    torch.manual_seed(107)
    owners = synthetic_owners(2, 259, 'mixed')
    plan = make_plan(owners, device='cuda')
    q = torch.randn(2, 32, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, 259, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    op = Attention(plan, splits=3, block_n=64)
    for change in (False, True):
        if change:
            owners[owners == 1] = 15
            plan.counts[:, 0] = 32
            plan.row_map[:, 0] = torch.arange(32, device='cuda', dtype=torch.int32)
            plan.owner_rows[:, 0] = 1
        actual, lse = op(q, k, v)
        expected, expected_lse = reference(q, k, v, owners)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.015)
        torch.testing.assert_close(lse, expected_lse.float(), atol=0.00005, rtol=0.00005)
