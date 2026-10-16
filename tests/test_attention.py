import pytest
import torch
from branchtile.attention import Attention, make_plan, reference, synthetic_owners


@pytest.mark.parametrize("pattern,batch,gqa,tokens,splits", [
    ("cross", 4, 8, 197, 1),
    ("mixed", 4, 8, 301, 3),
    ("full", 4, 8, 129, 8),
    ("empty", 4, 8, 31, 2),
    ("disjoint", 4, 4, 97, 1),
    ("tree", 3, 8, 127, 2),
    ("full", 1, 8, 17, 3),
])
@pytest.mark.parametrize("compact", [True, False])
def test_attention(pattern, batch, gqa, tokens, splits, compact):
    torch.manual_seed(123)
    heads, dim = 2, 128
    owners = synthetic_owners(heads, tokens, pattern, batch)
    # One head has no tokens owned by the final request, testing empty rows.
    owners[1] &= (1 << (batch-1))-1
    plan = make_plan(owners, batch, gqa, "cuda")
    q = torch.randn(heads, batch*gqa, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(heads, tokens, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    op = Attention(plan, splits=splits, block_n=32, compact=compact)
    actual, lse = op(q, k, v)
    expected, expected_lse = reference(q, k, v, owners, batch, gqa)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.015)
    torch.testing.assert_close(lse, expected_lse.float(), atol=0.00005, rtol=0.00005)
    assert torch.isfinite(actual).all()


def test_runtime_metadata_changes_both_m_paths():
    torch.manual_seed(456)
    owners = synthetic_owners(2, 257, "mixed")
    plan = make_plan(owners, device="cuda")
    q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 257, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    op = Attention(plan, splits=3, block_n=32)
    for change in (False, True):
        if change:
            q.normal_()
            # Reuse the compiled CTA, changing its first region from M16 to M32.
            owners[owners == 1] = 15
            plan.counts[:, 0] = 32
            plan.row_map[:, 0] = torch.arange(32, device="cuda", dtype=torch.int32)
            plan.owner_rows[:, 0] = 1
        actual, lse = op(q, k, v)
        expected, expected_lse = reference(q, k, v, owners)
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.015)
        torch.testing.assert_close(lse, expected_lse.float(), atol=0.00005, rtol=0.00005)


@pytest.mark.parametrize("compact", [True, False])
def test_region_partials(compact):
    torch.manual_seed(789)
    owners = synthetic_owners(2, 257, "mixed")
    plan = make_plan(owners, device="cuda")
    q = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 257, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    op = Attention(plan, splits=2, block_n=32, compact=compact, region_partials=True)
    actual, lse = op(q, k, v)
    expected, expected_lse = reference(q, k, v, owners)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.015)
    torch.testing.assert_close(lse, expected_lse.float(), atol=0.00005, rtol=0.00005)
