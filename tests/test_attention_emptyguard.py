"""GPU qualifications for root-owned, budgeted execution; not run by CPU audit."""
import pytest
import torch

from branchtile.attention import make_plan, reference
from branchtile.attention_emptyguard import Attention
from branchtile.attention_v2 import Attention as V2Attention


@pytest.mark.parametrize("region_partials", [False, True])
@pytest.mark.parametrize("pattern", ["all_empty", "empty_head", "uneven_shards"])
def test_empty_shards_preserve_state_and_reference(pattern, region_partials):
    # The very small regions and splits=8 deliberately create many shards
    # with begin==end even when the region as a whole contains tokens.
    torch.manual_seed(20261016)
    batch, gqa, heads, tokens, dim = 3, 8, 2, 19, 128
    owners = torch.zeros((heads, tokens), dtype=torch.int64)
    if pattern == "empty_head":
        owners[0] = torch.tensor([1, 2, 4, 3, 5, 6, 7, 0, 0, 7, 1, 2, 4, 3, 5, 6, 7, 0, 0])
    elif pattern == "uneven_shards":
        owners[0] = torch.tensor([1, 2, 4, 3, 5, 6, 7, 0, 0, 7, 1, 2, 4, 3, 5, 6, 7, 0, 0])
        owners[1] = torch.tensor([7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    plan = make_plan(owners, batch, gqa, "cuda")
    q = torch.randn((heads, batch * gqa, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((heads, tokens, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    guarded = Attention(plan, splits=8, block_n=32, region_partials=region_partials)
    baseline = V2Attention(plan, splits=8, block_n=32, region_partials=region_partials)
    actual, lse = guarded(q, k, v)
    original, original_lse = baseline(q, k, v)
    expected, expected_lse = reference(q, k, v, owners, batch, gqa)
    assert torch.equal(actual, original)
    assert torch.equal(lse, original_lse)
    torch.testing.assert_close(actual.float(), expected.float(), atol=0.006, rtol=0.015)
    torch.testing.assert_close(lse, expected_lse.float(), atol=0.00005, rtol=0.00005)
    assert not torch.isnan(actual).any()
    assert not torch.isnan(lse).any()
    if pattern == "all_empty":
        assert torch.count_nonzero(guarded.u) == 0
        assert torch.count_nonzero(guarded.l) == 0
        assert torch.isneginf(guarded.m).all()
        assert torch.count_nonzero(actual) == 0
        assert torch.isneginf(lse).all()
    elif pattern == "empty_head":
        assert torch.count_nonzero(actual[1]) == 0
        assert torch.isneginf(lse[1]).all()
