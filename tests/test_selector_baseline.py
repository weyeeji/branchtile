import pytest
import torch

from selector_baseline import (FusedKnownLSEScorer, qc_scores, reference_scores,
                               select_support, streaming_lse)


@pytest.mark.parametrize("n,budget", [(7,7),(36,36),(37,37),(81,51),(257,128)])
def test_stable_protected_ties(n, budget):
    scores = torch.ones(2, n)
    selected = select_support(scores, budget)
    expected = torch.zeros_like(selected)
    expected[:, :4] = True
    expected[:, max(0,n-32):] = True
    left = budget-int(expected[0].sum())
    expected[:, 4:4+left] = True
    assert torch.equal(selected, expected)


def test_fp64_causality_and_pool_padding():
    torch.manual_seed(8)
    q = torch.randn(1, 4, 3, 128).bfloat16()
    k = torch.randn(1, 2, 59, 128).bfloat16()
    positions = torch.tensor([53, 54, 55])
    actual = qc_scores(q, k, positions, 5, 49, chunk_size=13)
    expected, _ = reference_scores(q, k, positions, 5, 49, chunk_size=17)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-8)
    # Future full-history K must not alter either scores or normalization.
    changed = k.clone()
    changed[:, :, 56:] *= 100
    assert torch.equal(actual, qc_scores(q, changed, positions, 5, 49, chunk_size=13))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("g,w,n,start,end,bn", [
    (4,23,197,5,169,32), (8,32,301,7,279,64),
    (4,1,59,0,59,128), (4,7,43,2,39,64)])
def test_fused_score_fp64(g, w, n, start, end, bn):
    torch.manual_seed(2026)
    q = torch.randn(1, 2*g, w, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 2, n, 128, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(n-w, n, device="cuda")
    lse = streaming_lse(q, k, pos, 73)
    expected, _ = reference_scores(q, k, pos, start, end, 71)
    scorer = FusedKnownLSEScorer(q, k, pos, start, end, bn)
    actual = scorer.run(lse)
    torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=2e-8)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        masks = select_support(scorer.run(lse), min(64,end-start))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(masks, select_support(actual,min(64,end-start)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("h,n,budget", [(2,257,128),(8,6126,4096),(4,20896,4096)])
def test_native_topk_ties_and_graph(h,n,budget):
    torch.manual_seed(20261016)
    scores = torch.randn(h,n,device="cuda")
    # Many ties, including at selection boundaries, exercise lower-position policy.
    scores = (scores*8).round()
    expected = select_support(scores,budget,"argsort")
    actual = select_support(scores,budget,"flashinfer")
    assert torch.equal(actual,expected)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = select_support(scores,budget,"flashinfer")
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(actual,expected)
