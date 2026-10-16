"""CPU contract tests: reject routing changes, not merely large errors."""
import os
import sys
import torch
from scripts.vllm_tinygemm_gate_worker import bitwise_equal, routing_identity, bf16_comparison


def test_routing_exact_identity():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    weights = torch.tensor([[.5, .5]], dtype=torch.float32)
    ids = torch.tensor([[2, 5]], dtype=torch.int32)
    assert routing_identity(weights, ids, weights.clone(), ids.clone())
    assert not routing_identity(weights, ids, weights, ids.flip(-1))
    changed = weights.clone(); changed[0, 0] = torch.nextafter(changed[0, 0], torch.tensor(float('inf')))
    assert not routing_identity(weights, ids, changed, ids)
    assert not routing_identity(weights, ids, weights, ids.long())
    assert not routing_identity(weights, ids, torch.full_like(weights, float('nan')), ids)
    assert not bitwise_equal(torch.tensor([0.]), torch.tensor([-0.]))
    assert not torch.cuda.is_initialized()


def test_bf16_errors_are_descriptive():
    x = torch.tensor([1., -1., 0.], dtype=torch.bfloat16)
    y = x.clone(); y[0] = torch.nextafter(x[0], torch.tensor(float('inf'), dtype=x.dtype))
    report = bf16_comparison(y, x)
    assert report['different_elements'] == 1 and report['total_elements'] == 3
    assert report['max_BF16_encoding_steps'] == 1
    assert 'passed' not in report
