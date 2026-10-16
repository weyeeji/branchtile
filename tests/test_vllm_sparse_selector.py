"""CPU regression for APC-skipped observation Q outside source's own last32."""
from unittest.mock import patch

import pytest
import torch

from src.vllm_sparse_selector import FrozenQCSelector


def fixture():
    selector = FrozenQCSelector([list(range(180)), list(range(132))], (4, 100), 36)
    positions = torch.arange(100, 180)
    query = positions[:, None, None].expand(-1, 1, 4).float().clone()
    table = torch.arange(12)
    selector.observe(0, 0, positions, query, table, 16)
    cache = torch.zeros((12, 1, 16, 8))
    return selector, cache, table


def test_reuses_exact_prefix_q_outside_source_last32():
    selector, cache, table = fixture()
    def scores(q, k, positions, start, end):
        assert torch.equal(q[0, 0, :, 0], torch.arange(100, 132).float())
        return torch.arange(end-start).float()[None]
    with patch('src.vllm_sparse_selector.qc_scores', scores), patch('torch.cuda.synchronize'):
        selector.freeze(0, 1, cache, table)
    assert selector.records[0]['reused_other_question_prefix_Q'] == 32
    support = selector.logical_supports(0, 1, 132)[0]
    assert len(support) == 4 + 36 + 33
    assert len(torch.unique(support)) == len(support)
    assert torch.equal(support[-33:], torch.arange(100, 133))
    assert torch.isin(torch.arange(68, 100), support).all()


def test_rejects_reused_page_with_different_physical_identity():
    selector, cache, table = fixture()
    table = table.clone()
    table[6] = 11
    with pytest.raises(RuntimeError, match='Missing exact APC observation Q'):
        selector.freeze(0, 1, cache, table)
