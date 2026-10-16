"""CPU invariants and exact equivalence to the frozen pre-integration algorithm."""
import ast
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from src.joint_support_exchange import (
    FROZEN_SOURCE_SHA256, PROTOCOL_ID, PROTOCOL_SHA256,
    JointSupportExchangeConfig, apply_joint_exchange,
)

ROOT = Path(__file__).resolve().parents[1]


def config(epsilon=0, order='ascending'):
    return JointSupportExchangeConfig(PROTOCOL_ID, PROTOCOL_SHA256,
        epsilon, order, 2, 4, 32)


def frozen_reference():
    path = ROOT/'scripts/probe_joint_support_exchange.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_SOURCE_SHA256
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('score_sum', 'stable_order', 'exchange')]
    namespace = {'np':np, 'math':math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['exchange']


def fixture(seed=0, b=3, h=2, n=192, k=80):
    rng = np.random.default_rng(seed)
    scores = rng.choice(np.arange(950, 1001), size=(b,h,n)).astype(np.float32)/1000
    masks = np.zeros((b,h,n), dtype=bool)
    masks[:,:,:4] = True
    masks[:,:,-32:] = True
    candidates = np.arange(4,n-32)
    for q in range(b):
        for head in range(h):
            selected = candidates[np.argsort(-scores[q,head,candidates],kind='stable')[:k-36]]
            masks[q,head,selected] = True
    return scores, masks


@pytest.mark.parametrize('epsilon', [0,.001,.005,.01])
@pytest.mark.parametrize('order', ['ascending','descending'])
def test_matches_frozen_function_and_preserves_inputs(epsilon, order):
    scores, masks = fixture()
    original_scores, original_masks = scores.copy(), masks.copy()
    independent, actual, record = apply_joint_exchange(scores,masks,config(epsilon,order),
        document_start=34,common_source_tokens=scores.shape[-1]-2)
    protected = np.zeros(scores.shape[-1],dtype=bool)
    protected[:4] = True
    protected[-32:] = True
    request_order = list(range(len(scores)))
    if order == 'descending':request_order.reverse()
    for head in range(scores.shape[1]):
        expected,events = frozen_reference()(scores[:,head].astype(np.float64),masks[:,head],
            protected,epsilon,request_order,2)
        assert np.array_equal(actual[:,head],expected)
        assert record['records'][head]['events'] == events
    assert np.array_equal(scores,original_scores)
    assert np.array_equal(masks,original_masks)
    assert np.array_equal(independent,masks)
    assert np.all(actual.sum(-1) == masks.sum(-1))
    assert actual[:,:,protected].all()
    assert all(r['score_loss_fraction'] <= epsilon for h in record['records'] for r in h['requests'])
    assert all(h['new_geometry']['common_source_union_tokens'] <= h['independent_geometry']['common_source_union_tokens'] for h in record['records'])


def test_exact_tie_exchange_hashes_and_full_budget():
    scores = np.ones((2,1,64),dtype=np.float32)
    masks = np.zeros_like(scores,dtype=bool)
    masks[:,:,:4] = True
    masks[:,:,-32:] = True
    masks[0,0,4:8] = True
    masks[1,0,6:10] = True
    initial,actual,record = apply_joint_exchange(scores,masks,config(),document_start=34)
    assert actual.any(0).sum() == 40 and initial.any(0).sum() == 42
    assert all(r['score_loss_fraction'] == 0 for r in record['records'][0]['requests'])
    row = record['records'][0]['requests'][0]
    assert row['independent_mask_sha256'] != row['new_mask_sha256']
    assert row['new_mask_sha256'] == hashlib.sha256((np.flatnonzero(actual[0,0])+34).astype('<i8').tobytes()).hexdigest()
    masks[:] = True
    _,actual,_ = apply_joint_exchange(scores,masks,config(.01))
    assert actual.all()


def test_rejects_implicit_changed_or_invalid_contracts():
    with pytest.raises(ValueError,match='All opt-in fields'):
        JointSupportExchangeConfig.from_mapping({'epsilon':.001,'request_order':'ascending'})
    with pytest.raises(ValueError,match='epsilon'):
        config(.1)
    with pytest.raises(ValueError,match='order'):
        config(order='best')
    scores,masks = fixture()
    with pytest.raises(TypeError,match='float32'):
        apply_joint_exchange(scores.astype(np.float64),masks,config())
    with pytest.raises(ValueError,match='tail'):
        apply_joint_exchange(scores,masks,config(),common_source_tokens=10)
    with pytest.raises(ValueError,match='positive'):
        apply_joint_exchange(np.zeros_like(scores),masks,config())
    damaged = masks.copy();damaged[0,0,0] = False
    with pytest.raises(ValueError,match='protected'):
        apply_joint_exchange(scores,damaged,config())
    bad_scores = scores.copy();bad_scores[0,0,np.flatnonzero(~masks[0,0])[0]] = 2
    with pytest.raises(ValueError,match='top-k'):
        apply_joint_exchange(bad_scores,masks,config())
