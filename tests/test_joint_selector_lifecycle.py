"""Original selector compatibility and atomic opt-in cohort lifecycle on CPU."""
import copy
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.joint_support_exchange import JointSupportExchangeConfig, PROTOCOL_ID, PROTOCOL_SHA256
from src.vllm_sparse_selector import FrozenQCSelector


def configuration(epsilon=.01):
    from dataclasses import asdict
    return asdict(JointSupportExchangeConfig(PROTOCOL_ID,PROTOCOL_SHA256,epsilon,'ascending',2,4,32))


def fixture(cls=FrozenQCSelector, config=None):
    kwargs={} if config is None else {'joint_support_exchange':config}
    selector=cls([list(range(180)),list(range(132))],(4,100),44,**kwargs)
    table=torch.arange(12);cache=torch.zeros((12,2,16,8))
    positions=torch.arange(100,180)
    q=positions[:,None,None].expand(-1,2,4).float().clone()
    for layer in (0,1):selector.observe(layer,0,positions,q,table,16)
    return selector,cache,table


def scores(q,k,positions,start,end):
    rng=np.random.default_rng(int(positions[0]))
    return torch.from_numpy(rng.choice(np.arange(950,1001),size=(2,end-start)).astype(np.float32)/1000)


def fill(selector,cache,table):
    for layer in (0,1):
        for question in (0,1):selector.freeze(layer,question,cache,table)


def test_disabled_selector_matches_frozen_preintegration_source_bitwise():
    root=Path(__file__).resolve().parents[1]
    source=subprocess.check_output(['git','show','d078320:src/vllm_sparse_selector.py'],cwd=root,text=True)
    ns={'__name__':'src._frozen_selector_before_joint','__package__':'src'}
    exec(compile(source,'d078320:src/vllm_sparse_selector.py','exec'),ns)
    old,cache,table=fixture(ns['FrozenQCSelector']);actual,_,_=fixture()
    with patch.dict(ns,{'qc_scores':scores}),patch('src.vllm_sparse_selector.qc_scores',scores),patch('torch.cuda.synchronize'):
        fill(old,cache,table);fill(actual,cache,table)
    assert set(old.masks)==set(actual.masks)
    for key in old.masks:
        assert all(torch.equal(a,b) for a,b in zip(old.masks[key],actual.masks[key]))
        assert all(torch.equal(a,b) for a,b in zip(old.logical_supports(*key,180),actual.logical_supports(*key,180)))
    assert [r['mask_sha256'] for r in old.records]==[r['mask_sha256'] for r in actual.records]
    assert actual.joint_pending_scores=={} and actual.joint_transfers==[]
    assert 'joint_support_exchange' not in actual.summary()


def test_incomplete_rejected_then_atomic_finalize_once_and_original_preserved():
    selector,cache,table=fixture(config=configuration())
    with patch('src.vllm_sparse_selector.qc_scores',scores),patch('torch.cuda.synchronize'):
        selector.freeze(0,0,cache,table)
        with pytest.raises(RuntimeError,match='Incomplete'):
            selector.finalize_joint_supports(2,2)
        with pytest.raises(RuntimeError,match='finalized'):
            selector.logical_supports(0,0,180)
        fill(selector,cache,table)
    originals={key:[x.clone() for x in ids] for key,ids in selector.masks.items()}
    records=copy.deepcopy(selector.records)
    report=selector.finalize_joint_supports(2,2)
    assert report['layers']==2 and report['heads']==2 and report['requests']==2
    assert report['GPU_to_CPU_score_bytes']==0
    assert selector.joint_pending_scores=={} and selector.records==records
    for key,ids in originals.items():
        assert all(torch.equal(a,b) for a,b in zip(ids,selector.independent_masks[key]))
        assert all(len(x)==44 for x in selector.masks[key])
        assert all(torch.isin(torch.arange(68,100),x).all() for x in selector.masks[key])
    assert selector.mask_identity()['sha256']==report['new_mask_identity']['sha256']
    assert selector.mask_identity(selector.independent_masks)['sha256']==report['independent_mask_identity']['sha256']
    with pytest.raises(RuntimeError,match='already'):
        selector.finalize_joint_supports(2,2)


def test_invalid_later_layer_does_not_partially_replace_masks():
    selector,cache,table=fixture(config=configuration())
    with patch('src.vllm_sparse_selector.qc_scores',scores),patch('torch.cuda.synchronize'):
        fill(selector,cache,table)
    original=selector.mask_identity()['sha256']
    selector.joint_pending_scores[1,0][0,10]=np.nan
    with pytest.raises(ValueError,match='finite'):
        selector.finalize_joint_supports(2,2)
    assert selector.mask_identity()['sha256']==original
    assert selector.joint_report is None and len(selector.joint_pending_scores)==4
