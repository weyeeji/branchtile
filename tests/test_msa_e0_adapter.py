import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from msa_e0_adapter import index_plan


def test_exact_ragged_lengths_and_no_support_expansion():
    torch.manual_seed(17);owners=torch.zeros(2,400,dtype=torch.int64)
    lengths=[[127,128,129],[257,36,399]]
    expected=[]
    for h in range(2):
        for b in range(3):
            ids=torch.randperm(400)[:lengths[h][b]].sort().values
            owners[h,ids]|=1<<b;expected.append(ids+h*400)
    plan=index_plan(owners,3)
    assert plan['lengths']==sum(lengths,[]) and plan['payload_padding_tokens']==0
    assert torch.equal(plan['indices'],torch.cat(expected))
    for row,ids in enumerate(expected):
        lo,hi=plan['cu_seqlens_k'][row:row+2]
        assert torch.equal(plan['indices'][lo:hi],ids)
        valid=plan['q2k_indices'][0,row];valid=valid[valid>=0]
        represented=torch.cat([torch.arange(int(b)*128,int(b)*128+128) for b in valid])
        represented=represented[represented<len(ids)]
        assert torch.equal(represented,torch.arange(len(ids)))
    assert plan['masked_tail_lanes']>0


def test_invalid_owner_and_empty_row_rejected():
    with pytest.raises(ValueError):index_plan(torch.tensor([[0,0]],dtype=torch.int64),1)
    with pytest.raises(ValueError):index_plan(torch.tensor([[3,1]],dtype=torch.int64),1)
