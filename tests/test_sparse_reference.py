import pytest
import torch
from transformers import Qwen3Config
from transformers.cache_utils import DynamicCache
from sparse_reference import alias_cache,compress_cache


def fixture():
    config=Qwen3Config(num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16)
    data=[]
    for li in range(2):
        key=torch.arange(2*91*16).reshape(1,2,91,16).float()+li*100000
        data.append((key,key+3))
    cache=DynamicCache(ddp_cache_data=data,config=config)
    return config,cache


def test_full_budget_gather_identity_and_independent_append():
    config,cache=fixture()
    supports=[torch.ones(2,65,dtype=torch.bool) for _ in range(2)]
    gathered,positions=compress_cache(cache,supports,7,72,config)
    for a,b,ids in zip(cache.layers,gathered.layers,positions):
        assert torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
        assert torch.equal(ids,torch.arange(91)[None].expand(2,-1))
    cloned=alias_cache(cache,config)
    cloned.update(torch.zeros(1,2,1,16),torch.ones(1,2,1,16),0)
    assert cloned.layers[0].get_seq_length()==92
    assert cache.layers[0].get_seq_length()==91


def test_per_head_support_and_prefix_private_retention():
    config,cache=fixture()
    support=torch.zeros(2,65,dtype=torch.bool)
    support[0,:36]=True;support[1,-36:]=True
    compact,positions=compress_cache(cache,[support,support.flip(0)],7,72,config)
    for source,target,ids in zip(cache.layers,compact.layers,positions):
        assert target.get_seq_length()==7+36+19
        assert torch.equal(target.keys[:,:,:7],source.keys[:,:,:7])
        assert torch.equal(target.values[:,:,-19:],source.values[:,:,72:])
        assert bool((ids[:,1:]>ids[:,:-1]).all())
        for h in range(2):
            assert torch.equal(target.keys[0,h],source.keys[0,h,ids[h]])
    assert not torch.equal(positions[0][0],positions[0][1])


def test_reject_inconsistent_budgets_or_layer_lengths():
    config,cache=fixture()
    support=torch.ones(2,65,dtype=torch.bool);support[0,0]=False
    with pytest.raises(ValueError,match="same budget"):
        compress_cache(cache,[support,support],7,72,config)
    a=torch.ones(2,65,dtype=torch.bool);b=a.clone();b[:,0]=False
    with pytest.raises(ValueError,match="same physical"):
        compress_cache(cache,[a,b],7,72,config)
