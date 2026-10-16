"""CPU semantic checks; real gate/full-budget qualification uses actual GPU model."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastkvzip_reference import (NVIDIA,SharedMaskReference,global_support,original_nvidia_support,
                                protect_scores,upstream_class)


def test_global_native_ties_and_unequal_budgets():
    torch.manual_seed(123)
    scores=torch.randint(0,6,(4,1,3,71)).float()/8
    scores[0]*=.1
    for fraction in [.1,.25,.5,1.]:
        actual=global_support(scores,fraction)
        expected=original_nvidia_support(scores,fraction)
        assert torch.equal(actual,expected)
        assert actual.sum()==scores.numel()-int(scores.numel()*(1-fraction))
    assert global_support(scores,.5).sum(-1).unique().numel()>1


def test_original_nvidia_protection_short_and_long():
    cls=upstream_class(NVIDIA,'FastKVzipPress',{'_score_fast'})
    for length in [1,49,50,20896,32000]:
        raw=torch.rand((2,1,3,length))
        press=cls();press.n_sink=4;press.window_ratio=.02;press.window_size=4096
        press.score_val=[None,None]
        # Each gate returns fixed raw scores, isolating the upstream protection
        # method from the independently GPU-qualified gate arithmetic.
        class FixedGate:
            def __init__(self,scores):self.scores=scores
            def to(self,device):return self
            def __call__(self,hidden):return self.scores.clone()
        press.gates=[FixedGate(r) for r in raw]
        for i in range(2):press._score_fast(SimpleNamespace(layer_idx=i),torch.empty(1,length,1))
        actual,window=protect_scores(raw)
        assert torch.equal(actual,torch.stack(press.score_val))
        assert window==(max(1,int(length*.02)) if length<32000 else 4096)


def test_shared_mask_preserves_future_causality_and_private_tokens():
    mask=SharedMaskReference.__new__(SharedMaskReference)
    mask.active=True;mask.full=False;mask.shared_length=6;mask.calls=0;mask.seconds=0.
    keep=torch.ones((1,4,1,20),dtype=torch.bool)
    keep[:,0:2,:,1:4]=False;keep[:,2:4,:,3:6]=False;mask.keep=[keep]
    kwargs={'hidden_states':torch.empty(1,2,8),'past_key_values':SimpleNamespace(get_seq_length=lambda i:8),'attention_mask':None}
    _,modified=mask.hook(SimpleNamespace(layer_idx=0),(),kwargs)
    result=modified['attention_mask']
    assert result.shape==(1,4,2,10)
    assert result[...,0,9].eq(False).all() and result[...,1,9].all()
    assert result[...,6:8].all() and result[...,0,8].all()
    assert not result[0,0,0,1] and result[0,2,0,1]
    assert result[0,0,0,5] and not result[0,2,0,5]


def test_masked_attention_matches_unequal_head_physical_support_fp64():
    torch.manual_seed(44)
    mask=SharedMaskReference.__new__(SharedMaskReference)
    mask.active=True;mask.full=False;mask.shared_length=6;mask.calls=0;mask.seconds=0.
    keep=torch.ones((1,4,1,20),dtype=torch.bool)
    keep[:,0:2,:,1:5]=False;keep[:,2:4,:,3:5]=False;mask.keep=[keep]
    q=torch.randn(1,4,2,8,dtype=torch.float64)
    k=torch.randn(1,2,10,8,dtype=torch.float64);v=torch.randn_like(k)
    kwargs={'hidden_states':torch.empty(1,2,8),'past_key_values':SimpleNamespace(get_seq_length=lambda i:8),'attention_mask':None}
    _,modified=mask.hook(SimpleNamespace(layer_idx=0),(),kwargs)
    actual=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),
        attn_mask=modified['attention_mask'],is_causal=False)
    for head in range(4):
        for step in range(2):
            positions=modified['attention_mask'][0,head,step].nonzero().flatten()
            logits=q[0,head,step]@k[0,head//2,positions].T/(8**.5)
            expected=torch.softmax(logits,-1)@v[0,head//2,positions]
            torch.testing.assert_close(actual[0,head,step],expected,atol=1e-15,rtol=1e-14)
