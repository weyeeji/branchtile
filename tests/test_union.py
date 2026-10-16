import pytest
import torch
from branchtile.attention import make_plan, reference, synthetic_owners
from branchtile.union import FixedUnion


@pytest.mark.parametrize("pattern,b,g,n,s,bn", [
    ("mixed",4,8,301,7,64), ("cross",4,8,197,3,32),
    ("empty",4,8,23,3,64), ("tree",3,4,127,2,64),
    ("full",1,4,17,8,32), ("disjoint",4,4,137,4,64)])
def test_fixed_union(pattern,b,g,n,s,bn):
    torch.manual_seed(314)
    owners = synthetic_owners(2,n,pattern,b)
    owners[1] &= (1 << (b-1))-1
    plan = make_plan(owners,b,g,"cuda")
    q = torch.randn(2,b*g,128,device="cuda",dtype=torch.bfloat16)
    k = torch.randn(2,n,128,device="cuda",dtype=torch.bfloat16)
    v = torch.randn_like(k)
    out,lse = FixedUnion(plan,owners,s,bn)(q,k,v)
    ref,ref_lse = reference(q,k,v,owners,b,g)
    torch.testing.assert_close(out.float(),ref.float(),atol=0.006,rtol=0.015)
    torch.testing.assert_close(lse,ref_lse.float(),atol=0.00005,rtol=0.00005)
