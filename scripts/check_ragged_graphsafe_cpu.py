"""Exercise finite-plan guards with CPU buffers, without CUDA initialization."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    import torch
    from src.ragged_verify_graphsafe import FiniteShapeVerifyPlan
    from src.ragged_shared_verify_plan_v2 import ragged_metadata
    groups=[0]*4+[1]*4;prefixes=[256,272];counts=[4]*8
    starts=[267,259,256,270,280,276,272,276]
    tables=torch.zeros((8,20),dtype=torch.int32)
    for r,g in enumerate(groups):
        p=prefixes[g]//16;tables[r,:p]=torch.arange(p)+g*30
        tables[r,p:]=torch.arange(20-p)+100+r*4
    lens=lambda shift:[x+4+shift for x in starts]
    plan=FiniteShapeVerifyPlan.__new__(FiniteShapeVerifyPlan)
    plan.metadata=ragged_metadata(groups,prefixes,counts,lens(0),tables)
    plan.wrappers={n:SimpleNamespace(_paged_kv_indices_buf=torch.tensor(plan.metadata[n+'_ids'],dtype=torch.int32),
        _paged_kv_last_page_len_buf=torch.tensor(plan.metadata[n+'_last'],dtype=torch.int32)) for n in ('prefix','tail','full')}
    assert plan.update(groups,prefixes,counts,lens(1),tables)['reusable']
    before={n:(w._paged_kv_indices_buf.clone(),w._paged_kv_last_page_len_buf.clone()) for n,w in plan.wrappers.items()}
    assert not plan.update(groups,prefixes,counts,lens(2),tables)['reusable']
    for n,w in plan.wrappers.items():
        assert torch.equal(before[n][0],w._paged_kv_indices_buf) and torch.equal(before[n][1],w._paged_kv_last_page_len_buf)
    alternate=[8,8,8,4,1,1,1,1]
    assert not plan.update(groups,prefixes,alternate,[s+n for s,n in zip(starts,alternate)],tables)['reusable']
    assert not torch.cuda.is_initialized()
    print('Finite plan accepts same-page growth, rejects page crossing and changed query partition before mutation; CUDA not initialized')


if __name__=='__main__':main()
