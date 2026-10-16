"""CPU-only independent sanity checks for the actual-byte FP64 reference."""
import os
import sys

import torch

from scripts.vllm_native_fp8_kv_worker import fp64_from_actual_fp8_cache


def test_strided_cache_and_partial_pages():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    generator=torch.Generator().manual_seed(20261016)
    # Interleaved layer gaps plus permuted physical pages; the input to the
    # reference is byte storage, matching vLLM's quantized KV container.
    backing=torch.zeros((20,3,2,4,256),dtype=torch.uint8)
    cache=backing[:,1]
    decoded=torch.randn((20,2,4,256),generator=generator).clamp(-448,448).to(torch.float8_e4m3fn)
    cache.copy_(decoded.view(torch.uint8))
    tables=torch.tensor([[7,1,18,4,10],[7,1,12,5,13],[7,1,9,6,14]])
    q=torch.randn((3,16,128),generator=generator).bfloat16()
    positions=[10,14,18]
    result,lse=fp64_from_actual_fp8_cache(q,cache,tables,positions,128**-.5,.5,2.0,chunk=3)
    expected=[];expected_lse=[]
    for r,p in enumerate(positions):
        logical=torch.arange(p+1)
        kv=decoded.double()[tables[r,logical//4].long(),:,logical%4,:].permute(1,0,2)
        logits=torch.einsum('hgd,hnd->hgn',q[r].double().reshape(2,8,128),kv[...,:128]*.5)*(128**-.5)
        expected.append(torch.einsum('hgn,hnd->hgd',logits.softmax(-1),kv[...,128:]*2.0).reshape(16,128))
        expected_lse.append(logits.logsumexp(-1).reshape(16))
    assert torch.allclose(result,torch.stack(expected),atol=1e-13,rtol=1e-13)
    assert torch.allclose(lse,torch.stack(expected_lse),atol=1e-13,rtol=1e-13)
    # Unused future cells are never read; overwrite them with extreme values.
    for r,p in enumerate(positions):
        nxt=p+1
        if nxt<20:cache[tables[r,nxt//4],:,nxt%4,:]=0x7e
    other,_=fp64_from_actual_fp8_cache(q,cache,tables,positions,128**-.5,.5,2.0,chunk=7)
    assert torch.allclose(result,other,atol=1e-13,rtol=1e-13)
