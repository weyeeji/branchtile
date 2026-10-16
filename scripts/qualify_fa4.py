"""FA4 SM120 qualification against FP64 with frozen project tolerances."""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fa4_baseline import FullSourceMasked, GatheredDecode
from branchtile.attention import reference
from flash_attn.cute.interface import flash_attn_varlen_func, _flash_attn_fwd


def errors(out, lse, ref, ref_lse):
    values = {"output_max_abs":float((out.double()-ref).abs().max()),
              "lse_max_abs":float((lse.double()-ref_lse).abs().max())}
    torch.testing.assert_close(out.float(),ref.float(),atol=.006,rtol=.015)
    torch.testing.assert_close(lse.float(),ref_lse.float(),atol=.0003,rtol=.0001)
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--quick',action='store_true');a=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    torch.set_num_threads(4);torch.manual_seed(20261016)
    result={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'uid':os.geteuid(),'venv':sys.prefix,'torch':torch.__version__,
            'fa4':importlib.metadata.version('flash-attn-4'),'gpu':torch.cuda.get_device_name(0),
            'capability':torch.cuda.get_device_capability(),'tolerances':{'O_atol':.006,'O_rtol':.015,'LSE_atol':.0003,'LSE_rtol':.0001},
            'cases':[],'limitations':'Layout/allocator-pressure qualification, not full MiniMax 33B issue2860 process reproduction. No sentinel padding or upstream patch applied.'}
    def save():Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    def case(name,fn,expected_error=None):
        row={'name':name};start=time.monotonic()
        try:
            row.update(fn() or {});torch.cuda.synchronize()
            row['status']='failed_missing_expected_rejection' if expected_error else 'passed'
        except Exception as exc:
            row.update(status='expected_unsupported' if expected_error and expected_error in str(exc) else 'failed',error=str(exc),traceback=traceback.format_exc())
            if 'illegal memory' in str(exc).lower():
                result['cases'].append(row);save();raise
        row['seconds']=time.monotonic()-start;result['cases'].append(row);save();print(json.dumps(row),flush=True)
    for b,g in ([(1,4)] if a.quick else [(b,g) for b in (1,3,4) for g in (4,8)]):
        h,n,d=2,257,128
        q=torch.randn(h,b*g,d,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(h,n,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        owners=torch.randint(0,1<<b,(h,n),dtype=torch.int64)
        owners[:,0]=(1<<b)-1
        ref,ref_lse=reference(q,k,v,owners,b,g)
        for cls in (FullSourceMasked,GatheredDecode):
            def run(cls=cls):
                op=cls(q,k,v,owners,b);out,lse=op.run();torch.cuda.synchronize()
                return dict(errors(out,lse,ref,ref_lse),segments=h*b if cls is GatheredDecode else h,
                            q_shape=list(q.shape),source_shape=list(k.shape),B=b,GQA=g)
            case(f'{cls.__name__}_B{b}_G{g}',run)
    if not a.quick:
        # Unequal >=4-segment varlen, including a partial M tile, strided K/V
        # views of a fused tensor, and unrelated live allocations around them.
        pressure=[torch.empty(8*1024*1024+i*256,device='cuda',dtype=torch.uint8) for i in range(5)]
        qlens=[128,129,257,65];klens=[513,519,521,527];hq,hkv,d=8,2,128
        q=torch.randn(sum(qlens),hq,d,device='cuda',dtype=torch.bfloat16)
        fused=torch.randn(sum(klens),hkv,2*d,device='cuda',dtype=torch.bfloat16)
        k,v=fused.split(d,dim=-1)
        cuq=torch.tensor([0,*torch.tensor(qlens).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
        cuk=torch.tensor([0,*torch.tensor(klens).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
        def varlen():
            out,lse=flash_attn_varlen_func(q,k,v,cu_seqlens_q=cuq,cu_seqlens_k=cuk,max_seqlen_q=max(qlens),max_seqlen_k=max(klens),return_lse=True,num_splits=0)
            refs,lses=[],[];qb=kb=0
            for nq,nk in zip(qlens,klens):
                qr=q[qb:qb+nq].double().transpose(0,1)
                kr=k[kb:kb+nk].double().repeat_interleave(hq//hkv,dim=1).transpose(0,1)
                vr=v[kb:kb+nk].double().repeat_interleave(hq//hkv,dim=1).transpose(0,1)
                logits=qr@kr.transpose(-1,-2)/d**.5
                refs.append((logits.softmax(-1)@vr).transpose(0,1));lses.append(logits.logsumexp(-1));qb+=nq;kb+=nk
            return dict(errors(out,lse,torch.cat(refs),torch.cat(lses,dim=1)),q_lengths=qlens,k_lengths=klens,k_stride=list(k.stride()),v_storage_offset=v.storage_offset(),allocator_pressure_bytes=sum(x.numel() for x in pressure))
        case('varlen_four_unequal_fused_kv_views',varlen)
        q=torch.randn(2,1,4,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(2,128,1,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        case('sm120_split2_rejected',lambda:_flash_attn_fwd(q,k,v,num_splits=2),expected_error='only supports num_splits=1')
        def paged():
            return _flash_attn_fwd(q,k,v,page_table=torch.arange(2,device='cuda',dtype=torch.int32)[:,None],seqused_k=torch.full((2,),128,device='cuda',dtype=torch.int32),max_seqlen_k=128)
        case('sm120_paged_rejected',paged,expected_error='Paged KV not supported')
    result['status']='completed';result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()


if __name__=='__main__':main()
