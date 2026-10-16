"""Audit intact capture artifacts after the narrowly identified receipt failure.

Never edits a failed run. This audit grants no numerical qualification: the
consumer must independently test original AR O and full/cascade O against FP64.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    import torch
    directory=Path(args.run).resolve();summary_path=directory/'summary.json'
    summary=json.loads(summary_path.read_text())
    assert summary['status']=='failed'
    assert 'native_precision_receipt' in summary['error'] and "'kernel'" in summary['error']
    assert summary['engine']['enforce_eager'] is True
    for name,digest in summary['source_sha256'].items():
        assert sha(ROOT/name)==digest, ('Capture source changed; explicit historical review needed',name)
    assert len(summary['outputs'])==8 and all(len(x['tokens'])>=9 for x in summary['outputs'])
    report=dict(schema='ragged_capture_recovery_integrity_v1',
        status='capture_integrity_passed_pending_GPU_numerical_qualification',
        original_run_status='failed',run=str(directory),summary_sha256=sha(summary_path),
        reason='Final read-only receipt assumed every eager decode metadata had a kernel field; no subkernel identity is inferred.',
        native_backend_contract='Configured FLASHINFER; exact eager AR subkernel was not exported and remains unspecified.',
        sources=summary['source_sha256'],files=[],requires_original_AR_and_full_and_shared_FP64=True)
    template_positions=None
    for rank in (0,1):
        receipt_path=directory/f'rank{rank}'/'ragged_capture_receipt.json'
        receipt=json.loads(receipt_path.read_text())
        assert receipt['status']=='completed' and len(receipt['files'])==4
        assert receipt['source_sha256']==summary['source_sha256']['scripts/vllm_ragged_state_capture_worker.py']
        configured=next(x for x in summary['configured'] if x['rank']==rank)
        assert len(configured['layers'])==48 and all(x['backend']=='FLASHINFER' for x in configured['layers'])
        for record in receipt['files']:
            path=Path(record['path']);assert sha(path)==record['sha256']
            data=torch.load(path,map_location='cpu',weights_only=True)
            assert data['schema']=='real_ar_multitoken_attention_v1'
            assert data['rank']==rank and data['layer']==record['layer']
            assert data['q'].shape==data['native_o'].shape==(8,8,16,128)
            assert bool(torch.isfinite(data['q']).all() and torch.isfinite(data['native_o']).all())
            assert data['cache'].dtype==torch.uint8 and data['kv_semantic_dtype']=='float8_e4m3fn'
            assert list(data['cache'].shape[1:])==[2,16,256]
            assert data['positions'].shape==(8,8)
            assert torch.equal(data['positions'][:,1:],data['positions'][:,:-1]+1)
            assert data['seq_lens']==(data['positions'][:,-1]+1).tolist()
            assert all(not torch.equal(data['q'][r,t],data['q'][r,t+1]) for r in range(8) for t in range(7))
            if template_positions is None:template_positions=data['positions']
            else:assert torch.equal(data['positions'],template_positions)
            assert data['k_scale']>0 and data['v_scale']>0
            table=data['block_tables'];prefix=data['common_prefix_tokens']//16
            assert data['common_prefix_tokens']%16==0 and prefix>0
            assert bool((table[:,:prefix]==table[:1,:prefix]).all())
            assert int(data['positions'].min())>=prefix*16
            for r,length in enumerate(data['seq_lens']):
                ids=table[r,:(length+15)//16]
                assert int(ids.min())>=0 and int(ids.max())<data['cache'].shape[0]
            report['files'].append(dict(path=str(path),sha256=record['sha256'],rank=rank,layer=data['layer'],
                receipt_sha256=sha(receipt_path),query_shape=list(data['q'].shape),
                actual_consecutive_positions=True,adjacent_queries_distinct=True,
                cache_shape=list(data['cache'].shape),original_stride=data['original_cache_stride'],
                k_scale=data['k_scale'],v_scale=data['v_scale'],common_prefix_tokens=prefix*16))
    assert not torch.cuda.is_initialized()
    output=Path(args.output);assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Audited {len(report["files"])} complete captures; failed run retained; GPU FP64 qualification still required')


if __name__=='__main__':main()
