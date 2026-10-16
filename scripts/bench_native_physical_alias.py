#!/usr/bin/env python3
"""Same native sparse operation with shared versus replicated physical KV pages.

This is a causal address-mapping control, not a new method or a DRAM counter.
All arms retain the same B replicas, logical supports, plan and arithmetic.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from compare_attention import load_capture, graph_of, elapsed
from branchtile.attention import reference
from flashinfer_baseline import NativeSparseDecode


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest_tensor(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True)
    p.add_argument('--document', required=True)
    p.add_argument('--budget', required=True, type=int)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') and ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    config = json.loads(Path(a.protocol).read_text())
    for path, expected in config['code_sha256'].items():
        assert sha(ROOT / path) == expected, path
    assert a.budget in config['budgets']
    source = next(s for s in config['sources'] if s['id'] == a.document)
    directory = ROOT / source['directory']
    for path, expected in source['files_sha256'].items():
        assert sha(directory / path) == expected, path
    output = ROOT / a.output
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='starting', protocol_sha256=sha(a.protocol), document=a.document,
                  budget=a.budget, uid=os.getuid(), python=sys.executable,
                  CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  qualifications=[], measurements=[])
    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    def terminate(*_):
        raise RuntimeError('Bounded job terminated')
    signal.signal(signal.SIGTERM, terminate)
    save()
    start = time.perf_counter()
    try:
        torch.set_num_threads(1)
        q, k, v, owners, b, g, provenance = load_capture(directory, 15, a.budget, 64, 2)
        assert b == 3 and g == 8 and provenance['ready_lifetime_measured']
        h, n, d = k.shape
        assert h == 2 and d == 128
        dev = torch.cuda.get_device_properties(0)
        result.update(provenance=provenance, gpu_uuid=str(dev.uuid), gpu_name=dev.name,
                      L2_bytes=dev.L2_cache_size,
                      packages={p: importlib.metadata.version(p) for p in ('torch', 'flashinfer-python')})
        supports = [[torch.where((owners[hi] & (1 << bi)) != 0)[0] for hi in range(h)] for bi in range(b)]
        lengths = [[len(x) for x in row] for row in supports]
        unique_counts = [(owners[hi] != 0).sum().item() for hi in range(h)]
        result['logical_access'] = dict(per_request_head_tokens=lengths,
            shared_unique_tokens=sum(unique_counts), private_unique_tokens=sum(map(sum, lengths)),
            payload_bytes_per_token=2*d*k.element_size(),
            note='Unique logical payload only; no cache-line, DRAM, or L2 traffic measurement.')
        # One allocation is used by every arm. The shared-copy rotations and
        # per-request replicas therefore have equal total residency/allocation.
        replicas_k = k.reshape(1, h*n, d).repeat(1, b, 1)
        replicas_v = v.reshape(1, h*n, d).repeat(1, b, 1)
        for ri in range(b):
            assert torch.equal(replicas_k[:,ri*h*n:(ri+1)*h*n], k.reshape(1,h*n,d))
            assert torch.equal(replicas_v[:,ri*h*n:(ri+1)*h*n], v.reshape(1,h*n,d))
        q_bh = q.view(h,b,g,d).permute(1,0,2,3).contiguous().view(1,b*h*g,d)
        target, _ = reference(q,k,v,owners,b,g)
        target = target.view(h,b,g,d).permute(1,0,2,3).reshape(b,h*g,d)
        del k,v,q
        indices_by_arm = {}
        op = None
        result['plans'] = {}
        result['allocation'] = dict(replica_KV_bytes=2*replicas_k.numel()*replicas_k.element_size(),
                                    replicas=b, same_KV_allocation_every_arm=True)
        expected_bits, expected_lse = None, None
        for name in config['arms']:
            rotation = int(name[-1])
            per_request_copy = [(bi+rotation)%b for bi in range(b)] if name.startswith('private') else [rotation] * b
            ids = [[supports[bi][hi] + hi*n + per_request_copy[bi]*h*n for bi in range(b) for hi in range(h)]]
            for bi in range(b):
                for hi in range(h):
                    assert torch.equal(ids[0][bi*h+hi]-hi*n-per_request_copy[bi]*h*n, supports[bi][hi])
            indices_by_arm[name] = torch.cat(ids[0]).to(device='cuda', dtype=torch.int32)
            if op is None:
                op = NativeSparseDecode(q_bh, replicas_k, replicas_v, ids,
                    use_tensor_cores=True, fixed_split_size=None, disable_split_kv=False, normalize_lse=False)
                canonical = torch.empty((b,h*g,d), device='cuda', dtype=q_bh.dtype)
            assert op.indices.data_ptr()==op.wrapper._paged_kv_indices_buf.data_ptr()
            op.indices.copy_(indices_by_arm[name])
            actual_plan = list(op.wrapper._plan_info)
            assert op.kv[0].data_ptr() == replicas_k.data_ptr()
            assert op.kv[1].data_ptr() == replicas_v.data_ptr()
            def call(op=op, canonical=canonical):
                op.wrapper.run(op.q, op.kv, out=op.out, return_lse=False)
                canonical.copy_(op.out.view(b,h*g,d))
                return canonical
            out = call()
            torch.testing.assert_close(out.double(), target, atol=.006, rtol=.015)
            out_hash = digest_tensor(out)
            lse_out, lse = op.run()
            assert digest_tensor(lse_out)==out_hash
            lse_hash = digest_tensor(lse)
            if expected_bits is None:
                expected_bits = out_hash
                expected_lse = lse_hash
            assert out_hash == expected_bits, 'Only physical addresses changed; require bitwise O'
            assert lse_hash == expected_lse, 'Only physical addresses changed; require bitwise raw LSE'
            result['qualifications'].append(dict(arm=name, status='passed', output_sha256=out_hash,
                raw_lse_sha256=lse_hash,
                max_abs_error_fp64=float((out.double()-target).abs().max()),
                indices_sha256=digest_tensor(op.indices)))
            result['plans'][name] = dict(plan_info=actual_plan, plan_ms=op.plan_ms,
                workspace_bytes=op.workspace.numel(), resolved_backend=getattr(op.wrapper,'_backend',None),
                per_request_copy=per_request_copy,
                pointers={key:t.data_ptr() for key,t in dict(q=op.q,indices=op.indices,indptr=op.indptr,
                    workspace=op.workspace,out=op.out,canonical=canonical,k=op.kv[0],v=op.kv[1]).items()})
        assert all(r['pointers']==result['plans']['shared0']['pointers'] for r in result['plans'].values())
        original_output=canonical.clone()
        original_v=replicas_v[:,:h*n].clone()
        result['address_mutation_qualification']=[]
        del target
        flush_bytes=max(256*1024*1024,2*dev.L2_cache_size+4)
        flush=torch.zeros((flush_bytes+3)//4,device='cuda',dtype=torch.float32)
        result['flush_bytes']=flush.numel()*flush.element_size()
        rng=random.Random(config['random_seed'])
        summaries=[]
        for cache,reps in [('hot',config['hot_repetitions']),('cold',1)]:
            graphs={name:graph_of(call,reps) for name in config['arms']+['shared0_AA','private0_AA']}
            # Captured graph outputs must also be correct before timing.
            for name,graph in graphs.items():
                arm=name.removesuffix('_AA')
                op.indices.copy_(indices_by_arm[arm])
                graph.replay();torch.cuda.synchronize()
                assert digest_tensor(op.out)==expected_bits
                for replica in range(b):
                    replicas_v[:,replica*h*n:(replica+1)*h*n].fill_(17.)
                    graph.replay();torch.cuda.synchronize()
                    mapping=result['plans'][arm]['per_request_copy']
                    for bi in range(b):
                        if mapping[bi]==replica:
                            torch.testing.assert_close(canonical[bi],torch.full_like(canonical[bi],17.),atol=.05,rtol=0)
                            assert not torch.equal(canonical[bi],original_output[bi])
                        else:
                            assert torch.equal(canonical[bi],original_output[bi])
                    replicas_v[:,replica*h*n:(replica+1)*h*n].copy_(original_v)
                    graph.replay();torch.cuda.synchronize()
                    assert digest_tensor(canonical)==expected_bits
                    result['address_mutation_qualification'].append(dict(cache=cache,arm=name,replica=replica,status='passed'))
            rows=[]
            for block in range(config['paired_blocks']):
                order=list(graphs);rng.shuffle(order)
                for direction,sequence in [('random',order),('reverse',list(reversed(order)))]:
                    # Warm each arm itself before its hot timed chain. Otherwise
                    # the previous arm can dictate the initial cache contents.
                    timings={}
                    for name in sequence:
                        op.indices.copy_(indices_by_arm[name.removesuffix('_AA')])
                        if cache=='hot':
                            graphs[name].replay();torch.cuda.synchronize()
                        timings[name]=elapsed(graphs[name],reps,flush if cache=='cold' else None)
                    row=dict(cache=cache,block=block,direction=direction,order=sequence,us=timings)
                    rows.append(row);result['measurements'].append(row)
            item=dict(cache=cache, repetitions=reps,
                median_us={name:statistics.median(row['us'][name] for row in rows) for name in graphs},
                paired_ratios={})
            draws=np.random.default_rng(config['bootstrap_seed']).integers(0,config['paired_blocks'],(10000,config['paired_blocks']))
            for numerator,denominator in [('private'+str(i),'shared'+str(i)) for i in range(b)]+[('shared0','shared0_AA'),('private0','private0_AA')]:
                ratios=np.array([[next(r for r in rows if r['block']==block and r['direction']==direction)['us'][numerator]/next(r for r in rows if r['block']==block and r['direction']==direction)['us'][denominator] for direction in ['random','reverse']] for block in range(config['paired_blocks'])])
                boot=np.median(ratios[draws],axis=(1,2))
                item['paired_ratios'][numerator+'_over_'+denominator]=dict(median=float(np.median(ratios)),paired_block_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
            summaries.append(item);result['summary']=summaries;save()
            print(json.dumps(item),flush=True)
            del graphs
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
        import traceback
        traceback.print_exc()
    result.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  wall_seconds_post_import=time.perf_counter()-start)
    if torch.cuda.is_initialized():
        result['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    save()
    raise SystemExit(result['status']!='completed')


if __name__=='__main__':
    main()
