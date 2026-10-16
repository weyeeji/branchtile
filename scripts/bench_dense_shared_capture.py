"""Real dense captured page16 XQA O-only versus known shared-prefix cascade.

Fixed input operator diagnostic. Original page pitch and sharing are restored
from compact CPU artifacts; no model throughput or online-export speed claim.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True);p.add_argument('--qualification-result',required=True)
    p.add_argument('--output',required=True);p.add_argument('--batch',type=int,default=8);p.add_argument('--rows')
    p.add_argument('--blocks',type=int,default=30);args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and 1<=args.batch<=8 and args.blocks>=10
    out=Path(args.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'initializing','args':vars(args),'scope':__doc__,'checks':{},'measurements':[],
        'uid':os.geteuid(),'python':sys.executable,'claim_scope':'operator_only_known_mechanism_control'}
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        import numpy as np
        import torch
        from flashinfer.decode import xqa_batch_decode_with_kv_cache
        from src.dense_capture_fixture import load_dense_capture_fixture
        from src.vllm_dense_cascade_plan import LayerSharedDenseCascadePlan
        from scripts.compare_attention import graph_of,elapsed
        torch.set_num_threads(4);assert torch.cuda.get_device_capability()==(12,0)
        qualification=json.loads(Path(args.qualification_result).read_text())
        assert qualification['status']=='qualified' and qualification['all_passed']
        assert qualification['sources']['src/vllm_dense_cascade_plan.py']==hashlib.sha256((ROOT/'src/vllm_dense_cascade_plan.py').read_bytes()).hexdigest()
        rows=list(map(int,args.rows.split(','))) if args.rows else list(range(args.batch));assert len(rows)==args.batch
        fixture=load_dense_capture_fixture(args.capture,rows)
        report['provenance']=fixture['provenance'];report['qualification_receipt_sha256']=hashlib.sha256(Path(args.qualification_result).read_bytes()).hexdigest()
        dtype=fixture['cache_cpu'].dtype;shape=tuple(fixture['cache_cpu'].shape);stride=fixture['original_stride']
        span=(sum((n-1)*s for n,s in zip(shape,stride))+1)*fixture['cache_cpu'].element_size()
        report['restored_cache_span_bytes']=span
        assert span<22*1024**3, 'Restoring original page pitch would exceed this diagnostic allocation cap'
        started=time.perf_counter();cache=torch.empty_strided(shape,stride,device='cuda',dtype=dtype);cache.copy_(fixture['cache_cpu'])
        query=fixture['q_cpu'].cuda();original=fixture['original_o_cpu'].cuda();tables=fixture['tables_cpu'].cuda()
        seq_lens=fixture['seq_lens'];lengths=torch.tensor(seq_lens,device='cuda',dtype=torch.uint32)
        typed=cache.view(torch.float8_e4m3fn) if fixture['semantic_fp8'] and dtype==torch.uint8 else cache
        kv=(typed[...,:128],typed[...,128:]);assert all(x.untyped_storage().data_ptr()==cache.untyped_storage().data_ptr() for x in kv)
        native_workspace=torch.zeros(256*1024*1024,device='cuda',dtype=torch.uint8);native_out=torch.empty_like(query)
        torch.cuda.synchronize();report['offline_fixture_restore_seconds']=time.perf_counter()-started
        prefix=fixture['prefix_tokens'];tail=max(seq_lens)-prefix
        started=time.perf_counter();plan=LayerSharedDenseCascadePlan([cache],args.batch,prefix,max_tail_tokens=max(16,tail),
            k_scales=[fixture['k_scale']],v_scales=[fixture['v_scale']])
        report['candidate_prepare']=plan.prepare(tables,seq_lens,require_stable_topology=False)
        torch.cuda.synchronize();report['candidate_constructor_and_first_prepare_seconds']=time.perf_counter()-started
        def native():
            xqa_batch_decode_with_kv_cache(query,kv,native_workspace,tables,lengths,max_seq_len=max(seq_lens),
                bmm1_scale=128**-.5*fixture['k_scale'],bmm2_scale=fixture['v_scale'],kv_layout='HND',out=native_out)
            return native_out
        def candidate():return plan.run_layer(0,query)
        # Independent FP64 reconstruction reads the complete original support.
        payload=typed.double();expected=torch.empty_like(query,dtype=torch.float64);expected_lse=torch.empty(query.shape[:2],device='cuda',dtype=torch.float64)
        for r,length in enumerate(seq_lens):
            logical=torch.arange(length,device='cuda');physical=tables[r,logical//16].long()
            for h in range(2):
                values=payload[physical,h,logical%16];q=query[r,h*8:(h+1)*8].double()
                scores=(q@(values[:,:128]*fixture['k_scale']).T)*(128**-.5)
                expected[r,h*8:(h+1)*8]=torch.softmax(scores,dim=-1)@(values[:,128:]*fixture['v_scale'])
                expected_lse[r,h*8:(h+1)*8]=torch.logsumexp(scores,dim=-1)
        del payload
        for name,fn in [('native_xqa_O_only',native),('shared_prefix_cascade',candidate)]:
            value=fn();passed=bool(torch.isclose(value.double(),expected,atol=.006,rtol=.015).all())
            report['checks'][name]={'passed':passed,'max_abs_fp64':float((value.double()-expected).abs().max()),
                'relative_l2_fp64':float((value.double()-expected).norm()/expected.norm()),
                'max_abs_original_native_snapshot':float((value.float()-original.float()).abs().max())}
            save();assert passed,report['checks'][name]
        lse=plan.prefix_lse[0].double()*math.log(2)
        report['checks']['cascade_internal_lse']={'passed':bool(torch.isclose(lse,expected_lse,atol=.0003,rtol=.0001).all()),'max_abs':float((lse-expected_lse).abs().max())}
        assert report['checks']['cascade_internal_lse']['passed']
        native();report['checks']['restored_native_vs_original']={'passed':bool(torch.isclose(native_out.float(),original.float(),atol=.006,rtol=.015).all())}
        assert report['checks']['restored_native_vs_original']['passed']
        flush=torch.zeros(max(256*1024*1024,2*torch.cuda.get_device_properties(0).L2_cache_size),device='cuda',dtype=torch.uint8)
        report['flush_bytes']=flush.numel();report['summary']={};rng=random.Random(20261017)
        for cache_state,reps in [('hot',20),('cold',1)]:
            graphs={'native_xqa_O_only':graph_of(native,reps),'shared_prefix_cascade':graph_of(candidate,reps),'native_xqa_AA':graph_of(native,reps)}
            for name,graph in graphs.items():
                graph.replay();value=plan.outputs[0] if name=='shared_prefix_cascade' else native_out
                assert bool(torch.isclose(value.double(),expected,atol=.006,rtol=.015).all())
            for block in range(args.blocks):
                order=list(graphs);rng.shuffle(order)
                for direction,ordered in [('random',order),('reverse',list(reversed(order)))]:
                    times={name:elapsed(graphs[name],reps,flush if cache_state=='cold' else None) for name in ordered}
                    report['measurements'].append({'cache':cache_state,'block':block,'direction':direction,'us':times})
            selected=[r for r in report['measurements'] if r['cache']==cache_state]
            stats={'median_us':{name:statistics.median(r['us'][name] for r in selected) for name in graphs},'paired_ratios':{}}
            draws=np.random.default_rng(71).integers(0,args.blocks,(10000,args.blocks))
            for numerator,denominator in [('native_xqa_O_only','shared_prefix_cascade'),('native_xqa_O_only','native_xqa_AA')]:
                ratios=np.array([[r['us'][numerator]/r['us'][denominator] for r in selected if r['block']==block] for block in range(args.blocks)])
                ci=np.quantile(np.median(ratios[draws],axis=(1,2)),[.025,.975])
                stats['paired_ratios'][numerator+'_over_'+denominator]={'median':float(np.median(ratios)),'bootstrap95':ci.tolist()}
            report['summary'][cache_state]=stats;save()
        sources=['scripts/bench_dense_shared_capture.py','src/dense_capture_fixture.py','src/vllm_dense_cascade_plan.py']
        report['source_sha256']={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}
        report['limitations']=['Fixed real snapshot and one layer/rank, not end-to-end or independent task-quality evidence.',
            'Known FlashInfer shared-prefix decomposition, not a novel mechanism.',
            'Preparation, offline restore, compile and graph capture excluded from operator latency and reported separately where measured.',
            'Restored pitch preserves address spacing; compact physical page IDs differ from the original engine allocation.',
            'Artificial cold flush and CUDA-event quantization retained; intervals do not account for systematic timing error.']
        report['status']='completed';save();print(json.dumps({'output':str(out),'summary':report['summary']}),flush=True)
    except Exception as exc:
        report['status']='failed';report['error_type']=type(exc).__name__;report['error']=str(exc);report['traceback']=traceback.format_exc();save();raise

if __name__=='__main__':main()
