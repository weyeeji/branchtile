#!/usr/bin/env python3
"""Frozen same-native-kernel mask mechanism gate, not engine throughput."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/joint_e0_gate_v1'
visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES','')
import probe_saturated_greedy_cover as cover
import probe_joint_support_exchange as selector
# Both frozen helpers are CPU-only. Restore allocation before CUDA is touched.
os.environ['CUDA_VISIBLE_DEVICES']=visible_devices
import numpy as np
import torch
from compare_attention import load_capture, graph_of, elapsed
from branchtile.attention import reference
from flashinfer_baseline import NativeSparseDecode


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def cpu_masks(source,budget,protocol):
    directory=ROOT/source['directory']
    metadata=json.loads((directory/'metadata.json').read_text())
    start,end=metadata['document_token_range']
    common_n=metadata['actual_shared_document_token_range'][1]-start
    loaded=[]
    for item in source['selectors']:
        path=directory/item['name'];assert sha(path)==item['sha256']
        data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert data['observation_is_question_suffix_only']
        positions=data['observation_positions'].numpy()
        assert (positions>=end).all()
        loaded.append((data['pooled_scores'].numpy().astype(np.float64),data['supports'].get(budget,data['supports'].get(str(budget))).numpy().copy()))
    scores=np.stack([x[0] for x in loaded]); independent=np.stack([x[1] for x in loaded])
    b,h,n=scores.shape
    protected=selector.protected_mask(n)
    previous={tuple(r[k] for k in ('document','budget','kv_head','method','epsilon','request_order')):r
              for r in map(json.loads,(ROOT/protocol['exchange_results']).read_text().splitlines())}
    old_cover={tuple(r[k] for k in ('document','budget','kv_head','epsilon')):r
               for r in map(json.loads,(ROOT/protocol['cover_results']).read_text().splitlines())}
    variants={'independent':independent.copy(),'exchange':independent.copy(),'cover':independent.copy()}
    preparation=[]
    for head in range(h):
        for method in ('independent','exchange','cover'):
            tw,tc=time.perf_counter(),time.process_time()
            if method=='exchange':
                masks,_=selector.exchange(scores[:,head],independent[:,head],protected,.01,list(range(b)),2)
            elif method=='cover':
                masks,_=cover.run_cover(scores[:,head],independent[:,head],protected,common_n,.01)
            else:
                masks=independent[:,head].copy()
            variants[method][:,head]=masks
            item={'method':method,'kv_head':head,'selection_cpu_seconds':time.process_time()-tc,
                  'selection_wall_seconds':time.perf_counter()-tw,**selector.describe(scores[:,head],masks,independent[:,head],common_n)}
            prior=(old_cover[source['id'],budget,head,.01] if method=='cover' else
                   previous[source['id'],budget,head,'exchange',.01,'ascending'] if method=='exchange' else
                   previous[source['id'],budget,head,'independent',None,None])
            assert [q['selected_indices_sha256'] for q in item['per_request']]==[q['selected_indices_sha256'] for q in prior['per_request']]
            assert all(q['score_loss_fraction']<=.01 for q in item['per_request'])
            preparation.append(item)
    return variants,preparation,metadata


def functions_for(op,h,b,g,d,batching_order):
    canonical_out=torch.empty((b,h*g,d),dtype=op.out.dtype,device=op.out.device) if batching_order=='sensitivity_HB' else None
    canonical_lse=torch.empty((b,h*g),dtype=op.lse.dtype,device=op.lse.device) if batching_order=='sensitivity_HB' else None
    functions={}
    for contract in ('O_only','O_plus_rawLSE'):
        def raw(contract=contract):
            if contract=='O_only':
                op.wrapper.run(op.q,op.kv,out=op.out,return_lse=False)
                return op.out.view(op.original_shape)
            return op.run()
        def complete(raw=raw,contract=contract):
            raw()
            if batching_order=='production_BH':
                if contract=='O_plus_rawLSE':
                    return op.out.view(b,h*g,d),op.lse.view(b,h*g)
                return op.out.view(b,h*g,d)
            canonical_out.view(b,h,g,d).copy_(op.out.view(h,b,g,d).permute(1,0,2,3))
            if contract=='O_plus_rawLSE':
                canonical_lse.view(b,h,g).copy_(op.lse.view(h,b,g).permute(1,0,2))
                return canonical_out,canonical_lse
            return canonical_out
        functions[contract,'native_kernel']=raw
        functions[contract,'complete_layout']=complete
    return functions


def main():
    assert os.getuid()!=0 and sys.prefix!=sys.base_prefix
    assert visible_devices and ',' not in visible_devices
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document',required=True);parser.add_argument('--budget',type=int,required=True)
    parser.add_argument('--job-id',required=True);args=parser.parse_args()
    protocol=json.loads((BASE/'protocol.json').read_text())
    for file,digest in protocol['code_and_result_hashes'].items():
        assert sha(ROOT/file)==digest
    assert sha(Path(__file__))==protocol['script_sha256']
    output=BASE/f'{args.job_id}.json'
    assert not output.exists()
    source=next(s for s in protocol['sources'] if s['id']==args.document)
    result={'job_id':args.job_id,'document':args.document,'budget':args.budget,'status':'starting',
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),
            'uid':os.getuid(),'python':sys.executable,'CUDA_VISIBLE_DEVICES':visible_devices,
            'protocol_sha256':sha(BASE/'protocol.json'),'qualification':[],'measurements':[]}
    def save():
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(RuntimeError('Bounded job SIGTERM')))
    save(); started=time.perf_counter()
    try:
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        variants,preparation,meta=cpu_masks(source,args.budget,protocol)
        result['selector_preparation']=preparation
        directory=ROOT/source['directory']
        assert sha(directory/'metadata.json')==source['metadata_sha256']
        assert sha(directory/source['source_file']['name'])==source['source_file']['sha256']
        for file in source['decode_files']:
            assert sha(directory/file['name'])==file['sha256']
            dec=torch.load(directory/file['name'],map_location='cpu',weights_only=True,mmap=True)
            assert dec['metadata']==file['metadata']
            assert dec['metadata']['hooked_kv_matches_actual_cache'] and dec['metadata']['physical_full_prefix_pages_match']
        assert sha(directory/f'masks_l15_b{args.budget}.pt')==source['mask_files'][str(args.budget)]['sha256']
        q,k,v,owners,b,g,provenance=load_capture(directory,15,args.budget,64,2)
        assert b==meta['B'] and provenance['ready_lifetime_measured']
        h,n,d=k.shape
        prefix=meta['document_token_range'][0]
        common_n=meta['actual_shared_document_token_range'][1]-prefix
        expected=sum(torch.from_numpy(variants['independent'][qi,:,:common_n].copy()).long()*(1<<qi) for qi in range(b))
        assert torch.equal(owners[:,prefix:prefix+common_n],expected)
        result.update(provenance=provenance,q_shape=list(q.shape),k_shape=list(k.shape),
                      private_context=[file['metadata'] for file in source['decode_files']],
                      scope=protocol['scope'],packages={p:importlib.metadata.version(p) for p in ('torch','flashinfer-python','tilelang','numpy')})
        device=torch.cuda.get_device_properties(0)
        result.update(gpu_uuid=str(device.uuid),gpu_name=device.name,L2_bytes=device.L2_cache_size)
        flush_bytes=max(2*device.L2_cache_size+4,256*1024*1024)
        flush=torch.zeros((flush_bytes+3)//4,dtype=torch.float32,device='cuda')
        assert flush.numel()*flush.element_size()>2*device.L2_cache_size
        result['flush_bytes']=flush.numel()*flush.element_size()
        ops={}; functions={}; plans={}; plan_info={}; lengths=None
        tq=time.perf_counter()
        q_bh=q.view(h,b,g,d).permute(1,0,2,3).contiguous().view(1,b*h*g,d)
        torch.cuda.synchronize()
        result['export_HB_to_production_BH_query_preparation_seconds']=time.perf_counter()-tq
        result['production_query_note']='Exported HB queries reordered once for this replay; engine production input is already BH. The preparation is not timed as an attention call.'
        for method in protocol['methods']:
            own=owners.clone()
            own[:,prefix:prefix+common_n]=sum(torch.from_numpy(variants[method][qi,:,:common_n].copy()).long()*(1<<qi) for qi in range(b))
            supports=[[torch.where((own[head]&(1<<request))!=0)[0] for request in range(b)] for head in range(h)]
            current_lengths=[[len(ids) for ids in head] for head in supports]
            if lengths is None:lengths=current_lengths
            assert current_lengths==lengths
            for request,file in enumerate(source['decode_files']):
                md=file['metadata']
                private=md['query_token_position']+1-md['shared_prefix_end']
                assert all(current_lengths[head][request]==prefix+args.budget+private-md['private_document_tail_tokens'] for head in range(h))
            ref,lse=reference(q,k,v,own,b,g)
            for batching_order in protocol['batching_orders']:
                tw=time.perf_counter()
                if batching_order=='production_BH':
                    q_input,k_input,v_input=q_bh,k.view(1,h*n,d),v.view(1,h*n,d)
                    input_supports=[[supports[head][request]+head*n for request in range(b) for head in range(h)]]
                else:
                    q_input,k_input,v_input,input_supports=q,k,v,supports
                op=NativeSparseDecode(q_input,k_input,v_input,input_supports,use_tensor_cores=True,fixed_split_size=None,disable_split_kv=False,normalize_lse=False)
                fs=functions_for(op,h,b,g,d,batching_order)
                key=batching_order+'/'+method
                plans[key]={'prepare_wall_seconds':time.perf_counter()-tw,'native_plan_ms':op.plan_ms,
                           'workspace_bytes':op.workspace.numel(),'resolved_backend':getattr(op.wrapper,'_backend',None),
                           'plan_info':list(op.wrapper._plan_info),'support_lengths':current_lengths,
                           'native_query_shape':list(op.q.shape),'native_batch_order':batching_order,
                           'q_aliases_prepared_query':op.q.data_ptr()==q_input.data_ptr(),
                           'kv_aliases_source':op.kv[0].data_ptr()==k.data_ptr() and op.kv[1].data_ptr()==v.data_ptr()}
                assert plans[key]['q_aliases_prepared_query'] and plans[key]['kv_aliases_source']
                if batching_order not in plan_info:plan_info[batching_order]=plans[key]['plan_info']
                assert plans[key]['plan_info']==plan_info[batching_order],'Same lengths/configuration must resolve identical plan scheduling'
                for contract in protocol['contracts']:
                    for call_scope in protocol['call_scopes']:
                        item={'method':method,'contract':contract,'call_scope':call_scope,'batching_order':batching_order}
                        try:
                            out=fs[contract,call_scope]()
                            actual,raw_lse=out if contract=='O_plus_rawLSE' else (out,None)
                            target=ref
                            target_lse=lse
                            if call_scope=='complete_layout' or batching_order=='production_BH':
                                target=ref.reshape(h,b,g,d).permute(1,0,2,3).reshape(b,h*g,d)
                                target_lse=lse.reshape(h,b,g).permute(1,0,2).reshape(b,h*g)
                            if call_scope=='native_kernel' and batching_order=='production_BH':
                                target=target.reshape(1,b*h*g,d);target_lse=target_lse.reshape(1,b*h*g)
                            item['max_output_abs_error_fp64']=float((actual.double()-target).abs().max())
                            torch.testing.assert_close(actual.double(),target,atol=.006,rtol=.015)
                            if raw_lse is not None:
                                ln_lse=raw_lse.double()*math.log(2)
                                item['max_lse_abs_error_fp64']=float((ln_lse-target_lse).abs().max())
                                torch.testing.assert_close(ln_lse,target_lse,atol=5e-5,rtol=5e-5)
                            item['status']='passed'
                        except Exception as exc:
                            item.update(status='failed',error=repr(exc))
                        result['qualification'].append(item)
                ops[batching_order,method]=op;functions[batching_order,method]=fs
            result['plans']=plans;save()
            del ref,lse,own
        if any(r['status']!='passed' for r in result['qualification']):
            raise RuntimeError('At least one frozen own-mask qualification failed; no timing gate claim')
        rng=random.Random(protocol['random_seed'])
        summaries=[]
        for batching_order,contract in [(order,c) for order in protocol['batching_orders'] for c in protocol['contracts']]:
            for call_scope in protocol['call_scopes']:
                for cache,repetitions in (('hot',20),('cold',1)):
                    graphs={method:graph_of(functions[batching_order,method][contract,call_scope],repetitions) for method in protocol['methods']}
                    graphs['independent_AA']=graph_of(functions[batching_order,'independent'][contract,call_scope],repetitions)
                    subset=[]
                    for block in range(protocol['paired_blocks']):
                        order=list(graphs);rng.shuffle(order)
                        for direction,sequence in (('random',order),('reverse',list(reversed(order)))):
                            row={'batching_order':batching_order,'contract':contract,'call_scope':call_scope,'cache':cache,'block':block,'direction':direction,'order':sequence,
                                 'us':{method:elapsed(graphs[method],repetitions,flush if cache=='cold' else None) for method in sequence}}
                            subset.append(row);result['measurements'].append(row)
                    item={'batching_order':batching_order,'contract':contract,'call_scope':call_scope,'cache':cache,
                          'median_us':{method:statistics.median(r['us'][method] for r in subset) for method in graphs},'paired_ratios':{}}
                    draws=np.random.default_rng(protocol['bootstrap_seed']).integers(0,protocol['paired_blocks'],(10000,protocol['paired_blocks']))
                    for denominator in ('exchange','cover','independent_AA'):
                        ratios=np.array([[next(r for r in subset if r['block']==block and r['direction']==direction)['us']['independent']/
                                          next(r for r in subset if r['block']==block and r['direction']==direction)['us'][denominator]
                                          for direction in ('random','reverse')] for block in range(protocol['paired_blocks'])])
                        boot=np.median(ratios[draws],axis=(1,2))
                        item['paired_ratios']['independent_over_'+denominator]={'median':float(np.median(ratios)),
                            'paired_block_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
                    summaries.append(item);result['summary']=summaries;save();del graphs
                    print(json.dumps({'document':args.document,'budget':args.budget,**item}),flush=True)
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
        import traceback
        traceback.print_exc()
    result.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds_post_import=time.perf_counter()-started)
    if torch.cuda.is_initialized():
        result['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        result['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    save()
    raise SystemExit(result['status']!='completed')


if __name__=='__main__':main()
