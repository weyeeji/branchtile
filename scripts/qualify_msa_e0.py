"""Bounded two-capture qualification/timing of an exact-support MSA adapter."""
from __future__ import annotations

import argparse
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

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from msa_e0_adapter import MSAGatherE0
from branchtile.attention import reference
from compare_attention import load_capture,graph_of,elapsed
from flashinfer_baseline import NativeSparseDecode


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(output,lse,expected,expected_lse):
    row={'output_max_abs_fp64':float((output.double()-expected).abs().max()),
         'output_relative_l2_fp64':float(torch.linalg.vector_norm(output.double()-expected)/torch.linalg.vector_norm(expected)),
         'lse_max_abs_fp64':float((lse.double()-expected_lse).abs().max())}
    try:
        torch.testing.assert_close(output.float(),expected.float(),atol=.006,rtol=.015)
        torch.testing.assert_close(lse.float(),expected_lse.float(),atol=.0003,rtol=.0001)
        row['status']='passed'
    except AssertionError as error:row.update(status='failed',error=str(error))
    return row


def fixtures(result,save):
    generator=torch.Generator().manual_seed(171)
    owners=torch.zeros(2,400,dtype=torch.int64);lengths=[[127,128,129],[257,36,399]]
    for h in range(2):
        for b in range(3):owners[h,torch.randperm(400,generator=generator)[:lengths[h][b]]]|=1<<b
    result['fixtures']=[]
    for gqa in [4,8]:
        q=torch.randn(2,3*gqa,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(2,400,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        for partial in [torch.float32,torch.bfloat16]:
            start=time.perf_counter();op=MSAGatherE0(q,k,v,owners,3,partial_dtype=partial)
            expected,lse=reference(q,k,v,owners,3,gqa);output,got_lse=op.run();torch.cuda.synchronize()
            row={'gqa':gqa,'partial_dtype':str(partial),'lengths':op.plan_cpu['lengths'],
                'initialization_and_first_call_seconds':time.perf_counter()-start,**check(output,got_lse,expected,lse)}
            result['fixtures'].append(row);save()
            if row['status']!='passed':continue
            op.run();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):captured=op.run()
            q.mul_(.8);k.mul_(.9);v.add_(.05)
            expected,lse=reference(q,k,v,owners,3,gqa);graph.replay();torch.cuda.synchronize()
            row['graph_changed_query_and_payload']=check(*captured,expected,lse)
            assert row['graph_changed_query_and_payload']['status']=='passed'
            # Different request histories and non-128 tails are exercised by
            # all six independent reference rows; no support block expansion.
            row['exact_indices_and_masked_tail_qualification']=True;save()
    assert all(r['status']=='passed' for r in result['fixtures']),'MSA fixture failed frozen numerical contract'


def case(result,save,name,capture,layer,heads,prior_comparison):
    start=time.perf_counter()
    q,k,v,owners,b,g,provenance=load_capture(capture,layer,4096,256,heads)
    expected,expected_lse=reference(q,k,v,owners,b,g)
    row={'name':name,'status':'preparing','provenance':provenance,'q_shape':list(q.shape),'k_shape':list(k.shape),
        'source_payload_bytes':2*k.numel()*k.element_size(),'configuration_qualification':[]}
    result['cases'].append(row);save()
    assert b=={'main30b':3,'qmsum4b':4}[name]
    prior=json.loads(Path(prior_comparison).read_text())
    assert prior['status']=='completed' and prior['provenance']==provenance
    configs={regime:prior['selected'][regime]['fi_decode'] for regime in ['hot','cold']}
    row['prior_native_tuning_source']={'path':prior_comparison,'sha256':sha(prior_comparison),'selected_configs':configs}
    supports=[[torch.where((owners[h]&(1<<request))!=0)[0] for request in range(b)] for h in range(k.shape[0])]
    props=torch.cuda.get_device_properties(0)
    flush=torch.zeros(max(2*props.L2_cache_size,256*1024*1024)//4,dtype=torch.float32,device='cuda')
    operators={};native={}
    for partial in [torch.float32,torch.bfloat16]:
        label='msa_'+str(partial).split('.')[-1];t=time.perf_counter()
        op=MSAGatherE0(q,k,v,owners,b,partial_dtype=partial)
        output,lse=op.run();torch.cuda.synchronize()
        qualification={**check(output,lse,expected,expected_lse),'method':label,
            'metadata_plan_seconds':op.plan_seconds,'preparation_and_first_call_seconds':time.perf_counter()-t,
            'lengths':op.plan_cpu['lengths'],'blocks':op.plan_cpu['blocks'],
            'payload_copy_bytes_per_call':op.payload_copy_bytes_per_call,
            'gather_read_plus_write_bytes_per_call':2*op.payload_copy_bytes_per_call,
            'metadata_bytes':op.metadata_bytes,'payload_padding_tokens':0,
            'masked_tail_lanes':op.plan_cpu['masked_tail_lanes']}
        row['configuration_qualification'].append(qualification);save()
        if qualification['status']=='passed':operators[label]=op
    assert operators,'No MSA partial dtype passed frozen numerical contract'
    for regime,config in configs.items():
        t=time.perf_counter();op=NativeSparseDecode(q,k,v,supports,use_tensor_cores=config['tensor_cores'],
            fixed_split_size=config['split'],normalize_lse=True)
        output,lse=op.run();torch.cuda.synchronize()
        qualification={**check(output,lse,expected,expected_lse),'method':'native_e0_'+regime,
            'metadata_plan_seconds':op.plan_ms/1000,'preparation_and_first_call_seconds':time.perf_counter()-t,
            'payload_copy_bytes_per_call':0}
        row['configuration_qualification'].append(qualification);save()
        assert qualification['status']=='passed';native[regime]=op
    row['case_setup_and_qualification_seconds']=time.perf_counter()-start
    row['timing']={};rng=random.Random(132)
    for regime,reps in [('hot',20),('cold',1)]:
        funcs={label:op.run for label,op in operators.items()}
        funcs.update(native_e0=native[regime].run,native_e0_AA=native[regime].run)
        graphs={label:graph_of(fn,reps) for label,fn in funcs.items()}
        measurements=[]
        for block in range(12):
            order=list(graphs);rng.shuffle(order)
            for direction,labels in [('random',order),('reverse',list(reversed(order)))]:
                measurements.append({'block':block,'direction':direction,'order':labels,
                    'us':{label:elapsed(graphs[label],reps,flush if regime=='cold' else None) for label in labels}})
        row['timing'][regime]={'repetitions':reps,'measurements':measurements,
            'median_us':{label:statistics.median(m['us'][label] for m in measurements) for label in graphs},
            'paired_median_over_native_e0':{label:statistics.median(m['us'][label]/m['us']['native_e0'] for m in measurements) for label in operators},
            'native_AA_median_ratio':statistics.median(m['us']['native_e0']/m['us']['native_e0_AA'] for m in measurements)}
        save()
    # Component graphs are descriptive; cache effects mean their medians are
    # not additive. Attention-only numbers are never substituted for full call.
    row['components_hot_us']={}
    for label,op in operators.items():
        op.gather();row['components_hot_us'][label]={}
        for component,fn in [('gather',op.gather),('msa_attention_including_native_allocations_and_combine',op.attention)]:
            graph=graph_of(fn,20)
            row['components_hot_us'][label][component]=statistics.median(elapsed(graph,20) for _ in range(7))
        # Explicitly include CPU metadata creation, allocation and device copy
        # in a separate repeated one-shot wall measurement, without hidden JIT.
        seconds=[]
        for _ in range(5):
            torch.cuda.synchronize();t=time.perf_counter()
            fresh=MSAGatherE0(q,k,v,owners,b,partial_dtype=op.partial_dtype);fresh.run();torch.cuda.synchronize()
            seconds.append(time.perf_counter()-t)
        row['components_hot_us'][label]['one_shot_plan_gather_attention_wall_seconds']=seconds
    row['status']='completed';save();print(json.dumps({'name':name,'timing':{k:v['median_us'] for k,v in row['timing'].items()}}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/msa_e0_adapter/qualification_timing.json');args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    torch.set_num_threads(4);torch.manual_seed(20261016);torch.backends.cuda.matmul.allow_tf32=False
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    package=Path(importlib.util.find_spec('flashinfer').origin).parent
    sources=[Path(__file__),ROOT/'src/msa_e0_adapter.py',package/'msa_ops/sparse_decode.py',package/'msa_ops/cute_dsl/sparse_decode_sm12x.py']
    result={'status':'running','uid':os.geteuid(),'python':sys.executable,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'start_epoch':time.time(),'versions':{p:importlib.metadata.version(p) for p in ['torch','flashinfer-python','nvidia-cutlass-dsl','apache-tvm-ffi']},
        'source_sha256':{str(path):sha(path) for path in sources},'cases':[],
        'method':'FlashInfer MSA E0 gather adapter; not full author algorithm reproduction',
        'semantics':'Flatten physical head/request into native batch; gather exact source positions every call, ragged true lengths mask partial blocks. Already post-RoPE Q/K untouched. No payload padding or source-block support enlargement.',
        'complete_call':'Both K/V index_select gathers plus MSA native split/combine/output allocations and natural-log LSE. Metadata planning reported separately and also included in one-shot wall measurements.',
        'tolerance':{'output_atol':.006,'output_rtol':.015,'lse_atol':.0003,'lse_rtol':.0001},
        'scope':'Two exact-mask operator replay points and tail fixtures; no application speed or quality claim.',
        'official_api':'https://docs.flashinfer.ai/generated/flashinfer.msa_ops.msa_sparse_decode_attention.html'}
    def save():
        temp=output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2)+'\n');temp.replace(output)
    def interrupt(signum,frame):raise TimeoutError(f'External signal {signum}')
    signal.signal(signal.SIGTERM,interrupt);save()
    try:
        fixtures(result,save)
        case(result,save,'main30b','results/captures/main30b_tp2_docfinqa256_export/rank0',31,2,'results/fa4/compare_main30b_with_union.json')
        case(result,save,'qmsum4b','results/captures/official4b_boundary/qmsum3',23,8,'results/fa4/compare_qmsum4b_with_union.json')
        result['status']='completed'
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        result['end_epoch']=time.time();save()


if __name__=='__main__':main()
