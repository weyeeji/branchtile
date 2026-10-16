"""Shared native plan/GPU indices boundary qualifications using original-cache FP64."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from src.vllm_shared_decode_plan import LayerSharedDecodePlan


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--fixed-splits',default='default,64');args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    torch.set_num_threads(4);torch.manual_seed(2027)
    layers,heads,questions,prompt=4,2,3,1024
    backing=torch.randn(4000,layers,heads,16,256,device='cuda',dtype=torch.bfloat16)
    caches=[backing[:,layer] for layer in range(layers)]
    histories=[[[torch.randperm(prompt)[:96].sort().values for h in range(heads)] for q in range(questions)] for l in range(layers)]
    tables=torch.stack([torch.cat((torch.arange(64),torch.arange(64+q*512,64+(q+1)*512))) for q in range(questions)])
    records=[]
    def save():
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'status':'running','records':records},indent=2)+'\n')
    for split_string in args.fixed_splits.split(','):
        split=None if split_string=='default' else int(split_string)
        ops={b:LayerSharedDecodePlan(caches,histories,[prompt]*questions,'document-mask-epoch0',b,
            max_per_request=8192,fixed_split_size=split) for b in (1,3)}
        queries={b:torch.randn(layers,b,16,128,device='cuda',dtype=torch.bfloat16)*.25 for b in (1,3)}
        graphs={};epoch=0;current_histories=histories
        lengths=(96,127,128,129,255,256,257,511,512,513,1023,1024,1025,4095,4096,4097)
        cases=[(3,[0,1,2],length) for length in lengths]
        cases.extend([(1,[2],513),(3,[2,0,1],1025),(3,[1,2,0],1026),(1,[0],1027),(3,[0,1,2],4097)])
        for step,(batch,question_rows,length) in enumerate(cases):
            op=ops[batch];q=queries[batch];q.normal_(std=.25)
            backing.add_(.001953125)
            if step==18:
                epoch=1
                current_histories=[[[torch.randperm(prompt)[:96].sort().values for h in range(heads)] for z in range(questions)] for l in range(layers)]
            table=tables[question_rows]+(1800 if epoch else 0)
            table_gpu=table.to('cuda',dtype=torch.int32)
            positions=[prompt+length-96-1+(i%2) for i in range(batch)]
            item={'fixed_split_pages':split,'step':step,'batch':batch,'question_order':question_rows,
                'support_length_base':length,'query_positions':positions,'history_epoch':epoch}
            started=time.perf_counter()
            try:
                identity=f'document-mask-epoch{epoch}'
                if op.history_id!=identity:
                    try:
                        op.prepare(table_gpu,question_rows,positions,identity)
                        raise AssertionError('Missing identity invalidation guard')
                    except ValueError as exc:
                        item['stale_prefix_identity_rejected']=str(exc)
                    op.set_histories(current_histories,identity)
                before=op.plan_calls
                prep=op.prepare(table_gpu,question_rows,positions,identity)
                assert op.plan_calls-before==1
                item['prepare']=prep
                recapture=batch not in graphs or prep['plan_topology_changed_requires_recapture']
                if recapture:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for layer in range(layers):op.run_layer(layer,q[layer])
                    torch.cuda.current_stream().wait_stream(stream)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        for layer in range(layers):op.run_layer(layer,q[layer])
                    graphs[batch]=graph
                graphs[batch].replay()
                torch.cuda.synchronize()
                item['prepare_capture_replay_complete_seconds']=time.perf_counter()-started
                item['recaptured_for_new_batch_or_topology']=recapture
                checks=[]
                for layer in range(layers):
                    expected=torch.empty_like(q[layer],dtype=torch.float64)
                    for request,question in enumerate(question_rows):
                        for head in range(heads):
                            logical=torch.cat((current_histories[layer][question][head],torch.arange(prompt,positions[request]+1)))
                            pages=table[request,logical//16].to('cuda');offsets=(logical%16).to('cuda')
                            kv=caches[layer][pages,head,offsets].double()
                            score=q[layer,request,head*8:(head+1)*8].double() @ kv[:,:128].T / 128**.5
                            expected[request,head*8:(head+1)*8]=score.softmax(-1) @ kv[:,128:]
                    actual=op.outputs[layer].double()
                    checks.append({'layer':layer,'frozen_output_tolerance_passed':bool(torch.isclose(actual,expected,atol=.006,rtol=.015).all()),'max_abs':float((actual-expected).abs().max())})
                item['checks']=checks;item['status']='qualified' if all(c['frozen_output_tolerance_passed'] for c in checks) else 'failed'
            except Exception as exc:
                known_unsupported='new batch size should not exceed padded batch size' in str(exc)
                item.update(status='unsupported_native_graph_configuration' if known_unsupported else 'failed',error=repr(exc))
                if known_unsupported:
                    try:
                        op.run_layer(0,q[0])
                        raise RuntimeError('Failed plan remained executable')
                    except AssertionError:
                        item['failed_plan_execution_rejected']=True
            records.append(item);save()
            if item['status']=='failed':break
        del ops,graphs,queries
    success=len(records)==21*len(args.fixed_splits.split(',')) and all(r['status'] in ('qualified','unsupported_native_graph_configuration') for r in records)
    path=Path(args.output)
    unsupported=sum(r['status']=='unsupported_native_graph_configuration' for r in records)
    status='qualified_supported_configurations' if success and unsupported else ('qualified' if success else 'failed')
    path.write_text(json.dumps({'status':status,'unsupported_native_graph_cases':unsupported,'records':records,
        'uid':os.geteuid(),'python':sys.executable,'packages':{x:importlib.metadata.version(x) for x in ('torch','flashinfer-python','triton')},
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/vllm_shared_decode_plan.py','scripts/qualify_shared_decode_plan.py']},
        'frozen_output_tolerance':{'atol':.006,'rtol':.015},'source_cache_shape':list(caches[0].shape),'source_cache_stride':list(caches[0].stride()),
        'scope':'Existing native plan sharing across four layers with GPU address mapping. Explicit recapture if native topology changes. Includes split/page boundaries, B3→B1→B3, request reorder, physical-page remap and rejected stale prefix identity. No application performance claim.'},indent=2)+'\n')
    print(json.dumps({'status':status,'cases':len(records),'layer_checks':sum(len(r.get('checks',[])) for r in records),'output':str(path)}))
    assert success


if __name__=='__main__':main()
