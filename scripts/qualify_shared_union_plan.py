"""Original-cache FP64 qualification of persistent masked union graph metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.vllm_shared_union_plan import LayerSharedUnionPlan


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    torch.set_num_threads(4);torch.manual_seed(2028)
    layers,heads,questions,prompt=4,2,3,1024
    backing=torch.randn(7200,layers,heads,16,256,device='cuda',dtype=torch.bfloat16)
    caches=[backing[:,l] for l in range(layers)]
    histories=[[[torch.cat((torch.randperm(1008)[:80].sort().values,torch.arange(1008,1024))) for h in range(heads)] for q in range(questions)] for l in range(layers)]
    tables=torch.stack([torch.cat((torch.arange(63),torch.arange(64+q*1024,64+(q+1)*1024))) for q in range(questions)])
    workspace=torch.zeros(256*1024*1024,device='cuda',dtype=torch.uint8)
    ops={b:LayerSharedUnionPlan(caches,histories,[prompt]*3,'epoch0',b,1008,max_generated=12289,workspace=workspace) for b in (1,3)}
    queries={b:torch.randn(layers,b,16,128,device='cuda',dtype=torch.bfloat16)*.25 for b in (1,3)}
    graphs={};records=[]
    cases=[(3,[0,1,2],n) for n in (0,1,7,8,15,16,17,127,128,129,4095,4096,4097,8191,8192,8193,12288)]
    cases.extend([(1,[2],17),(3,[2,0,1],18),(3,[1,2,0],19),(1,[0],20),(3,[0,1,2],21)])
    for step,(batch,order,tail) in enumerate(cases):
        op=ops[batch];q=queries[batch];q.normal_(std=.25);backing.add_(.001953125)
        table=tables[order]+(3600 if step>=19 else 0);gpu_table=table.to('cuda',dtype=torch.int32)
        positions=[prompt+tail-1+(i%2) for i in range(batch)]
        item={'step':step,'batch':batch,'question_order':order,'generated_tail_base':tail,'physical_epoch':int(step>=19)}
        try:
            if step==19:
                invalid=gpu_table.clone();invalid[1,0]+=1
                for name,table_argument,identity in [('prefix_identity',gpu_table,'changed'),('shared_page_identity',invalid,'epoch0')]:
                    try:
                        op.prepare(table_argument,order,positions,identity)
                        raise AssertionError('Invalidation was not rejected')
                    except ValueError as exc:item[name+'_rejected']=str(exc)
                    try:
                        op.run_layer(0,q[0]);raise RuntimeError('Invalid plan remained executable')
                    except AssertionError:item[name+'_execution_blocked']=True
            started=time.perf_counter();prep=op.prepare(gpu_table,order,positions,'epoch0');item['prepare']=prep
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
            graphs[batch].replay();torch.cuda.synchronize()
            item['prepare_capture_replay_complete_seconds']=time.perf_counter()-started;item['recaptured']=recapture
            checks=[]
            for layer in range(layers):
                expected=torch.empty_like(q[layer],dtype=torch.float64)
                for request,question in enumerate(order):
                    for head in range(heads):
                        logical=torch.cat((histories[layer][question][head],torch.arange(prompt,positions[request]+1)))
                        pages=table[request,logical//16].to('cuda');offsets=(logical%16).to('cuda')
                        kv=caches[layer][pages,head,offsets].double()
                        scores=q[layer,request,head*8:(head+1)*8].double() @ kv[:,:128].T /128**.5
                        expected[request,head*8:(head+1)*8]=scores.softmax(-1) @kv[:,128:]
                actual=op.outputs[layer].double()
                checks.append({'layer':layer,'frozen_output_tolerance_passed':bool(torch.isclose(actual,expected,atol=.006,rtol=.015).all()),'max_abs':float((actual-expected).abs().max())})
            item['checks']=checks;item['status']='qualified' if all(c['frozen_output_tolerance_passed'] for c in checks) else 'failed'
        except Exception as exc:item.update(status='failed',error=repr(exc))
        records.append(item)
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps({'status':'running','records':records},indent=2)+'\n')
        if item['status']=='failed':break
    passed=len(records)==len(cases) and all(r['status']=='qualified' for r in records)
    path.write_text(json.dumps({'status':'qualified' if passed else 'failed','records':records,'uid':os.geteuid(),'python':sys.executable,
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['src/vllm_shared_union_plan.py','scripts/qualify_shared_union_plan.py']},
        'frozen_output_tolerance':{'atol':.006,'rtol':.015},'source_cache_shape':list(caches[0].shape),'source_cache_stride':list(caches[0].stride()),
        'scope':'Existing union with one padded length/native plan across layers, original-cache FP64 reference. Changing Q/KV, byte/page/split boundaries, generated suffix up to12289, B3/B1/reorder, physical remap and explicit invalidation. No application speed/quality claim.'},indent=2)+'\n')
    print(json.dumps({'status':'qualified' if passed else 'failed','cases':len(records),'checks':sum(len(r.get('checks',[])) for r in records)}))
    assert passed


if __name__=='__main__':main()
