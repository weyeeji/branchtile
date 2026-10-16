"""CPU-only descriptive probability changes on fixed teacher paths.

These metrics do not establish generated-answer quality or revise logit gates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    import torch
    assert not torch.cuda.is_initialized()
    root=Path(a.directory);summary=json.loads((root/'summary.json').read_text())
    temperature=summary['sampling']['temperature'];top_p=summary['sampling']['top_p']
    basepath=root/'rank0/e0_natural_0.pt';ownerpath=root/'rank0/owner_forced_0.pt'
    baseline=torch.load(basepath,map_location='cpu',weights_only=True);owner=torch.load(ownerpath,map_location='cpu',weights_only=True)
    def probs(x):return torch.softmax(x.double()/temperature,dim=0)
    def nucleus(p):
        sorted_p,order=torch.sort(p,descending=True)
        keep=(torch.cumsum(sorted_p,0)-sorted_p)<top_p
        q=torch.zeros_like(p);q[order[keep]]=sorted_p[keep];return q/q.sum()
    rows=[]
    for key in sorted(set(baseline)&set(owner)):
        x,y=baseline[key],owner[key];px,py=probs(x),probs(y);nx,ny=nucleus(px),nucleus(py)
        mix=(px+py)/2
        js=.5*((px*torch.log((px/mix).clamp_min(1e-300))).sum()+(py*torch.log((py/mix).clamp_min(1e-300))).sum())
        rows.append({'key':key,'cosine':float(torch.nn.functional.cosine_similarity(x.double(),y.double(),dim=0)),
            'raw_top1_equal':int(x.argmax())==int(y.argmax()),'softmax_total_variation':float((px-py).abs().sum()/2),
            'top_p_total_variation':float((nx-ny).abs().sum()/2),'softmax_js_nats':float(js),
            'baseline_top1_probability':float(px.max()),'owner_top1_probability':float(py.max()),
            'logit_mean_shift':float((y-x).double().mean()),'centered_logit_rmse':float(((y-y.mean())-(x-x.mean())).double().square().mean().sqrt())})
    def stats(name):
        vals=torch.tensor([r[name] for r in rows],dtype=torch.float64)
        return {'min':float(vals.min()),'median':float(vals.median()),'max':float(vals.max()),'mean':float(vals.mean())}
    result={'status':'descriptive_fixed_path_distribution_audit','vectors':len(rows),'temperature':temperature,'top_p':top_p,
        'softmax_total_variation':stats('softmax_total_variation'),'top_p_total_variation':stats('top_p_total_variation'),
        'softmax_js_nats':stats('softmax_js_nats'),'rows':rows,'qualification_gate_relaxed':False,
        'quality_or_noninferiority_proved':False,'cuda_initialized':False,
        'scope':'CPU mathematical softmax/nucleus distributions, not a reproduction of vLLM sampler RNG or implementation. Top-p retains the crossing token. Actual answer quality requires natural generations and a task metric.',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'summary.json',basepath,ownerpath)}}
    out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256')}))
if __name__=='__main__':main()
