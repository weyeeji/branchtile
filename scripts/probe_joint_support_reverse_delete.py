#!/usr/bin/env python3
"""CPU strong controls for score-constrained union minimization, not novelty."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from probe_joint_support_exchange import (ROOT, PROTOCOL, protected_mask, score_sum,
                                         stable_order, describe, exchange, sha)
OUT=ROOT/'results/joint_support_reverse_delete_v1'


def reverse_delete(scores, baseline, protected, epsilon, universe):
    b,n=scores.shape
    k=int(baseline[0].sum())-int(protected.sum())
    original=np.array([score_sum(scores[q],baseline[q]) for q in range(b)])
    active=(baseline.any(axis=0) if universe=='original_union' else np.ones(n,dtype=bool))
    ids=np.flatnonzero(active & ~protected)
    ranked=[stable_order(ids,scores[q],descending=True) for q in range(b)]
    masks=np.repeat(protected[None,:],b,axis=0)
    for q in range(b): masks[q,ranked[q][:k]]=True
    assert all(score_sum(scores[q],masks[q])==original[q] for q in range(b))
    if epsilon==0:
        # Unique natural top-k fixes its minimum union. Preserve exact original
        # supports; tie cases in artificial fixtures use positive epsilon.
        assert np.array_equal(masks,baseline)
        return masks,dict(removed_from_start=0,accepted_decisions=0,zero_loss_shortcut=True)
    priority=(scores/original[:,None]).sum(axis=0)
    ordered=stable_order(ids,priority)
    cursor=[k]*b
    losses=np.zeros(b,dtype=np.float64)
    accepted=0
    for t in ordered:
        affected=np.flatnonzero(masks[:,t])
        replacements={}; valid=True
        for q in affected:
            while cursor[q]<len(ranked[q]) and not active[ranked[q][cursor[q]]]: cursor[q]+=1
            if cursor[q]==len(ranked[q]): valid=False; break
            nxt=int(ranked[q][cursor[q]])
            delta=float(scores[q,t]-scores[q,nxt])
            assert delta>=0
            new_loss=losses[q]+delta
            if new_loss>epsilon*original[q]*(1+1e-12):
                valid=False;break
            # Recompute the actual selected sum at a boundary; a rounding guard
            # must not reject a mathematically feasible exact-boundary fixture.
            if new_loss>epsilon*original[q]*(1-1e-12):
                candidate=masks[q].copy();candidate[t]=False;candidate[nxt]=True
                new_loss=original[q]-score_sum(scores[q],candidate)
                if new_loss>epsilon*original[q] or new_loss/original[q]>epsilon:
                    valid=False;break
            replacements[int(q)]=(nxt,new_loss)
        if not valid: continue
        active[t]=False;accepted+=1
        for q,(nxt,new_loss) in replacements.items():
            masks[q,t]=False;masks[q,nxt]=True;losses[q]=new_loss;cursor[q]+=1
    assert masks[:,protected].all() and np.all(masks.sum(axis=1)==baseline.sum(axis=1))
    assert np.all(~masks | active[None,:])
    for q in range(b):
        loss=original[q]-score_sum(scores[q],masks[q])
        assert loss/original[q]<=epsilon and loss<=epsilon*original[q]
        allowed=np.flatnonzero(active & ~protected)
        expected=stable_order(allowed,scores[q],descending=True)[:k]
        assert np.array_equal(np.sort(expected),np.flatnonzero(masks[q] & ~protected))
    return masks,dict(removed_from_start=accepted,accepted_decisions=accepted,
                     retained_candidate_union=int(active.sum()),zero_loss_shortcut=False,
                     exact_fsum_boundary_recheck_relative_guard=1e-12)


def fixtures():
    cases=[(np.array([[102,101,100],[102,100,101],[100,102,101]],dtype=float),2,.005),
           (np.array([[100,1,99],[1,100,99]],dtype=float),1,.01)]
    rows=[]
    for scores,k,eps in cases:
        baseline=np.zeros_like(scores,dtype=bool)
        for q in range(len(scores)): baseline[q,stable_order(np.arange(scores.shape[1]),scores[q],True)[:k]]=True
        protected=np.zeros(scores.shape[1],dtype=bool)
        a,_=exchange(scores,baseline,protected,eps,list(range(len(scores))))
        d,_=exchange(scores,baseline,protected,eps,list(reversed(range(len(scores)))))
        assert np.array_equal(a,baseline) and np.array_equal(d,baseline)
        result={}
        for universe in ['original_union','all_positions']:
            out,_=reverse_delete(scores,baseline,protected,eps,universe)
            result[universe]=int(out.any(axis=0).sum())
        assert result['all_positions']==k
        rows.append(dict(scores=scores.tolist(),k=k,epsilon=eps,exchange_union=int(a.any(axis=0).sum()),
                         reverse_delete_union=result,optimal_union=k))
    return rows


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    tests=fixtures()
    if '--self-test' in sys.argv: print(json.dumps(tests));return
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert not (OUT/'results.jsonl').exists()
    source_protocol=json.loads(PROTOCOL.read_text())
    assert sha(PROTOCOL)==protocol['exchange_protocol_sha256']
    rows=[];start=time.perf_counter();cpu=time.process_time()
    for source in source_protocol['sources']:
        directory=ROOT/source['directory'];meta=json.loads((directory/'metadata.json').read_text())
        assert sha(directory/'metadata.json')==source['metadata_sha256'] and meta['export_semantics_version']==2
        ds,de=meta['document_token_range'];cs,ce=meta['actual_shared_document_token_range'];assert cs==ds
        n=de-ds;common_n=ce-ds;protected=protected_mask(n)
        assert n-common_n<=32 and protected[common_n:].all()
        loaded=[]
        for item in source['selector_files']:
            path=directory/item['name'];assert sha(path)==item['sha256']
            loaded.append(torch.load(path,map_location='cpu',weights_only=False))
        for budget in source_protocol['budgets']:
            for head in range(2):
                scores=np.stack([x['pooled_scores'][head].numpy().astype(np.float64) for x in loaded])
                baseline=np.stack([x['supports'].get(budget,x['supports'].get(str(budget)))[head].numpy() for x in loaded])
                assert np.all(baseline.sum(axis=1)==budget) and baseline[:,protected].all()
                for eps in protocol['epsilon']:
                    for universe in protocol['universes']:
                        t=time.perf_counter();out,events=reverse_delete(scores,baseline,protected,eps,universe)
                        row=dict(document=source['id'],document_sha256=meta['document_sha256'],rank=0,layer=15,
                                 kv_head=head,B=meta['B'],budget=budget,epsilon=eps,universe=universe,
                                 selection_wall_seconds=time.perf_counter()-t,events=events,
                                 **describe(scores,out,baseline,common_n))
                        rows.append(row)
        print(json.dumps(dict(document=source['id'],rows=len(rows))),flush=True)
    assert len(rows)==protocol['expected_rows']==96 and not torch.cuda.is_initialized()
    (OUT/'results.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    summary=dict(rows=len(rows),protocol_sha256=sha(OUT/'protocol.json'),script_sha256=sha(Path(__file__)),
                 results_sha256=sha(OUT/'results.jsonl'),fixtures=tests,cpu_seconds=time.process_time()-cpu,
                 wall_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 cuda_initialized=torch.cuda.is_initialized(),gpu_hours=0,aggregates=[])
    for eps in protocol['epsilon']:
        for universe in protocol['universes']:
            rs=[r for r in rows if r['epsilon']==eps and r['universe']==universe]
            u=[r['actual_common_source_logical']['union_reduction_fraction'] for r in rs]
            summary['aggregates'].append(dict(epsilon=eps,universe=universe,cells=len(rs),
                union_reduction_min_median_max=[min(u),float(np.median(u)),max(u)],
                max_request_score_loss=max(q['score_loss_fraction'] for r in rs for q in r['per_request']),
                selection_wall_seconds=sum(r['selection_wall_seconds'] for r in rs)))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['fixtures','aggregates']}),flush=True)
if __name__=='__main__':main()
