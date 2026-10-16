"""CPU source-identity, ragged support and paired timing audit for MSA E0."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from msa_e0_adapter import index_plan


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tsha(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    directory=ROOT/'results/msa_e0_adapter';path=directory/'qualification_timing.json'
    data=json.loads(path.read_text());assert data['status']=='completed'
    assert all(sha(p)==s for p,s in data['source_sha256'].items())
    for fixture in data['fixtures']:
        assert fixture['status']=='passed' and fixture['graph_changed_query_and_payload']['status']=='passed'
    result={'status':'passed','source':str(path),'source_sha256':sha(path),'cases':[],
        'scope':'Exact-support E0 mechanism adapter only. Both gathers, native attention/combine and natural-log LSE are included in full-call GPU timing. Plan/allocation/copy CPU wall costs are also reported. No application throughput or author-algorithm claim.',
        'interpretation':'Both natural points are slower than the independently selected native sparse-decode baseline. No full-call speed advantage was found. Component medians are descriptive and not additive.',
        'paired_interval_method':'Resample 12 measurement blocks jointly (random/reverse retained together), 10000 draws, median ratio to native E0; descriptive operator timing interval only.',
        'physical_gpu_identity_read_after_completed_job':subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip()}
    rng=np.random.default_rng(124)
    for case in data['cases']:
        provenance=case['provenance'];capture=Path(provenance['capture']);layer=provenance['layer'];budget=provenance['budget']
        files=[capture/'metadata.json',capture/f'source_l{layer}.pt',capture/f'masks_l{layer}_b{budget}.pt']
        meta=json.loads(files[0].read_text());source=torch.load(files[1],map_location='cpu',weights_only=True)
        masks=torch.load(files[2],map_location='cpu',weights_only=True)['supports']
        active=provenance['original_active_requests'];b=len(active);h=case['k_shape'][0]
        prefix=source['prefix_k'].shape[2];doc=source['document_k'].shape[2]
        owners=[torch.full((h,prefix),(1<<b)-1,dtype=torch.int64),
            sum(masks[qi,:h].long()*(1<<ri) for ri,qi in enumerate(active))]
        private=[]
        for ri,qi in enumerate(active):
            file=capture/f'decode_q{qi}_l{layer}_s{provenance["step"]}.pt';files.append(file)
            dec=torch.load(file,map_location='cpu',weights_only=True);length=dec['private_suffix_k'].shape[2]
            private.append(length);owners.append(torch.full((h,length),1<<ri,dtype=torch.int64))
        owners=torch.cat(owners,dim=1);plan=index_plan(owners,b)
        assert list(owners.shape)==case['k_shape'][:2]
        common_counts=masks[active,:h].sum(-1).tolist()
        tail=meta['document_token_range'][1]-(prefix+doc)
        assert tail>=0 and all(count+tail==budget for counts in common_counts for count in counts)
        assert all(plan['lengths'][head*b+ri]==prefix+common_counts[ri][head]+private[ri] for head in range(h) for ri in range(b))
        for q in case['configuration_qualification']:
            assert q['status']=='passed'
            if q['method'].startswith('msa'):
                assert q['lengths']==plan['lengths'] and q['blocks']==plan['blocks'] and q['payload_padding_tokens']==0
                assert q['payload_copy_bytes_per_call']==len(plan['indices'])*128*2*2
        summary={'name':case['name'],'capture_files_sha256':{str(p):sha(p) for p in files},
            'owners_sha256':tsha(owners),'source_indices_sha256':tsha(plan['indices']),
            'cu_seqlens_sha256':tsha(plan['cu_seqlens_k']),'q2k_sha256':tsha(plan['q2k_indices']),
            'batch':b,'kv_heads':h,'gqa':case['q_shape'][1]//b,
            'selected_shared_document_tokens':common_counts,'document_tail_tokens_in_each_private_history':tail,
            'private_suffix_tokens':private,'prefix_tokens':prefix,'true_compact_lengths':plan['lengths'],
            'last_block_valid_tokens':[(n-1)%128+1 for n in plan['lengths']],
            'payload_padding_tokens':0,'no_support_enlargement':True,'timing':{},
            'component_hot_us':case['components_hot_us']}
        for regime,measure in case['timing'].items():
            rows=measure['measurements'];blocks=sorted({r['block'] for r in rows})
            assert len(blocks)==12 and len(rows)==24
            summary['timing'][regime]={'median_us':measure['median_us'],'native_AA_median_ratio':measure['native_AA_median_ratio'],'ratios':{}}
            for method in ['msa_float32','msa_bfloat16']:
                ratios=np.array([[next(r for r in rows if r['block']==block and r['direction']==direction)['us'][method]/
                    next(r for r in rows if r['block']==block and r['direction']==direction)['us']['native_e0']
                    for direction in ['random','reverse']] for block in blocks])
                sampled=np.median(ratios[rng.integers(0,len(blocks),size=(10000,len(blocks)))],axis=(1,2))
                summary['timing'][regime]['ratios'][method]={'median':float(np.median(ratios)),
                    'paired_block_bootstrap95':np.quantile(sampled,[.025,.975]).tolist()}
        result['cases'].append(summary)
    events=[json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines() if line.strip()]
    events=[event for event in events if event['name']=='msa_e0_adapter_two_natural']
    ends=[event for event in events if event['event']=='end']
    assert len(ends)==1 and ends[0]['exit_code']==0 and ends[0]['gpu_hours']<=.2
    (directory/'gpu_ledger.json').write_text(json.dumps({'events':events,'total_gpu_hours':ends[0]['gpu_hours'],'budget_gpu_hours':.2},indent=2)+'\n')
    result['total_gpu_hours']=ends[0]['gpu_hours']
    (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','cases':len(result['cases']),'gpu_hours':result['total_gpu_hours']}))


if __name__=='__main__':main()
