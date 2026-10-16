#!/usr/bin/env python3
"""CPU geometry audit; reconstruct unchanged QC masks, never read decode Q."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from capture_model import select_support, set_partitions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/natural_capacity_boundary_v1'
DOCUMENTS = ['docfinqa256', 'publicdev1_256', 'publicdev6_256']
BUDGETS = [4096, 8192, 16384, 32768, 65536, 131072, 'full']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def work(counts, batch, gqa, block_n=64):
    # Counts already aggregate equal owner sets; chronological reordering is
    # not charged here. All quantities are optimistic logical work, not time.
    counts = np.asarray(counts, dtype=np.int64)
    dynamic_tiles = sum(math.ceil(int(c)/block_n)*math.ceil(o.bit_count()*gqa/16)
                        for row in counts for o,c in enumerate(row) if o and c)
    dynamic_n_tiles = sum(math.ceil(int(c)/block_n) for row in counts for o,c in enumerate(row) if o and c)
    union = int(counts[:,1:].sum())
    partitions = []
    for partition in set_partitions(batch):
        lengths = [[sum(int(row[o]) for o in range(1,1<<batch) if o & group)
                    for group in partition] for row in counts]
        tiles = sum(math.ceil(n/block_n)*math.ceil(group.bit_count()*gqa/16)
                    for row in lengths for n,group in zip(row,partition))
        partitions.append({'partition':partition, 'logical_KV_token_visits':sum(map(sum,lengths)),
                           'logical_M16_N64_tiles':tiles,
                           'logical_N64_load_tiles':sum(math.ceil(n/block_n) for row in lengths for n in row)})
    best = min(partitions,key=lambda p:(p['logical_M16_N64_tiles'],p['logical_KV_token_visits'],p['partition']))
    relaxed_tiles = 0
    for row in counts:
        relaxed_tiles += min(sum(math.ceil(sum(int(row[o]) for o in range(1,1<<batch) if o&group)/block_n)
                                *math.ceil(group.bit_count()*gqa/16) for group in partition)
                             for partition in set_partitions(batch))
    pareto = [p for p in partitions if not any(
        q['logical_KV_token_visits'] <= p['logical_KV_token_visits'] and
        q['logical_M16_N64_tiles'] <= p['logical_M16_N64_tiles'] and
        (q['logical_KV_token_visits'],q['logical_M16_N64_tiles']) != (p['logical_KV_token_visits'],p['logical_M16_N64_tiles'])
        for q in partitions)]
    full = next(p for p in partitions if p['partition']==[(1<<batch)-1])
    return {'block_N':block_n,'M_tile':16,'owner_counts_by_head':counts.tolist(),
        'dynamic_optimistic_KV_token_visits':union,'dynamic_optimistic_M16_N64_tiles':dynamic_tiles,
        'dynamic_optimistic_N64_load_tiles':dynamic_n_tiles,
        'full_union':full,'best_fixed_MMA_same_partition_all_heads':best,
        'relaxed_best_fixed_MMA_allow_per_head_partition_tiles':relaxed_tiles,
        'dynamic_over_best_fixed_MMA':dynamic_tiles/best['logical_M16_N64_tiles'],
        'best_fixed_KV_over_dynamic':best['logical_KV_token_visits']/union,
        'full_union_MMA_over_dynamic':full['logical_M16_N64_tiles']/dynamic_tiles,
        'fixed_partition_Pareto_frontier':pareto,
        'fixed_partition_count':len(partitions),
        'dynamic_dominated_or_matched_by_fixed':any(p['logical_KV_token_visits']<=union and p['logical_M16_N64_tiles']<=dynamic_tiles for p in partitions)}


def main():
    assert os.getuid()!=0 and sys.prefix!=sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    started=time.perf_counter(); inputs={}; rows=[]; old_checks=[]; inventory=[]
    def load(path):
        inputs[str(path.relative_to(ROOT))]=sha(path)
        return torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    for document in DOCUMENTS:
        for rank in (0,1):
            directory=ROOT/f'results/captures/main30b_tp2_{document}_suffix_v2/rank{rank}'
            meta=json.loads((directory/'metadata.json').read_text())
            inputs[str((directory/'metadata.json').relative_to(ROOT))]=sha(directory/'metadata.json')
            assert meta['export_semantics_version']==2 and meta['gqa']==8 and meta['kv_heads']==2
            batch=meta['B']; n=meta['document_token_range'][1]-meta['document_token_range'][0]
            common_n=meta['actual_shared_document_token_range'][1]-meta['document_token_range'][0]
            private_doc=n-common_n
            assert private_doc==meta['private_protected_document_tail_tokens'] and private_doc<=32
            for layer in meta['layers']:
                src=load(directory/f'source_l{layer}.pt')
                prefix=src['prefix_k'].shape[2]
                assert src['document_k'].shape[1:]==(2,common_n,128)
                del src
                sizes={}
                for step in (16,64,256):
                    assert meta['ready_lifetime_measured_by_step'][str(step)]
                    tails=[]; events=[]
                    for request in range(batch):
                        dec=load(directory/f'decode_q{request}_l{layer}_s{step}.pt')
                        tails.append(dec['private_suffix_k'].shape[2])
                        events.append(dec['metadata']['same_event_query_positions'])
                        assert dec['metadata']['physical_full_prefix_pages_match'] and dec['metadata']['hooked_kv_matches_actual_cache']
                        # No decode Q tensor values enter this CPU audit.
                    assert all(e==events[0] for e in events)
                    sizes[step]=tails
                    full_tokens=2*(prefix+common_n+sum(tails))
                    inventory.append({'document':document,'rank':rank,'layer':layer,'step':step,'B':batch,'Hkv':2,'GQA':8,
                        'full_source_token_entries':full_tokens,'full_source_BF16_bytes':full_tokens*512,
                        'full_source_FP8_payload_only_bytes':full_tokens*256,'same_forward_ready':True})
                selectors=[load(directory/f'selector_q{q}_l{layer}.pt') for q in range(batch)]
                assert all(s['observation_is_question_suffix_only'] and s['pooled_scores'].shape==(2,n) and torch.isfinite(s['pooled_scores']).all() for s in selectors)
                for budget in BUDGETS:
                    effective=n if budget=='full' else min(budget,n)
                    supports=torch.stack([select_support(s['pooled_scores'],effective) for s in selectors])
                    assert bool(supports.sum(-1).eq(effective).all())
                    if budget in meta['budgets']:
                        old=load(directory/f'masks_l{layer}_b{budget}.pt')['supports']
                        assert torch.equal(old,supports[:,:,:common_n])
                        assert all(torch.equal(s['supports'][budget],supports[q]) for q,s in enumerate(selectors))
                        old_checks.append({'document':document,'rank':rank,'layer':layer,'budget':budget,'bitwise_restored':True})
                    owners=sum(supports[q,:,:common_n].to(torch.int16)*(1<<q) for q in range(batch))
                    counts=np.stack([np.bincount(owners[h].numpy(),minlength=1<<batch) for h in range(2)])
                    counts[:,(1<<batch)-1]+=prefix
                    for request,tail in enumerate(sizes[64]):
                        counts[:,1<<request]+=tail
                    geometry=work(counts,batch,8)
                    unique=geometry['dynamic_optimistic_KV_token_visits']
                    row={'document':document,'rank':rank,'layer':layer,'step':64,'B':batch,'Hkv':2,'GQA':8,
                         'requested_budget':budget,'effective_full_document_budget':effective,
                         'common_document_tokens':common_n,'private_protected_document_tail_tokens':private_doc,
                         'private_suffix_lengths':sizes[64],
                         'restored_full_document_support_sha256':hashlib.sha256(supports.numpy().tobytes()).hexdigest(),
                         'unique_BF16_payload_bytes':unique*512,'unique_FP8_payload_only_bytes':unique*256,
                         'above_96MiB_BF16':unique*512>96*1024**2,
                         'above_96MiB_FP8_payload_only':unique*256>96*1024**2,
                         **geometry}
                    rows.append(row)
    historical=[]
    for family in ('official4b_boundary','old30b_diagnostic'):
        for directory in sorted((ROOT/'results/captures'/family).glob('qmsum*')):
            path=directory/'metadata.json'
            if not path.exists() or 'failed' in directory.name:continue
            meta=json.loads(path.read_text())
            files=list(directory.glob('source_l*.pt'))
            if not files:continue
            inputs[str(path.relative_to(ROOT))]=sha(path)
            for source in files:
                layer=int(source.stem.split('_l')[1]); src=load(source)
                h=src['document_k'].shape[1]
                base=src['prefix_k'].shape[2]+src['document_k'].shape[2]
                candidates=[]
                for step in (1,16,64,256):
                    tails=[]
                    for q in range(meta['B']):
                        p=directory/f'decode_q{q}_l{layer}_s{step}.pt'
                        if p.exists():tails.append(load(p)['private_suffix_k'].shape[2])
                    if tails:candidates.append({'step':step,'available_B':len(tails),
                        'full_source_BF16_bytes':h*(base+sum(tails))*512})
                historical.append({'capture':str(directory.relative_to(ROOT)),'layer':layer,'Hkv':h,
                    'GQA':meta.get('gqa'),'base_tokens':base,'states':candidates,
                    'max_full_source_BF16_bytes':max((c['full_source_BF16_bytes'] for c in candidates),default=None),
                    'availability':'decode states present' if candidates else 'source only; no matching decode states',
                    'same_forward_ready':False,'scope':'Serial diagnostic captures; footprint upper bound, not common-ready evidence.'})
                del src
    for name in ('scripts/capture_model.py','scripts/export_rank_capture.py','src/branchtile/attention.py','src/partition_baseline.py'):
        inputs[name]=sha(ROOT/name)
    result={'schema':1,'scope':'CPU mask/geometry audit only; no GPU runs or mask export into production.',
        'uid':os.getuid(),'python':sys.executable,'CUDA_initialized':torch.cuda.is_initialized(),'GPU_hours':0,
        'elapsed_seconds':time.perf_counter()-started,'input_sha256':inputs,
        'restored_old_budget_checks':old_checks,'full_source_inventory':inventory,
        'historical_boundary_inventory':historical,'geometry_rows':rows,
        'limitations':['All logical work uses an optimistic split=1 M16/N64 count, with no preparation, state/register pressure, split/merge, pipeline or bandwidth timing.',
            'Best fixed MMA uses one partition across heads; a per-head relaxed bound is also reported. Pareto comparisons do not predict a runtime winner.',
            'BF16 payload is actual capture dtype; FP8 numbers are byte-volume estimates only, not restored FP8 operands or a numerical qualification.',
            'Footprint above L2 is not proof of a cache miss. Other allocations and cache-line/index traffic are not included.',
            'Full source remains allocated; logical support does not release physical capacity.',
            'Frozen suffix-only observation scores choose masks; decode queries/outputs are not used to select budgets or supports.',
            'Only existing same-forward B3/B4 main captures are restored. B8 historical slow variants are not substituted.']}
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(OUT/'audit.json'),'geometry_rows':len(rows),'old_masks_restored':len(old_checks),
                      'full_source_rows':len(inventory),'historical_rows':len(historical),'GPU_hours':0}))


if __name__=='__main__':main()
