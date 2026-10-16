"""CPU-only bounds: observed support unions versus actual 16-token pages."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lifecycle_model(lane_memberships, batch, prefix_tokens, block_size=16):
    """Closed-cohort integer allocator model, with no actual KV allocation.

    Each lane is one captured (TP rank, layer, KV head). All lanes consume
    the same block IDs, as in the full-attention pool. Departure orders are
    enumerated through all remaining-consumer subsets; time between departures
    is an artificial equal interval. This is not a serving trace or prediction.
    """
    prefix_blocks=math.ceil(prefix_tokens/block_size)
    signatures=1<<batch
    counts=torch.stack([torch.bincount(x.long(),minlength=signatures)
                        for x in lane_memberships])
    capacity_by_signature=(counts.max(0).values+block_size-1)//block_size
    capacity_by_signature[0]=0
    max_tokens=max(x.numel() for x in lane_memberships)
    union_blocks=math.ceil(max_tokens/block_size)
    packed=torch.zeros((len(lane_memberships),union_blocks*block_size),dtype=torch.int16)
    signature_sorted=torch.zeros_like(packed)
    for lane,x in enumerate(lane_memberships):
        packed[lane,:x.numel()]=x
        signature_sorted[lane,:x.numel()]=x.sort().values
    rows=[]
    for active in range(1,signatures):
        live=(packed&active).ne(0).view(len(lane_memberships),union_blocks,block_size).any(2).any(0)
        live_sorted=(signature_sorted&active).ne(0).view(len(lane_memberships),union_blocks,block_size).any(2).any(0)
        aligned=sum(int(capacity_by_signature[s]) for s in range(1,signatures) if s&active)
        rows.append({'active_consumer_bits':active,'active_consumers':active.bit_count(),
                     'original_order_union_blocks':prefix_blocks+int(live.sum()),
                     'per_lane_signature_sorted_blocks':prefix_blocks+int(live_sorted.sum()),
                     'consumer_signature_aligned_blocks':prefix_blocks+aligned})
    steps=[]
    for active_count in range(batch,0,-1):
        group=[r for r in rows if r['active_consumers']==active_count]
        steps.append({'active_consumers':active_count,
                      'original_order_union_mean_blocks':statistics.mean(r['original_order_union_blocks'] for r in group),
                      'per_lane_signature_sorted_mean_blocks':statistics.mean(r['per_lane_signature_sorted_blocks'] for r in group),
                      'consumer_signature_aligned_mean_blocks':statistics.mean(r['consumer_signature_aligned_blocks'] for r in group)})
    result={'scope':'Synthetic closed cohort, no arrivals, equal time per departure, all orders uniformly weighted; only captured lanes, not all 48 layers.',
            'prefix_blocks_reserved_separately':prefix_blocks,
            'initial_original_order_union_blocks':prefix_blocks+union_blocks,
            'initial_consumer_signature_aligned_blocks':prefix_blocks+int(capacity_by_signature.sum()),
            'consumer_signature_capacity_blocks':capacity_by_signature.tolist(),
            'mean_block_steps_original_order_union':sum(r['original_order_union_mean_blocks'] for r in steps),
            'mean_block_steps_per_lane_signature_sorted':sum(r['per_lane_signature_sorted_mean_blocks'] for r in steps),
            'mean_block_steps_consumer_signature_aligned':sum(r['consumer_signature_aligned_mean_blocks'] for r in steps),
            'departure_steps':steps,'all_active_subsets':rows}
    # Different consumer IDs never change who may read a token.
    assert len(rows)==signatures-1
    assert rows[-1]['original_order_union_blocks']==result['initial_original_order_union_blocks']
    assert rows[-1]['consumer_signature_aligned_blocks']==result['initial_consumer_signature_aligned_blocks']
    return result


def main():
    torch.set_num_threads(4)
    captures=['main30b_tp2_docfinqa256_export','main30b_tp2_publicdev1_256_export','main30b_tp2_publicdev6_256_export']
    rows=[]
    for capture in captures:
        folder=Path('results/captures')/capture
        meta=json.loads((folder/'rank0/metadata.json').read_text())
        meta1=json.loads((folder/'rank1/metadata.json').read_text())
        for key in ['B','layers','document_token_range','actual_shared_document_token_range']:
            assert meta[key]==meta1[key],(capture,key)
        ds,de=meta['actual_shared_document_token_range'];n=de-ds
        assert de%16==0
        for budget in [4096,8192,16384,32768]:
            globally_live=torch.zeros(de,dtype=torch.bool);globally_live[:ds]=True
            counts=[];independent_counts=[];head_group_page_counts=[];files={};lane_memberships=[]
            for rank in range(2):
                for layer in meta['layers']:
                    file=folder/f'rank{rank}'/f'masks_l{layer}_b{budget}.pt'
                    masks=torch.load(file,map_location='cpu',weights_only=True)['supports'].bool()
                    assert masks.shape==(meta['B'],2,n)
                    union=masks.any(0);u=union.sum(-1)
                    codes=(masks.to(torch.int16)*(2**torch.arange(meta['B'],dtype=torch.int16))[:,None,None]).sum(0).to(torch.int16)
                    lane_memberships.extend(codes[h,union[h]] for h in range(2))
                    counts.extend(u.tolist());independent_counts.extend(masks.sum((0,2)).tolist())
                    head_group_page_counts.append((ds+int(u.max())+15)//16)
                    globally_live[ds:de]|=union.any(0)
                    files[str(file)]=sha(file)
            live_pages=globally_live.view(-1,16).any(-1)
            freed=int((~live_pages).sum())
            rows.append({'capture':capture,'B':meta['B'],'budget':budget,'observed_layers':meta['layers'],
                'observed_ranks':[0,1],'shared_document_tokens':n,'original_document_tail_in_private':meta['document_token_range'][1]-de,
                'source_shared_blocks':de//16,'source_shared_bytes_per_rank_all48layers':de*48*2*128*2*2,
                'union_token_fraction_mean_observed_layer_heads':statistics.mean(counts)/n,
                'union_over_sum_independent_selected_mean':sum(counts)/sum(independent_counts),
                'two_heads_per_page_compact_page_fraction_mean_observed_layer_rank':statistics.mean(head_group_page_counts)/(de//16),
                'without_compaction_free_blocks_upper_bound_observed4of48layers':freed,
                'without_compaction_free_fraction_upper_bound':freed/(de//16),
                'without_compaction_free_bytes_per_rank_upper_bound':freed*16*48*2*128*2*2,
                'fullpool_compact_blocks_lower_bound_observed4of48layers':math.ceil((ds+max(counts))/16),
                'synthetic_departure_allocator_model':lifecycle_model(lane_memberships,meta['B'],ds),
                'source_metadata_sha256':{str(folder/f'rank{rank}/metadata.json'):sha(folder/f'rank{rank}/metadata.json') for rank in range(2)},
                'source_mask_sha256':files})
    result={'status':'completed','gpu_hours':0,'scope':'CPU mask geometry only, not allocated-memory measurement or quality evidence.',
        'limits':['Observed layers are only 0/15/31/47; uncaptured layers can only reduce the whole-pool free-block bound.',
            'The per-head union compact fraction is a storage lower bound for an allocator allowing layer/head-independent placement, not a native vLLM capacity claim.',
            'Reclamation assumes every old consumer releases source references after exact remapping; dense/prefill/future-query consumers can pin the source.',
            'The preallocated tensor is not returned to CUDA by BlockPool.free_blocks. Admission-capacity and peak-board-memory metrics differ.'],
        'rows':rows}
    out=Path('results/pivot_candidates/capacity_bounds.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:print(json.dumps({k:r[k] for k in ['capture','budget','union_token_fraction_mean_observed_layer_heads','union_over_sum_independent_selected_mean','without_compaction_free_fraction_upper_bound']}))


if __name__=='__main__':main()
