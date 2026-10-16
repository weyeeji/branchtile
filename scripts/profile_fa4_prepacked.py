"""Optimistic attention-only lower bound after one-time FA4 payload packing."""
import json
import os
from pathlib import Path
import statistics
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from compare_attention import load_capture,graph_of,elapsed
from fa4_baseline import GatheredDecode,GatheredUnion,FullSourceMasked
from branchtile.attention import reference


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    torch.set_num_threads(4)
    rows=[]
    for name in ('main30b','qmsum4b'):
        record=json.loads((ROOT/f'results/fa4/compare_{name}_with_union.json').read_text())
        assert record['status']=='completed'
        p=record['provenance']
        q,k,v,owners,b,g,_=load_capture(p['capture'],p['layer'],p['budget'],p['step'],len(p['selected_kv_heads']))
        ref,rl=reference(q,k,v,owners,b,g)
        flush=torch.zeros(256*1024*1024//4,device='cuda')
        for regime in ('hot','cold'):
            for method,cls in [('fa4_gather_e0',GatheredDecode),('fa4_gather_union',GatheredUnion)]:
                config=record['selected'][regime][method]
                op=cls(q,k,v,owners,b,pack_gqa=config['pack_gqa'],tile_mn=tuple(config['tile_mn']))
                op.run()
                fn=op.run_attention_only if cls is GatheredDecode else lambda op=op:FullSourceMasked.run(op)
                out,lse=fn()
                torch.testing.assert_close(out.float(),ref.float(),atol=.006,rtol=.015)
                torch.testing.assert_close(lse.float(),rl.float(),atol=.0003,rtol=.0001)
                reps=20 if regime=='hot' else 1
                graph=graph_of(fn,reps)
                times=[elapsed(graph,reps,flush if regime=='cold' else None) for _ in range(10)]
                rows.append({'capture':name,'method':method,'cache':regime,'configuration':config,
                    'attention_only_median_us':statistics.median(times),'samples_us':times,
                    'packed_payload_bytes':op.payload_copy_bytes_per_call})
    result={'scope':'Optimistic static-input attention-only diagnostic. K/V packed once before timing; preparation, payload refresh and private-suffix growth omitted. This is not a complete source-layout transition or application baseline. No paired-ratio claim is made from these separate samples.',
            'purpose':'Check that charging full gather every replay is not the cause of FA4 losing on the selected low-batch long-decode shapes.',
            'rows':rows}
    (ROOT/'results/fa4/prepacked_attention_lower_bound.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
