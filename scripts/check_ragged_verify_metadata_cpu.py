"""CPU support-set and shrinking-batch checks; never initializes CUDA."""
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    import torch
    from scripts.qualify_ragged_shared_verify import synthetic_fixture
    from src.ragged_shared_verify_plan import ragged_metadata
    fixture=synthetic_fixture('bf16')
    checks=0
    for rows in ([0,1,2,3,4,5,6,7],[2,0,5],[0,4],[6]):
        old_groups=[fixture['groups'][r] for r in rows]
        unique=sorted(set(old_groups))
        groups=[unique.index(g) for g in old_groups]
        prefix=[fixture['prefixes'][g] for g in unique]
        table=fixture['block_tables'][rows]
        for depth in (1,2,4,8):
            counts=[depth]*len(rows)
            lens=[int(fixture['positions'][r,0])+depth for r in rows]
            meta=ragged_metadata(groups,prefix,counts,lens,table)
            for r in range(len(rows)):
                group=groups[r]
                prefix_ids=meta['prefix_ids'][meta['prefix_indptr'][group]:meta['prefix_indptr'][group+1]]
                tail_ids=meta['tail_ids'][meta['tail_indptr'][r]:meta['tail_indptr'][r+1]]
                for j in range(depth):
                    endpoint=lens[r]-depth+j+1
                    full=[(int(table[r,t//16]),t%16) for t in range(endpoint)]
                    separate=[(prefix_ids[t//16],t%16) for t in range(prefix[group])]
                    separate += [(tail_ids[t//16],t%16) for t in range(endpoint-prefix[group])]
                    assert full==separate
                    checks+=1
    assert not torch.cuda.is_initialized()
    print(f'{checks} CPU exact-support checks passed; CUDA not initialized')


if __name__=='__main__':main()
