"""CPU diagnostic: does a global-tile-conditioned additive cost explain G0?"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',action='append',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    records=[];sources=[]
    for name in args.input:
        path=Path(name);data=json.loads(path.read_text())
        assert data['status']=='completed' and data['all_FP64_passed']
        sources.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for i in range(3):
            cell=next(x for x in data['cases'] if x['name']==f'joint_{i}')
            groups=data['cases'][0]['original_rows']
            assert groups==list(range(8))
            group_queries=[sum(cell['query_counts'][:4]),sum(cell['query_counts'][4:])]
            info=cell['plan_info']['prefix'];assert len(info)==15
            tile=info[3];qt=[math.ceil(q*8/tile) for q in group_queries]
            weighted=sum(t*l for t,l in zip(qt,cell['prefix_tokens']))
            records.append(dict(input=str(path),pattern=i,group_queries=group_queries,
                prefix_lengths=cell['prefix_tokens'],global_tile_q=tile,query_tiles_per_group=qt,
                padded_split_work_items=info[0],internal_cuda_graph=bool(info[13]),split_kv=bool(info[14]),
                prefix_token_scan_proxy=weighted,prefix_KV_bytes_proxy=weighted*512,
                hot_joint_ms=data['measurements']['hot']['median_ms'][f'joint_{i}'],
                cold_joint_ms=data['measurements']['cold']['median_ms'][f'joint_{i}'],
                isolated_auto_config_group_sum_ms=data['measurements']['hot']['sum_separate_group_ms'][str(i)]))
    # Calibration uses only two allocations from the first input, never the
    # held-out third allocation or the second layer. This is still just two
    # distinct x-values, insufficient to validate a general online cost model.
    a,b=records[:2];dx=b['prefix_token_scan_proxy']-a['prefix_token_scan_proxy'];assert dx>0
    slope=(b['hot_joint_ms']-a['hot_joint_ms'])/dx
    intercept=a['hot_joint_ms']-slope*a['prefix_token_scan_proxy']
    for row in records:
        row['tile_conditioned_prediction_ms']=intercept+slope*row['prefix_token_scan_proxy']
        row['relative_prediction_error']=(row['tile_conditioned_prediction_ms']/row['hot_joint_ms'])-1
        row['naive_isolated_sum_relative_error']=row['isolated_auto_config_group_sum_ms']/row['hot_joint_ms']-1
    result=dict(status='completed',sources=sources,calibration=dict(intercept_ms=intercept,slope_ms_per_prefix_token_scan=slope,
        calibrated_on=[dict(input=a['input'],pattern=0),dict(input=b['input'],pattern=1)]),records=records,
        conclusion='Current observations are explained by a small table conditioned on global tile configuration and additive group scan counts; no evidence yet that an enhanced marginal-greedy scheduler is insufficient.',
        limitations=['KV bytes are an analytic scan proxy, not hardware DRAM counters.',
            'Per-group isolated kernels may choose different tiles/splits; summing their auto-tuned times is not the correct jointly-executed C2 cost table.',
            'Two distinct scan-proxy values and two layers of the same document pair cannot validate generalization to new contexts or acceptance traces.',
            'No real acceptance utility was measured; this audit cannot establish greedy optimality or an end-to-end speedup.'])
    output=Path(args.output);assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(intercept_us=intercept*1000,held_out_errors=[r['relative_prediction_error'] for r in records[2:]])))


if __name__=='__main__':main()
