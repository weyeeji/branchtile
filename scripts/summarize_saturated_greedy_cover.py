#!/usr/bin/env python3
"""Validate fixed cover results and join every earlier control, CPU only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/saturated_greedy_cover_v1'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def cell(row):
    return tuple(row[k] for k in ('document','budget','kv_head'))


def stats(values):
    a=np.array(values,dtype=np.float64)
    return {'count':len(a),'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max()),'sum':float(a.sum())}


def main():
    assert os.getuid()!=0 and sys.prefix!=sys.base_prefix
    run=json.loads((BASE/'run_summary.json').read_text())
    rows=read(BASE/'results.jsonl')
    protocol=json.loads((BASE/'protocol.json').read_text())
    exchange_path=ROOT/'results/joint_support_exchange_v1/results.jsonl'
    reverse_path=ROOT/'results/joint_support_reverse_delete_v1/results.jsonl'
    exchange=read(exchange_path)
    reverse=read(reverse_path)
    independent={cell(r):r for r in exchange if r['method']=='independent'}
    unique={cell(r)+(r['epsilon'],) for r in rows}
    assert len(unique)==len(rows)==run['rows']
    if run['status']=='completed':
        assert len(rows)==protocol['expected_rows']==48
    for row in rows:
        assert row['selected_entries']==row['B']*row['budget']
        assert row['physical_capacity_released_bytes']==0
        initial=independent[cell(row)]
        for q,p in enumerate(row['per_request']):
            assert p['selected_tokens']==row['budget']
            assert p['original_selected_score']==initial['per_request'][q]['original_selected_score']
            assert p['score_loss_fraction']<=row['epsilon']
            if row['epsilon']==0:
                assert p['selected_indices_sha256']==initial['per_request'][q]['selected_indices_sha256']
        if row['epsilon']:
            assert row['cover']['all_original_fsum_constraints_satisfied']
            assert row['cover']['all_exact_cover_constraints_satisfied']
    aggregates=[]
    for epsilon in protocol['epsilon']:
        for budget in [None]+protocol['budgets']:
            rr=[r for r in rows if r['epsilon']==epsilon and (budget is None or r['budget']==budget)]
            if not rr:
                continue
            aggregates.append({'epsilon':epsilon,'budget':budget,'head_cells':len(rr),
                'union_reduction_fraction':stats([r['actual_common_source_logical']['union_reduction_fraction'] for r in rr]),
                'owner_N64_tile_reduction_fraction':stats([r['actual_common_source_logical']['owner_tile_reduction_fraction'] for r in rr]),
                'selection_cpu_seconds':stats([r['selection_cpu_seconds'] for r in rr]),
                'selection_wall_seconds':stats([r['selection_wall_seconds'] for r in rr]),
                'maximum_request_score_loss_fraction':max(p['score_loss_fraction'] for r in rr for p in r['per_request']),
                'greedy_unused_tokens_removed':stats([r['cover'].get('unused_greedy_tokens_removed_by_refill',0) for r in rr]),
                'marginal_evaluations':stats([r['cover'].get('marginal_evaluations',0) for r in rr])})
    joined=[]
    for row in rows:
        if row['epsilon']==0:
            continue
        controls=[(f"exchange_{r['request_order']}",r) for r in exchange if cell(r)==cell(row) and r['method']=='exchange' and r['epsilon']==row['epsilon']]
        controls += [(f"reverse_delete_{r['universe']}",r) for r in reverse if cell(r)==cell(row) and r['epsilon']==row['epsilon']]
        assert len(controls)==4
        for name,other in controls:
            assert [p['original_selected_score'] for p in row['per_request']]==[p['original_selected_score'] for p in other['per_request']]
            joined.append({k:row[k] for k in ('document','budget','kv_head','epsilon')}|{
                'control':name,'cover_union_tokens':row['actual_common_source_logical']['union_tokens'],
                'control_union_tokens':other['actual_common_source_logical']['union_tokens'],
                'cover_minus_control_union_tokens':row['actual_common_source_logical']['union_tokens']-other['actual_common_source_logical']['union_tokens'],
                'cover_owner_tiles':row['actual_common_source_logical']['owner_partition_N64_tiles'],
                'control_owner_tiles':other['actual_common_source_logical']['owner_partition_N64_tiles'],
                'cover_selection_wall_seconds':row['selection_wall_seconds'],
                'control_selection_wall_seconds':other['selection_wall_seconds'],
                'cover_selection_cpu_seconds':row['selection_cpu_seconds'],
                'control_selection_cpu_seconds':other.get('selection_cpu_seconds')})
    comparisons=[]
    for control in sorted({r['control'] for r in joined}):
        for epsilon in [None,.001,.005,.01]:
            rr=[r for r in joined if r['control']==control and (epsilon is None or r['epsilon']==epsilon)]
            if not rr:
                continue
            delta=[r['cover_minus_control_union_tokens'] for r in rr]
            comparison={'control':control,'epsilon':epsilon,'matched_head_cells':len(rr),
                'cover_smaller_union':sum(v<0 for v in delta),'same_union':sum(v==0 for v in delta),'cover_larger_union':sum(v>0 for v in delta),
                'cover_minus_control_union_tokens':stats(delta),
                'cover_selection_cpu_seconds':stats([r['cover_selection_cpu_seconds'] for r in rr]),
                'cover_selection_wall_seconds':stats([r['cover_selection_wall_seconds'] for r in rr]),
                'control_selection_wall_seconds':stats([r['control_selection_wall_seconds'] for r in rr]),
                'warning':'All fixed controls retained. Selector timers enclose each method call and exclude input load and subsequent geometry reporting. Separate single runs are not a controlled latency benchmark; whole-run wall times cover different grids and must not be compared as per-cell latency.'}
            if all(r['control_selection_cpu_seconds'] is not None for r in rr):
                comparison['control_selection_cpu_seconds']=stats([r['control_selection_cpu_seconds'] for r in rr])
            comparisons.append(comparison)
    summary={'status':run['status'],'rows':len(rows),'unique_rows':len(unique),
             'identity_rows':sum(r['epsilon']==0 for r in rows),'positive_epsilon_rows':sum(r['epsilon']>0 for r in rows),
             'source_result_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [BASE/'results.jsonl',exchange_path,reverse_path]},
             'aggregates':aggregates,'pairwise_controls':comparisons,'all_pairwise_cells':joined,
             'run_accounting':run,'validation':{'same_original_scores':True,'budgets_unchanged':True,'epsilon_zero_supports_identical':True,'all_per_request_fsum_bounds_passed':True},
             'scope':'Classical full-universe greedy cover control. No global-optimality/approximation guarantee, no attention/task-quality result, no novelty claim, no physical capacity release, no GPU. Cells and shared-document requests are correlated.'}
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'status':summary['status'],'rows':len(rows),'aggregates':aggregates,'pairwise_controls':comparisons},indent=2))


if __name__=='__main__':
    main()
