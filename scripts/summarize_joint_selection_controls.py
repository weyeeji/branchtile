"""Descriptive proxy geometry and preparation costs, not a deployment result."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    groups={};provenance={}
    for folder in ['joint_support_exchange_v1','joint_support_reverse_delete_v1','joint_support_relaxed_prefix_v1','saturated_greedy_cover_v1']:
        p=ROOT/'results'/folder/'results.jsonl';provenance[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        for r in map(json.loads,p.read_text().splitlines()):
            if r['epsilon']!=.01:continue
            if folder=='joint_support_exchange_v1': name='exchange '+r['request_order']
            elif folder=='joint_support_reverse_delete_v1': name='reverse-delete '+r['universe']
            elif folder=='joint_support_relaxed_prefix_v1':name='independent relaxed prefix'
            else:name='classical saturated cover'
            groups.setdefault(name,[]).append(r)
    canonical=None;rows=[]
    for name,rs in groups.items():
        keys={(r['document'],r['budget'],r['kv_head'],r['document_sha256']) for r in rs}
        assert len(rs)==len(keys)==12
        if canonical is None:canonical=keys
        assert keys==canonical
        u=[r['actual_common_source_logical']['union_reduction_fraction'] for r in rs]
        t=[r['selection_wall_seconds'] for r in rs]
        assert all(r['physical_capacity_released_bytes']==0 for r in rs)
        assert all(q['score_loss_fraction']<=.01 for r in rs for q in r['per_request'])
        rows.append(dict(method=name,cells=12,cardinality='at most original k' if 'relaxed' in name else 'exact original k',
            union_reduction_min=min(u),union_reduction_median=statistics.median(u),union_reduction_max=max(u),
            selection_wall_seconds_min=min(t),selection_wall_seconds_median=statistics.median(t),selection_wall_seconds_max=max(t),
            max_request_score_loss=max(q['score_loss_fraction'] for r in rs for q in r['per_request'])))
    out=ROOT/'results/joint_selection_controls_summary.json'
    result=dict(scope='Corrected suffix-only, three documents/rank0/layer15, 4096/16384 budgets, two heads; epsilon=.01',
        sources_sha256=provenance,rows=rows,limitations=['No actual attention quality, generation quality, GPU speedup, physical cache release or algorithm novelty claim.',
        'Range bars span 12 heterogeneous paired cells, not confidence intervals or 12 independent documents.',
        'Wall times are one run of different CPU implementations per cell; input load and subsequent geometry excluded; no repeated latency distribution or implementation-optimality claim.',
        'Relaxed prefix uses a budget upper bound; other methods retain exact original cardinality.',
        'Classic cover is a greedy heuristic, not a natural-data exact oracle; its optional-set objective is classical submodular cover.',
        'Single-layer 4K/16K geometry does not generalize to all48 layers/32K production; separate capture has smaller shared-union reduction.'])
    out.write_text(json.dumps(result,indent=2)+'\n')
    with out.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    labels=[r['method'].replace(' ','\n',1).replace('original_union','original union').replace('all_positions','all positions') for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,prefix,scale in [(axes[0],'union_reduction',100),(axes[1],'selection_wall_seconds',1)]:
        mid=[r[prefix+'_median']*scale for r in rows]
        errs=[[mid[i]-r[prefix+'_min']*scale for i,r in enumerate(rows)],
              [r[prefix+'_max']*scale-mid[i] for i,r in enumerate(rows)]]
        ax.bar(range(len(rows)),mid,color=['#346aa0','#729ac0','#b37b47','#d0a881','#858585','#497f58'])
        ax.errorbar(range(len(rows)),mid,yerr=errs,fmt='none',ecolor='black',capsize=3)
        ax.set_xticks(range(len(rows)),labels,rotation=20,ha='right');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Common-source logical union reduction (%)')
    axes[1].set_ylabel('Single-run selector wall seconds / cell');axes[1].set_yscale('log')
    fig.suptitle('1% pooled-score loss allowance: geometry and CPU preparation are separate\n3 documents × 2 budgets × 2 heads, layer15; medians and min–max, not confidence intervals\nRelaxed prefix uses ≤k, others exact k; no quality, GPU-speed or capacity claim',fontsize=10)
    fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)
    print(json.dumps(dict(output=str(out),rows=len(rows))))
if __name__=='__main__':main()
