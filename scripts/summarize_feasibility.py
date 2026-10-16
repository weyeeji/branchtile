"""Rebuild a compact audit table and scientific figure from paired raw results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    records, unqualified = [], []
    patterns = [('v2', 'results/paired/paired_v2_cache_tuned_*.json'),
                ('v3', 'results/kernel_ablation_v3/paired_cross_*.json'),
                ('v2+union', 'results/union_paired/union_*.json'),
                ('old30b', 'results/natural_paired/*.json'),
                ('official4b', 'results/official4b_paired/*.json'),
                ('latest4b', 'results/latest4b_paired/*.json'),
                ('latest-old30b', 'results/latest_old30b_paired/*.json'),
                ('latest-v2', 'results/latest_synthetic_paired/*.json'),
                ('native-union4b', 'results/native_union_paired/*.json')]
    for version, pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            d = json.loads(path.read_text())
            if 'summary' not in d:
                unqualified.append({'file':str(path.relative_to(ROOT)), 'status':d.get('status'), 'reason':d.get('reason')})
                continue
            for cache, s in d['summary'].items():
                times = s['median_us']
                best_name = min((m for m in times if m != 'compact'), key=times.get)
                records.append({'file': str(path.relative_to(ROOT)), 'version': version,
                                'cache': cache, 'input': 'natural' if 'capture' in d['provenance'] else 'synthetic',
                                'q_shape': str(d['q_shape']), 'source_tokens': d['k_shape'][1],
                                'native_lse_contract': d.get('native_lse_contract', 'native log2; conversion only in validation'),
                                'candidate_us': times['compact'], 'fixed_us': times['fixed'],
                                'union_us': times.get('union'), 'native_us': times['native'],
                                'native_raw_us': times.get('native_raw'),
                                'native_union_us': times.get('native_union'),
                                'native_union_raw_us': times.get('native_union_raw'),
                                'best_control': best_name,
                                'best_control_over_candidate': times[best_name]/times['compact'],
                                'fixed_over_candidate': s['fixed_over_compact']['median'],
                                'fixed_over_candidate_ci95': s['fixed_over_compact']['paired_block_bootstrap95'],
                                'fixed_aa': s['fixed_AA_median_ratio']})
    dest = ROOT/'results/summary'
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'paired_comparisons.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    report = {'scope': 'operator qualification; source manifests and limitations remain in raw files',
              'paired_configurations': len(records)//2, 'rows': len(records),
              'candidate_fixed_ratio_above_1_15': sum(r['fixed_over_candidate'] >= 1.15 for r in records),
              'unqualified_files': unqualified,
              'records': records}
    (dest/'paired_comparisons.json').write_text(json.dumps(report, indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5.2,len(records)*0.13+1.7)), constrained_layout=True)
    for ax, cache in zip(axes, ['hot', 'cold']):
        rows = [r for r in records if r['cache']==cache]
        labels = []
        for r in rows:
            p = Path(r['file']).stem
            if 'qmsum' in p:
                part = p[p.index('qmsum'):].replace('_allheads', '').replace('_b4096','')
                label = ('old30b' if r['version']=='v2+union' else r['version'])+' '+part
            else:
                label = r['version']+' '+('mixed' if 'mixed' in p else 'cross')+' '+str(r['source_tokens']//1024)+'K'
                if 'g4' in p:
                    label += ' G4'
            labels.append(label)
        y = np.arange(len(rows))
        values = np.array([r['fixed_over_candidate'] for r in rows])
        bounds = np.array([r['fixed_over_candidate_ci95'] for r in rows])
        ax.barh(y, values, color=['#456c9c' if r['input']=='synthetic' else '#bd7845' for r in rows], height=0.65)
        ax.errorbar(values,y,xerr=np.maximum(0,np.stack([values-bounds[:,0],bounds[:,1]-values])),fmt='none',ecolor='#222222',capsize=2)
        ax.axvline(1, color='black', linewidth=1)
        ax.axvline(1.15,color='#a32626',linestyle='--',linewidth=1)
        ax.set_yticks(y,labels,fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0,1.25)
        ax.set_xlabel('Fixed-region latency / dynamic latency (higher is better)')
        ax.set_title(cache.capitalize()+' cache; paired bootstrap 95% CI')
    fig.suptitle('Dynamic query-row compaction does not meet the 1.15 progression threshold',fontsize=12)
    fig.savefig(dest/'dynamic_vs_fixed.png',dpi=180)
    fig.savefig(dest/'dynamic_vs_fixed.pdf')
    plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}))


if __name__ == '__main__':
    main()
