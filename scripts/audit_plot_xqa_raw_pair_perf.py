"""Audit all frozen full-call cases and plot paired-block speed ratios."""
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
rng = np.random.default_rng(20261017)
rows = []
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
for ax, doc in zip(axes, (1, 7)):
    labels = []
    plot = {regime: [] for regime in ('hot', 'cold')}
    for rank in (0, 1):
        for layer in (0, 15, 31, 47):
            name = f'train{doc}_r{rank}_l{layer}'
            path = ROOT / f'results/xqa_raw_pair_perf_v1/{name}.json'
            d = json.loads(path.read_text())
            assert d['status'] == 'completed' and all(d['checks'].values())
            for file, sha in d['source_sha256'].items():
                assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == sha
            for record in (d['capture'], d['qualification']):
                assert hashlib.sha256((ROOT / record['path']).read_bytes()).hexdigest() == record['sha256']
            labels.append(f'R{rank}/L{layer}')
            for regime, t in d['timing'].items():
                blocks = t['blocks_ms']
                assert len(blocks) == 30 and len(t['orders']) == 30
                arms = set(t['median_ms'])
                assert all(set(b) == arms and all(v > 0 for v in b.values()) for b in blocks)
                assert all(set(order) == arms and len(order) == len(arms) for order in t['orders'])
                ratios = np.array([b['native_pdl'] / b['paired_complete'] for b in blocks])
                aa = np.array([b['native_pdl'] / b['native_pdl_AA'] for b in blocks])
                idx = rng.integers(0, 30, size=(20000, 30))
                ci = np.quantile(np.median(ratios[idx], axis=1), [.025, .975])
                aa_ci = np.quantile(np.median(aa[idx], axis=1), [.025, .975])
                median = float(np.median(ratios))
                assert abs(median - t['native_over_pair']['paired_median_ratio']) < 1e-12
                row = {'name': name, 'regime': regime, 'ratio': median, 'ci': ci.tolist(),
                       'AA_median': float(np.median(aa)), 'AA_ci': aa_ci.tolist(),
                       'native_us': 1000*t['median_ms']['native_pdl'],
                       'paired_us': 1000*t['median_ms']['paired_complete'],
                       'unshared_us': 1000*t['median_ms']['unshared_checkpoint'],
                       'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                rows.append(row)
                plot[regime].append(row)
    for regime, color, offset in [('hot', '#2465a7', -.06), ('cold', '#bd5d29', .06)]:
        values = plot[regime]
        y = np.array([v['ratio'] for v in values])
        lo = np.array([v['ci'][0] for v in values])
        hi = np.array([v['ci'][1] for v in values])
        ax.errorbar(np.arange(8)+offset, y, yerr=[y-lo, hi-y], marker='o',
                    capsize=3, lw=1.2, label=regime, color=color)
    ax.axhline(1, ls='--', color='#777', lw=1)
    ax.set(title=f'Training document {doc}', xticks=np.arange(8), xticklabels=labels,
           ylim=(.97, 1.52))
    ax.tick_params(axis='x', labelsize=8, rotation=35)
    ax.grid(axis='y', alpha=.2)
axes[0].set_ylabel('Native XQA (PDL on) / complete paired call')
axes[0].legend(frameon=False)
fig.text(.08, .025, 'Actual BF16 B8 initial states with private tails; complete Q pack, metadata copies, raw state and tail/merge included.\n'
         '30 randomized paired blocks; 20,000-resample 95% intervals. Cold: 256 MiB flush before each sample.\n'
         'Host preparation and graph capture reported separately. Operator measurements only; no end-to-end claim.', fontsize=8)
fig.subplots_adjust(left=.08, right=.985, top=.90, bottom=.28, wspace=.13)
out = ROOT / 'results/xqa_raw_pair_perf_audit_v1'
out.mkdir(exist_ok=True)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(out / f'complete_call.{ext}', dpi=180)
plt.close(fig)
result = {'status': 'completed_operator_only', 'real_cases': 16, 'paired_blocks_per_regime': 30,
          'all_sources_capture_qualification_hashes_match': True,
          'resampling': 'Median paired-block ratios; all blocks retained; 20,000 resamples',
          'rows': rows}
(out / 'audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({regime: [min(r['ratio'] for r in rows if r['regime']==regime),
                         max(r['ratio'] for r in rows if r['regime']==regime)] for regime in ('hot', 'cold')}))
