"""Plot qualified prefix-only pair costs; never infer natural E2E from subsets."""
import hashlib
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    output = ROOT / 'results/xqa_pair_survivor_figure_v1'
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261017)
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for tag, color, offset in [('train1', '#2465a7', -.06), ('train7', '#bd5d29', .06)]:
        medians, lows, highs = [], [], []
        for batch in (2, 4, 6, 8):
            path = ROOT / f'results/xqa_pair_padding_natural_subset_v3/{tag}_b{batch}.json'
            d = json.loads(path.read_text())
            assert d['status'] == 'bitwise_qualified_and_benchmarked_prefix_only'
            assert d['paired_fixed_split_bitwise_qualified']
            native = np.array([b['us']['native'] for b in d['timing_blocks']])
            pair = np.array([b['us']['pair_fixed_split'] for b in d['timing_blocks']])
            assert len(native) == 20
            idx = rng.integers(0, len(native), size=(20000, len(native)))
            ratio = float(np.median(native) / np.median(pair))
            ci = np.quantile(np.median(native[idx], axis=1) / np.median(pair[idx], axis=1), [.025, .975])
            medians.append(ratio); lows.append(ci[0]); highs.append(ci[1])
            rows.append(dict(document=tag, batch=batch, ratio=ratio, paired_block_bootstrap_95ci=ci.tolist(),
                             source=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        xs = np.array([2, 4, 6, 8]) + offset
        ax.errorbar(xs, medians, yerr=[np.array(medians)-lows, np.array(highs)-medians],
                    marker='o', capsize=3, lw=1.7, color=color, label=tag)
    ax.axhline(1, ls='--', color='#666666', lw=1)
    ax.set(xticks=[2, 4, 6, 8], xlabel='Selected active requests (B)',
           ylabel='Native latency / exact-pair latency', ylim=(.88, 1.72),
           title='BF16 shared-prefix XQA: preserved native split order')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False, loc='upper left')
    ax.text(2.08, .91, 'B2: native fallback is preferable', fontsize=9, color='#555555')
    fig.text(.09, .035, 'Initial B8 Q subsets; membership from natural survival traces, not later-step Q.\n'
             'Prefix only; Q packing + O restoration included. 20 paired blocks; 95% bootstrap intervals.\n'
             'No private tails or complete-model/task speedup measured by this figure.', fontsize=8.3, color='#444444')
    fig.subplots_adjust(left=.12, right=.97, top=.89, bottom=.24)
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(output / f'qualified_prefix_pair_by_batch.{suffix}', dpi=180)
    plt.close(fig)
    receipt = dict(status='completed', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   estimates=rows, resampling='20,000 paired blocks; ratio of arm medians, all 20 original blocks',
                   scope='Qualified common-prefix only, initial Q subsets; not natural E2E or later-step replay')
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'conditions': len(rows)}))


if __name__ == '__main__':
    main()
