"""Plot qualified same-state graph costs beside all four natural-task seeds."""
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
    sources = [ROOT / 'results/dense_cascade_four_seed_audit_v1/audit.json']
    sources += [ROOT / f'results/dense_cascade_snapshot_v1/train1_{dtype}/analysis.json'
                for dtype in ('bf16', 'fp8')]
    task = json.loads(sources[0].read_text())
    graphs = [json.loads(p.read_text()) for p in sources[1:]]
    out = ROOT / 'results/real_scene_opportunity_figure_v1'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={'width_ratios': [1, 1.5]})
    y = np.arange(2)
    native = [g['statistics']['model_warm']['max_rank_event']['median_cuda_us']['native']/1000 for g in graphs]
    cascade = [g['statistics']['model_warm']['max_rank_event']['median_cuda_us']['cascade']/1000 for g in graphs]
    for offset, values, color, label in ((-.17, native, '#788494', 'Native'),
                                         (.17, cascade, '#087e8b', 'Cascade')):
        bars = a.barh(y+offset, values, height=.31, color=color, label=label)
        a.bar_label(bars, labels=[f'{v:.2f}' for v in values], padding=4, fontsize=9)
    a.set_yticks(y, ['BF16 KV', 'FP8 KV'])
    a.invert_yaxis()
    a.set_xlim(0, 16.8)
    a.set_xlabel('Whole-model graph time (ms; lower is better)')
    a.set_title('One real B8 input and prior KV history', loc='left', fontsize=11)
    a.legend(loc='lower right', frameon=False)
    labels = []
    plot_data = []
    for i, condition in enumerate(task['conditions']):
        values = [pair['arms']['cascade']['cold_seconds']/pair['arms']['native']['cold_seconds']
                  for pair in condition['pairs']]
        native_total = sum(pair['arms']['native']['cold_seconds'] for pair in condition['pairs'])
        cascade_total = sum(pair['arms']['cascade']['cold_seconds'] for pair in condition['pairs'])
        pooled = cascade_total/native_total
        b.scatter(values, i+np.linspace(-.12, .12, len(values)), color='#aeb6bf', s=30,
                  label='Individual seed' if i == 0 else None, zorder=3)
        b.scatter([pooled], [i], marker='D', color='#a13636', s=48,
                  label='Ratio of four-seed total times' if i == 0 else None, zorder=4)
        b.annotate(f'{pooled:.3f}', (pooled, i), xytext=(7, -15), textcoords='offset points', fontsize=9)
        labels.append(condition['condition'].replace('train1_', 'Document 1 / ').replace('train7_', 'Document 2 / ').upper())
        plot_data.append({'condition': condition['condition'], 'per_seed_cascade_over_native': values,
                          'summed_cascade_over_native': pooled})
    b.axvline(1, color='#303840', linestyle='--', linewidth=1)
    b.set_yticks(range(len(labels)), labels)
    b.invert_yaxis()
    b.set_xlabel('Complete task time / native time (lower is better)')
    b.set_title('Natural EOS, cold anchor + complete generation', loc='left', fontsize=11)
    b.legend(loc='upper center', bbox_to_anchor=(.5, -.18), frameon=False, fontsize=8)
    b.grid(axis='x', alpha=.15)
    fig.text(.02, .025, 'Left excludes per-step host preparation and prefill. Right retains all output lengths and quality outcomes; 2 documents, 4 seeds, no capped answers.\nThese measure different boundaries. Existing cascade is a control, not a new contribution; task quality non-inferiority is not established.', fontsize=8)
    fig.subplots_adjust(left=.075, right=.985, bottom=.29, top=.9, wspace=.6)
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(out / f'opportunity_and_task_outcomes.{suffix}', dpi=170)
    plt.close(fig)
    receipt = {'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               'task_plot_data': plot_data,
               'scope': 'Descriptive evidence, not a confidence statement across independent documents.'}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(out)


if __name__ == '__main__':
    main()
