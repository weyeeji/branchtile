"""Descriptive source-audited development quality, not formal noninferiority."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
paths = {
    'A4 QC-SnapKV': ('results/sparse_reference/train8_quality.json', 'results/factual_audit/train8_unblinded_v1.json'),
    'A2 FastKVzip mask': ('results/fastkvzip_reference/train8_quality.json', 'results/factual_audit/fastkvzip_train8_unblinded_v1.json'),
}
fractions = ['0.1', '0.25', '0.5', '1.0']
rows = []
for method, (quality_path, factual_path) in paths.items():
    quality = json.loads((ROOT / quality_path).read_text())['cohorts']
    factual = json.loads((ROOT / factual_path).read_text())['cohorts']
    for fraction in fractions:
        q, f = quality[fraction], factual[fraction]
        assert q['questions'] == f['answers'] == 32
        rows.append(dict(method=method, nominal_retention=float(fraction),
            rouge_l_f1_percent=q['macro_f1_percent']['L'], factual_pass=f['factual_pass'],
            adequate_coverage=f['coverage_counts'].get('2', 0),
            factual_and_adequate=f['fully_covered_and_factual'], uncertain_answers=f['uncertain_answers']))
output = ROOT / 'results/factual_audit'
with (output / 'development_quality.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
for method, color, marker in [('A4 QC-SnapKV', '#177ba4', 'o'), ('A2 FastKVzip mask', '#b35224', 's')]:
    subset = [r for r in rows if r['method'] == method]
    for ax, metric in zip(axes, ['rouge_l_f1_percent', 'adequate_coverage', 'factual_and_adequate']):
        ax.plot(range(4), [r[metric] for r in subset], color=color, marker=marker, label=method, linewidth=1.8)
        ax.set_xticks(range(4), ['10%', '25%', '50%', '100%'])
        ax.set_xlabel('Nominal retained fraction')
        ax.grid(axis='y', alpha=.2)
for ax in axes[1:]:
    ax.set_ylim(0, 32); ax.set_yticks([0, 8, 16, 24, 32]); ax.set_ylabel('Answers out of 32')
axes[0].set_ylabel('Original Perl ROUGE-L F1 (%)')
axes[0].set_title('Reference overlap')
axes[1].set_title('Adequate source-supported coverage')
axes[2].set_title('Factual pass AND adequate coverage')
axes[0].legend(frameon=False, loc='lower right', fontsize=9)
fig.suptitle('QMSum train development: 8 meetings, 7 source clusters, one seed', y=.98)
fig.text(.5, .025, 'Single LLM-assisted budget-blind judge; no human or formal noninferiority claim. Full answers/judgments are shared.\nA4 uses per-head document budgets; A2 uses global shared-prefix budgets. Mask-reference latency is not physical-compression speed.', ha='center', fontsize=8.3)
fig.tight_layout(rect=(0, .13, 1, .93))
fig.savefig(output / 'development_quality.png', dpi=180)
fig.savefig(output / 'development_quality.pdf')
(output / 'development_quality_provenance.json').write_text(json.dumps({
    'inputs_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for pair in paths.values() for p in pair},
    'scope': 'Descriptive development only. Same full32 answers and judgments are reused, not independent samples. No confidence claim or correctness-based energy denominator.'}, indent=2) + '\n')
