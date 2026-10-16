"""Static descriptive repeat figure, without treating phases/tokens as replicates."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-directory', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    source = Path(args.input)
    result = json.loads(source.read_text())
    assert result['status'] == 'complete_repetition_audit'
    assert result['all_three_runs_reproduced_documents'] == 6
    assert result['selected_document_denominator'] == 8 and result['selected_question_denominator'] == 98
    documents = [d for d in result['documents'] if d['status'] == 'all_three_runs_reproduced']
    assert len(documents) == 6
    out = Path(args.output_directory)
    assert not out.exists(), out
    out.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = ['ready_generation', 'net_cold_with_unamortized_per_document_setup']
    rows = []
    for document in documents:
        for run in range(3):
            rows.append({'evaluation_index': document['evaluation_index'], 'repeat_index': run,
                'ready_ratio': document['metrics'][metrics[0]]['run_ratios'][run],
                'net_ratio': document['metrics'][metrics[1]]['run_ratios'][run],
                'native_AA_max_over_min': document['AA_max_over_min_by_run'][run],
                'root_reported_possible_CUDA_import_overlap': document['evaluation_index'] == 12 and run == 2})
    with (out / 'data.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    labels = ['Original primary', 'Repeat 1', 'Repeat 2']
    colors = ['#222222', '#0072B2', '#D55E00']
    markers = ['D', 'o', 's']
    for ax, metric, title in zip(axes, metrics, ['Generation after shared anchor', 'Cold task + full per-document setup']):
        for run in range(3):
            ax.scatter([i + (run - 1) * .2 for i in range(6)],
                [d['metrics'][metric]['run_ratios'][run] for d in documents],
                color=colors[run], marker=markers[run], s=28, label=labels[run], zorder=3)
        ax.axhline(1, color='#888888', linewidth=.8)
        ax.set_xticks(range(6), [str(d['evaluation_index']) for d in documents])
        ax.set_xlabel('Frozen evaluation document index')
        ax.set_ylabel('Native / candidate time')
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=.18)
        ax.set_ylim(bottom=.995)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Same-input repeats on six feasible held-out documents', fontsize=11)
    fig.text(.02, .035, '8 selected documents; 2 context failures retained. No phase/token confidence intervals.\n'
        'Possible unrecorded CUDA-import overlap in document 12 repeat 2 is retained; see companion incident audit.', fontsize=7)
    fig.tight_layout(rect=(0, .12, 1, .96))
    for extension in ['pdf', 'svg']:
        fig.savefig(out / f'document_repeats.{extension}', metadata={'Creator': Path(__file__).name})
    plt.close(fig)
    receipt = {'source_audit': str(source), 'source_audit_sha256': sha(source),
        'plot_script_sha256': sha(Path(__file__)), 'matplotlib_version': matplotlib.__version__,
        'files': {p.name: sha(p) for p in sorted(out.iterdir())},
        'scope': 'Descriptive process repeats on six identical documents; original first-run primary unchanged; no uncertainty interval or additional unseen samples inferred.'}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output_directory': str(out), 'document_count': len(documents), 'process_count': len(rows)}))


if __name__ == '__main__':
    main()
