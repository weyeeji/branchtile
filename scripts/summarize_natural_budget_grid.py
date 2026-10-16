"""One-document natural-budget development evidence; never a speedup claim."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    folders = ['natural_compiled','natural_budget8192','natural_budget16384','natural_budget32768','natural_fullbudget']
    rows, provenance, dense_tokens = [], {}, []
    for folder in folders:
        directory = ROOT/'results/main30b_sparse'/folder
        path = directory/'natural_execution_summary.json'
        report = json.loads(path.read_text())
        assert report['status'] == 'completed'
        raw_path = ROOT/report['source']; raw_bytes = raw_path.read_bytes()
        assert hashlib.sha256(raw_bytes).hexdigest() == report['source_sha256']
        raw = json.loads(raw_bytes)
        assert raw['document_id'] == 'docfinqa-c0f11a67aae850fc'
        assert raw['sampling'] == dict(temperature=.6,top_p=.95,seed=20261016,max_tokens=8192,ignore_eos=False)
        dense_tokens.append(next(r['answers'] for r in raw['runs'] if r['method']=='dense'))
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        capture = report['one_time_costs']['graph_capture']
        for row in report['rows']:
            assert row['score']['denominator'] == 3 and len(row['answers']) == 3
            assert all(w['shadow_check_count']==w['logit_copy_count']==0 for w in row['workers'])
            rows.append(dict(budget=raw['budget'], budget_label='full' if folder=='natural_fullbudget' else str(raw['budget']//1024)+'K',
                method=row['method'], exact_correct=round(row['score']['exact_accuracy']*3),
                rounded_correct=round(row['score']['rounded_accuracy']*3), questions=3,
                cap_answers=sum(a['finish_reason']=='length' for a in row['answers']),
                output_tokens=row['total_output_tokens'], generate_wall_seconds=row['complete_generate_wall_seconds'],
                anchor_wall_seconds=row['dense_anchor_wall_seconds'], anchor_plus_generate_wall_seconds=row['anchor_plus_generate_wall_seconds'],
                initialization_wall_seconds=raw['initialization_wall_seconds'],
                selector_dense_warmup_wall_seconds=raw['selector_dense_warmup_wall_seconds'],
                stochastic_sampler_warmed_before_all_arms='stochastic_sampler_warmup_wall_seconds' in raw,
                max_rank_all_candidate_graph_capture_seconds=max(r['capture_seconds'] for r in capture),
                max_rank_all_candidate_additional_torch_bytes=max(r['additional_torch_allocated_bytes'] for r in capture)))
    dense_equal = all([a['tokens'] for a in answers]==[a['tokens'] for a in dense_tokens[0]] for answers in dense_tokens)
    native_path = ROOT/'results/main30b_sparse/native_flashinfer_natural/natural_execution_summary.json'
    native = json.loads(native_path.read_text())
    assert native['status'] == 'completed' and native['document_id'] == raw['document_id']
    assert hashlib.sha256((ROOT/native['source']).read_bytes()).hexdigest() == native['source_sha256']
    native_control = [dict(method=r['method'],repeat=r['repeat'],
        generate_wall_seconds=r['complete_generate_wall_seconds'],
        anchor_plus_generate_wall_seconds=r['anchor_plus_generate_wall_seconds'],
        output_tokens=r['total_output_tokens'],rounded_correct=round(r['score']['rounded_accuracy']*3)) for r in native['rows']]
    provenance[str(native_path.relative_to(ROOT))] = hashlib.sha256(native_path.read_bytes()).hexdigest()
    result = dict(scope='One exploratory financial report / three questions / one seed; different natural output trajectories',
        all_five_dense_answer_token_sequences_identical=dense_equal, source_sha256=provenance, rows=rows,
        independent_native_flashinfer_control=native_control,
        limitations=['Not a quality noninferiority or causal application speedup experiment.',
            'Exact/rounded FINAL measures numerical endpoint only, not fidelity of the full rationale or reproduced tables.',
            'All candidate graph sets resident in every arm; native standalone backend may use less memory.',
            'One-time selector warmup and all-candidate capture are separately charged; graph max-rank duration is not an exact driver wall measurement or per-method cost.',
            '4K predates explicit stochastic sampler warmup; its first dense JIT affects timing only and is retained.',
            'Grid dense used vLLM FLASH_ATTN; standalone official FLASHINFER/XQA control is shown separately with different natural outputs and no candidate graphs resident.',
            'Budgets were the original development grid; outcomes do not freeze a formally validated budget.'])
    out = ROOT/'results/main30b_sparse/natural_budget_grid_summary.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    with out.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    labels=['4K','8K','16K','32K','full']
    for method,color in [('dense','#777777'),('e0','#2563a6'),('union','#c56320')]:
        rs=[r for r in rows if r['method']==method]
        for ax,field in zip(axes,['anchor_plus_generate_wall_seconds','output_tokens','rounded_correct']):
            ax.plot(range(5),[r[field] for r in rs],marker='o',color=color,label='dense FA' if method=='dense' else method)
            ax.set_xticks(range(5),labels);ax.set_xlabel('Frozen document budget');ax.grid(alpha=.2)
    for ax,field in zip(axes,['anchor_plus_generate_wall_seconds','output_tokens','rounded_correct']):
        ax.axhline(sum(r[field] for r in native_control)/len(native_control),color='#388653',linestyle='--',label='native XQA (separate)')
    axes[0].set_ylabel('Anchor + generation seconds')
    axes[1].set_ylabel('Total generated tokens (3 requests)')
    capped=next(r for r in rows if r['method']=='e0' and r['budget_label']=='4K')
    axes[1].annotate('1 answer capped',xy=(0,capped['output_tokens']),xytext=(.45,capped['output_tokens']-1500),
                     arrowprops=dict(arrowstyle='->',color='#2563a6'),fontsize=9)
    axes[2].set_ylabel('Rounded FINAL correct / 3');axes[2].set_yticks([0,1,2,3]);axes[2].set_ylim(-.1,3.1)
    axes[0].legend(frameon=False)
    fig.suptitle('Single-report development screen: different outputs; no speedup or noninferiority claim\nPreparation/capture recorded separately; 4K first dense arm includes sampler JIT',fontsize=11)
    fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'))
    plt.close(fig)
    print(json.dumps(dict(output=str(out),dense_tokens_equal=dense_equal,rows=len(rows))))


if __name__ == '__main__':
    main()
