"""Two-document exploratory native-XQA matrix, preserving separate costs."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    protocol=ROOT/'results/main30b_sparse/fi_natural_expansion_protocol_v1.json'
    config=ROOT/'configs/main30b_fi_natural_expansion_v1.json'
    rows=[];sources={};dense={}
    for job in json.loads(config.read_text()):
        directory=ROOT/job['argv'][job['argv'].index('--output')+1]
        rp=directory/'natural_execution_summary.json';r=json.loads(rp.read_text())
        rawpath=ROOT/r['source'];assert hashlib.sha256(rawpath.read_bytes()).hexdigest()==r['source_sha256']
        raw=json.loads(rawpath.read_text());assert raw['status']==r['status']=='completed'
        assert raw['engine_configuration']['attention_config']['backend']=='FLASHINFER'
        assert len(raw['runs'])==3 and raw['B']==3
        sources[str(rp.relative_to(ROOT))]=hashlib.sha256(rp.read_bytes()).hexdigest()
        dd=next(run for run in raw['runs'] if run['method']=='dense')
        dense.setdefault(raw['document_id'],[]).append([a['tokens'] for a in dd['answers']])
        for run in r['rows']:
            assert run['score']['denominator']==3 and all(w['shadow_check_count']==w['logit_copy_count']==0 for w in run['workers'])
            rows.append(dict(document=raw['document_id'],budget=raw['budget'],budget_request=raw['budget_request'],
                method=run['method'],questions=3,exact_correct=round(run['score']['exact_accuracy']*3),
                rounded_correct=round(run['score']['rounded_accuracy']*3),all_eos=run['all_natural_stop'],
                output_tokens=run['total_output_tokens'],generation_seconds=run['complete_generate_wall_seconds'],
                cold_anchor_seconds=run['dense_anchor_wall_seconds'],cold_anchor_plus_generation_seconds=run['anchor_plus_generate_wall_seconds'],
                initialization_seconds=raw['initialization_wall_seconds'],selector_warmup_seconds=raw['selector_dense_warmup_wall_seconds'],
                sampler_warmup_seconds=raw['stochastic_sampler_warmup_wall_seconds'],
                max_rank_all_candidate_capture_seconds=max(x['capture_seconds'] for x in raw['graph_capture']),
                max_rank_all_candidate_added_torch_bytes=max(x['additional_torch_allocated_bytes'] for x in raw['graph_capture'])))
    assert len(rows)==12
    ledger=[json.loads(s) for s in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()]
    charges=[r for r in ledger if r['event']=='end' and r['name'] in {j['name'] for j in json.loads(config.read_text())}]
    assert len(charges)==4 and all(r['exit_code']==0 for r in charges)
    out=ROOT/'results/main30b_sparse/fi_natural_expansion_summary.json'
    result=dict(scope='Two development financial reports, six questions, one seed; all cells exploratory',
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),configuration_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        sources_sha256=sources,rows=rows,allocated_gpu_hours=sum(r['gpu_hours'] for r in charges),
        dense_tokens_identical_across_budget_runs={k:all(v==values[0] for v in values) for k,values in dense.items()},
        dense_repeat_counts={k:len(values) for k,values in dense.items()},
        limitations=['No formal noninferiority, new algorithm, causal speedup, or rationale factuality claim.',
          'Changing backend changes trajectories even at full budget; changes cannot all be attributed to deleting KV.',
          'Natural output lengths differ, all EOS failures/caps would remain in denominator.',
          'All candidate graphs remain resident; graph capture is max-rank duration, not exact driver wall time or per-method cost.',
          'Selector/initialization/sampler/capture must be charged for service economics; cold anchor-plus-generation alone excludes these.',
          'One original document32K point was chosen based on earlier exploratory evidence; the new development document and budget cells were frozen before this run.',
          'Dense repetitions across budget cells are the same seed, not additional independent samples.'])
    out.write_text(json.dumps(result,indent=2)+'\n')
    with out.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    labels=['Original\n32K','Second\n16K','Second\n32K','Second\nfull']
    for method,color in [('dense','#388653'),('e0','#2563a6'),('union','#c56320')]:
        rr=[r for r in rows if r['method']==method]
        for ax,field in zip(axes,['cold_anchor_plus_generation_seconds','output_tokens','rounded_correct']):
            ax.plot(range(4),[r[field] for r in rr],marker='o',label=method,color=color)
            ax.set_xticks(range(4),labels);ax.grid(alpha=.2)
    axes[0].set_ylabel('Cold anchor + generation seconds');axes[0].legend(frameon=False)
    axes[1].set_ylabel('Total generated tokens, 3 requests')
    axes[2].set_ylabel('Rounded FINAL correct / 3');axes[2].set_yticks([0,1,2,3]);axes[2].set_ylim(-.1,3.1)
    fig.suptitle('Official native XQA development control: all natural EOS, different outputs\nPreparation/capture costs separate; two documents and one seed do not establish quality equivalence',fontsize=11)
    fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)
    print(json.dumps(dict(output=str(out),rows=len(rows),gpu_hours=result['allocated_gpu_hours'])))
if __name__=='__main__':main()
