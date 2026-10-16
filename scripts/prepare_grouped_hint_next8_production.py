"""Freeze repaired-build next-eight production evaluation before new inference."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    assert not path.exists(),path
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    manifest_path=ROOT/'data/manifests/docfinqa_evaluation_natural_next8_cohort6plus_v1.json'
    protocol_path=ROOT/'configs/native_natural_evaluation_next8_cohort6plus_v1.json'
    feasibility_path=ROOT/'results/evaluation_natural_next8_feasibility_v1/audit.json'
    manifest=json.loads(manifest_path.read_text());protocol=json.loads(protocol_path.read_text())
    feasibility=json.loads(feasibility_path.read_text())
    assert feasibility['selected_manifest_sha256']==sha(manifest_path)
    assert len(manifest['groups'])==8 and sum(g['B'] for g in manifest['groups'])==98
    assert protocol['manifest']==str(manifest_path.relative_to(ROOT))
    ledger_path=ROOT/'results/job_ledger.jsonl';ledger_text=ledger_path.read_text()
    ledger=[json.loads(line) for line in ledger_text.splitlines()]
    assert all(g['document_id'] not in ledger_text for g in manifest['groups']), 'New input has already been scheduled'
    late_path=ROOT/'results/raw_pair_eval2_grouped_hint_model_diag_cpu_audit_v1.json'
    late=json.loads(late_path.read_text())
    late_summary=ROOT/'results/raw_pair_eval2_grouped_hint_model_diag_v1/summary.json'
    assert json.loads(late_summary.read_text())['grouped_hint_regression_passed']
    assert late['status']=='known_regression_audited' and late['summary_sha256']==sha(late_summary)
    offline_paths=[ROOT/f'results/grouped_hint_model_qual_v1/train26_bf16/rank{rank}/qualification_raw_pair.json' for rank in (0,1)]
    offline=[]
    for rank,path in enumerate(offline_paths):
        rec=json.loads(path.read_text());assert rec['rank']==rank and rec['qualification_passed']
        assert rec['qualified_batches']==list(range(6,22)) and rec['qualification_steps']==125
        assert rec['build_receipt']['native_group']==8 and rec['build_receipt']['fold']==2
        assert rec['build_receipt']['patch_sha256']==sha(ROOT/'src/xqa_raw_grouped_hint_build.py')
        assert 'scripts/vllm_native_raw_pair_grouped_hint_worker.py' in rec['source_sha256']
        for name,expected in rec['source_sha256'].items():assert sha(ROOT/name)==expected
        offline.append(rec)
    assert offline[0]['source_sha256']==offline[1]['source_sha256']
    assert offline[0]['execution_fingerprint']==offline[1]['execution_fingerprint']
    costs=[]
    for doc in ('train26','train33'):
        path=ROOT/f'results/grouped_hint_model_qual_v1/{doc}_bf16/summary.json'
        audit=path.with_name('independent_analysis.json');a=json.loads(audit.read_text());s=json.loads(path.read_text())
        assert a['status']=='qualified' and a['summary_sha256']==sha(path)
        ends=[r for r in ledger if r.get('event')=='end' and '--output' in r.get('argv',[]) and
            (ROOT/r['argv'][r['argv'].index('--output')+1]).resolve()==path.parent.resolve()]
        assert len(ends)==1 and ends[0]['exit_code']==0 and not ends[0]['timed_out']
        costs.append({'document':doc,'summary_path':str(path.relative_to(ROOT)),'summary_sha256':sha(path),
            'independent_audit_sha256':sha(audit),'ledger_receipt':ends[0],
            'qualification_phase_seconds':s['runs'][0]['full_phase_control_wall_seconds'],
            'candidate_strict_setup_seconds':s['candidate_total_setup_wall_seconds']})
    driver=ROOT/'scripts/benchmark_native_raw_pair_grouped_hint_production.py'
    nodes=[n for n in ast.walk(ast.parse(driver.read_text())) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets)]
    assert len(nodes)==1
    names=ast.literal_eval(nodes[0].value)+['scripts/audit_grouped_hint_model_e2e.py',
        'scripts/audit_grouped_hint_production_setup.py','scripts/summarize_grouped_hint_next8_production.py',str(Path(__file__).relative_to(ROOT))]
    template=json.loads((ROOT/'configs/native_raw_pair_heldout_production_eval2_jobs.json').read_text())[0]
    documents=[]
    for group,record in zip(manifest['groups'],feasibility['records'],strict=True):
        assert group['document_id']==record['document_id']
        index=group['original_evaluation_group_index'];fits=record['feasibility'].startswith('context_fits')
        row={'document_id':group['document_id'],'evaluation_index':index,'B':group['B'],
            'context_feasible':fits,'max_prompt_tokens':max(record['prompt_lengths'])}
        if fits:
            job=copy.deepcopy(template);output=f'results/grouped_hint_next8_production_v1/eval{index}_bf16'
            job.update(name=f'grouped_hint_next8_eval{index}_v1',gpus=[6,7],timeout_seconds=1800)
            job['argv']=['.venv-engine/bin/python',str(driver.relative_to(ROOT)),'--protocol',str(protocol_path.relative_to(ROOT)),
                '--document-id',group['document_id'],'--offline-qualification-records',*[str(p) for p in offline_paths],
                '--phase-kind','natural','--output',output]
            path=ROOT/f'configs/grouped_hint_next8_eval{index}_jobs.json';save(path,[job])
            row.update(job_name=job['name'],launch_manifest=str(path.relative_to(ROOT)),summary_path=output+'/summary.json',frozen_argv=job['argv'])
        else:row['reason']='Frozen max prompt plus8192 output exceeds163840; retain complete document/questions, no replacement or cap reduction.'
        documents.append(row)
    assert sum(d['context_feasible'] for d in documents)==6
    plan={'status':'grouped_hint_next8_primary_frozen_before_new_inference','mechanism_version':'grouped_hint_v2',
        'evaluation_role':'new_validation_after_known_first8_regression','selection_commit':'26569b2',
        'selection_manifest':str(manifest_path.relative_to(ROOT)),'selection_manifest_sha256':sha(manifest_path),
        'protocol':str(protocol_path.relative_to(ROOT)),'protocol_sha256':sha(protocol_path),
        'frozen_source_sha256':{name:sha(ROOT/name) for name in names},'driver_commit':'5954169','worker_commit':'5954169',
        'offline_inputs':[{'path':str(p),'sha256':sha(p)} for p in offline_paths],
        'offline_qualification_cost':{'primary_receipts':'new train26 both ranks125 states coveringB6..21',
            'training_qualification_runs':costs,'scope':'Actual offline training qualification costs reported separately, not rerun in validation or replaced by old-build receipts.'},
        'known_regression_gate':{'audit_path':str(late_path.relative_to(ROOT)),'audit_sha256':sha(late_path),
            'summary_sha256':sha(late_summary),'scope':'175 native-trajectory states x48 layers x2 ranks; original first8 remain known regression, not new validation.'},
        'primary_metric':'net_cold_with_unamortized_per_document_setup',
        'primary_definition':'mean(native anchor+generate) / (mean(candidate anchor+generate) + offline preflight + candidate configuration + actual production graph capture + setup receipt export), charged once per document without amortization.',
        'aggregate_rules':'Both geometric mean per-document ratio and ratio of summed times, conditional on paired completions; all8 documents/98questions in coverage/failure denominators. No selection of the more favorable statistic.',
        'qualification_scope':'New source/model/device receipts only. Complete output tokens/APC/finishes/graph trajectories must match; not every hidden value in each new document is inspected.',
        'execution_rule':'Six feasible documents, fresh process each, natural native/candidate/candidate/native ABBA. Frozen policy unchanged. GPU6/7 placeholders are reassigned by root. All failures and caps retained.',
        'old_version_policy':'Never substitute old-build receipts/results or the known first8 outcomes into the new-validation aggregate.',
        'ledger_sha256_before_launch':hashlib.sha256(ledger_text.encode()).hexdigest(),'documents':documents}
    save(ROOT/'results/grouped_hint_next8_production_launch_v1.json',plan)
    print(json.dumps({'status':plan['status'],'jobs':[d['launch_manifest'] for d in documents if d['context_feasible']]},indent=2))


if __name__=='__main__':main()
