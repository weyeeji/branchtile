"""Freeze production-only heldout launches and analysis before any heldout GPU run."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    assert not path.exists(),path
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    previous=json.loads((ROOT/'results/native_raw_pair_heldout_launch_v1.json').read_text())
    selection=ROOT/'data/manifests/docfinqa_evaluation_natural_first8_cohort6plus_v1.json'
    protocol=ROOT/'configs/native_natural_evaluation_first8_cohort6plus_v1.json'
    assert sha(protocol)==previous['protocol_sha256']
    offline_paths=[ROOT/f'results/native_raw_pair_general_policy_qual_v1/train26_bf16/rank{rank}/qualification_raw_pair.json' for rank in (0,1)]
    for rank,path in enumerate(offline_paths):
        record=json.loads(path.read_text())
        assert record['rank']==rank and record['qualification_passed'] and record['qualified_batches']==list(range(6,22))
        for name,expected in record['source_sha256'].items():assert sha(ROOT/name)==expected
    offline_summary_path=ROOT/'results/native_raw_pair_general_policy_qual_v1/train26_bf16/summary.json'
    offline_summary=json.loads(offline_summary_path.read_text());assert offline_summary['status']=='completed'
    ledger=[json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()]
    ends=[r for r in ledger if r.get('event')=='end' and r.get('name')=='native_raw_pair_general_policy_qual_train26_v1']
    assert len(ends)==1 and ends[0]['exit_code']==0
    assert not any(r.get('event')=='start' and 'heldout' in r.get('name','') for r in ledger),'Freeze must precede heldout GPU inference'
    driver=ROOT/'scripts/benchmark_native_raw_pair_production_setup.py'
    trees=ast.parse(driver.read_text())
    source_nodes=[n for n in ast.walk(trees) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets)]
    assert len(source_nodes)==1
    source_names=ast.literal_eval(source_nodes[0].value)
    source_names+=['scripts/audit_native_raw_pair_production_setup.py','scripts/summarize_raw_pair_heldout_production.py',str(Path(__file__).relative_to(ROOT))]
    documents=[]
    for old in previous['documents']:
        row=copy.deepcopy(old)
        if old['context_feasible']:
            original=json.loads((ROOT/old['launch_manifest']).read_text());assert len(original)==1
            job=copy.deepcopy(original[0]);index=old['evaluation_index']
            job.update(name=f'native_raw_pair_heldout_production_eval{index}_v1',gpus=[6,7],timeout_seconds=1800)
            output=f'results/native_raw_pair_heldout_production_v1/eval{index}_bf16'
            job['argv']=['.venv-engine/bin/python',str(driver.relative_to(ROOT)),'--protocol',str(protocol.relative_to(ROOT)),
                '--document-id',old['document_id'],'--offline-qualification-records',*[str(p) for p in offline_paths],
                '--phase-kind','natural','--output',output]
            launch=ROOT/f'configs/native_raw_pair_heldout_production_eval{index}_jobs.json'
            save(launch,[job]);row.update(launch_manifest=str(launch.relative_to(ROOT)),job_name=job['name'],summary_path=output+'/summary.json',frozen_argv=job['argv'])
        documents.append(row)
    report={'status':'primary_production_protocol_frozen_before_heldout_inference',
        'selection_manifest':str(selection.relative_to(ROOT)),'selection_manifest_sha256':sha(selection),
        'protocol':str(protocol.relative_to(ROOT)),'protocol_sha256':sha(protocol),
        'frozen_source_sha256':{name:sha(ROOT/name) for name in source_names},
        'driver_commit':'0521958','worker_commit':'debcb81',
        'offline_inputs':[{'path':str(p),'sha256':sha(p)} for p in offline_paths],
        'offline_qualification_cost':{'source_summary':str(offline_summary_path.relative_to(ROOT)),
            'source_summary_sha256':sha(offline_summary_path),'ledger_receipt':ends[0],
            'qualification_phase_seconds':offline_summary['runs'][0]['full_phase_control_wall_seconds'],
            'candidate_strict_setup_seconds':offline_summary['candidate_total_setup_wall_seconds'],
            'scope':'One-time training numerical qualification; separately reported and not rerun or secretly included as a heldout task.'},
        'primary_metric':'net_cold_with_unamortized_per_document_setup',
        'primary_definition':'mean(native anchor+generate) / (mean(candidate anchor+generate) + offline preflight + candidate configuration + single-production-mode graph capture + setup receipt export). Charge all measured setup once per document, without cross-document amortization.',
        'secondary_metrics':['ready generation','cold anchor+generation','full phase control','full phase control plus per-document setup','fresh-process component estimate explicitly labeled'],
        'aggregate_rules':'Report both per-document-ratio geometric mean and ratio of summed task times, conditional on paired completed documents; retain fixed 8-document/101-question failure and coverage denominators. Do not choose the better statistic after observing results.',
        'qualification_scope':'Offline training exact source/model/device fingerprints cover B6..21. Heldout full output tokens/APC/exit/graph trajectories must match across ABBA. Every heldout hidden value is not inspected.',
        'strict_alternative':'Earlier strict nine-phase launch manifests remain unexecuted alternatives/numerical diagnostics; never replace the primary production-only result by whichever timing is faster.',
        'execution_rule':'Six context-feasible preselected documents each use a fresh process and natural native/candidate/candidate/native phases; preserve two context failures. GPU6/7 entries are scheduling placeholders, not a requirement to overlap jobs. No future lengths or runtime timing drives dispatch.',
        'documents':documents}
    save(ROOT/'results/native_raw_pair_heldout_production_launch_v1.json',report)
    print(json.dumps({'status':report['status'],'jobs':[d['launch_manifest'] for d in documents if d['context_feasible']],'offline_qualification_gpu_hours':ends[0]['gpu_hours']},indent=2))


if __name__=='__main__':
    main()
