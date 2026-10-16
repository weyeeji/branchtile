"""CPU audit of natural cohort survival, actual capacity, and native task costs."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path):
    from scripts.screen_application import extract_final
    from src.financial_answer_metric import score_pairs
    report=json.loads(path.read_text())
    assert report['status']=='completed',report.get('error')
    terminal=[]
    for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines():
        row=json.loads(line)
        argv=row.get('argv',[])
        if row.get('event')=='end' and '--output' in argv and (ROOT/argv[argv.index('--output')+1]).resolve()==path.parent.resolve():
            terminal.append(row)
    assert len(terminal)==1 and terminal[0]['exit_code']==0 and not terminal[0]['timed_out']
    for name,expected in report['source_sha256'].items():
        assert digest(ROOT/name)==expected,name
    assert digest(ROOT/report['manifest'])==report['manifest_sha256']
    assert len(report['runs'])==1
    run=report['runs'][0]
    assert run['natural_eos_enabled'] and not run['sampling']['ignore_eos']
    assert run['sampling']==report['protocol']['sampling']
    assert len(run['answers'])==report['B']
    assert report['engine']['max_num_seqs']==report['B']
    assert report['engine']['compilation_config']['cudagraph_capture_sizes']==list(range(1,report['B']+1))
    all_steps=[]
    graphs=[]
    artifacts=[]
    for worker in run['workers']:
        step_path=Path(worker['all_steps_metadata'])
        graph_path=Path(worker['graph_metadata'])
        steps=json.loads(step_path.read_text())
        graph=json.loads(graph_path.read_text())
        assert steps['rank']==graph['rank']==worker['rank']
        assert steps['phase']==graph['phase']=='native_natural'
        assert len(steps['steps'])==worker['all_steps_count']
        assert len(graph['replays'])==worker['replays']
        all_steps.append(steps['steps'])
        graphs.append(graph['replays'])
        artifacts += [{'path':str(p),'sha256':digest(p)} for p in (step_path,graph_path)]
    assert len(all_steps)==len(graphs)==2
    keys=('batch','questions','positions','num_scheduled_tokens','is_prefilling','has_prefill','num_tokens','num_tokens_after_padding')
    assert [[{k:r[k] for k in keys} for r in rows] for rows in all_steps][0]==[
        {k:r[k] for k in keys} for r in all_steps[1]]
    assert [(r['batch'],r['questions'],r['positions']) for r in graphs[0]]==[
        (r['batch'],r['questions'],r['positions']) for r in graphs[1]]
    for row in all_steps[0]:
        assert len(row['questions'])==len(set(row['questions']))==row['batch']
        assert all(0<=q<report['B'] for q in row['questions'])
        assert sum(row['num_scheduled_tokens'])==row['num_tokens']
    pure=[r for r in all_steps[0] if not r['has_prefill'] and r['num_tokens']==r['batch']]
    counts=Counter(r['batch'] for r in pure)
    intervals=defaultdict(float)
    prefilling_span=0.0
    for first,second in zip(all_steps[0],all_steps[0][1:]):
        duration=second['prepare_inputs_started_monotonic_seconds']-first['prepare_inputs_started_monotonic_seconds']
        assert duration>=0
        if not first['has_prefill'] and first['num_tokens']==first['batch']:
            intervals[first['batch']]+=duration
        else:
            prefilling_span+=duration
    answers=run['answers']
    assert sum(len(a['tokens']) for a in answers)==run['total_output_tokens']
    assert all(a['request_metrics'] is not None for a in answers)
    preemptions=[a['request_metrics']['num_preemptions'] for a in answers]
    assert sum(preemptions)==run['total_request_preemptions']
    group=next(g for g in json.loads((ROOT/report['manifest']).read_text())['groups'] if g['document_id']==report['args']['document_id'])
    assert len(group['questions'])==group['natural_question_count']==report['B']
    predictions=[]
    for answer in answers:
        final,_,parse=extract_final(answer['text'],thinking=True)
        predictions.append(final if answer['finish_reason']=='stop' and parse=='closed_thinking' else '')
    # Metric internals contain tuples, which JSON receipts serialize as lists.
    rescored=json.loads(json.dumps(score_pairs(predictions,[q['answer'] for q in group['questions']])))
    assert rescored==run['quality']['score']
    capacity=[]
    for before,after in zip(report['native_before'],report['native_after'],strict=True):
        assert before['rank']==after['rank']
        assert before['allocated_cache_token_capacity']==after['allocated_cache_token_capacity']
        assert before['cache_unique_storage_bytes']==after['cache_unique_storage_bytes']
        capacity.append({'rank':after['rank'],'token_slots':after['allocated_cache_token_capacity'],
            'pool_GiB':after['cache_unique_storage_bytes']/2**30,
            'peak_allocated_GiB':after['peak_allocated_bytes']/2**30,'device_free_GiB_after':after['device_free_bytes']/2**30})
    assert abs(run['anchor_wall_seconds']+run['complete_generate_wall_seconds']-run['cold_shared_document_total_wall_seconds'])<1e-8
    step_total=sum(counts.values())
    interval_total=sum(intervals.values())
    return {'status':'qualified','summary_sha256':digest(path),'audit_source_sha256':digest(Path(__file__)),
        'document_id':report['args']['document_id'],'B':report['B'],'source_hashes_match':True,
        'clean_exit_receipt':terminal[0],
        'rank_all_step_and_graph_trajectories_equal':True,'artifacts':artifacts,
        'output_lengths':[len(a['tokens']) for a in answers],'finish_reasons':[a['finish_reason'] for a in answers],
        'cap_hits':run['cap_hits'],'preemptions_per_question':preemptions,'total_preemptions':sum(preemptions),
        'quality':run['quality'],'actual_capacity':capacity,
        'initialization_and_native_graph_setup_seconds':report['native_engine_initialization_and_graph_setup_wall_seconds'],
        'configuration_and_initial_receipt_seconds':report['configuration_and_initial_receipt_wall_seconds'],
        'anchor_prefill_and_one_token_seconds':run['anchor_wall_seconds'],
        'complete_generate_seconds':run['complete_generate_wall_seconds'],
        'cold_anchor_plus_generate_seconds':run['cold_shared_document_total_wall_seconds'],
        'generation_plus_record_export_seconds':run['generate_plus_record_export_wall_seconds'],
        'pure_decode_step_counts_by_B':dict(sorted(counts.items())),
        'pure_decode_steps_total':step_total,
        'fraction_of_pure_decode_steps_at_least_B':{str(b):sum(n for k,n in counts.items() if k>=b)/step_total if step_total else None for b in (2,4,8,16)},
        'CPU_between_prepare_interval_seconds_by_pure_decode_B':dict(sorted(intervals.items())),
        'fraction_of_observed_pure_decode_boundary_span_at_least_B':{str(b):sum(n for k,n in intervals.items() if k>=b)/interval_total if interval_total else None for b in (2,4,8,16)},
        'observed_prefill_or_mixed_between_prepare_span_seconds':prefilling_span,
        'observed_all_between_prepare_span_seconds':prefilling_span+interval_total,
        'all_prepared_batches':len(all_steps[0]),'full_native_graph_replays':len(graphs[0]),
        'limitations':['Between-prepare intervals include model execution, scheduler and host effects; they are not GPU kernel timings. First-entry and final-completion gaps are excluded.',
            'Pure-decode step counts and actual scheduled batch sizes are the direct cohort observations; request count alone does not establish cohort persistence.',
            'Natural native baseline only; no candidate speedup or generalization across applications is established.']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.output)
    assert not path.exists(),path
    report=audit(Path(args.input))
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
