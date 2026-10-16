"""CPU audit of exact dense-cascade whole-model qualification and wall timings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path):
    report = json.loads(path.read_text())
    assert report['status'] == 'completed', report.get('error', report['status'])
    for source, expected in report['source_sha256'].items():
        assert digest(ROOT / source) == expected, source
    output = {'summary': str(path), 'summary_sha256': digest(path), 'source_hashes_match': True,
              'B': report['B'], 'dtype': report['args']['kv_dtype'],
              'common_prefix_tokens': report['common_prefix_tokens'],
              'candidate_plan_and_capture_wall_seconds': report['candidate_plan_and_capture_wall_seconds'],
              'phases': [], 'comparisons': {}}
    by_phase = {}
    for run in report['runs']:
        graphs, cascade = [], []
        for worker in run['workers']:
            graphs.append(json.loads(Path(worker['graph_metadata']).read_text()))
            cascade.append(json.loads(Path(worker['cascade_record']).read_text()))
        assert len(graphs) == len(cascade) == 2
        trajectories = [[(r['batch'], r['questions'], r['positions'], r['candidate_graph_replayed'])
                         for r in graph['replays']] for graph in graphs]
        assert trajectories[0] == trajectories[1], (run['phase'], 'rank trajectories differ')
        for rank in cascade:
            assert rank['method'] == run['method'] and rank['validation'] == run['validation']
            assert len(rank['prepare_records']) == sum(row[3] for row in trajectories[0])
            assert all(p['actual_shared_physical_prefix_checked'] and not p['source_payload_copy']
                       and not p['tail_plan_topology_changed'] for p in rank['prepare_records'])
        qualification_count = 0
        if run['validation']:
            assert run['same_Q_native_all_layers_passed']
            for rank in cascade:
                assert len(rank['checks']) == len(rank['prepare_records']) > 0
                for step in rank['checks']:
                    assert [r['layer'] for r in step['layers']] == list(range(48))
                    assert all(r['same_Q_native_O_passed'] for r in step['layers'])
                    qualification_count += len(step['layers'])
        answers = run['answers']
        assert sum(len(a['tokens']) for a in answers) == run['total_output_tokens']
        if not run['natural_eos_enabled']:
            assert all(len(a['tokens']) == run['sampling']['max_tokens'] for a in answers)
        assert abs(run['anchor_wall_seconds']+run['complete_generate_wall_seconds']-
                   run['cold_shared_document_total_wall_seconds']) < 1e-8
        phase = {'phase': run['phase'], 'method': run['method'], 'validation': run['validation'],
                 'same_Q_layer_checks_two_ranks': qualification_count,
                 'natural_eos_enabled': run['natural_eos_enabled'],
                 'output_tokens': [len(a['tokens']) for a in answers],
                 'finish_reasons': [a['finish_reason'] for a in answers],
                 'cap_hits': [i for i,a in enumerate(answers) if a['finish_reason'] == 'length'],
                 'num_cached_tokens': [a['num_cached_tokens'] for a in answers],
                 'batch_steps': dict(sorted(Counter(r[0] for r in trajectories[0]).items())),
                 'candidate_batch_steps': dict(sorted(Counter(r[0] for r in trajectories[0] if r[3]).items())),
                 'rank_trajectories_equal': True,
                 'complete_generate_wall_seconds': run['complete_generate_wall_seconds'],
                 'anchor_wall_seconds': run['anchor_wall_seconds'],
                 'cold_shared_document_total_wall_seconds': run['cold_shared_document_total_wall_seconds'],
                 'additional_prepare_host_seconds_per_rank': [sum(p['complete_prepare_host_seconds'] for p in c['prepare_records']) for c in cascade],
                 'prepare_ms_per_candidate_step_per_rank': [statistics.median(p['complete_prepare_host_seconds']*1000 for p in c['prepare_records']) if c['prepare_records'] else None for c in cascade]}
        output['phases'].append(phase)
        by_phase[run['phase']] = (run, trajectories[0])
    fixed = ['native_A0', 'cascade_0', 'cascade_1', 'native_A1']
    if all(p in by_phase for p in fixed):
        runs = [by_phase[p][0] for p in fixed]
        wall = [r['complete_generate_wall_seconds'] for r in runs]
        cold = [r['cold_shared_document_total_wall_seconds'] for r in runs]
        native_tokens = [a['tokens'] for a in runs[0]['answers']]
        output['comparisons']['fixed'] = {
            'native_AA_tokens_equal': native_tokens == [a['tokens'] for a in runs[3]['answers']],
            'candidate_repeat_tokens_equal': [a['tokens'] for a in runs[1]['answers']] == [a['tokens'] for a in runs[2]['answers']],
            'native_candidate_tokens_equal': native_tokens == [a['tokens'] for a in runs[1]['answers']],
            'all_step_positions_and_questions_equal': all([(b,q,pos) for b,q,pos,_ in by_phase[p][1]] == [(b,q,pos) for b,q,pos,_ in by_phase[fixed[0]][1]] for p in fixed),
            'all_APC_hits_equal': all([a['num_cached_tokens'] for a in r['answers']] == [a['num_cached_tokens'] for a in runs[0]['answers']] for r in runs),
            'ready_group_native_over_candidate': statistics.mean([wall[0],wall[3]])/statistics.mean(wall[1:3]),
            'cold_document_native_over_candidate': statistics.mean([cold[0],cold[3]])/statistics.mean(cold[1:3]),
            'native_AA_max_over_min': max(wall[0],wall[3])/min(wall[0],wall[3]),
            'scope': 'Two fixed-length repeats each; descriptive ratio, no population CI or natural quality claim.'}
    if all(p in by_phase for p in ['native_natural','cascade_natural']):
        n,c = [by_phase[p][0] for p in ['native_natural','cascade_natural']]
        output['comparisons']['natural'] = {
            'output_tokens_exactly_equal': [a['tokens'] for a in n['answers']] == [a['tokens'] for a in c['answers']],
            'output_lengths_equal': [len(a['tokens']) for a in n['answers']] == [len(a['tokens']) for a in c['answers']],
            'observed_ready_group_wall_ratio': n['complete_generate_wall_seconds']/c['complete_generate_wall_seconds'],
            'observed_cold_document_wall_ratio': n['cold_shared_document_total_wall_seconds']/c['cold_shared_document_total_wall_seconds'],
            'scope': 'Natural task wall ratio; changed tokens can change workload. Task quality requires independent answer scoring.'}
    output['limits'] = ['Same-Q per-layer tolerance is qualification, not bitwise trajectory identity or task accuracy.',
                        'Candidate setup cost shown separately; cold document figures include anchor+generation, not first-use graph setup.',
                        'Native metadata preparation remains charged on cascade path; host phase timing includes all additional work.',
                        'Existing cascade algorithm is a control, not algorithmic novelty.']
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    report=audit(Path(args.input))
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
