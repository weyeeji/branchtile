"""Descriptive cross-run comparison; never interprets it as a paired causal speedup."""
import collections
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    paths = {name: ROOT / f'results/{directory}/train1/summary.json' for name, directory in [
        ('pair_old_strict', 'gqa4_thinking_survivor_e2e_v1'), ('quad_new_production', 'gqa4_thinking_quad_production_v1')]}
    summaries = {name: json.loads(p.read_text()) for name, p in paths.items()}
    audits = {name: json.loads((p.parent / 'independent_analysis.json').read_text()) for name,p in paths.items()}
    for name, summary in summaries.items():
        assert summary['status'] == 'completed' and audits[name]['status'] == 'qualified'
        assert audits[name]['summary_sha256'] == sha(paths[name])
    old, new = summaries.values()
    shared_fields = ['document_sha256', 'prompt_token_hashes', 'prompt_lengths', 'common_prefix_tokens', 'B']
    assert all(old[field] == new[field] for field in shared_fields)
    old_runs = [r for r in old['runs'] if r['phase'].startswith('natural_')]
    new_runs = [r for r in new['runs'] if r['phase'].startswith('natural_')]
    equality = {}
    for old_run, new_run in zip(old_runs,new_runs,strict=True):
        assert old_run['phase'] == new_run['phase']
        fields = ['tokens', 'num_cached_tokens', 'finish_reason', 'stop_reason']
        matches = {field: [a[field] for a in old_run['answers']] == [a[field] for a in new_run['answers']] for field in fields}
        matches['scored_quality'] = old_run['quality'] == new_run['quality']
        matches['sampling'] = old_run['sampling'] == new_run['sampling']
        matches['rank_graph_trajectories'] = old_run['graph_trajectory_sha256_by_rank'] == new_run['graph_trajectory_sha256_by_rank']
        equality[old_run['phase']] = matches
    rows = {}
    for name, summary in summaries.items():
        runs = [r for r in summary['runs'] if r['phase'].startswith('natural_')]
        candidate = runs[1]
        record = json.loads(Path(candidate['workers'][0]['raw_pair_record']).read_text())
        graph = json.loads(Path(candidate['workers'][0]['graph_metadata']).read_text())['replays']
        counts = collections.Counter(r['batch'] for r in graph if r['candidate_graph_replayed'])
        rows[name] = {'summary_sha256': sha(paths[name]), 'independent_audit_sha256': sha(paths[name].parent / 'independent_analysis.json'),
            'enabled_batches': summary['enabled_batches'], 'candidate_replays': record['candidate_replays'],
            'total_full_graph_replays': len(graph), 'candidate_fraction': record['candidate_replays'] / len(graph),
            'candidate_steps_by_batch': dict(counts), 'natural_ratios_with_own_native_ABBA': summary['comparisons']['natural']['native_over_candidate'],
            'natural_output_lengths': [len(a['tokens']) for a in candidate['answers']],
            'quality': {k:candidate['quality']['score'][k] for k in ['denominator', 'exact_accuracy', 'rounded_accuracy', 'prediction_parse_failures', 'gold_parse_failures']},
            'cap_hits': candidate['cap_hits'], 'preemptions': candidate['total_request_preemptions']}
    setup_names = ['offline_preflight_wall_seconds', 'candidate_configuration_wall_seconds', 'candidate_graph_capture_wall_seconds', 'setup_receipt_export_wall_seconds']
    setup = {k:new[k] for k in setup_names}
    n = statistics.mean(new_runs[i]['cold_shared_document_total_wall_seconds'] for i in (0,3))
    c = statistics.mean(new_runs[i]['cold_shared_document_total_wall_seconds'] for i in (1,2))
    result = {'status': 'descriptive_cross_run_audit', 'source_sha256': sha(Path(__file__)),
        'shared_inputs_equal': shared_fields, 'cross_run_equality': equality, 'results': rows,
        'quad_primary_net': {'native_cold_mean_seconds': n, 'candidate_cold_mean_seconds': c,
            'unamortized_setup_components_seconds': setup, 'unamortized_setup_sum_seconds': sum(setup.values()),
            'native_over_candidate_plus_entire_setup': n/(c+sum(setup.values()))},
        'limits': ['Independent sessions, different graph setup contracts and source versions; this is not a directly paired causal quad-versus-pair measurement.',
            'Only each implementation versus its own native ABBA is paired. Extra observed B4/5 execution demonstrates coverage, not its isolated contribution to speed.',
            'No heldout claim; frozen P120528/B4..8/private<=8192 development scope only.',
            'The previous task-quality audit remains applicable if the output trajectory is identical; 0/8 strict/rounded scores are not repaired by faster execution.',
            'Old strict run overlapped recorded PAT visibility exposure; do not use old absolute timings as a clean cross-run baseline.']}
    out = ROOT / 'results/gqa4_pair_quad_cross_run_audit_v1.json'
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_cross_run_equality_fields': all(all(v.values()) for v in equality.values()), 'quad_net': result['quad_primary_net']},indent=2))


if __name__ == '__main__':
    main()
