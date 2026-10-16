"""CPU-only audit of this frozen attempt; no model or CUDA imports."""
import hashlib
import gzip
import json
from pathlib import Path
import statistics

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((RUN / name).read_text())


def spread(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}


def counts(checks):
    rows = sum(c['batch'] for c in checks)
    return {
        'tensor_points': len(checks), 'rows': rows, 'logit_elements': rows * 128,
        'selected_ID_and_weight_elements_each': rows * 8,
        'native_recompute_matches_actual_graph_points': sum(c['native_same_input_matches_actual_graph_bitwise'] for c in checks),
        'candidate_logits_bitwise_equal_points': sum(c['candidate_logits_bitwise_equal_native'] for c in checks),
        'bitwise_drop_in_passed_points': sum(c['passed_exact_contract'] for c in checks),
        'bitwise_drop_in_passed_rows': sum(sum(c['exact_routing_identity_by_row']) for c in checks),
        'different_logit_elements': sum(c['candidate_vs_native_BF16']['different_elements'] for c in checks),
        'different_ordered_selected_ID_elements': sum(c['changed_selected_id_elements'] for c in checks),
        'different_selected_weight_elements_in_order': sum(c['changed_selected_weight_elements'] for c in checks),
        'different_expert_sets_rows': sum(set(a) != set(b) for c in checks for a, b in
            zip(c['selected_ids_native'], c['selected_ids_candidate'])),
        'finite_native_and_candidate_points': sum(c['native_error']['finite'] and c['candidate_error']['finite'] for c in checks),
        'max_abs_candidate_vs_native': max(c['candidate_vs_native_BF16']['max_abs'] for c in checks),
        'max_BF16_encoding_steps': max(c['candidate_vs_native_BF16']['max_BF16_encoding_steps'] for c in checks),
        'max_selected_weight_abs_difference': max(c['max_selected_weight_abs_difference'] for c in checks),
        'native_max_abs_vs_FP64': spread([c['native_error']['max_abs_vs_FP64'] for c in checks]),
        'candidate_max_abs_vs_FP64': spread([c['candidate_error']['max_abs_vs_FP64'] for c in checks]),
        'native_relative_L2_vs_FP64': spread([c['native_error']['relative_l2_vs_FP64'] for c in checks]),
        'candidate_relative_L2_vs_FP64': spread([c['candidate_error']['relative_l2_vs_FP64'] for c in checks]),
    }


def main():
    summary = read('summary.json'); ledger = read('ledger_events.json')
    assert summary['operationally_completed']
    assert summary['status'] == 'completed_not_qualified_bitwise_drop_in'
    assert ledger[-1]['exit_code'] == 0 and not ledger[-1]['timed_out']
    assert all(all(values) for values in summary['cold_AA'].values())
    assert summary['timing_workers']['status'].startswith('skipped_on_both_ranks')
    assert not list(RUN.glob('rank*/timing.json'))
    protocol_path = ROOT / summary['protocol']
    assert sha(protocol_path) == summary['protocol_sha256']
    source_matches = {name: sha(ROOT / name) == digest
                      for name, digest in summary['protocol_contents']['source_sha256'].items()}
    assert all(source_matches.values())
    by_rank = []; all_checks = []; qualification = []
    receipts = {}
    for rank in (0, 1):
        q = read(f'rank{rank}/qualification.json'); qualification.append(q)
        c = q['checks']; all_checks.extend(c)
        assert len(c) == 288
        assert {(x['phase'], x['batch'], x['layer']) for x in c} == {
            (phase, batch, layer) for phase in ('full_A', 'full_B') for batch in (1, 2, 3) for layer in range(48)}
        assert len(q['cold_AA_points']) == 144 and all(x['bitwise_equal'] for x in q['cold_AA_points'])
        assert len(q['poison_overwrite_receipts']) == 6 and all(q['poison_overwrite_receipts'].values())
        assert not q['qualified']
        snapshot = RUN / f'rank{rank}/actual_gate_snapshots.pt.gz'
        assert hashlib.sha256(gzip.decompress(snapshot.read_bytes())).hexdigest() == q['snapshot_sha256']
        receipts[str(snapshot.relative_to(ROOT))] = {
            'compressed_sha256': sha(snapshot), 'uncompressed_sha256': q['snapshot_sha256'],
            'storage': 'Deterministic gzip mtime=0 committed; original .pt also retained locally'}
        by_rank.append({'rank': rank, 'counts': counts(c),
            'by_batch': {str(b): counts([x for x in c if x['batch'] == b]) for b in (1, 2, 3)},
            'qualification_wall_seconds_including_JIT': q['qualification_wall_seconds'],
            'tinygemm_JIT_and_first_call_seconds': q['tinygemm_JIT_and_first_call_seconds'],
            'cold_AA_tensor_points_all_bitwise': 144, 'poison_overwrite_receipts_passed': 6})
    a = qualification[0]['checks']; b = qualification[1]['checks']
    cross_rank = all(x == y for x, y in zip(a, b))
    assert cross_rank
    a_only = [x for x in a if x['phase'] == 'full_A']
    b_lookup = {(x['batch'], x['layer']): x for x in a if x['phase'] == 'full_B'}
    assert all({k: v for k, v in x.items() if k != 'phase'} ==
               {k: v for k, v in b_lookup[x['batch'], x['layer']].items() if k != 'phase'} for x in a_only)
    changed = [x for x in a_only if not x['passed_exact_contract']]
    unique_cases = [{k: c[k] for k in ('batch', 'layer', 'exact_routing_identity_by_row',
                    'changed_selected_id_elements', 'changed_selected_weight_elements',
                    'candidate_vs_native_BF16', 'native_error', 'candidate_error',
                    'max_selected_weight_abs_difference')} for c in changed]
    result = {
        'schema': 1, 'status': 'completed_not_qualified_conservative_bitwise_drop_in',
        'script_sha256': sha(Path(__file__)),
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            [RUN / 'summary.json', RUN / 'ledger_events.json', protocol_path] +
            sorted(RUN.glob('rank*/*.json'))},
        'frozen_source_matches_current': source_matches,
        'operational_exit_code': 0, 'timed_out': False,
        'allocated_seconds': ledger[-1]['allocated_seconds'], 'P1_GPU_hours': ledger[-1]['gpu_hours'],
        'conservative_bitwise_qualification': False, 'candidate_performance_measured': False,
        'candidate_installed_in_model': False, 'natural_quality_measured': False,
        'global_counts_including_repeated_phases_and_ranks': counts(all_checks),
        'rank_breakdown': by_rank, 'rank_check_records_exactly_identical': cross_rank,
        'full_A_full_B_numerical_records_exactly_identical_except_phase': True,
        'rank0_full_A_counts_without_AA_or_TP_duplication': counts(a_only),
        'rank0_full_A_changed_cases': unique_cases,
        'sample_scope': 'One document, three questions. First native graph state of each B1/B2/B3; 48 layers. A/B and TP ranks are exact repeats here, not independent quality samples.',
        'changed_cases_candidate_lower_FP64_relative_L2': sum(c['candidate_error']['relative_l2_vs_FP64'] <
            c['native_error']['relative_l2_vs_FP64'] for c in changed),
        'changed_cases_candidate_higher_FP64_relative_L2': sum(c['candidate_error']['relative_l2_vs_FP64'] >
            c['native_error']['relative_l2_vs_FP64'] for c in changed),
        'initialization_seconds': summary['initialization_wall_seconds'],
        'warmup_seconds': summary['warmup_wall_seconds'],
        'native_instrumented_phases': [{k: r[k] for k in ('phase', 'cold_anchor_wall_seconds',
            'instrumented_generate_wall_seconds', 'anchor_cached_tokens')} for r in summary['runs']],
        'capture_cost_by_rank': [{k: g[k] for k in ('rank', 'wall_seconds', 'additional_allocated_bytes',
            'persistent_copy_buffer_bytes')} for g in summary['instrumented_native_graph_capture']],
        'local_tensor_artifacts': receipts,
        'interpretation': [
            'All actual graph receipts, cold A/A and native same-input reproductions pass. Thus the observed difference is between two GEMM execution paths, not failed graph capture or native instability at these points.',
            'Each of nine rank0/full_A changed points has exactly one logit differing by one adjacent BF16 encoding. Eight have lower FP64 relative-L2 error with TinyGEMM; one has higher error. This is descriptive and no candidate-derived tolerance is introduced.',
            'Every selected expert SET remains unchanged. B3 layer17 row2 (zero-based) swaps the order of experts 18 and83 as a BF16 tie appears; ordered FP32 weights also differ.',
            'The conservative bitwise direct-substitution contract fails. This does not show incorrect TinyGEMM mathematics, task degradation, performance failure, or permanent baseline ineligibility.',
            'Frozen stop rule skips both ranks performance measurement; no throughput, application speedup or natural output conclusion is available. Any approximate substitution requires a separately frozen model-level numerical/quality experiment.',
        ]}
    (RUN / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'counts': result['global_counts_including_repeated_phases_and_ranks'],
                      'P1_GPU_hours': result['P1_GPU_hours']}, indent=2))


if __name__ == '__main__':
    main()
