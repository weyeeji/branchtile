"""CPU-only post-run audit; preserves raw driver results and ledger status."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--job-name', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    out = ROOT / args.output
    summary = json.loads((out / 'summary.json').read_text())
    protocol = summary['protocol_contents']
    ledger = [json.loads(line) for line in (ROOT / 'results/job_ledger.jsonl').read_text().splitlines()]
    events = [row for row in ledger if row['name'] == args.job_name and row['pid'] == summary['pid']]
    assert [row['event'] for row in events] == ['start', 'end']
    end = events[1]
    source_ok = {name: sha(ROOT / name) == digest for name, digest in protocol['source_sha256'].items()}
    artifacts_ok = {name: sha(ROOT / name) == digest for name, digest in protocol['artifact_sha256'].items()}
    assert all(source_ok.values()) and all(artifacts_ok.values())
    raw_files = {}
    for path in sorted(out.rglob('*.json')):
        if path.name == 'audit.json':
            continue
        raw_files[str(path.relative_to(ROOT))] = {
            'sha256': sha(path), 'bytes': path.stat().st_size,
            'observed_mtime_utc': dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
        }
    all_checks, ranks = [], []
    expected_keys = {f'q{q}_s{s}' for q, n in enumerate(protocol['tokens_by_question']) for s in range(1, n+1)}
    expected_checks = {(phase, b, layer) for phase in ('full_A', 'eager_teacher')
                       for b in (1, 2, 3) for layer in protocol['reference_layers']}
    for rank in (0, 1):
        directory = out / f'rank{rank}'
        checks = json.loads((directory / 'checks.json').read_text())['checks']
        assert len(checks) == len(expected_checks)
        assert {(c['phase'], c['batch'], c['layer']) for c in checks} == expected_checks
        assert all(c['passed'] and c['reference_lse_finite'] and c['same_quantized_input'] for c in checks)
        assert all(c['actual_query_dtype'] == c['backend_query_dtype'] == 'torch.bfloat16' for c in checks)
        assert all(c['shared_complete_prefix_pages_verified'] and c['complete_prefix_private_suffix_and_generated_tail'] for c in checks)
        assert all(c['Q_scale_tensor_equals_host_float_unit'] for c in checks)
        phase_data = {phase: json.loads((directory / f'{phase}.json').read_text())
                      for phase in ('full_A', 'full_B', 'eager_teacher')}
        def steps(phase):
            return [(row['batch'], row['questions'], row['positions']) for row in phase_data[phase]['steps']]
        assert steps('full_A') == steps('full_B') == steps('eager_teacher')
        for phase, data in phase_data.items():
            assert {row['batch'] for row in data['steps']} == {1, 2, 3}
            assert all(row['native_FULL_graph_replayed'] == (phase != 'eager_teacher') for row in data['steps'])
            if rank == 0:
                assert set(data['logits']) == expected_keys
                assert all(row['all_finite'] for row in data['logits'].values())
                if phase != 'full_A':
                    assert all(row['max_abs_vs_full_A'] == row['relative_l2_vs_full_A'] == 0
                               and row['top1_equal_full_A'] for row in data['logits'].values())
        config = json.loads((directory / 'configuration.json').read_text())
        assert set(map(int, config['layers'])) == set(range(48))
        assert all(all(v == 1 for v in layer['scales'].values())
                   and all(v == 1 for v in layer['scale_floats'].values())
                   for layer in config['layers'].values())
        all_checks.extend(checks)
        ranks.append({'rank': rank, 'checks': len(checks), 'all_passed': True,
                      'pure_step_counts_by_B': dict(Counter(row['batch'] for row in phase_data['full_A']['steps'])),
                      'same_phase_step_sequence': True, 'max_abs_vs_own_FP64': max(c['max_abs'] for c in checks),
                      'max_relative_l2_vs_own_FP64': max(c['relative_l2'] for c in checks),
                      'max_same_input_eager_vs_FULL_abs': max(c.get('same_input_eager_vs_FULL_max_abs', 0) for c in checks),
                      'cache_payload_bytes_all_layers': sum(layer['logical_payload_bytes'] for layer in config['layers'].values()),
                      'capacity_tokens': config['deployment']['cache_num_gpu_blocks'] * 16,
                      'peak_allocated_bytes_by_phase': {phase: data['deployment']['torch_peak_allocated_bytes'] for phase, data in phase_data.items()},
                      'graph_instrumentation': json.loads((directory / 'graph_capture.json').read_text())})
    assert all(all(values) for values in summary['cold_AA'].values())
    assert all(summary['eager_teacher_same_tokens_A']) and all(summary['eager_teacher_same_APC_A'])
    numerical_complete = summary['status'] == 'qualified_instrumented_native_FP8KV_unit'
    process_clean = end['exit_code'] == 0 and not end['timed_out']
    summary_mtime = (out / 'summary.json').stat().st_mtime
    seconds_before_end = dt.datetime.fromisoformat(end['time_utc']).timestamp() - summary_mtime
    quantization = [c['current_token_quantization'] for c in all_checks if 'current_token_quantization' in c]
    audit = {
        'schema': 1, 'job_name': args.job_name, 'pid': summary['pid'],
        'raw_driver_status': summary['status'],
        'frozen_protocol_overall_status': 'passed_clean_exit' if numerical_complete and process_clean else 'timeout_failed' if end['timed_out'] else 'failed',
        'numerical_and_trajectory_contract_complete': numerical_complete,
        'process_clean_exit': process_clean,
        'status_interpretation': 'Both axes are required by the frozen protocol. A complete numerical receipt does not override a timeout. Raw summary and ledger remain unchanged.',
        'ledger_events': events, 'P1_allocated_GPU_hours': end['gpu_hours'],
        'summary_observed_mtime_seconds_before_ledger_end': seconds_before_end,
        'timestamp_scope': 'Filesystem mtime observed after the run; raw log independently records shutdown. Neither is a durable fsync timestamp or a substitute for clean exit.',
        'source_hash_checks': source_ok, 'artifact_hash_checks': artifacts_ok,
        'initialization_wall_seconds': summary['initialization_wall_seconds'],
        'ranks': ranks, 'numerical_check_count': len(all_checks),
        'max_abs_vs_own_FP64': max(c['max_abs'] for c in all_checks),
        'max_relative_l2_vs_own_FP64': max(c['relative_l2'] for c in all_checks),
        'logit_diagnostics': summary['logit_diagnostics'], 'cold_AA': summary['cold_AA'],
        'current_KV_quantization_descriptive': {
            'points': len(quantization), 'max_abs': max(q['max_abs_vs_unquantized_KV'] for q in quantization),
            'max_relative_l2': max(q['relative_l2_vs_unquantized_KV'] for q in quantization),
            'observed_abs_above_448_count': sum(q['unquantized_abs_gt_448_count'] for q in quantization),
            'not_all_prompt_KV': True, 'not_task_quality': True},
        'limitations': ['Instrumented fixed-length 8/12/16 tails, ignore_eos=True; no natural task-quality evidence.',
                        'FP64 reference uses actual FP8 cache and BF16 Q; no BF16-KV equivalence claim.',
                        'Native XQA exposes O only; no native LSE qualification.',
                        'eager_teacher disables FULL graph dispatch but retains torch.compile mode 3.',
                        'Extra persistent Q/O/K/V copies and graphs affect memory and timing; no application speed claim.',
                        'Unit-scale configuration only; timeout does not establish calibrated FP8 failure.',
                        'Independent native precision baseline, unrelated to the stopped joint-selector mechanism.'],
        'raw_artifacts': raw_files,
        'log': {'path': str(Path('logs') / f'{args.job_name}.log'), 'sha256': sha(ROOT / 'logs' / f'{args.job_name}.log')},
        'audit_script_sha256': sha(Path(__file__)),
    }
    (out / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({key: audit[key] for key in ('frozen_protocol_overall_status', 'numerical_check_count',
        'max_abs_vs_own_FP64', 'max_relative_l2_vs_own_FP64', 'P1_allocated_GPU_hours')}, indent=2))


if __name__ == '__main__':
    main()
