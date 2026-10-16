"""Independent CPU receipt audit for the device-metadata full-model control."""
import argparse
import json
from pathlib import Path
import statistics

from audit_dense_cascade_reference_model import audit as reference_audit, digest


def audit(path):
    result = reference_audit(path)
    summary = json.loads(path.read_text())
    assert summary['engine']['worker_extension_cls'].endswith('DenseCascadeDeviceWorker')
    result['audit_source_sha256'] = digest(Path(__file__))
    result['configuration_wall_seconds'] = summary['configuration_wall_seconds']
    result['capture_by_rank'] = summary['capture']
    result['device_metadata_phases'] = []
    for run in summary['runs']:
        ranks = []
        for worker in run['workers']:
            record_path = Path(worker['cascade_record'])
            graph_path = Path(worker['graph_metadata'])
            record = json.loads(record_path.read_text())
            graph = json.loads(graph_path.read_text())
            assert record['rank'] == graph['rank'] == worker['rank']
            assert record['phase'] == graph['phase'] == run['phase']
            assert worker['device_metadata_valid'] and record['device_metadata_valid']
            replay = [r for r in graph['replays'] if r['candidate_graph_replayed']]
            prepares = record['prepare_records']
            assert len(replay) == len(prepares)
            used = sorted({r['batch'] for r in replay})
            receipts = record['device_error_receipts']
            assert sorted(map(int, receipts)) == used
            for receipt in receipts.values():
                assert receipt['flags'] == 0 and receipt['valid']
                assert not receipt['tail_bounds_error'] and not receipt['prefix_identity_error']
                assert receipt['tail_plan_info_unchanged']
                assert receipt['host_plan_calls_after_initialization'] == 0
            bindings = []
            for i, (step, prepared) in enumerate(zip(replay, prepares)):
                assert prepared['batch'] == step['batch'] >= 2
                assert prepared['seq_lens'] == [int(p)+1 for p in step['positions']]
                common = prepared['common_prefix_tokens']
                assert common == summary['common_prefix_tokens'] and common % 16 == 0
                tails = [n-common for n in prepared['seq_lens']]
                assert tails == prepared['private_tail_tokens'] and min(tails) > 0
                assert prepared['private_tail_pages'] == [(t+15)//16 for t in tails]
                for key in ('host_plan_calls_this_prepare', 'tail_plan_calls_this_prepare', 'prefix_plan_calls_this_prepare'):
                    assert prepared[key] == 0
                for key in ('device_metadata_inside_model_graph', 'device_prefix_identity_checked_each_graph', 'cohort_request_subset_and_CPU_prefix_epochs_checked'):
                    assert prepared[key]
                assert prepared['layers_sharing_plan'] == 48
                if prepared['prefix_binding'] is not None:
                    binding = prepared['prefix_binding']
                    assert binding['physical_prefix_checked'] and binding['prefix_pages'] == common//16
                    assert binding['host_elapsed_seconds'] >= 0
                    bindings.append(i)
            assert bindings == ([0] if prepares else [])
            steady = [p['complete_prepare_host_seconds']*1e6 for p in prepares[1:]]
            ranks.append({'rank': worker['rank'], 'record_sha256': digest(record_path),
                          'graph_sha256': digest(graph_path), 'used_candidate_batches': used,
                          'candidate_steps': len(prepares), 'prefix_binding_steps': bindings,
                          'device_error_receipts': receipts, 'fallbacks': record['fallbacks'],
                          'steady_prepare_host_median_us': statistics.median(steady) if steady else None,
                          'first_binding_prepare_host_us': prepares[0]['complete_prepare_host_seconds']*1e6 if prepares else None})
        export = run['phase_validation_and_record_export_wall_seconds']
        assert export >= 0
        validated = run['generate_plus_phase_validation_wall_seconds']
        assert abs(validated-run['complete_generate_wall_seconds']-export) < 1e-8
        result['device_metadata_phases'].append({'phase': run['phase'], 'ranks': ranks,
            'phase_validation_and_record_export_wall_seconds': export,
            'generate_plus_phase_validation_wall_seconds': validated,
            'anchor_plus_generate_plus_phase_validation_wall_seconds': run['anchor_wall_seconds']+validated})
    result['limits'].extend([
        'Device error receipts are read at phase end. Raw generation excludes that validation/export RPC; the inclusive wall figure is reported separately.',
        'No per-step host plan calls are permitted by runtime guards; source/receipt checks do not replace independent GPU metadata qualification.',
        'Steady prepare timing is measured on the stated phase; checked qualification synchronizes for numeric checks and is not a production-latency estimate.',
        'Device metadata and existing cascade are implementation controls, not claimed algorithmic novelty.',
    ])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), output
    report = audit(Path(args.input))
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
