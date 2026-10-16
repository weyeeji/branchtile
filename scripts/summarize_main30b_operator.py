"""CPU-only audit of the bounded official-main B3 operator comparison."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = []
    for path in sorted((ROOT / 'results/main30b_operator').glob('*.json')):
        data = json.loads(path.read_text())
        if 'tuning_planned' not in data:
            continue
        assert data['batch'] == 3 and data['gqa'] == 8
        assert len(data['partitions']) == 5 and len(data['hierarchies']) == 3
        assert data['q_shape'] == [2, 24, 128]
        assert data['provenance']['ready_lifetime_measured']
        assert data['provenance']['private_protected_document_tail_tokens'] == 2
        coverage = {}
        for method in sorted({row['method'] for row in data['tuning']}):
            rows = [row for row in data['tuning'] if row['method'] == method]
            key = 'tree' if method == 'e4_native' else 'partition'
            structures = {json.dumps(row[key]) for row in rows if key in row}
            passed_structures = {json.dumps(row[key]) for row in rows if key in row and row['status'] == 'ok'}
            coverage[method] = {'executed': len(rows), 'qualified': sum(row['status'] == 'ok' for row in rows),
                                'structures': len(structures), 'qualified_structures': len(passed_structures),
                                'failed_reasons': dict(collections.Counter(row['error'] for row in rows if row['status'] != 'ok'))}
        for cache in ('hot', 'cold'):
            rows = [row for row in data['measurements'] if row['cache'] == cache]
            if not rows:
                continue
            assert len(rows) == 60
            for block in range(30):
                pair = {row['direction']: row for row in rows if row['block'] == block}
                assert pair['random']['order'] == list(reversed(pair['reverse']['order']))
        item = {'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'provenance': data['provenance'], 'q_shape': data['q_shape'], 'k_shape': data['k_shape'],
                'gpu_uuid': data['gpu_uuid'], 'tuning_complete': data['tuning_complete'], 'coverage': coverage,
                'summary': data['summary'], 'selected': {cache: {method: value['config'] for method, value in choices.items()}
                                                        for cache, choices in data['selected'].items()},
                'search_seconds_including_jit_and_failures': data['search_seconds_including_jit_and_failures']}
        planning_path = path.with_name(path.stem + '_planning.json')
        if planning_path.exists():
            planning = json.loads(planning_path.read_text())
            assert planning['source_sha256'] == item['sha256']
            assert planning['gpu_uuid'] != item['gpu_uuid']
            followup = {'file': str(planning_path.relative_to(ROOT)), 'different_physical_gpu': True, 'cache': {}}
            for cache, values in planning['cache'].items():
                scenarios = {}
                for reuse, scenario in values['amortization_scenarios'].items():
                    costs = scenario['cost_us_by_method']
                    control = min((method for method in costs if not method.startswith('dynamic_')), key=costs.get)
                    candidate = min((method for method in costs if method.startswith('dynamic_')), key=costs.get)
                    scenarios[reuse] = {**scenario, 'best_control': control, 'best_dynamic': candidate,
                                        'best_control_over_dynamic_cost_ratio': costs[control] / costs[candidate]}
                followup['cache'][cache] = {'median_gpu_us': values['median_gpu_us'], 'paired_ratios': values['paired_ratios'],
                                            'amortization_scenarios': scenarios}
            item['second_card'] = followup
        else:
            item['second_card'] = {'pending': True}
        records.append(item)
    ledger = [json.loads(line) for line in (ROOT / 'results/job_ledger.jsonl').read_text().splitlines() if line]
    jobs = [row for row in ledger if row.get('name', '').startswith('main30b_operator_')]
    ended = [row for row in jobs if row['event'] == 'end']
    ended_pids = {row['pid'] for row in ended}
    pending = [row for row in jobs if row['event'] == 'start' and row['pid'] not in ended_pids]
    report = {'scope': 'Official Qwen3-30B-A3B-Thinking-2507-FP8 TP2, true B3 ready groups, same exact captured dense-trajectory payload/masks; operator evidence only.',
              'records': records,
              'natural_document_count': len({r['provenance']['document_sha256'] for r in records}),
              'budget': {'limit_gpu_hours': 1, 'actual_allocated_gpu_hours': sum(row['gpu_hours'] for row in ended),
                         'includes_failures': True, 'finished_jobs': len(ended), 'pending': pending},
              'comparison_rules': ['All five B3 fixed partitions use native and own union; all three binary trees execute exact intersection/residual nodes and LSE merge.',
                                   'No padded fourth request is constructed. Dynamic v3 aligns its internal accumulator to 32 rows while Q/O remain the true 24 rows.',
                                   'Dynamic v2/v3 and controls are independently selected by cache from the finite grid using qualification before timing, then paired without reselection.',
                                   'Complete GPU calls include query packing, output restoration, ln-LSE conversion and partial merging where needed. CPU plans and search costs are explicit.',
                                   'Random and reversed observations stay in the same block for 10000 paired bootstrap resamples; AA is retained.',
                                   'Second physical-card replay uses first-card selected configurations, includes explicit native full union, and measures every selected backend preparation.'],
              'integration_gate': 'Require stable >=1.15 strongest-qualified-control/dynamic ratio on prespecified natural inputs after defensible preparation amortization and second-card replication; earlier dynamic no-go evidence remains.',
              'limitations': ['One document with three natural requests; layers, steps, ranks and bootstrap blocks are not independent documents.',
                              'Ready metadata establishes same-forward eligibility, not scheduler ready-window duration, sparse model quality, throughput/JCT or energy.',
                              'Rank-local timings on separate inputs do not measure TP2 synchronized full-model decode speed.',
                              'Prepare amortization counts 1/16/64/256 are scenarios, not observed static-plan lifetimes; suffix growth may require updates.',
                              'E4 is a binary hierarchy mechanism port, not a full QUILT planner reproduction.',
                              'Frozen numerical tolerance is inherited; failures are reported and excluded, without claiming dominance over unqualified native configurations.']}
    output = ROOT / 'results/main30b_operator/summary.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'files': len(records), 'gpu_hours': report['budget']['actual_allocated_gpu_hours'], 'pending_jobs': len(pending)}))


if __name__ == '__main__':
    main()
