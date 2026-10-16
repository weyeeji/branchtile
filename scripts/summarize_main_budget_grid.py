"""Audit all pre-specified larger budgets without dropping negative settings."""
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
paths = [ROOT / 'results/main30b_operator/main30b_operator_rank0_l15_s64_b4096.json']
paths += [ROOT / f'results/main30b_budget_grid/rank0_l15_s64_b{budget}.json' for budget in [8192, 16384, 32768]]
records = []
for budget, path in zip([4096, 8192, 16384, 32768], paths):
    data = json.loads(path.read_text())
    assert data['tuning_complete'] and data['batch'] == 3 and data['gqa'] == 8
    assert len(data['partitions']) == 5 and len(data['hierarchies']) == 3
    assert data['provenance']['ready_lifetime_measured']
    methods = collections.Counter(row['method'] for row in data['tuning'])
    failures = [row for row in data['tuning'] if row['status'] != 'ok']
    for cache in ['hot', 'cold']:
        rows = [row for row in data['measurements'] if row['cache'] == cache]
        assert len(rows) == 60
        for block in range(30):
            pair = {r['direction']: r for r in rows if r['block'] == block}
            assert pair['random']['order'] == list(reversed(pair['reverse']['order']))
    records.append({'budget': budget, 'file': str(path.relative_to(ROOT)),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'gpu_uuid': data['gpu_uuid'],
        'provenance': data['provenance'], 'tuning_configurations': dict(methods),
        'tuning_failures_retained': failures, 'summary': data['summary'],
        'selected': {cache: {method: item['config'] for method, item in values.items()}
                     for cache, values in data['selected'].items()}})
assert len({r['provenance']['document_sha256'] for r in records}) == 1
ledger = [json.loads(line) for line in (ROOT / 'results/job_ledger.jsonl').read_text().splitlines()]
ended = [r for r in ledger if r['event'] == 'end' and r['name'].startswith('main30b_budget_grid_')]
assert len(ended) == 3 and all(r['exit_code'] == 0 for r in ended)
result = {'scope': 'One development document, rank0/layer15/true-ready step64; four original support budgets. Budgets/ranks/timing blocks are not independent documents.',
    'cost_scope': 'Paired hot/cold complete GPU calls include output/LSE conversion and merge; CPU planning/search not included in these call medians. Selected configurations are per-capture hindsight tuning oracles.',
    'protocol': 'results/main30b_budget_grid/protocol.json',
    'additional_allocated_P2_gpu_hours': sum(r['gpu_hours'] for r in ended),
    'limitations': ['Larger budgets were scheduled after the4K negative evidence, before their own measurements; this is a coverage extension, not a new holdout.',
                   'Native configs failing the frozen output/LSE tolerance are retained and excluded; a failure does not prove the candidate dominates a fully qualified native alternative.',
                   'Reported selected best_control is selected on tuning and can differ from the minimum during paired replay. All control families remain visible.',
                   'No quality, engine throughput, natural static-plan lifetime, energy or novel-mechanism claim follows from this replay.'],
    'records': records}
(ROOT / 'results/main30b_budget_grid/summary.json').write_text(json.dumps(result, indent=2) + '\n')
for row in records:
    print(row['budget'], {cache: row['summary'][cache]['median_us'] for cache in ['hot', 'cold']})
