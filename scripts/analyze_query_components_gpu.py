"""CPU paired-block analysis of the four frozen wave33 kernel experiments."""
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for path in sorted((ROOT / 'results/query_components_gpu_v1').glob('*.json')):
        data = json.loads(path.read_text())
        assert data['status'] == 'qualified_and_benchmarked_prefix_only'
        assert data['all_numerical_checks_passed'] and all(c['passed'] for c in data['checks'])
        assert data['args']['blocks'] == 20 and data['args']['replays'] == 10 and not data['args']['cold']
        blocks = {}
        for r in data['timing_blocks']:
            assert r['method'] not in blocks.setdefault(r['block'], {})
            blocks[r['block']][r['method']] = r['cuda_us_per_call']
        assert len(blocks) == 20 and all(len(r) == 3 for r in blocks.values())
        methods = ('native_xqa_O_only', 'BF16_QK_control', 'two_FP8_QK')
        medians = {m: statistics.median(b[m] for b in blocks.values()) for m in methods}
        deltas = [b['BF16_QK_control'] - b['two_FP8_QK'] for b in blocks.values()]
        rng = random.Random(20261017)
        boot = sorted(statistics.median(rng.choices(deltas, k=len(deltas))) for _ in range(10000))
        rows.append({'source': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'shape': data['shape'], 'block_m': data['args']['block_m'], 'median_us': medians,
                     'component_latency_reduction_vs_control_ratio_of_medians': 1 - medians['two_FP8_QK'] / medians['BF16_QK_control'],
                     'component_speedup_vs_XQA_ratio_of_medians': medians['native_xqa_O_only'] / medians['two_FP8_QK'],
                     'paired_delta_control_minus_component_us': deltas,
                     'paired_delta_median_us': statistics.median(deltas),
                     'paired_delta_median_bootstrap_95pct': [boot[250], boot[9749]],
                     'blocks_component_faster_than_control': sum(x > 0 for x in deltas),
                     'resources': data['resources'], 'checks': data['checks']})
    assert len(rows) == 4
    result = {'schema': 1, 'status': 'CPU_existing_GPU_result_review',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'rows': rows,
              'limitations': ['Bootstrap resamples 20 timing blocks only, not independent documents or generated requests.',
                              'Historical source raw files are retained; four cases are not a broad qualification.',
                              'Two real cases use one layer and one rank from each native FP8 document state, complete shared prefix only.',
                              'Q grouping, compact K/V export and page-layout conversions are outside timing. The candidate is not yet a zero-copy paged engine implementation.',
                              'Incremental QK mechanism comparison shares the same custom layout and BF16 PV. Large wins versus XQA cannot be solely attributed to the new QK mechanism.',
                              'No end-to-end acceleration or model task-quality result follows from these standalone timings.']}
    target = ROOT / 'results/platform/query_components_gpu_analysis_20261017.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    for r in rows:
        print(Path(r['source']).name, r['paired_delta_median_us'], r['paired_delta_median_bootstrap_95pct'], r['blocks_component_faster_than_control'])


if __name__ == '__main__':
    main()
