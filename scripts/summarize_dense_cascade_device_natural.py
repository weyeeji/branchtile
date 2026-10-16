"""Audit all four device-metadata conditions, including negative results and costs."""
import argparse
import json
from pathlib import Path
import statistics

from audit_dense_cascade_device_model import audit
from audit_dense_cascade_four_seeds import ROOT, METRIC_ID, score_run, sha


def terminal(source, ledger):
    matched = []
    for row in ledger:
        argv = row.get('argv', [])
        if row.get('event') == 'end' and '--output' in argv:
            if (ROOT / argv[argv.index('--output')+1]).resolve() == source.parent.resolve():
                matched.append(row)
    assert len(matched) == 1 and matched[0]['exit_code'] == 0 and not matched[0]['timed_out'], source
    return matched[0]


def summarize(directory):
    ledger = [json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()]
    output = {'status': 'completed', 'metric_id': METRIC_ID, 'conditions': [],
              'metric_source_sha256': sha(ROOT/'src/financial_answer_metric.py'),
              'extraction_source_sha256': sha(ROOT/'scripts/screen_application.py'),
              'analysis_sources': {str(p.relative_to(ROOT)): sha(p) for p in
                  (Path(__file__).resolve(), ROOT/'scripts/audit_dense_cascade_device_model.py',
                   ROOT/'scripts/audit_dense_cascade_reference_model.py', ROOT/'scripts/audit_dense_cascade_four_seeds.py')},
              'limits': [
                  'One seed on each of two documents per dtype; descriptive observations, no population confidence interval.',
                  'Natural outputs may change length and MoE routing; native/candidate task wall is not an isolated algorithm speedup.',
                  'The inclusive wall includes phase-end device flag validation and record export; the latter is experimental instrumentation.',
                  'Conservative first-use candidate cost includes all configured production and checked graphs. Production capture components are separately available in the individual audit.',
                  'Old/new implementation wall timings are historical, not contemporaneously paired; only exact token equality is an unambiguous trajectory observation.',
                  'The underlying cascade and device metadata optimization are established techniques, not claimed novelty.']}
    for doc in ('train1', 'train7'):
        for dtype in ('bf16', 'fp8'):
            condition = f'{doc}_{dtype}'
            path = directory/condition/'summary.json'
            data = json.loads(path.read_text())
            audited = audit(path)
            old_root = 'dense_cascade_reference_natural_v1' if dtype == 'bf16' else 'dense_cascade_natural_v1'
            old_path = ROOT/'results'/old_root/condition/'summary.json'
            old = json.loads(old_path.read_text())
            assert old['status'] == 'completed'
            for key in ('protocol', 'prompt_token_hashes', 'document_sha256', 'common_prefix_tokens'):
                assert old[key] == data[key], (condition, key)
            assert {k:v for k,v in old['engine'].items() if k != 'worker_extension_cls'} == {
                k:v for k,v in data['engine'].items() if k != 'worker_extension_cls'}
            group = next(g for g in json.loads((ROOT/data['protocol']['manifest']).read_text())['groups']
                         if g['document_id'] == data['args']['document_id'])
            assert group['document_sha256'] == data['document_sha256']
            old_runs = {r['phase']:r for r in old['runs']}
            tokens_equal_by_phase = {}
            for run in data['runs']:
                assert len(run['answers']) == len(old_runs[run['phase']]['answers']) == 8
                tokens_equal_by_phase[run['phase']] = [a['tokens'] == b['tokens'] for a,b in
                    zip(run['answers'],old_runs[run['phase']]['answers'])]
            arms = {}
            for run in data['runs']:
                if not run['natural_eos_enabled']:
                    continue
                method = run['method']
                assert method not in arms
                previous = next(r for r in old['runs'] if r['natural_eos_enabled'] and r['method'] == method)
                assert previous['sampling'] == run['sampling']
                score, details = score_run(run, group['questions'])
                phase = next(p for p in audited['phases'] if p['phase'] == run['phase'])
                device = next(p for p in audited['device_metadata_phases'] if p['phase'] == run['phase'])
                arms[method] = {'sampling': run['sampling'], 'ready_seconds': run['complete_generate_wall_seconds'],
                    'cold_seconds': run['cold_shared_document_total_wall_seconds'],
                    'ready_including_validation_export_seconds': run['generate_plus_phase_validation_wall_seconds'],
                    'cold_including_validation_export_seconds': device['anchor_plus_generate_plus_phase_validation_wall_seconds'],
                    'output_tokens': run['total_output_tokens'], 'cap_hits': phase['cap_hits'], 'score': score,
                    'APC_hits': phase['num_cached_tokens'],
                    'answers': details, 'batch_steps': phase['batch_steps'],
                    'candidate_batch_steps': phase['candidate_batch_steps'],
                    'device_receipts': device['ranks'],
                    'old_implementation_tokens_equal_per_question': [a['tokens'] == b['tokens'] for a,b in zip(run['answers'],previous['answers'])],
                    'old_implementation_total_tokens': previous['total_output_tokens'],
                    'old_implementation_ready_seconds': previous['complete_generate_wall_seconds'],
                    'old_implementation_cold_seconds': previous['cold_shared_document_total_wall_seconds']}
            assert set(arms) == {'native', 'cascade'}
            n, c = arms['native'], arms['cascade']
            assert n['sampling'] == c['sampling']
            assert n['APC_hits'] == c['APC_hits']
            ratios = {key:n[key]/c[key] for key in ('ready_seconds', 'cold_seconds',
                'ready_including_validation_export_seconds', 'cold_including_validation_export_seconds')}
            ratios['conservative_first_use_cold_including_all_candidate_setup'] = n['cold_including_validation_export_seconds'] / (
                c['cold_including_validation_export_seconds'] + data['candidate_plan_and_capture_wall_seconds'])
            fixed_runs = {r['phase']:r for r in data['runs'] if not r['natural_eos_enabled']}
            fixed_inclusive = {p:fixed_runs[p]['generate_plus_phase_validation_wall_seconds'] for p in
                ('native_A0', 'native_A1', 'cascade_0', 'cascade_1')}
            fixed = audited['comparisons']['fixed']
            fixed['ready_native_over_candidate_including_validation_export'] = statistics.mean(
                fixed_inclusive[p] for p in ('native_A0','native_A1')) / statistics.mean(
                fixed_inclusive[p] for p in ('cascade_0','cascade_1'))
            output['conditions'].append({'condition': condition, 'sources': {
                str(path):sha(path), str(old_path):sha(old_path)}, 'clean_exit': terminal(path,ledger),
                'qualified_layer_checks': audited['phases'][0]['same_Q_layer_checks_two_ranks'],
                'pair_failed_but_reference_passed': audited['phases'][0]['pair_failed_but_reference_passed_count'],
                'old_implementation_tokens_equal_per_phase_and_question': tokens_equal_by_phase,
                'candidate_setup_seconds': data['candidate_plan_and_capture_wall_seconds'],
                'fixed': fixed, 'natural_arms': arms, 'natural_native_over_candidate': ratios})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/dense_cascade_device_natural_v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    assert not path.exists(), path
    result = summarize(Path(args.root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps([{k:c[k] for k in ('condition','fixed','natural_native_over_candidate')} for c in result['conditions']], indent=2))


if __name__ == '__main__':
    main()
