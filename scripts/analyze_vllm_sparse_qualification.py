"""CPU audit of full-budget keys, local kernel and across-layer output/logit errors."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    args = p.parse_args()
    root = Path(args.directory)
    data = json.loads((root/'summary.json').read_text())
    reference_phase = data.get('reference_phase', 'dense_A')
    report = {'source': str(root/'summary.json'), 'source_sha256': hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
              'scope': 'Full-budget backend numerical diagnostics, not application quality or speed evidence.', 'methods': {}}
    report['reference_phase'] = reference_phase
    base_logits = torch.load(root/f'rank0/{reference_phase}.pt', map_location='cpu', weights_only=True)
    reference_keys = set(base_logits)
    expected_keys = {f'q{q}_s{step}' for q, answer in enumerate(data['runs'][0]['answers']) for step in range(1, len(answer['tokens'])+1)}
    report['dense_reference_key_completeness'] = {'expected_keys': sorted(expected_keys), 'actual_keys': sorted(reference_keys),
        'expected_present': expected_keys <= reference_keys, 'contiguous_per_question': all(
            sorted(int(key.rsplit('_s',1)[1]) for key in reference_keys if key.startswith(f'q{q}_')) == list(range(1, 1+sum(key.startswith(f'q{q}_') for key in reference_keys))) for q in range(data['B']))}
    for run in data['runs'][1:]:
        phase = run['phase']
        actual_logits = torch.load(root/f'rank0/{phase}.pt', map_location='cpu', weights_only=True)
        keys = set(actual_logits)
        logit_rows = []
        for key in sorted(reference_keys & keys):
            x, y = base_logits[key].double(), actual_logits[key].double()
            diff = y-x
            lpx, lpy = x.log_softmax(0), y.log_softmax(0)
            logit_rows.append({'key': key, 'max_abs': float(diff.abs().max()), 'rmse': float(diff.square().mean().sqrt()),
                              'mean_shift': float(diff.mean()), 'centered_rmse': float((diff-diff.mean()).square().mean().sqrt()),
                              'forward_kl': float((lpx.exp()*(lpx-lpy)).sum()), 'top1_equal': int(x.argmax()) == int(y.argmax())})
        token_matches = data.get('tokens_equal_reference', data.get('greedy_tokens_equal_dense_A'))[phase]
        teacher_forced = run.get('teacher_forced_to_dense_A', False) or run.get('teacher_forced_to_reference', False)
        item = {'generated_tokens_equal_reference': token_matches,
                'greedy_matches_dense_A': token_matches if reference_phase == 'dense_A' else None,
                'teacher_forced_to_dense_A': teacher_forced and reference_phase == 'dense_A',
                'teacher_forced_to_reference': teacher_forced,
                'token_equality_is_forced_and_not_natural_greedy_evidence': teacher_forced,
                'logit_key_sets_equal': keys == reference_keys, 'missing_keys': sorted(reference_keys-keys), 'extra_keys': sorted(keys-reference_keys),
                'logits': logit_rows, 'max_logit_abs': max(r['max_abs'] for r in logit_rows),
                'max_logit_kl': max(r['forward_kl'] for r in logit_rows),
                'unforced_top1_disagreements': sum(not r['top1_equal'] for r in logit_rows),
                'exact_logits': all(r['max_abs']==0 for r in logit_rows), 'ranks': {}}
        for rank in (0, 1):
            metadata = json.loads((root/f'rank{rank}/{phase}.json').read_text())
            records = metadata['records']
            by_layer = collections.defaultdict(list)
            for row in records:
                by_layer[row['layer']].append(row.get('max_abs_output_vs_dense', row.get('same_mask_reference', {}).get('max_abs',0)))
            result = {'local_kernel_checks': len(records), 'local_kernel_all_frozen_output_tolerance_passed': all(r.get('dense_output_tolerance_passed', r.get('same_mask_reference', {}).get('output_tolerance_passed',False)) for r in records) if records else None,
                      'independent_fp64_probes': [{'layer':r['layer'], **r['independent_fp64_probe']} for r in records if 'independent_fp64_probe' in r],
                      'local_error_by_layer': {str(layer): {'max_abs': max(values), 'median_abs_max': statistics.median(values)} for layer, values in sorted(by_layer.items())}}
            base_path, actual_path = root/f'rank{rank}/{reference_phase}_layer_outputs.pt', root/f'rank{rank}/{phase}_layer_outputs.pt'
            if base_path.exists() and actual_path.exists():
                base = torch.load(base_path, map_location='cpu', weights_only=True)
                actual = torch.load(actual_path, map_location='cpu', weights_only=True)
                accumulated = collections.defaultdict(list)
                for key in sorted(set(base) & set(actual)):
                    layer = int(re.match(r'l(\d+)_', key)[1])
                    difference = (actual[key]-base[key]).double()
                    accumulated[layer].append({'key': key, 'max_abs': float(difference.abs().max()),
                        'rmse': float(difference.square().mean().sqrt()),
                        'relative_l2': float(difference.norm() / base[key].double().norm().clamp_min(1e-12))})
                result['same_history_keys_equal'] = set(base) == set(actual)
                result['same_history_comparison_valid_for_short_path'] = all(token_matches)
                result['actual_output_difference_by_layer'] = {str(layer): {'maximum_abs': max(r['max_abs'] for r in rows),
                    'maximum_relative_l2': max(r['relative_l2'] for r in rows), 'median_rmse': statistics.median(r['rmse'] for r in rows), 'rows': rows}
                    for layer, rows in sorted(accumulated.items())}
                result['layer_output_files_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (base_path, actual_path)}
            item['ranks'][str(rank)] = result
        report['methods'][phase] = item
    report['interpretation'] = [
        'Frozen local output tolerance remains atol=.006 plus rtol=.015; local native-vs-FA checks use the same current hidden state.',
        'Across-layer actual-output differences compare separate complete runs. They are same-token-history comparisons only while greedy histories agree.',
        'Logit errors and KL are diagnostics. No new logit threshold is inferred from candidate results, and eight-token agreement is not a formal full-generation qualification.',
        'For teacher-forced calibration, generated-token equality is imposed. Only the captured pre-forcing logits/top1 are numerical evidence; no natural greedy consistency is inferred.',
        'A/A, query-key completeness and raw snapshots must be checked before interpreting drift; bootstrap or multiple layers cannot create independent documents.',
        'Reference checks and CPU snapshots add work. Instrumented wall times are not application performance.']
    report['reference_local_checks'] = {}
    for rank in (0,1):
        metadata = json.loads((root/f'rank{rank}/{reference_phase}.json').read_text())
        records = metadata['records']
        checks = [r.get('dense_output_tolerance_passed',r.get('same_mask_reference',{}).get('output_tolerance_passed')) for r in records]
        report['reference_local_checks'][str(rank)] = {'count':len(checks),'all_passed':all(v is True for v in checks) if checks else None,
            'fp64_probes':[{'layer':r['layer'],**r['independent_fp64_probe']} for r in records if 'independent_fp64_probe' in r]}
    path = root/'numerical_audit.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({phase: {'max_logit_abs': r['max_logit_abs'], 'max_kl': r['max_logit_kl'], 'all_output_tokens_equal': all(r['generated_tokens_equal_reference']), 'teacher_forced':r['teacher_forced_to_dense_A'], 'keys_equal':r['logit_key_sets_equal']} for phase,r in report['methods'].items()}))


if __name__ == '__main__':
    main()
