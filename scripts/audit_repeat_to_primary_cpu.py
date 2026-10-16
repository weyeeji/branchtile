"""Supplemental one-repeat provenance/trajectory audit; does not replace the frozen primary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', required=True)
    parser.add_argument('--repeat', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    paths = [ROOT / args.primary, ROOT / args.repeat]
    summaries = [json.loads(p.read_text()) for p in paths]
    assert all(s['status'] == 'completed' for s in summaries)
    first, repeated = summaries
    keys = ['source_sha256', 'mechanism', 'protocol_sha256', 'manifest_sha256', 'document_sha256',
        'prompt_token_hashes', 'prompt_lengths', 'common_prefix_tokens', 'B', 'offline_inputs']
    checks = {key: first[key] == repeated[key] for key in keys}
    phases = []
    assert len(first['runs']) == len(repeated['runs']) == 4
    for a,b in zip(first['runs'],repeated['runs'],strict=True):
        assert a['phase'] == b['phase']
        fields = {key:[r[key] for r in a['answers']] == [r[key] for r in b['answers']]
            for key in ['tokens', 'text', 'finish_reason', 'stop_reason', 'num_cached_tokens']}
        fields.update({key:a[key] == b[key] for key in ['sampling', 'quality', 'graph_trajectory_sha256_by_rank']})
        phases.append({'phase': a['phase'], 'fields_equal': fields,
            'primary_generate_seconds': a['complete_generate_wall_seconds'], 'repeat_generate_seconds': b['complete_generate_wall_seconds'],
            'primary_cold_seconds': a['cold_shared_document_total_wall_seconds'], 'repeat_cold_seconds': b['cold_shared_document_total_wall_seconds'],
            'primary_preemptions': a['total_request_preemptions'], 'repeat_preemptions': b['total_request_preemptions'],
            'primary_cap_hits': a['cap_hits'], 'repeat_cap_hits': b['cap_hits']})
    passed = all(checks.values()) and all(all(p['fields_equal'].values()) for p in phases)
    output = ROOT / args.output
    assert not output.exists()
    output.write_text(json.dumps({'status': 'same_inputs_outputs_and_trajectory' if passed else 'reproduction_difference_retained',
        'audit_sha256': sha(Path(__file__)), 'summary_sha256': {str(p.relative_to(ROOT)):sha(p) for p in paths},
        'input_and_source_equality': checks, 'phases': phases,
        'scope': 'Same-input repeated execution stability only; not another unseen document. Original first result remains primary; neither faster nor slower repeat replaces it.'},indent=2)+'\n')
    print(json.dumps({'status': passed, 'output': str(output.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
