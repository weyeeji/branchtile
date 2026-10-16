"""Independent completeness/source/bitwise-receipt audit of frozen raw-pair gate."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
folder = ROOT / 'results/xqa_raw_pair_gpu_v1'
expected = ['synthetic'] + [f'train{d}_r{r}_l{l}' for d in (1, 7)
                            for r in (0, 1) for l in (0, 15, 31, 47)]
required = ['native_pdl_bits', 'paired_bits', 'native_AA_10_bits',
            'native_FP64_pass', 'paired_FP64_pass', 'inputs_not_mutated',
            'graph_initial_bits', 'graph_changed_input_bits',
            'graph_swapped_pair_bits', 'invalid_prefix_rejected']
rows = []
for name in expected:
    path = folder / (name + '.json')
    report = json.loads(path.read_text())
    assert report['status'] == 'qualified_pair_only', name
    for source, sha in report['source_sha256'].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == sha, source
    assert len(report['checks']) == (4 if name == 'synthetic' else 1)
    for check in report['checks']:
        assert all(check[k] is True for k in required), (name, check)
        assert check['different_bit_values'] == 0 and check['maxabs'] == 0
        if name != 'synthetic':
            assert check['captured_original_O_pass'] is True
    if name != 'synthetic':
        capture = report['capture']
        assert hashlib.sha256((ROOT / capture['path']).read_bytes()).hexdigest() == capture['sha256']
        assert capture['payload_cast_performed'] is False
    rows.append({'name': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'case_count': len(report['checks']), 'passed': True})
out = ROOT / 'results/xqa_raw_pair_gate_audit_v1.json'
out.write_text(json.dumps({'status': 'qualified_operator_correctness_only',
    'scope': '16 actual BF16 initial decode states plus four synthetic cases; full private tails; no performance or model claim',
    'bitwise_definition': 'contiguous int16 views, including signed zero',
    'real_cases': 16, 'synthetic_cases': 4, 'required_boolean_checks': required,
    'all_current_sources_and_real_capture_hashes_match': True, 'rows': rows}, indent=2) + '\n')
print(out)
