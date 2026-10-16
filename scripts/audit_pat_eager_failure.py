"""CPU-only independent audit of the failed first PAT complete-call comparison."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'results/pat_raw_pair_eager_v1/train1_r0_l15.json'
OUT = ROOT / 'results/pat_eager_failure_audit_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


r = json.loads(RESULT.read_text())
assert r['status'] == 'failed'
checks = {}
checks['runner_sources_match'] = all(sha(ROOT / p) == h for p, h in r['source_sha256'].items())
checks['capture_matches'] = sha(r['capture']['path']) == r['capture']['sha256']
br = r['build_receipt']
checks['original_failed_build_receipt_matches'] = sha(br['path']) == br['sha256']
b = json.loads(Path(br['path']).read_text())
checks['artifact_matches'] = sha(b['module']) == b['module_sha256']
checks['original_source_matches'] = all(sha(ROOT / 'build/pat_prior' / p) == h for p, h in b['source_sha256'].items())
checks['compiled_source_matches'] = all(sha(ROOT / 'build/pat_cuda13_compat' / p) == h for p, h in b['generated_sha256'].items())
checks['build_guard_failure_disclosed'] = (br['original_status'] == 'failed' and br['CPU_only_guard_failure_preserved'] is True)
checks['same_cache_alias_and_legal_tiles'] = (r['PAT_contract']['K_V_alias_original_cache'] is True and
    len(r['PAT_contract']['MNWs']) == 10 and [16, 128, 1] not in r['PAT_contract']['MNWs'])
checks['default_stream_multistream'] = r['environment']['stream'] == 0 and r['PAT_contract']['original_multistream'] is True
checks['initial_all_FP64_pass'] = all(r['checks']['initial'][n]['FP64']['passed'] for n in ('native_pdl', 'native_pdl_AA', 'PAT_default', 'raw_pair_complete'))
checks['final_native_AA_raw_exact_and_FP64_pass'] = all(
    r['checks']['final'][n]['FP64']['passed'] and r['checks']['final'][n]['native_bit_mismatch_count'] == 0
    for n in ('native_pdl', 'native_pdl_AA', 'raw_pair_complete'))
checks['final_PAT_failed'] = not r['checks']['final']['PAT_default']['FP64']['passed']
checks['captured_native_output_exact'] = r['checks']['final']['captured_native_O']['native_bit_equal'] is True
checks['paired_eager_dual_metric_complete'] = all(
    len(r['timing'][regime][metric]['blocks_ms']) == r['args']['blocks']
    for regime in ('hot', 'cold') for metric in ('default_stream_event', 'synchronized_wall'))
assert all(checks.values()), checks
report = {'status': 'audited_failed_numerical_qualification', 'result': str(RESULT.relative_to(ROOT)),
          'result_sha256': sha(RESULT), 'checks': checks,
          'initial_PAT_FP64': r['checks']['initial']['PAT_default']['FP64'],
          'final_PAT_FP64': r['checks']['final']['PAT_default']['FP64'],
          'performance_evidence_valid': False,
          'interpretation': 'Initial output passed; repeated-call final output failed. Keep all timing as invalid diagnostic data, not a superiority result. Cause is unresolved.',
          'next_test': 'Same frozen artifact, official default versus DISABLE_STREAM=1; synchronize/check each repeated call. No numerical/source patch.',
          'scope': 'CPU provenance and saved-result audit; no GPU execution.'}
assert not OUT.exists()
OUT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'checks': len(checks)}))
