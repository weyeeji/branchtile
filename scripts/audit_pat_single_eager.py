"""CPU-only audit of qualified PAT single-stream matched eager result."""
import hashlib
import json
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'results/pat_raw_pair_eager_single_v1/train1_r0_l15.json'
out = ROOT / 'results/pat_single_eager_audit_v1.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ratio(blocks, n, d):
    values = [b[n] / b[d] for b in blocks]
    rng = random.Random(20261017)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
    return {'paired_median': statistics.median(values), 'bootstrap_95ci': [boot[49], boot[1949]]}

r = json.loads(path.read_text())
assert r['status'] == 'completed_qualified_eager_only'
assert all(sha(ROOT / p) == h for p, h in r['source_sha256'].items())
assert sha(r['capture']['path']) == r['capture']['sha256']
assert sha(r['build_receipt']['path']) == r['build_receipt']['sha256']
build = json.loads(Path(r['build_receipt']['path']).read_text())
assert sha(build['module']) == build['module_sha256']
assert all(sha(ROOT/'build/pat_cuda13_compat'/p) == h for p,h in build['generated_sha256'].items())
assert r['environment']['DISABLE_STREAM'] == '1'
assert r['PAT_contract']['original_multistream'] is False
checks = [r['checks']['initial'], r['checks']['final']]
ratios = {}
for regime, timing in r['timing'].items():
    assert len(timing['outside_timing_block_checks']) == r['args']['blocks'] == 20
    checks += timing['outside_timing_block_checks']
    ratios[regime] = {}
    for metric in ('default_stream_event', 'synchronized_wall'):
        m = timing[metric]
        for key,n,d in [('AA','native_pdl','native_pdl_AA'),('native_over_PAT','native_pdl','PAT_single_stream'),
                        ('native_over_raw','native_pdl','raw_pair_complete'),('PAT_over_raw','PAT_single_stream','raw_pair_complete')]:
            assert ratio(m['blocks_ms'],n,d) == m[key]
        ratios[regime][metric] = {key: m[key] for key in ('AA','native_over_PAT','native_over_raw','PAT_over_raw')}
for c in checks:
    assert all(c[name]['finite'] and c[name]['FP64']['passed'] for name in ('native_pdl','native_pdl_AA','PAT_single_stream','raw_pair_complete'))
    assert all(c[name]['native_bit_mismatch_count'] == 0 for name in ('native_pdl_AA','raw_pair_complete'))
    assert c['captured_native_O']['native_bit_equal']
trace = json.loads(Path(r['profile_trace']).read_text())
kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
streams = sorted(set(e['args']['stream'] for e in kernels))
assert len(kernels) == 12 and len(streams) == 1
pat_kernels = [e['name'] for e in kernels if 'pat::' in e['name']]
assert len(pat_kernels) == 5 and sum('gather_kernel' in n for n in pat_kernels) == 1
report = {'status':'audited_qualified_single_state_eager_comparison', 'result':str(path.relative_to(ROOT)),
          'result_sha256':sha(path),'sources_and_artifact_match':True,'FP64_checkpoints':len(checks),
          'all_arms_FP64_pass':True,'native_AA_raw_always_bit_exact':True,
          'PAT_native_bit_equality_not_required':True,'same_stream_profile_kernel_count':len(kernels),
          'profiler_stream_ids':streams,'PAT_selected_kernels':pat_kernels,'recomputed_paired_ratios':ratios,
          'setup':r['setup'],
          'limitations':['One real B8 train1 rank0 layer15 state only; no end-to-end or whole-method generalization.',
            'Official DISABLE_STREAM=1 is used because the retained default-path repeated-call result failed.',
            'PAT FP64 correctness does not imply native bit equality or equal natural generated-token trajectory.',
            'Cold AA uncertainty is large; report it and avoid a precise cold-effect conclusion.',
            'Prepared-metadata complete calls exclude separately reported tree setup/upload and online metadata readback.']}
assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'FP64_checkpoints':len(checks)}))
