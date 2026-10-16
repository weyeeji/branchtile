"""CPU-only result audit for PAT fork/join, device cut and GQA4 quad gates."""
import hashlib
import json
from pathlib import Path
import random
import statistics
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_gate(r):
    assert all(sha(ROOT/p)==h for p,h in r['source_sha256'].items())
def ratio(blocks,n,d):
    values=[b[n]/b[d] for b in blocks];rng=random.Random(20261017)
    boot=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(2000))
    return {'paired_median_ratio':statistics.median(values),'bootstrap_95ci':[boot[49],boot[1949]]}

pat={}
for mode in ['default','nondefault']:
    p=ROOT/f'results/pat_stream_compat_qual_v1/{mode}.json';r=json.loads(p.read_text());source_gate(r)
    assert r['status']=='completed_repeatability_diagnostic' and r['PAT_numerical_pass'] is False
    assert r['first_failure']==0 and len(r['iterations'])==30
    assert r['checks']['initial']['PAT_default']['FP64']['passed']
    assert all(x['FP64']['passed'] is False and x['FP64']['failed_values']==4238 for x in r['iterations'])
    assert (r['environment']['stream']==0)==(mode=='default')
    assert sha(r['build_receipt']['path'])==r['build_receipt']['sha256']
    for phase in ['initial','final']:
        assert all(r['checks'][phase][n]['FP64']['passed'] and r['checks'][phase][n]['native_bit_mismatch_count']==0 for n in ['native_pdl','native_pdl_AA','raw_pair_complete'])
    pat[mode]={'path':str(p.relative_to(ROOT)),'sha256':sha(p),'failed_repeats':30,'first_failure':0,
               'final_FP64':r['checks']['final']['PAT_default']['FP64']}

p=ROOT/'results/xqa_raw_dynamic_cut_qual_v1.json';r=json.loads(p.read_text());source_gate(r)
assert r['status']=='qualified' and len(r['cases'])==70
assert set(c['B'] for c in r['cases'])=={6,7,8,9,14,19,21}
assert all(c['same_graph'] and c['native_AA_bits'] and c['native_candidate_bits'] and c['max_abs']==0 for c in r['cases'])
assert sum('FP64_pass' in c for c in r['cases'])==21
assert all(c.get('FP64_pass',True) is True for c in r['cases'])
assert len(r['negative_controls'])==7 and all(c['wrong_external_cut_rejected'] and c['invalid_update_unchanged'] for c in r['negative_controls'])
dynamic={'path':str(p.relative_to(ROOT)),'sha256':sha(p),'cases':70,'FP64_selected_row_checks':21,'negative_control_batches':7,
         'scope':'Standalone synthetic graph/ABI gate; no resolution of actual eval2 trajectory failure.'}
quad=[]
paths=sorted((ROOT/'results/gqa4_quad_operator_v1').glob('*.json'));assert len(paths)==10
for p in paths:
    r=json.loads(p.read_text());source_gate(r)
    assert r['status']=='qualified_and_measured' and all(v is True for v in r['checks'].values())
    assert sha(r['capture']['path'])==r['capture']['sha256']
    row={'path':str(p.relative_to(ROOT)),'sha256':sha(p),'B':r['batch'],'extra_tail':r['args']['extra_tail'],'native_S':r['S'],
         'same_native_original_capture_layout':r['args']['extra_tail']==0,
         'all_reported_qualification_checks_pass':True}
    for regime in ['hot','cold']:
        data=r[regime]
        for field,n,d in [('native_over_candidate','native','candidate'),('AA','native','AA'),('native_over_unshared','native','unshared_checkpoint')]:
            assert ratio(data['blocks_ms'],n,d)==data[field]
        if 'pair_over_quad' in data:assert ratio(data['blocks_ms'],'pair','candidate')==data['pair_over_quad']
        row[regime]={k:data[k] for k in ['native_over_candidate','AA','pair_over_quad'] if k in data}
    quad.append(row)
report={'status':'audited_three_independent_operator_gates','PAT_explicit_fork_join':pat,
        'PAT_verdict':'Explicit ready/done fork-join and current-stream gather alone do not repair the repeated-call failure. Preserve negative result; no compatible-multistream timing may be claimed.',
        'dynamic_cut':dynamic,'quad':quad,
        'quad_limits':['One real captured layer/state, B subsets are reordered actual Q not natural later survivor activations.',
                       '8192 added tails and remapped layout are synthetic stress, not captured natural continuations.',
                       'No whole-model or natural task acceleration established by these ten cases.',
                       'Quad over already-qualified pair increment is small; original native speedup mostly belongs to the shared-state mechanism, not the new four-member variant.']}
out=ROOT/'results/pat_dynamic_quad_independent_audit_v1.json';assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'PAT_failed_repeats':60,'dynamic_cases':70,'quad_cells':10}))
