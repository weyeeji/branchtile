"""CPU-only independent audit of the four frozen v2 native-capture screens."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def paired(rows, numerator, denominator):
    values=np.array([np.log(row['us'][numerator]/row['us'][denominator]) for row in rows])
    rng=np.random.default_rng(20261017)
    boot=np.median(values[rng.integers(0,len(values),(20000,len(values)))],axis=1)
    return {'median_paired_ratio':float(np.exp(np.median(values))),
            'paired_bootstrap_95_percent_interval':np.exp(np.quantile(boot,[.025,.975])).tolist(),
            'positive_blocks':int((values>0).sum()),'blocks':len(values),
            'all_blocks_retained':True,'bootstrap_unit':'randomized within-block paired timings'}

def main():
    report={'status':'auditing','cases':{},'old_v1_real_cases':'Invalid ABI: int64 page table was reinterpreted by XQA as int32. Synthetic v1 was int32 and is unaffected.'}
    for name in ('train1_bf16_r0_l15','train7_bf16_r0_l47','train1_fp8_r0_l15','train7_fp8_r0_l47'):
        p=ROOT/'results/xqa_pair_padding_native_capture_v2'/f'{name}.json'
        d=json.loads(p.read_text())
        assert d['status']=='bitwise_qualified_and_benchmarked_prefix_only'
        assert all(sha(k)==v for k,v in d['sources'].items())
        assert sha(ROOT/d['args']['capture'])==d['capture_sha256']
        assert d['capture_ABI']['runtime_table_dtype']=='torch.int32' and d['capture_ABI']['page_roundtrip_bytes_exact']
        assert d['capture_ABI']['k_scale']==d['capture_ABI']['v_scale']==1.
        assert d['production_entry_bitwise_match']
        fixed=[r for r in d['checks'] if r['method']=='pair_fixed_split']
        aa=[r for r in d['checks'] if r['method']=='native_AA']
        assert len(fixed)==len(aa)==6 and all(r['bitwise'] and r['finite'] and not r['different_elements'] for r in fixed+aa)
        assert all(not r['bitwise'] for r in d['checks'] if r['method']=='pair_auto_split')
        oracle=d['FP64_reference']['samples']; assert len(oracle)==16
        assert all(s['errors']['native']==s['errors']['pair_fixed_split'] for s in oracle)
        assert all(v['finite'] for s in oracle for v in s['errors'].values())
        grids={}; resources={}
        for method in ('native','pair_fixed_split','pair_auto_split'):
            expected={'native':[10,2,8],'pair_fixed_split':[10,2,4],'pair_auto_split':[21,2,4]}[method]
            trace=Path(d['launch_profiles'][method]['trace'])
            events=json.loads(trace.read_text())['traceEvents']
            kernels=[e for e in events if e.get('cat')=='kernel' and 'kernel_mha' in e.get('name','')]
            assert kernels and all(e['args']['grid']==expected for e in kernels)
            grids[method]=expected
            resources[method]={k:kernels[0]['args'][k] for k in ('registers per thread','shared memory','block')}
        report['cases'][name]={'source_result_sha256':sha(p),'source_hashes_verified':True,
            'ABI':d['capture_ABI'],'six_fixed_split_bitwise_checks':True,'six_auto_split_negative_checks':True,
            'FP64_native_and_fixed_identical_errors':True,
            'FP64_max_absolute_error':max(s['errors']['native']['max_abs'] for s in oracle),
            'FP64_max_relative_l2_error':max(s['errors']['native']['relative_l2'] for s in oracle),
            'median_us':d['median_us'],'native_over_fixed':paired(d['timing_blocks'],'native','pair_fixed_split'),
            'nativeAA_over_fixed':paired(d['timing_blocks'],'native_AA','pair_fixed_split'),
            'native_over_nativeAA':paired(d['timing_blocks'],'native','native_AA'),
            'verified_actual_grids':grids,'resources':resources}
    report['status']='audited'
    report['limitations']=['Prefix-only, not growing private suffix or end-to-end.','Six bounded input states per capture do not prove universal bit equivalence.',
        'Bootstrap captures observed within-run timing variability, not independent-run/device/workload variation.',
        'FP64 is a sampled mathematical reference, not native arithmetic emulation or an independent model quality test.',
        'CUPTI occupancy estimates report impossible zero active blocks despite successful kernels; do not treat them as physical occupancy measurements.',
        'Existing GQA16 public kernel adapter is an opportunity diagnostic, not a novel implementation.']
    output=ROOT/'results/xqa_pair_native_capture_audit_v2/audit.json'; output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{'paired':v['native_over_fixed'],'AA':v['native_over_nativeAA']} for k,v in report['cases'].items()},indent=2))
if __name__=='__main__': main()
