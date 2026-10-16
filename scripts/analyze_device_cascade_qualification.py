"""Independent CPU audit of bounded device-metadata qualification evidence."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/dense_cascade_device_qual_v1');p.add_argument('--output',default='results/platform/device_cascade_qualification_v1_audit_20261017.json');a=p.parse_args()
    rows=[]
    for f in sorted((ROOT/a.input).glob('*.json')):
        j=json.loads(f.read_text());assert j['status']=='qualified' and j['all_passed']
        for path,digest in j['sources'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        assert all(x['passed'] for x in j['checks']+j['internal_LSE_checks'])
        assert all(x['valid'] and x['flags']==0 and x['tail_plan_info_unchanged'] and x['host_plan_calls_after_initialization']==0 for x in j['device_error_receipts'])
        for r in j['rejections']:
            receipt=r['receipt'];assert not receipt['valid']
            if 'prefix' in r['case'] or 'shared' in r['case']:assert receipt['prefix_identity_error']
            else:assert receipt['tail_bounds_error']
        batches=list(map(int,j['args']['batches'].split(',')));layers=j['args']['layers']
        tailcases=list(map(int,j['args']['tail_cases'].split(','))) if 'tail_cases' in j['args'] else [1,15,16,17,127,129]
        assert len(j['checks'])==len(batches)*len(tailcases)*layers*2
        assert len(j['internal_LSE_checks'])==len(batches)*len(tailcases)*layers
        assert len(j['device_error_receipts'])==len(batches)*len(tailcases)
        if 'sticky_flag_checks' in j:
            assert len(j['sticky_flag_checks'])==len(batches)
            assert all(x['passed'] for x in j['sticky_flag_checks'])
            assert len(j['host_bind_rejections'])==sum(b>1 for b in batches)
            assert all(x['passed'] for x in j['host_bind_rejections'])
        rows.append({'source':str(f.relative_to(ROOT)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
            'args':j['args'],'cache':j['cache'],'O_checks':len(j['checks']),'LSE_checks':len(j['internal_LSE_checks']),
            'valid_replay_receipts':len(j['device_error_receipts']),'explicit_epoch_binds':len(j['prepare_receipts']),
            'rejected_cases':len(j['rejections']),
            'largest_candidate_O_relative_l2':max(x['relative_l2'] for x in j['checks'] if x['method']=='shared_prefix_cascade'),
            'largest_candidate_O_abs':max(x['max_abs'] for x in j['checks'] if x['method']=='shared_prefix_cascade'),
            'largest_LSE_abs':max(x['max_abs'] for x in j['internal_LSE_checks']),
            'source_hashes_verified':j['sources'],'coverage':j.get('coverage'),
            'sticky_flag_checks':j.get('sticky_flag_checks'),'host_bind_rejections':j.get('host_bind_rejections')})
    assert len(rows)==4
    result={'status':'CPU_independent_qualification_audit','rows':rows,
        'totals':{k:sum(r[k] for r in rows) for k in ('O_checks','LSE_checks','valid_replay_receipts','explicit_epoch_binds','rejected_cases')},
        'limits':['Fixed batch gets its own newly constructed plan and graph; this does not validate a single graph changing its batch shape.',
         'Phase-boundary flags reject outputs after execution. They do not prevent invalid physical cache addresses from being dereferenced.',
         'Position input must be contiguous and all physical IDs allocator-valid; only logical tail bounds and bound-prefix identity are checked.',
         'Output/metadata checks do not establish full-model sampling quality or speedup.']}
    out=ROOT/a.output;out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['totals']))
if __name__=='__main__':main()
