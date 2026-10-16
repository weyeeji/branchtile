"""CPU audit of completed page16 native-XQA/cascade qualifications."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows=[]
    for name in a.inputs:
        path=Path(name);r=json.loads(path.read_text());assert r['status']=='qualified' and r['all_passed']
        assert all(x['passed'] for x in r['checks']) and all(x['passed'] for x in r['internal_LSE_checks'])
        batches=list(map(int,r['args']['batches'].split(',')));expected={(m,b,t,l) for m in ('native_xqa_O_only','shared_prefix_cascade') for b in batches for t in (1,15,16,17,127,129) for l in (0,1)}
        assert {(x['method'],x['batch'],x['tail'],x['layer']) for x in r['checks']}==expected
        assert len(r['checks'])==len(expected) and len(r['internal_LSE_checks'])==len(expected)//2
        assert {x['batch'] for x in r['rejections']}=={b for b in batches if b>1}
        for source,digest in r['sources'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
        rows.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dtype':r['args']['dtype'],
            'nonunit_scales':r['args']['nonunit_scales'],'prefix_tokens':r['args']['prefix_tokens'],'O_checks':len(r['checks']),
            'LSE_checks':len(r['internal_LSE_checks']),'recaptures':len(r['recaptures']),'shared_page_divergence_rejections':len(r['rejections']),
            'max_O_absolute_error':max(x['max_abs'] for x in r['checks']),'max_LSE_absolute_error':max(x['max_abs'] for x in r['internal_LSE_checks']),
            'all_source_hashes_current':True})
    result={'status':'independent_cpu_coverage_audit_passed','rows':rows,'total_O_checks':sum(x['O_checks'] for x in rows),
        'total_LSE_checks':sum(x['LSE_checks'] for x in rows),'scope':'Synthetic values with actual fused page16 stride ABI. B shrink uses separately captured fixed-B plans, not illegal shape-changing replay. No performance or novel-mechanism claim.'}
    out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
