"""Separate layer-output and final-logit gates in a frozen model qualification."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.input);summary=json.loads((root/'summary.json').read_text());records=[];sources={}
    for path in sorted(root.glob('rank*/*_natural_*_graph.json')):
        data=json.loads(path.read_text());checks=[c for row in data['replays'] for c in row.get('shadow_checks',[])]
        records.append({'rank':data['rank'],'method':data['method'],'checks':len(checks),'layers':sorted(set(x['layer'] for x in checks)),
                        'all_passed':bool(checks) and all(x['frozen_output_tolerance_passed'] for x in checks),
                        'max_abs':max((x['max_abs'] for x in checks),default=None)})
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    sources[str(root/'summary.json')]=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()
    logits=summary['qualification_same_teacher_path_logits'];threshold=logits['cosine_threshold']
    result={'status':summary['status'],'output_shadow_checks':sum(x['checks'] for x in records),
            'all_recorded_output_shadows_passed':all(x['all_passed'] for x in records),'output_records':records,
            'logit_gate_passed':logits['passed'],'logit_cosine_threshold':threshold,
            'logit_failures':[x for x in logits['checks'] if not x['finite'] or x['cosine']<threshold],
            'all_logit_top1_equal':all(x['top1_equal'] for x in logits['checks']),
            'all_logits_finite':all(x['finite'] for x in logits['checks']),
            'minimum_logit_cosine':min(x['cosine'] for x in logits['checks']),
            'runner_label_erratum':'qualification_all_shadow_checks_passed and the assertion text combine output and logit gates; consult these separate independently reconstructed fields.',
            'source_sha256':sources,'performance_evidence':False,'gate_relaxed':False}
    out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
