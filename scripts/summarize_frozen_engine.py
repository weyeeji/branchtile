"""CPU gate for exact-mask local execution; no global quality threshold fitted."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_native_calibration import normalized_trace


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',default='results/main30b_sparse/frozen_4096_qualification')
    args=parser.parse_args()
    root=Path(args.directory)
    summary=json.loads((root/'summary.json').read_text())
    audit=json.loads((root/'numerical_audit.json').read_text())
    reference=summary['reference_phase']
    runmap={r['phase']:r for r in summary['runs']}
    rows=[]
    for rank in (0,1):
        base=json.loads((root/f'rank{rank}/{reference}.json').read_text())
        base_trace=normalized_trace(base['forward_trace'])
        base_cached=[a['num_cached_tokens'] for a in runmap[reference]['answers']]
        for phase in runmap:
            path=root/f'rank{rank}/{phase}.json'
            data=json.loads(path.read_text())
            records=data['records']
            probes=[r['independent_fp64_probe'] for r in records if 'independent_fp64_probe' in r]
            rows.append({'rank':rank,'phase':phase,'metadata_file':str(path),'metadata_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'same_forward_trace':normalized_trace(data['forward_trace'])==base_trace,
                'same_cached_token_counts':[a['num_cached_tokens'] for a in runmap[phase]['answers']]==base_cached,
                'same_frozen_mask_file':data['frozen_mask_file_sha256']==base['frozen_mask_file_sha256'],
                'mask_file':data['frozen_mask_file'],'mask_sha256':data['frozen_mask_file_sha256'],
                'layer_question_mask_count':data['selector']['frozen_layer_question_masks'],
                'selector_prepare_seconds':data['selector']['selector_prepare_seconds'],
                'local_same_mask_checks':len(records),
                'all_local_frozen_output_tolerance_passed':bool(records) and all(r['same_mask_reference']['output_tolerance_passed'] for r in records),
                'fp64_probe_count':len(probes),'all_fp64_probes_passed':len(probes)==4 and all(p['candidate']['frozen_output_tolerance_passed'] for p in probes),
                'pure_decode_bypassed_dense_layer_calls':sum(r['pure_decode_bypassed_dense'] for r in records),
                'mixed_prefill_redundant_dense_layer_calls':sum(r['mixed_redundant_dense'] for r in records)})
    conditions=('same_forward_trace','same_cached_token_counts','same_frozen_mask_file','all_local_frozen_output_tolerance_passed','all_fp64_probes_passed')
    aa=audit['methods']['e0_simt_B']['exact_logits']
    keys=all(r['logit_key_sets_equal'] for r in audit['methods'].values())
    qualified=all(all(r[c] for c in conditions) and r['layer_question_mask_count']==48*summary['B'] for r in rows) and aa and keys
    result={'status':'local_execution_qualified' if qualified else 'local_execution_failed',
        'document_id':summary['document_id'],'B':summary['B'],'budget':summary['budget'],'steps':summary['steps_cap'],
        'reference':reference,'AA_logits_exact':aa,'all_method_logit_key_sets_equal':keys,
        'frozen_output_tolerance':{'atol':.006,'rtol':.015},'rows':rows,
        'logit_diagnostics':{phase:{k:r[k] for k in ('max_logit_abs','max_logit_kl','unforced_top1_disagreements','teacher_forced_to_reference')} for phase,r in audit['methods'].items()},
        'scope':'Actual vLLM TP2 exact-mask attention execution, independent native and FP64 local qualifications with matched mask/APC/forward paths.',
        'limitations':['Eight teacher-forced tokens do not establish complete natural generation quality.','No global-logit acceptance threshold is fitted to this candidate.','Python metadata, temporary selector full-K gather, native reference checks and activation snapshots are instrumented costs; these wall times are not application speed measurements.','Pure decode bypass is implemented; mixed prefill still pays original dense work before replacing decode rows.','This engine run uses eager execution. Separately qualified native graph operator replay is not yet fullgraph engine integration.']}
    path=root/'execution_qualification.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'local_checks':sum(r['local_same_mask_checks'] for r in rows),'fp64_probes':sum(r['fp64_probe_count'] for r in rows)}))


if __name__=='__main__':
    main()
