"""Describe independent native calibration without fitting a candidate tolerance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/main30b_sparse'


def normalized_trace(rows):
    return [{'query_start_loc':r['query_start_loc'],'seq_lens':r['seq_lens'],
             'num_actual_tokens':r['num_actual_tokens'],
             'num_computed_tokens_before_forward':r['num_computed_tokens_before_forward'],
             'requests':[{k:v for k,v in q.items() if k not in ('request_id','slot')} for q in r['requests']]} for r in rows]


def main():
    records=[]
    for name in ('native_calibration_train0','native_calibration_train1','native_calibration_train1_matched_apc'):
        directory=BASE/name
        path=directory/'summary.json'
        auditpath=directory/'numerical_audit.json'
        if not path.exists() or not auditpath.exists():
            records.append({'directory':str(directory.relative_to(ROOT)),'status':'pending'})
            continue
        data,audit=json.loads(path.read_text()),json.loads(auditpath.read_text())
        phases={r['phase']:r for r in data['runs']}
        item={'directory':str(directory.relative_to(ROOT)),'status':data['status'],
              'document_id':data['document_id'],'document_sha256':data['document_sha256'],'B':data['B'],
              'prompt_lengths':data['prompt_lengths'],'prefix_policy':data.get('prefix_policy','not recorded'),
              'files_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,auditpath)},
              'methods':{},'cache_and_forward_path':{}}
        for rank in (0,1):
            ref=json.loads((directory/f'rank{rank}/dense_A.json').read_text())
            rt=normalized_trace(ref.get('forward_trace',[]))
            item['cache_and_forward_path'][str(rank)]={}
            for phase in phases:
                meta=json.loads((directory/f'rank{rank}/{phase}.json').read_text())
                trace=normalized_trace(meta.get('forward_trace',[]))
                item['cache_and_forward_path'][str(rank)][phase]={
                    'num_cached_tokens':[a.get('num_cached_tokens') for a in phases[phase]['answers']],
                    'trace_recorded':bool(trace),'trace_equal_dense_A':trace==rt if trace and rt else None,
                    'normalized_trace_sha256':hashlib.sha256(json.dumps(trace,sort_keys=True).encode()).hexdigest() if trace else None,
                    'first_forward':trace[0] if trace else None}
        for phase,method in audit['methods'].items():
            probes=[p for rank in method['ranks'].values() for p in rank['independent_fp64_probes']]
            item['methods'][phase]={'logit_rows':len(method['logits']),'max_logit_abs':method['max_logit_abs'],
                'max_forward_kl':method['max_logit_kl'],'unforced_top1_disagreements':method['unforced_top1_disagreements'],
                'exact_logits':method['exact_logits'],'key_sets_equal':method['logit_key_sets_equal'],
                'teacher_forced_token_equality_is_not_natural_greedy_evidence':method['teacher_forced_to_reference'],
                'local_output_checks':sum(r['local_kernel_checks'] for r in method['ranks'].values()),
                'all_local_output_checks_passed':all(r['local_kernel_all_frozen_output_tolerance_passed'] is True for r in method['ranks'].values()) if probes else None,
                'independent_fp64_probe_count':len(probes),
                'all_native_and_FA_fp64_probes_passed':all(p['candidate']['frozen_output_tolerance_passed'] and p['dense']['frozen_output_tolerance_passed'] for p in probes) if probes else None}
        item['AA_exact']=item['methods']['dense_B']['exact_logits']
        reference_cached_tokens=[a.get('num_cached_tokens') for a in phases['dense_A']['answers']]
        item['fully_matched_cache_forward_path']=all(
            all(r['trace_recorded'] and r['trace_equal_dense_A'] and r['num_cached_tokens']==reference_cached_tokens for r in byphase.values())
            for byphase in item['cache_and_forward_path'].values())
        item['interpretation']='Backend-only attribution requires matched APC/forward path and A/A. Raw ranges remain diagnostics; no logit threshold is fit here.'
        records.append(item)
    result={'scope':'Independent native full-history numerical calibration. Different layers/heads/tokens do not increase independent report count.',
            'frozen_local_output_tolerance':{'atol':.006,'rtol':.015},'new_logit_acceptance_threshold':None,
            'records':records,'limitations':['Teacher forcing measures logits under equal token history, not natural task quality.','Initial train1 retained-cache run had A/A drift and is not used for backend-only attribution.','Graph operator qualification is separate from real vLLM fullgraph integration and complete preparation costs.']}
    path=BASE/'native_calibration_summary.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({r['directory']:{'status':r['status'],'AA_exact':r.get('AA_exact'),'matched':r.get('fully_matched_cache_forward_path')} for r in records}))


if __name__=='__main__':
    main()
