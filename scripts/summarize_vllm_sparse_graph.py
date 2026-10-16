"""Audit actual graph capture/replay and full Q/cache shadow output evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
    root=Path(args.directory);data=json.loads((root/'summary.json').read_text())
    reference=data['reference_phase'];runs={r['phase']:r for r in data['runs']}
    rows=[];passed=True
    ref_logits=torch.load(root/f'rank0/{reference}.pt',map_location='cpu',weights_only=True)
    diagnostics={};no_shadow_invariance={}
    for phase,run in runs.items():
        actual=torch.load(root/f'rank0/{phase}.pt',map_location='cpu',weights_only=True)
        keys=set(actual)==set(ref_logits);delta=[]
        for key in sorted(set(actual)&set(ref_logits)):
            a,b=actual[key].double(),ref_logits[key].double();la,lb=a.log_softmax(0),b.log_softmax(0)
            delta.append({'key':key,'max_abs':float((a-b).abs().max()),'forward_kl':float((lb.exp()*(lb-la)).sum()),'unforced_top1_equal':int(a.argmax())==int(b.argmax())})
        diagnostics[phase]={'key_sets_equal':keys,'rows':delta,'max_abs':max(r['max_abs'] for r in delta),
            'max_forward_kl':max(r['forward_kl'] for r in delta),'unforced_top1_disagreements':sum(not r['unforced_top1_equal'] for r in delta),
            'teacher_forced_to_reference':run['teacher_forced_to_reference']}
        passed &= keys and len(delta)==data['B']*data['steps_cap']
        if run.get('shadow_validation_enabled') is False:
            source='union_graph' if run['method']=='union' else reference
            with_shadow=torch.load(root/f'rank0/{source}.pt',map_location='cpu',weights_only=True)
            invariant=set(with_shadow)==set(actual) and all(torch.equal(actual[k],with_shadow[k]) for k in actual)
            no_shadow_invariance[phase]={'reference_phase':source,'all_pre_forcing_logits_exact':invariant,
                'scope':'Attention shadows and layer snapshots disabled; logits still copied for this equivalence check, so this is not uninstrumented throughput.'}
            passed &= invariant
    for rank in (0,1):
        base=json.loads((root/f'rank{rank}/{reference}.json').read_text())
        base_graph=json.loads((root/f'rank{rank}/{reference}_graph.json').read_text())
        base_sequence=[(r['batch'],r['questions'],r['positions']) for r in base_graph['replays']]
        for phase,run in runs.items():
            path=root/f'rank{rank}/{phase}_graph.json';graph=json.loads(path.read_text())
            meta=json.loads((root/f'rank{rank}/{phase}.json').read_text());replays=graph['replays']
            shadow=run.get('shadow_validation_enabled',True)
            checks=[c for r in replays for c in r.get('shadow_checks',[])]
            method=run['method'];capture=graph['capture_calls']
            capture_complete=all(capture.get(f'{method}:B{b}:L{l}',0)>=2 for b in range(1,data['B']+1) for l in range(48))
            row={'rank':rank,'phase':phase,'metadata_file':str(path),'metadata_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'capture_candidate_calls_complete_B1_B2_B3_all48layers':capture_complete,
                'actual_graph_replays':len(replays),'actual_replay_batches':[r['batch'] for r in replays],
                'same_ready_query_sequence':[(r['batch'],r['questions'],r['positions']) for r in replays]==base_sequence,
                'same_frozen_mask_hash':meta['frozen_mask_file_sha256']==base['frozen_mask_file_sha256'],
                'same_APC_hits':[a['num_cached_tokens'] for a in run['answers']]==[a['num_cached_tokens'] for a in runs[reference]['answers']],
                'shadow_validation_enabled':shadow,
                'shadow_checks':len(checks),'all_shadow_output_checks_passed':bool(checks) and all(c['frozen_output_tolerance_passed'] for c in checks),
                'shadow_max_abs_output_difference':max((c['max_abs'] for c in checks),default=None),
                'positive_actual_query_norm':bool(checks) and all(c['query_l2']>0 for c in checks),
                'layers_with_output_norm_changed_across_replays':sum(len({r['shadow_checks'][l]['output_l2'] for r in replays})>1 for l in range(48)) if shadow else None,
                'dense_attention_pure_decode_calls':sum(r['dense_attention_executed_for_pure_decode'] for r in replays),
                'median_metadata_complete_host_ms':1000*statistics.median(r['metadata_complete_host_seconds'] for r in replays),
                'scope':'Preparation includes required metadata work. Shadow validation is additional work included in application wall time. Host graph submit time is not GPU execution time.'}
            native_checks=[c['native_full_reference'] for c in checks if 'native_full_reference' in c]
            if native_checks:
                row['native_full_same_query_checks']={
                    'count':len(native_checks),'expected_count':len(checks),
                    'all_frozen_output_checks_passed':all(c['frozen_output_tolerance_passed'] for c in native_checks),
                    'max_abs':max(c['max_abs'] for c in native_checks),
                    'same_actual_query_cache_full_history':all(c['same_actual_query_cache_full_history'] for c in native_checks)}
                passed &= len(native_checks)==len(checks) and row['native_full_same_query_checks']['all_frozen_output_checks_passed']
            if all('shared_native_prepare' in r for r in replays):
                row['shared_native_plan']={
                    'one_plan_call_per_step':all(r['shared_native_prepare']['native_plan_calls']==1 for r in replays),
                    'layers_sharing_plan':sorted({r['shared_native_prepare']['layers_sharing_plan'] for r in replays}),
                    'median_gpu_map_submit_host_ms':1000*statistics.median(r['shared_native_prepare']['gpu_map_submit_host_seconds'] for r in replays),
                    'median_native_plan_complete_host_ms':1000*statistics.median(r['shared_native_prepare']['native_plan_complete_host_seconds'] for r in replays),
                    'median_extra_shadow_metadata_host_ms':1000*statistics.median(r['extra_shadow_metadata_host_seconds'] for r in replays),
                    'scope':'One native length plan shared by all layers, with one GPU kernel mapping all persistent indices; existing engineering control, not a novel method. GPU map time is host submission only; all GPU work remains in end-to-end wall time.'}
            rows.append(row)
            passed &= all(row[k] for k in ('capture_candidate_calls_complete_B1_B2_B3_all48layers','same_ready_query_sequence','same_frozen_mask_hash','same_APC_hits'))
            passed &= (row['all_shadow_output_checks_passed'] and row['positive_actual_query_norm']) if shadow else len(checks)==0
            passed &= len(replays)==data['steps_cap']-1 and len(checks)==(48*len(replays) if shadow else 0) and row['dense_attention_pure_decode_calls']==0
    aa=diagnostics['e0_graph_B']['max_abs']==0;passed &= aa
    shared=all('shared_native_plan' in r for r in rows)
    included=all(c.get('memory_includes_shared_workspace_and_histories',False) for c in data['graph_capture'])
    report={'status':'graph_execution_qualified' if passed else 'graph_execution_failed','AA_logits_exact':aa,
        'frozen_output_tolerance':{'atol':.006,'rtol':.015},'rows':rows,'logit_diagnostics':diagnostics,'no_shadow_invariance':no_shadow_invariance,
        'capture':data['graph_capture'],'memory_reporting_correction':('capture additional_torch_allocated_bytes includes shared 256MiB workspace, histories and persistent metadata; total model/cache/retained original graph memory is separate.' if included else 'capture additional_torch_allocated_bytes excludes the preallocated shared 256MiB float workspace and tiny dummy Q buffers; include those separately in complete memory accounting.'),
        'scope':'Actual new vLLM FULL graphs contain candidate attention, and each replay uses changed actual queries/real cache with same-mask independent native shadows.',
        'limitations':['Eight teacher-forced steps are not complete natural generation quality or an accepted global-logit contract.','No uninstrumented application throughput or speedup is reported.',('Shared native planning removes per-layer Python rebuild from the main path. Shadow validation and logit copying add measured work, and their disablement must be explicit.' if shared else '48-layer Python metadata construction dominates this prototype; next control must reuse native plans across layers and update indices on GPU.'),'All B1/B2/B3 graphs were captured; this run only exercises B3 replay. B3 to B1 to B3 and prefix remapping boundary coverage remain separate required tests.',f"The qualifier uses compile mode {data['engine_configuration']['compilation_config']['mode']}, FULL graphs and 0.80 KV memory utilization. Final performance must match the strongest native production configuration/resources."]}
    path=root/'graph_execution_qualification.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'AA_exact':aa,'shadow_checks':sum(r['shadow_checks'] for r in rows),'metadata_medians_ms':[r['median_metadata_complete_host_ms'] for r in rows]}))


if __name__=='__main__':main()
