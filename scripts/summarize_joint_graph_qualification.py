"""Audit frozen variants, same-mask shadows, history restoration and E0 A/A."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
    root=Path(args.directory);data=json.loads((root/'summary.json').read_text())
    assert data['qualification'] and data['status']=='completed'
    expected=['independent_A','epsilon0_A','epsilon001','epsilon01','epsilon0_B']
    assert [r['phase'] for r in data['runs']]==expected
    rows=[];passed=True
    for rank in (0,1):
        frozen=data['frozen_variants'][rank]
        assert frozen['rank']==rank and frozen['all_variants_frozen_before_any_candidate_graph']
        independent=frozen['independent_mask_identity']
        for name,variant in frozen['variants'].items():
            if variant['enabled']:
                passed &= variant['independent_mask_identity']['sha256']==independent
                passed &= variant['layers']==48 and variant['requests']==data['B'] and variant['heads']==2
                passed &= all(q['score_loss_fraction']<=variant['configuration']['epsilon']
                    for l in variant['layer_records'] for h in l['records'] for q in h['requests'])
        base_sequence=None;capture_counts=None
        for run in data['runs']:
            phase=run['phase'];path=root/f'rank{rank}/{phase}_graph.json'
            graph=json.loads(path.read_text());meta=json.loads((root/f'rank{rank}/{phase}.json').read_text())
            replays=graph['replays'];checks=[c for r in replays for c in r['shadow_checks']]
            installed=run['variant_installation'][rank]['installation']
            selector=frozen['variants'][run['variant']]
            expected_mask=(selector['new_mask_identity'] if selector['enabled'] else selector['mask_identity'])['sha256']
            ids={p['new_history_id'] for p in installed['plans']};assert len(ids)==1
            history_id=next(iter(ids))
            sequence=[(r['batch'],r['questions'],r['positions']) for r in replays]
            if base_sequence is None:base_sequence=sequence
            if capture_counts is None:capture_counts=graph['capture_calls']
            mixed=[r['same_mask_reference'] for r in meta['records'] if 'same_mask_reference'in r]
            row={'rank':rank,'phase':phase,'variant':run['variant'],'raw_graph':str(path),
                'raw_graph_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'active_mask_identity':installed['active_mask_identity'],
                'expected_new_mask_matches':installed['active_mask_identity']==expected_mask,
                'same_query_sequence':sequence==base_sequence,'replays':len(replays),'same_mask_shadow_checks':len(checks),
                'all_shadow_checks_passed':bool(checks) and all(c['frozen_output_tolerance_passed'] for c in checks),
                'shadow_max_abs':max(c['max_abs'] for c in checks),
                'no_original_mask_dense_oracle':all('native_full_reference' not in c and c['reference_method']=='e0_simt' for c in checks),
                'mixed_same_mask_checks':len(mixed),'all_mixed_checks_passed':all(c['output_tolerance_passed'] for c in mixed),
                'checked_plan_history_identity_matches':all(r['shared_native_prepare']['history_id']==history_id for r in replays),
                'no_plan_topology_change':all(not r['shared_native_prepare']['plan_topology_changed_requires_recapture'] for r in replays),
                'same_E0_graph_capture_counts':graph['capture_calls']==capture_counts,
                'all_B_graphs_captured_once':all(graph['capture_calls'].get(f'e0:B{b}:L{l}',0)>=2 for b in range(1,data['B']+1) for l in range(48)),
                'capture_buffer_addresses_unchanged':all(p['all_capture_buffer_addresses_unchanged'] for p in installed['plans']),
                'plans_invalidated_after_install':all(p['prepared_false_until_next_checked_native_plan'] for p in installed['plans']),
                'install_complete_wall_seconds':installed['install_complete_wall_seconds'],
                'history_H2D_bytes':sum(p['history_H2D_bytes'] for p in installed['plans']),
                'pure_decode_bypasses_dense':all(not r['dense_attention_executed_for_pure_decode'] for r in replays)}
            gate=['expected_new_mask_matches','same_query_sequence','all_shadow_checks_passed','no_original_mask_dense_oracle',
                'all_mixed_checks_passed','checked_plan_history_identity_matches','no_plan_topology_change',
                'same_E0_graph_capture_counts','all_B_graphs_captured_once','capture_buffer_addresses_unchanged',
                'plans_invalidated_after_install','pure_decode_bypasses_dense']
            passed &= all(row[k] for k in gate) and len(replays)==data['steps_cap']-1 and len(checks)==48*len(replays)
            rows.append(row)
    before=torch.load(root/'rank0/epsilon0_A.pt',map_location='cpu',weights_only=True)
    after=torch.load(root/'rank0/epsilon0_B.pt',map_location='cpu',weights_only=True)
    keys=set(before)==set(after) and len(before)==data['B']*data['steps_cap']
    exact=keys and all(torch.equal(before[k],after[k]) for k in before)
    passed &= exact
    runs={r['phase']:r for r in data['runs']}
    same_apc=all([a['num_cached_tokens'] for a in r['answers']]==[a['num_cached_tokens'] for a in data['runs'][0]['answers']] for r in data['runs'])
    same_teacher=all([a['tokens'] for a in r['answers']]==[a['tokens'] for a in data['runs'][0]['answers']] for r in data['runs'])
    passed &= same_apc and same_teacher
    result={'status':'joint_graph_execution_qualified' if passed else 'joint_graph_execution_failed',
        'source_summary_sha256':hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
        'same_APC_hits':same_apc,'forced_input_trajectory_matches':same_teacher,
        'epsilon0_restore_AA_logits_exact':exact,'complete_logit_keys':keys,
        'frozen_local_output_tolerance':{'atol':.006,'rtol':.015},'rows':rows,
        'total_same_new_mask_shadow_checks':sum(r['same_mask_shadow_checks'] for r in rows),
        'scope':'All masks frozen before any candidate output; same E0 graph/buffers, explicit equal-length installs and checked plan identity; per-variant same-mask SIMT references.',
        'limitations':['Only four teacher-forced steps; no natural task-quality or global-logit equivalence contract.',
            'Pure B3 replay here; B1/B2 captured and existing equal-length/long-tail fixtures remain separate evidence.',
            'Classical coverage control required before system superiority claims; no novel selection/execution algorithm claim.']}
    (root/'joint_graph_qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':result['total_same_new_mask_shadow_checks'],'restore_AA_exact':exact}))


if __name__=='__main__':main()
