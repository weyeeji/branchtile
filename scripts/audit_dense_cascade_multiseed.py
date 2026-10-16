"""Audit all prespecified additional seeds without selecting by output length."""
import argparse
import json
from pathlib import Path
import statistics
from audit_dense_cascade_reference_model import audit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    path=Path(args.input)
    raw=json.loads(path.read_text())
    report=audit(path)
    runs={r['phase']:r for r in raw['runs']}
    paired=[]
    for seed,order in zip(raw['seed_protocol']['additional_sampling_seeds'],raw['seed_protocol']['method_orders']):
        n,c=(runs[f'{m}_natural_seed{seed}'] for m in ('native','cascade'))
        assert n['sampling'] == c['sampling'] and n['sampling_seed'] == c['sampling_seed'] == seed
        assert [a['num_cached_tokens'] for a in n['answers']] == [a['num_cached_tokens'] for a in c['answers']]
        paired.append({'seed':seed,'prespecified_order':order,
            'ready_group_native_over_candidate':n['complete_generate_wall_seconds']/c['complete_generate_wall_seconds'],
            'cold_document_native_over_candidate':n['cold_shared_document_total_wall_seconds']/c['cold_shared_document_total_wall_seconds'],
            'ready_native_seconds':n['complete_generate_wall_seconds'],'ready_candidate_seconds':c['complete_generate_wall_seconds'],
            'cold_native_seconds':n['cold_shared_document_total_wall_seconds'],'cold_candidate_seconds':c['cold_shared_document_total_wall_seconds'],
            'native_token_lengths':[len(a['tokens']) for a in n['answers']],
            'candidate_token_lengths':[len(a['tokens']) for a in c['answers']],
            'native_cap_hits':[a['question_index'] for a in n['answers'] if a['finish_reason']=='length'],
            'candidate_cap_hits':[a['question_index'] for a in c['answers'] if a['finish_reason']=='length']})
    report['paired_additional_seeds']=paired
    report['additional_seeds_descriptive']={
        'ready_ratio_of_total_walls':sum(r['ready_native_seconds'] for r in paired)/sum(r['ready_candidate_seconds'] for r in paired),
        'cold_ratio_of_total_walls':sum(r['cold_native_seconds'] for r in paired)/sum(r['cold_candidate_seconds'] for r in paired),
        'ready_median_paired_ratio':statistics.median(r['ready_group_native_over_candidate'] for r in paired),
        'scope':'Additional prespecified seeds only; combine original seed20261016 in final reporting, including negative outcomes. Three added seeds are not a robust generalization claim.'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'paired_additional_seeds':paired,'additional_seeds_descriptive':report['additional_seeds_descriptive']},indent=2))

if __name__=='__main__':
    main()
