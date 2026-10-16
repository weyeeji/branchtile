"""Frozen A2 development summary; compare same-stack A4 without speed claims."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

from calibrate_qmsum_rouge import score_pairs


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    return {'count':len(values),'minimum':min(values),'median':statistics.median(values),
            'mean':statistics.mean(values),'maximum':max(values),'sum':sum(values)}


def aggregate(rows):
    return {'questions':len(rows),'natural_eos':sum(r['natural_eos'] for r in rows),
        'cap_hits':sum(r['hit_cap'] for r in rows),'empty_outputs':sum(not r['prediction'].strip() for r in rows),
        'unexpected_delimiter_outputs':sum(any(t in r['prediction'] for t in ['<think>','</think>','<|im_start|>','<|im_end|>']) for r in rows),
        'macro_f1_percent':{m:100*statistics.mean(r['rouge'][m]['f1'] for r in rows) for m in ['1','2','L']},
        'generated_token_distribution':distribution([r['generated_tokens'] for r in rows]),
        'cost_seconds':{k:distribution([r[k] for r in rows]) for k in ['question_prefill_seconds','decode_seconds',
            'question_mask_prepare_host_seconds','decode_mask_prepare_host_seconds']}}


def blind_queue(rows,output):
    shuffled=rows.copy();random.Random(20261016).shuffle(shuffled);queue=[];mapping={}
    for i,row in enumerate(shuffled):
        key=f'A2-{i+1:04d}';source=Path(row['document_path']).resolve()
        queue.append({'item_id':key,'question':row['question'],'gold_reference':row['reference'],'answer':row['prediction'],
            'document_path':str(source),'document_sha256':row['document_sha256'],
            'document_line_start':1,'document_line_end':len(source.read_text().splitlines())})
        mapping[key]={k:row[k] for k in ['group_index','question_index','fraction','seed','model_revision','source_path']}
        mapping[key]['answer_sha256']=hashlib.sha256(row['prediction'].encode()).hexdigest()
    qp=output.with_name(output.stem+'_blind_audit.jsonl');mp=output.with_name(output.stem+'_blind_audit_mapping.json')
    qp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in queue))
    mp.write_text(json.dumps({'status':'pending; no factual verdict assigned','items':mapping},indent=2)+'\n')
    return {'status':'pending independent LLM-assisted review, not human evaluation','items':len(queue),
        'queue':str(qp),'queue_sha256':sha(qp),'separate_mapping':str(mp),'mapping_sha256':sha(mp),
        'protocol':'Reviewer reads only queue and original sources; do not expose arm mapping before verdicts are frozen.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--a4-summary',default='results/sparse_reference/train8_quality.json')
    args=p.parse_args();data=json.loads(Path(args.input).read_text())
    assert data['status']=='completed' and data['mode']=='natural'
    manifest=json.loads(Path(data['manifest']).read_text());assert sha(data['manifest'])==data['manifest_sha256']
    a4=json.loads(Path(args.a4_summary).read_text());a4_tokens={}
    for source in a4['sources']:
        raw=json.loads(Path(source['path']).read_text())
        assert raw['model_revision']==data['model_revision'] and raw['manifest_sha256']==data['manifest_sha256']
        assert all(raw['generation'][k]==data['generation'][k] for k in ['temperature','top_p','seed','max_new_tokens'])
        for group in raw['groups']:
            for question in group['questions']:
                variant=next(v for v in question['variants'] if v['fraction']==1.)
                a4_tokens[(group['group_index'],question['question_index'])]=(question['input_ids_sha256'],variant['generated_ids'])
    identity_path=Path('results/qmsum_identity/qmsum_train_exploratory8_identity.json')
    identities={r['group_index']:r for r in json.loads(identity_path.read_text())['groups']}
    cohorts=collections.defaultdict(list);seen=set();shared=[];full_checks=[]
    for group in data['groups']:
        gi=group['group_index'];gold=manifest['groups'][gi]
        assert group['status']=='completed' and group['document_sha256']==gold['document_sha256']
        assert len(group['questions'])==group['original_natural_B']==len(gold['questions'])
        shared.append({k:group[k] for k in ['group_index','shared_prefill_tokens','document_range','protection_window',
            'shared_document_prefill_including_gate_seconds','gate_seconds','protection_prepare_seconds','shared_supports']})
        for question in group['questions']:
            qi=question['question_index'];key=(gi,qi)
            assert key not in seen;seen.add(key)
            assert question['status']=='completed' and question['question_sha256']==gold['questions'][qi]['question_sha256']
            assert question['input_ids_sha256']==a4_tokens[key][0]
            for variant in question['variants']:
                assert variant['status']=='completed' and variant['shared_history_reused'] and not variant['physical_compression']
                positions=variant['absolute_decode_positions']
                assert positions==list(range(question['prompt_tokens'],question['prompt_tokens']+len(positions)))
                assert variant['final_physical_cache_length']==variant['initial_physical_cache_length']+len(positions)
                assert variant['generated_tokens']==len(variant['generated_ids'])==len(positions)+1
                support=next(s for s in group['shared_supports'] if s['fraction']==variant['fraction'])
                assert support['support_sha256']==variant['shared_support_sha256']
                if variant['fraction']==1.:
                    full_checks.append({'group_index':gi,'question_index':qi,
                        'a4_fullbudget_token_ids_equal':variant['generated_ids']==a4_tokens[key][1]})
                row={k:variant[k] for k in ['fraction','seed','natural_eos','hit_cap','generated_tokens','question_prefill_seconds',
                    'decode_seconds','question_mask_prepare_host_seconds','decode_mask_prepare_host_seconds']}
                row.update(group_index=gi,question_index=qi,question_sha256=question['question_sha256'],
                    document_path=gold['document_path'],document_sha256=gold['document_sha256'],
                    meeting_id=identities[gi]['meeting_id'],higher_source_cluster=identities[gi]['higher_source_cluster'],
                    question=gold['questions'][qi]['question'],prediction=variant['generated_text'],reference=gold['questions'][qi]['answer'],
                    source_path=args.input,model_revision=data['model_revision'],shared_support_sha256=variant['shared_support_sha256'],
                    actual_document_fraction=support['retained_document_pairs']/support['total_document_pairs'],
                    actual_system_fraction=support['retained_system_pairs']/support['total_system_pairs'])
                cohorts[str(variant['fraction'])].append(row)
    expected={(gi,qi) for gi,g in enumerate(manifest['groups']) for qi in range(len(g['questions']))}
    assert seen==expected,'Frozen train8 matrix incomplete'
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    result={'method':data['method'],'scope':'Frozen train8 development only, single seed; not formal noninferiority or factuality evidence.',
        'source':args.input,'source_sha256':sha(args.input),'manifest_sha256':data['manifest_sha256'],
        'generation':data['generation'],'semantics':data['semantics'],
        'metric':'Original calibrated Perl ROUGE-1.5.5 author-linked flags; per-question F1 macro percent.',
        'calibration':'results/metric_calibration_qmsum.json','metric_limit':'Published fixture differs by up to0.011score points.',
        'source_identity_overlay':str(identity_path),'source_identity_sha256':sha(identity_path),
        'source_clusters':len({r['higher_source_cluster'] for r in identities.values()}),
        'cost_limit':'Instrumented dense-mask HF reference, no physical compression/performance claim. Gate is included in shared document prefill; host mask preparation is included in prefill/decode; decode includes periodic checkpoint writes. Shared costs are counted once per document/budget, not once per question.',
        'initialization_seconds':{k:data[k] for k in ['model_initialization_seconds','gate_initialization_seconds']},
        'shared_document_costs_and_allocations':shared,'cohorts':{},'fullbudget_a4_checks':full_checks,
        'fullbudget_a4_all_token_ids_equal':all(r['a4_fullbudget_token_ids_equal'] for r in full_checks),
        'a4_source':args.a4_summary,'a4_source_sha256':sha(args.a4_summary),
        'a4_comparison_limit':'Same model, sampling and full-budget input identity. Nominal budgets have different denominators: A2 is system+document and global unequal head/layer allocation; A4 is document-only equal per-head counts. Report exact retained fractions; do not claim matched physical memory or application speed.'}
    for fraction,rows in sorted(cohorts.items(),key=lambda x:float(x[0])):
        rows.sort(key=lambda r:(r['group_index'],r['question_index']))
        assert len(rows)==len(expected) and {(r['group_index'],r['question_index']) for r in rows}==expected
        prefix=output.parent/'perl'/(output.stem+'_fraction'+fraction.replace('.','p'))
        measured=score_pairs([r['prediction'] for r in rows],[r['reference'] for r in rows],output_prefix=prefix)
        for row,metric in zip(rows,measured['per_example']):row['rouge']={k:v for k,v in metric.items() if k!='index'}
        result['cohorts'][fraction]={**aggregate(rows),'documents':len(data['groups']),
            'official_bootstrap_average_percent':measured['reported_average_percent'],
            'per_document':{str(gi):aggregate([r for r in rows if r['group_index']==gi]) for gi in range(len(data['groups']))},
            'raw_perl_path':str(prefix.with_suffix('.perl.txt')),'raw_perl_sha256':sha(prefix.with_suffix('.perl.txt')),
            'per_question':rows}
        print(json.dumps({'fraction':fraction,'ROUGE':result['cohorts'][fraction]['macro_f1_percent']}),flush=True)
    full=result['cohorts']['1.0'];all_rows=[]
    for f,cohort in result['cohorts'].items():
        cohort['delta_vs_a2_fullbudget_points']={m:cohort['macro_f1_percent'][m]-full['macro_f1_percent'][m] for m in ['1','2','L']}
        cohort['delta_vs_a4_same_nominal_fraction_points']={m:cohort['macro_f1_percent'][m]-a4['cohorts'][f]['macro_f1_percent'][m] for m in ['1','2','L']}
        for gi,doc in cohort['per_document'].items():
            doc['delta_vs_a2_fullbudget_points']={m:doc['macro_f1_percent'][m]-full['per_document'][gi]['macro_f1_percent'][m] for m in ['1','2','L']}
        for row,baseline in zip(cohort['per_question'],full['per_question']):
            row['delta_vs_a2_fullbudget_points']={m:100*(row['rouge'][m]['f1']-baseline['rouge'][m]['f1']) for m in ['1','2','L']}
        all_rows.extend(cohort['per_question'])
    result['blind_factual_audit']=blind_queue(all_rows,output)
    fields=['fraction','group_index','question_index','meeting_id','higher_source_cluster','natural_eos','hit_cap','generated_tokens',
            'actual_document_fraction','actual_system_fraction','question_prefill_seconds','decode_seconds','rouge1','rouge2','rougeL']
    with output.with_suffix('.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in all_rows:
            flat={k:row[k] for k in fields if k in row};flat.update({f'rouge{m}':100*row['rouge'][m]['f1'] for m in ['1','2','L']});writer.writerow(flat)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
