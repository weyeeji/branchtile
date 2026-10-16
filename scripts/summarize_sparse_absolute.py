"""Absolute A4 development metrics, exact mask audit and explicit reuse counts."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from selector_baseline import select_support
from sparse_reference import tensor_hash
from sparse_budget_plan import digest,load_full_sources,resolve_budget,validate_reuse
from calibrate_qmsum_rouge import score_pairs


def distribution(values):
    return {'count':len(values),'minimum':min(values),'median':statistics.median(values),
            'mean':statistics.mean(values),'maximum':max(values),'sum':sum(values)}


def aggregate(rows):
    return {'questions':len(rows),'new_generations':sum(r['new_generation_performed'] for r in rows),
        'reused_rows':sum(not r['new_generation_performed'] for r in rows),
        'outputs_token_identical_to_prior_full':sum(r['token_identical_to_prior_full'] for r in rows),
        'natural_eos':sum(r['natural_eos'] for r in rows),'cap_hits':sum(r['hit_cap'] for r in rows),
        'empty_outputs':sum(not r['prediction'].strip() for r in rows),
        'unexpected_delimiter_outputs':sum(any(t in r['prediction'] for t in ['<think>','</think>','<|im_start|>','<|im_end|>']) for r in rows),
        'macro_f1_percent':{m:100*statistics.mean(r['rouge'][m]['f1'] for r in rows) for m in ['1','2','L']},
        'generated_token_distribution':distribution([r['generated_tokens'] for r in rows]),
        'actual_document_fraction':distribution([r['actual_document_fraction'] for r in rows]),
        'new_reference_cost_seconds':{key:distribution([r[key] for r in rows if r['new_generation_performed'] and key in r])
            for key in ['decode_seconds','gather_seconds','topk_and_support_seconds',
                        'mask_export_seconds','variant_reference_seconds_excluding_final_checkpoint']
            if any(r['new_generation_performed'] and key in r for r in rows)}}


def blind_queue(rows,output):
    new=[r for r in rows if r['new_generation_performed']]
    random.Random(20261016).shuffle(new);queue=[];mapping={}
    for i,row in enumerate(new):
        key=f'A4ABS-{i+1:04d}';source=Path(row['document_path']).resolve()
        queue.append({'item_id':key,'question':row['question'],'gold_reference':row['reference'],'answer':row['prediction'],
            'document_path':str(source),'document_sha256':row['document_sha256'],
            'document_line_start':1,'document_line_end':len(source.read_text().splitlines())})
        mapping[key]={k:row[k] for k in ['group_index','question_index','requested_budget','budget_tokens',
            'seed','model_revision','source_generation_id']}
        mapping[key]['answer_sha256']=hashlib.sha256(row['prediction'].encode()).hexdigest()
    qp=output.with_name(output.stem+'_blind_audit_new56.jsonl')
    mp=output.with_name(output.stem+'_blind_audit_mapping.json')
    qp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in queue))
    mp.write_text(json.dumps({'status':'pending; no verdict assigned','items':mapping},indent=2)+'\n')
    old_queue_path=ROOT/'results/sparse_reference/train8_quality_blind_audit.jsonl'
    old_map_path=ROOT/'results/sparse_reference/train8_quality_blind_audit_mapping.json'
    old_queue={r['item_id']:r for r in map(json.loads,old_queue_path.read_text().splitlines())}
    old_map=json.loads(old_map_path.read_text())['items']
    previous={(r['group_index'],r['question_index']):key for key,r in old_map.items() if r['fraction']==1.}
    links=[]
    for row in rows:
        if row['new_generation_performed']:continue
        old_id=previous[row['group_index'],row['question_index']];old=old_queue[old_id]
        assert row['prediction']==old['answer'] and row['reference']==old['gold_reference']
        assert row['question']==old['question'] and row['document_sha256']==old['document_sha256']
        assert str(Path(row['document_path']).resolve())==old['document_path']
        links.append({k:row[k] for k in ['group_index','question_index','requested_budget','budget_tokens',
                        'source_generation_id','source_path','source_sha256']})
        links[-1].update(prior_item_id=old_id,source_question_gold_answer_identical=True)
    rp=output.with_name(output.stem+'_prior_audit_reuse_mapping.json')
    rp.write_text(json.dumps({'scope':'Identity checks only. No prior verdict read. Root may inherit identical-answer judgments; duplicate nominal rows are not independent samples.',
        'prior_queue_sha256':digest(old_queue_path),'prior_map_sha256':digest(old_map_path),'links':links},indent=2)+'\n')
    return {'status':'pending independent LLM-assisted review, not human evaluation','new_items':len(queue),
        'queue':str(qp),'queue_sha256':digest(qp),'separate_mapping':str(mp),'mapping_sha256':digest(mp),
        'prior_identity_reuse_mapping':str(rp),'prior_identity_reuse_mapping_sha256':digest(rp)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/sparse_reference_absolute/natural_train8.json')
    p.add_argument('--output',default='results/sparse_reference_absolute/train8_quality.json')
    args=p.parse_args();torch.set_num_threads(4)
    data=json.loads(Path(args.input).read_text())
    assert data['status']=='completed' and data['mode']=='natural' and data['budget_mode']=='tokens'
    assert data['requested_budgets']==[4096,8192,16384,32768,'full']
    manifest=json.loads(Path(data['manifest']).read_text());assert digest(data['manifest'])==data['manifest_sha256']
    prior,sources=load_full_sources([s['path'] for s in data['reuse_sources']],data)
    assert sources==data['reuse_sources']
    assert all(json.loads(Path(s['path']).read_text())['torch']==data['torch'] for s in sources)
    identity_path=ROOT/'results/qmsum_identity/qmsum_train_exploratory8_identity.json'
    identities={r['group_index']:r for r in json.loads(identity_path.read_text())['groups']}
    cohorts=collections.defaultdict(list);seen=set();checks=[];shared=[];generated=set()
    for group in data['groups']:
        gi=group['group_index'];gold=manifest['groups'][gi];ds,de=group['document_range']
        assert group['document_sha256']==gold['document_sha256']
        shared.append({'group_index':gi,'dense_prefix_prefill_seconds':group['dense_prefix_prefill_seconds'],
            'questions':[{k:q[k] for k in ['question_index','selector_observer_seconds',
            'question_prefill_including_instrumented_selector_seconds']} for q in group['questions']]})
        for question in group['questions']:
            qi=question['question_index'];key=(gi,qi);assert key not in seen;seen.add(key)
            assert question['status']=='completed' and len(question['variants'])==5
            validate_reuse(prior[key],group,question)
            for variant in question['variants']:
                assert variant['status']=='completed'
                resolved=resolve_budget(variant['requested_budget'],'tokens',de-ds)
                assert all(variant[k]==v for k,v in resolved.items())
                budget=variant['budget_tokens'];positions=variant['absolute_decode_positions']
                assert positions==list(range(question['prompt_tokens'],question['prompt_tokens']+len(positions)))
                assert variant['generated_tokens']==len(variant['generated_ids'])==len(positions)+1
                assert variant['initial_physical_cache_length']==question['prompt_tokens']-(de-ds)+budget
                assert variant['final_physical_cache_length']==variant['initial_physical_cache_length']+len(positions)
                assert all(variant[k] for k in ['all_layers_equal_physical_length','system_prefix_retained',
                    'entire_private_suffix_retained','sorted_original_positions'])
                if variant['new_generation_performed']:
                    assert not resolved['saturated_to_full'] and variant['generation_origin']=='new_generation'
                    assert variant['source_generation_id'] not in generated;generated.add(variant['source_generation_id'])
                    artifact=Path(variant['mask_artifact']);assert digest(artifact)==variant['mask_artifact_sha256']
                    saved=torch.load(artifact,map_location='cpu',weights_only=True)
                    assert saved['budget']==budget and saved['document_range']==[ds,de]
                    assert len(saved['supports'])==36
                    for layer,support in enumerate(saved['supports']):
                        assert support.shape==(8,de-ds) and (support.sum(-1)==budget).all()
                        assert saved['scores'][layer].shape==support.shape and torch.isfinite(saved['scores'][layer]).all()
                        assert torch.equal(support,select_support(saved['scores'][layer],budget))
                        assert tensor_hash(support)==variant['support_sha256'][layer]
                    checks.append({'group_index':gi,'question_index':qi,'budget_tokens':budget,
                        'artifact_sha256_verified':True,'all_36_layer_supports_exact_original_selector_cpu_replay':True,
                        'all_heads_exact_budget':True,'original_absolute_positions_and_private_cache_append':True})
                else:
                    assert resolved['saturated_to_full'] and variant['generation_origin']=='reused_prior_full_budget'
                    assert variant['generated_ids']==prior[key]['variant']['generated_ids']
                    assert variant['generated_text']==prior[key]['variant']['generated_text']
                    assert variant['source_generation_id']==prior[key]['provenance']['source_generation_id']
                    assert 'decode_seconds' not in variant and 'historical_reference_cost_seconds' in variant
                row={k:variant[k] for k in ['requested_budget','budget_tokens','actual_document_fraction',
                    'generation_origin','new_generation_performed','source_generation_id','seed','natural_eos','hit_cap','generated_tokens']}
                for k,v in variant.items():
                    if '_seconds' in k or k in ['source_path','source_sha256']:row[k]=v
                row.update(group_index=gi,question_index=qi,document_path=gold['document_path'],document_sha256=gold['document_sha256'],
                    question=gold['questions'][qi]['question'],reference=gold['questions'][qi]['answer'],prediction=variant['generated_text'],
                    model_revision=data['model_revision'],meeting_id=identities[gi]['meeting_id'],
                    higher_source_cluster=identities[gi]['higher_source_cluster'],
                    token_identical_to_prior_full=variant['generated_ids']==prior[key]['variant']['generated_ids'])
                cohorts[str(variant['requested_budget'])].append(row)
    expected={(gi,qi) for gi,g in enumerate(manifest['groups']) for qi in range(len(g['questions']))}
    assert seen==expected and len(generated)==56 and len(checks)==56
    output=Path(args.output);result={'scope':'Frozen train8 development, one seed, official 4B Instruct; not formal noninferiority, factuality, or main-model proof.',
        'source':args.input,'source_sha256':digest(args.input),'generation':data['generation'],'manifest_sha256':data['manifest_sha256'],
        'budget_semantics':'Each layer and KV head retains min(document_length, requested_tokens); complete system prefix and question/generated suffix retained. 4K=4096.',
        'nominal_rows':160,'new_generations':56,'reused_rows':104,'unique_prior_full_generations':32,
        'unique_generation_sources_total':88,'source_identity_overlay':str(identity_path),'source_identity_sha256':digest(identity_path),
        'unsaturated_documents_per_budget':{'4096':8,'8192':5,'16384':1,'32768':0,'full':0},
        'higher_source_clusters':len({r['higher_source_cluster'] for r in identities.values()}),
        'metric':'Original calibrated Perl ROUGE-1.5.5 author-linked flags; per-question F1 macro percent.',
        'metric_limit':'Published calibration fixture differs by up to 0.011 score points.',
        'new_model_initialization_seconds':data['model_initialization_seconds'],'new_shared_prefill_costs':shared,
        'cost_limit':'HF instrumented reference only, not production serving performance. Historical costs are nested and excluded from new cost aggregates. Selector is part of question prefill; shared per-question costs are counted once, not per budget.',
        'cohorts':{}}
    for budget,rows in cohorts.items():
        rows.sort(key=lambda r:(r['group_index'],r['question_index']))
        assert len(rows)==32 and {(r['group_index'],r['question_index']) for r in rows}==expected
        prefix=output.parent/'perl'/(output.stem+'_budget'+budget)
        measured=score_pairs([r['prediction'] for r in rows],[r['reference'] for r in rows],output_prefix=prefix)
        for row,metric in zip(rows,measured['per_example']):row['rouge']={k:v for k,v in metric.items() if k!='index'}
        result['cohorts'][budget]={**aggregate(rows),'documents':8,'raw_perl_path':str(prefix.with_suffix('.perl.txt')),
            'raw_perl_sha256':digest(prefix.with_suffix('.perl.txt')),'official_bootstrap_average_percent':measured['reported_average_percent'],
            'per_document':{str(gi):aggregate([r for r in rows if r['group_index']==gi]) for gi in range(8)},'per_question':rows}
        print(json.dumps({'budget':budget,'ROUGE':result['cohorts'][budget]['macro_f1_percent']}),flush=True)
    full=result['cohorts']['full'];all_rows=[]
    for budget,cohort in result['cohorts'].items():
        cohort['delta_vs_full_points']={m:cohort['macro_f1_percent'][m]-full['macro_f1_percent'][m] for m in ['1','2','L']}
        for gi,doc in cohort['per_document'].items():
            doc['delta_vs_full_points']={m:doc['macro_f1_percent'][m]-full['per_document'][gi]['macro_f1_percent'][m] for m in ['1','2','L']}
        for row,baseline in zip(cohort['per_question'],full['per_question']):
            row['delta_vs_full_points']={m:100*(row['rouge'][m]['f1']-baseline['rouge'][m]['f1']) for m in ['1','2','L']}
        all_rows.extend(cohort['per_question'])
    result['blind_factual_audit']=blind_queue(all_rows,output)
    fields=['requested_budget','group_index','question_index','meeting_id','higher_source_cluster','budget_tokens',
        'actual_document_fraction','new_generation_performed','generation_origin','source_generation_id',
        'natural_eos','hit_cap','generated_tokens','rouge1','rouge2','rougeL']
    with output.with_suffix('.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in all_rows:
            flat={k:row[k] for k in fields if k in row};flat.update({f'rouge{m}':100*row['rouge'][m]['f1'] for m in ['1','2','L']});writer.writerow(flat)
    events=[json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines() if line.strip()]
    events=[e for e in events if e['name'].startswith('sparse_reference_absolute_')]
    ends=[e for e in events if e['event']=='end'];hours=sum(e['gpu_hours'] for e in ends)
    assert len(ends)==2 and all(e['exit_code']==0 for e in ends) and hours<=.3
    (output.parent/'gpu_ledger.json').write_text(json.dumps({'events':events,'total_gpu_hours':hours,'budget_gpu_hours':.3},indent=2)+'\n')
    (output.parent/'train8_checks.json').write_text(json.dumps({'status':'passed','new_mask_artifacts':56,'checks':checks,
        'reused_full_token_and_text_identity_rows':104,'source_sha256':digest(args.input)},indent=2)+'\n')
    result['total_new_gpu_hours_including_qualification']=hours
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
