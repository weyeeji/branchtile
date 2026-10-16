"""Score complete natural outputs with the frozen financial metric and report costs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.screen_application import extract_final
from src.financial_answer_metric import score_pairs,METRIC_ID


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):
    if not values:return None
    a=sorted(values)
    return {'count':len(a),'min':a[0],'p10':a[int((len(a)-1)*.1)],'median':statistics.median(a),
        'p90':a[int((len(a)-1)*.9)],'max':a[-1],'mean':statistics.mean(a)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
    directory=Path(args.directory);source=directory/'summary.json';data=json.loads(source.read_text())
    manifest=Path(data['manifest']);assert sha(manifest)==data['manifest_sha256']
    group=next(g for g in json.loads(manifest.read_text())['groups'] if g['document_id']==data['document_id'])
    questions=group['questions'];rows=[];reference_masks={}
    for run in data['runs']:
        predictions=[];answers=[]
        assert len(run['answers'])==len(questions)
        for q,a in zip(questions,run['answers']):
            final,reasoning,parse=extract_final(a['text'],thinking=True)
            eligible=a['finish_reason']=='stop' and parse=='closed_thinking'
            predictions.append(final if eligible else '')
            answers.append({'question_id':q['question_sha256'],'gold':q['answer'],'finish_reason':a['finish_reason'],
                'stop_reason':a['stop_reason'],'output_tokens':len(a['tokens']),'num_cached_tokens':a['num_cached_tokens'],
                'thinking_parse':parse,'final_text':final,'raw_text_sha256':hashlib.sha256(a['text'].encode()).hexdigest(),
                'reasoning_characters':len(reasoning),'completion_eligible':eligible,
                'raw_output':str(source)+'#'+run['phase']+':q'+str(a['question_index'])})
        score=score_pairs(predictions,[q['answer'] for q in questions])
        for a,sc in zip(answers,score.pop('per_example')):a['score']=sc
        workers=[]
        for rank in (0,1):
            path=directory/f'rank{rank}/{run["phase"]}_graph.json';meta_path=directory/f'rank{rank}/{run["phase"]}.json'
            graph=json.loads(path.read_text());meta=json.loads(meta_path.read_text());replays=graph['replays']
            masks=meta['frozen_mask_file_sha256'];reference_masks.setdefault(rank,masks)
            by_batch={}
            for batch in (1,2,3):
                # Consecutive same-B preparation-start intervals contain engine
                # scheduling/model work and metadata; they are not kernel-only time.
                intervals=[b['step_prepare_started_monotonic_seconds']-a['step_prepare_started_monotonic_seconds']
                    for a,b in zip(replays,replays[1:]) if a['batch']==b['batch']==batch]
                by_batch[str(batch)]={'step_interval_ms':stats([v*1000 for v in intervals]),
                    'attention_prepare_host_ms':stats([r['complete_attention_prepare_host_seconds']*1000 for r in replays if r['batch']==batch]),
                    'shared_native_prepare_host_ms':stats([r['metadata_complete_host_seconds']*1000 for r in replays if r['batch']==batch and 'metadata_complete_host_seconds' in r])}
            transitions=[]
            for r in replays:
                state=(r['batch'],tuple(r['questions']))
                if not transitions or transitions[-1]['state']!=list((state[0],list(state[1]))):
                    transitions.append({'state':[state[0],list(state[1])],'positions':r['positions']})
            workers.append({'rank':rank,'raw_graph':str(path),'raw_sha256':sha(path),'graph_replays':len(replays),
                'replay_batch_counts':dict(Counter(r['batch'] for r in replays)),
                'ready_group_transitions':transitions,'same_frozen_mask_hash_across_methods':masks==reference_masks[rank],
                'shadow_check_count':sum(len(r.get('shadow_checks',[])) for r in replays),'logit_copy_count':meta['logits_count'],
                'mixed_redundant_dense_layer_calls':sum(r['mixed_redundant_dense'] for r in meta['records']),
                'by_batch':by_batch,'selector_summary':{k:v for k,v in meta.get('selector',{}).items() if k!='records'}})
        rows.append({'phase':run['phase'],'method':run['method'],'repeat':run['repeat'],
            'complete_generate_wall_seconds':run['complete_generate_wall_seconds'],'dense_anchor_wall_seconds':run['anchor_wall_seconds'],
            'anchor_plus_generate_wall_seconds':run['anchor_wall_seconds']+run['complete_generate_wall_seconds'],
            'all_natural_stop':all(a['finish_reason']=='stop' for a in answers),
            'any_cap_hit':any(a['finish_reason']=='length' for a in answers),'total_output_tokens':run['total_output_tokens'],
            'descriptive_output_tokens_per_wall_second':run['tokens_per_generate_wall_second'],
            'score':score,'answers':answers,'workers':workers})
    wanted={(m,r) for m in data['methods'] for r in range(data['repeats'])}
    actual={(r['method'],r['repeat']) for r in rows}
    result={'status':'completed' if data['status']=='completed' and actual==wanted else 'pending',
        'source':str(source),'source_sha256':sha(source),'document_id':data['document_id'],
        'metric_id':METRIC_ID,'metric_source_sha256':sha(ROOT/'src/financial_answer_metric.py'),
        'thinking_extraction_source_sha256':sha(ROOT/'scripts/screen_application.py'),
        'metric_protocol':'results/metric_calibration_docfinqa.md','independent_documents':1,'questions_per_run':len(questions),
        'pending_method_repeats':sorted(wanted-actual),'rows':rows,
        'budget':data['budget'],'budget_request':data.get('budget_request',str(data['budget'])),
        'one_time_costs':{k:data.get(k) for k in ('initialization_wall_seconds','selector_dense_warmup_wall_seconds','stochastic_sampler_warmup_wall_seconds','graph_capture','production_configuration')},
        'limitations':data['limitations']+['FINAL exact/rounded match is not official execution accuracy or factual reasoning assessment.','Natural trajectories differ. Neither JCT ratio nor token/s ratio isolates executor speed or proves quality noninferiority.',('Stochastic sampler was warmed before all measured arms; warmup cost remains separate.' if 'stochastic_sampler_warmup_wall_seconds' in data else 'First dense run included previously unwarmed stochastic-sampler JIT, as recorded in the job log; no causal method speedup is inferred.'),'CPU preparation-start intervals include model and scheduler work, not synchronized kernel latency; rank intervals are not independent samples.']}
    path=directory/'natural_execution_summary.json';path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':result['status'],'runs':[{'method':r['method'],'natural':r['all_natural_stop'],'tokens':r['total_output_tokens'],'wall':r['complete_generate_wall_seconds'],'exact':r['score']['exact_accuracy'],'rounded':r['score']['rounded_accuracy']} for r in rows]}))


if __name__=='__main__':main()
