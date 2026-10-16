"""Unified four-seed wall/quality audit including original positive and negative runs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_dense_cascade_model import audit as audit_original
from scripts.audit_dense_cascade_reference_model import audit as audit_reference
from scripts.screen_application import extract_final
from src.financial_answer_metric import METRIC_ID,score_pairs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def production_ast(path):
    class StripQualification(ast.NodeTransformer):
        def visit_If(self,node):
            if ast.unparse(node.test)=="state['validation']":
                node.body=[ast.Pass()]
            return self.generic_visit(node)
    tree=ast.parse(Path(path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    return {n.name:ast.dump(StripQualification().visit(n),include_attributes=False)
            for n in cls.body if isinstance(n,ast.FunctionDef)}


def score_run(run,questions):
    predictions,details=[],[]
    for i,(answer,question) in enumerate(zip(run['answers'],questions)):
        assert answer['question_index']==i
        final,_,parse=extract_final(answer['text'],thinking=True)
        eligible=answer['finish_reason']=='stop' and parse=='closed_thinking'
        predictions.append(final if eligible else '')
        details.append({'question_index':i,'final':final,'gold':question['answer'],'parse':parse,
            'eligible':eligible,'finish_reason':answer['finish_reason'],'output_tokens':len(answer['tokens']),
            'text_sha256':hashlib.sha256(answer['text'].encode()).hexdigest()})
    score=score_pairs(predictions,[q['answer'] for q in questions])
    for detail,item in zip(details,score.pop('per_example')):
        detail['score']=item
    return score,details


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--additional-root',default='results/dense_cascade_multiseed_v1')
    p.add_argument('--original-fp8-root',default='results/dense_cascade_natural_v1')
    p.add_argument('--original-bf16-root',default='results/dense_cascade_reference_natural_v1')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    destination=Path(args.output)
    assert not destination.exists()
    old,new=(ROOT/'scripts/vllm_dense_cascade_worker.py',ROOT/'scripts/vllm_dense_cascade_reference_worker.py')
    old_ast,new_ast=production_ast(old),production_ast(new)
    assert old_ast==new_ast,'Old/new worker method bodies differ outside qualification'
    ledger=[json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()]
    def terminal_for(source):
        matches=[]
        for row in ledger:
            argv=row.get('argv',[])
            if row.get('event')=='end' and '--output' in argv and (ROOT/argv[argv.index('--output')+1]).resolve()==source.parent.resolve():
                matches.append(row)
        assert len(matches)==1 and matches[0]['exit_code']==0 and not matches[0]['timed_out'], (str(source),matches)
        return matches[0]
    output={'status':'completed','metric_id':METRIC_ID,'metric_source_sha256':sha(ROOT/'src/financial_answer_metric.py'),
        'extraction_source_sha256':sha(ROOT/'scripts/screen_application.py'),
        'worker_methods_AST_equal_after_removing_validation_branch':True,
        'worker_original_sha256':sha(old),'worker_reference_sha256':sha(new),'conditions':[],
        'limits':['Two independent documents; four seeds are repeated stochastic outcomes, not new documents.',
                  'Retains original seed20261016 and all caps/parse failures/negative timings; no length-based filtering.',
                  'Natural request wall includes changed tokens/MoE routes and is not isolated algorithm speedup.',
                  'Frozen FINAL scoring is not official program execution accuracy; original prompt and label limitations retained.',
                  'Complete cold cost is anchor+generation, excluding shared engine initialization. Candidate graph setup is reported separately.',
                  'Source worker production method bodies match after stripping only numerical qualification; this is a source audit, not formal semantic proof.']}
    for doc in ('train1','train7'):
        for dtype in ('bf16','fp8'):
            condition=f'{doc}_{dtype}'
            original_path=Path(args.original_bf16_root if dtype=='bf16' else args.original_fp8_root)/condition/'summary.json'
            extra_path=Path(args.additional_root)/condition/'summary.json'
            orig,extra=(json.loads(path.read_text()) for path in (original_path,extra_path))
            original_audit=(audit_reference if dtype=='bf16' else audit_original)(original_path)
            extra_audit=audit_reference(extra_path)
            assert orig['args']['document_id']==extra['args']['document_id']
            assert orig['prompt_token_hashes']==extra['prompt_token_hashes']
            assert orig['protocol']==extra['protocol']
            # FP8 original uses original gate worker, reference version changes qualification only.
            engine_orig={k:v for k,v in orig['engine'].items() if k!='worker_extension_cls'}
            engine_extra={k:v for k,v in extra['engine'].items() if k!='worker_extension_cls'}
            assert engine_orig==engine_extra
            seed_protocol=extra['seed_protocol']
            expected_seeds=[seed_protocol['already_measured_sampling_seed'],*seed_protocol['additional_sampling_seeds']]
            assert len(expected_seeds)==len(set(expected_seeds))==4
            group=next(g for g in json.loads((ROOT/extra['protocol']['manifest']).read_text())['groups'] if g['document_id']==extra['args']['document_id'])
            assert group['document_sha256']==orig['document_sha256']==extra['document_sha256']
            buckets={}
            for source,data in ((original_path,orig),(extra_path,extra)):
                for run in data['runs']:
                    if run['natural_eos_enabled']:
                        key=(run['sampling']['seed'],run['method'])
                        assert key not in buckets
                        buckets[key]=(source,run)
            assert set(buckets)=={(seed,method) for seed in expected_seeds for method in ('native','cascade')}
            item={'condition':condition,'document_id':extra['args']['document_id'],
                'manifest_sha256':sha(ROOT/extra['protocol']['manifest']),
                'sources':[{'path':str(p),'sha256':sha(p)} for p in (original_path,extra_path)],
                'source_and_trajectory_audits_passed':True,'pairs':[],
                'clean_exit_ledger_receipts':[terminal_for(p) for p in (original_path,extra_path)],
                'candidate_setup_seconds':{'original_session':orig['candidate_plan_and_capture_wall_seconds'],'additional_session':extra['candidate_plan_and_capture_wall_seconds']}}
            for seed in expected_seeds:
                arms={}
                for method in ('native','cascade'):
                    source,run=buckets[seed,method]
                    score,details=score_run(run,group['questions'])
                    assert len(details)==8
                    arms[method]={'source':str(source),'phase':run['phase'],'sampling':run['sampling'],
                        'ready_seconds':run['complete_generate_wall_seconds'],'cold_seconds':run['cold_shared_document_total_wall_seconds'],
                        'anchor_seconds':run['anchor_wall_seconds'],'total_output_tokens':run['total_output_tokens'],
                        'EOS':sum(a['finish_reason']=='stop' for a in run['answers']),
                        'cap_hits':sum(a['finish_reason']=='length' for a in run['answers']),
                        'APC_hits':[a['num_cached_tokens'] for a in run['answers']],'score':score,'answers':details}
                n,c=arms['native'],arms['cascade']
                assert n['sampling']==c['sampling'] and n['APC_hits']==c['APC_hits']
                item['pairs'].append({'seed':seed,'arms':arms,'ready_native_over_candidate':n['ready_seconds']/c['ready_seconds'],
                    'cold_native_over_candidate':n['cold_seconds']/c['cold_seconds'],
                    'exact_accuracy_delta_candidate_minus_native':c['score']['exact_accuracy']-n['score']['exact_accuracy'],
                    'rounded_accuracy_delta_candidate_minus_native':c['score']['rounded_accuracy']-n['score']['rounded_accuracy']})
            totals={m:{key:sum(pair['arms'][m][key] for pair in item['pairs']) for key in ('ready_seconds','cold_seconds','total_output_tokens','EOS','cap_hits')} for m in ('native','cascade')}
            item['four_seed_totals']=totals
            item['four_seed_ratios']={key:totals['native'][key]/totals['cascade'][key] for key in ('ready_seconds','cold_seconds')}
            item['four_seed_quality']={m:{metric:statistics.mean(pair['arms'][m]['score'][metric] for pair in item['pairs']) for metric in ('exact_accuracy','rounded_accuracy')} for m in ('native','cascade')}
            item['ready_faster_seed_count']=sum(pair['ready_native_over_candidate']>1 for pair in item['pairs'])
            item['cold_faster_seed_count']=sum(pair['cold_native_over_candidate']>1 for pair in item['pairs'])
            item['actual_two_sessions_candidate_setup_total_seconds']=sum(item['candidate_setup_seconds'].values())
            output['conditions'].append(item)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:c[k] for k in ('condition','four_seed_ratios','four_seed_quality','ready_faster_seed_count','cold_faster_seed_count')} for c in output['conditions']],indent=2))

if __name__=='__main__':
    main()
