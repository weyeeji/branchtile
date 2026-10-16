"""CPU artifact/budget/identity verification after the complete A2 cohort."""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sparse_reference import tensor_hash


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    directory=ROOT/'results/fastkvzip_reference'
    source=directory/'natural_train8.json';data=json.loads(source.read_text())
    assert data['status']=='completed' and data['mode']=='natural'
    manifest=json.loads(Path(data['manifest']).read_text());assert sha(data['manifest'])==data['manifest_sha256']
    assert len(data['groups'])==len(manifest['groups'])==8
    checks=[];outputs=0
    for group in data['groups']:
        ds,de=group['document_range'];window=group['protection_window']
        for record in group['shared_supports']:
            artifact=Path(record['mask_artifact']);assert sha(artifact)==record['mask_artifact_sha256']
            loaded=torch.load(artifact,map_location='cpu',weights_only=True)
            keep,scores,raw=(loaded[k] for k in ['supports','scores','raw_scores'])
            assert keep.shape==(36,8,de) and scores.shape==raw.shape==(36,1,8,de)
            assert torch.isfinite(scores).all() and torch.isfinite(raw).all()
            assert tensor_hash(keep)==record['support_sha256']
            assert tensor_hash(scores)==group['score_sha256'] and tensor_hash(raw)==group['raw_score_sha256']
            assert int(keep.sum())==keep.numel()-int(keep.numel()*(1-record['fraction']))
            assert keep.sum(-1).tolist()==record['layer_head_counts']
            assert keep[...,:4].all() and keep[...,-window:].all()
            assert int(keep[...,:ds].sum())==record['retained_system_pairs']
            assert int(keep[...,ds:de].sum())==record['retained_document_pairs']
            queries=[next(v for v in q['variants'] if v['fraction']==record['fraction']) for q in group['questions']]
            assert len(queries)==group['original_natural_B']==4
            assert all(v['shared_support_sha256']==record['support_sha256'] for v in queries)
            checks.append({'group_index':group['group_index'],'fraction':record['fraction'],
                'source_artifact_sha256_verified':True,'global_budget_and_all_head_counts_verified':True,
                'native_protected_positions_retained':True,'same_support_reused_for_every_question':True,
                'finite_raw_and_protected_gate_scores':True})
        for question in group['questions']:
            assert question['status']=='completed' and len(question['variants'])==4
            for variant in question['variants']:
                assert variant['status']=='completed';outputs+=1
                positions=variant['absolute_decode_positions']
                assert positions==list(range(question['prompt_tokens'],question['prompt_tokens']+len(positions)))
                assert variant['final_physical_cache_length']==question['prompt_tokens']+len(positions)
    events=[json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines() if line.strip()]
    events=[e for e in events if e['name'].startswith('fastkvzip_nvidia_global_mask_')]
    ends=[e for e in events if e['event']=='end'];hours=sum(e['gpu_hours'] for e in ends)
    assert len(ends)==2 and all(e['exit_code']==0 for e in ends) and hours<=.4
    ledger={'events':events,'total_gpu_hours':hours,'budget_gpu_hours':.4,'all_allocated_time_including_qualification':True}
    (directory/'gpu_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    report={'status':'passed','outputs':outputs,'shared_supports':len(checks),'checks':checks,
        'source_path':str(source),'source_sha256':sha(source),'total_gpu_hours':hours,
        'scope':'Artifact and semantic verification; no factuality, noninferiority, or production-performance verdict.'}
    (directory/'train8_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    # Compare only source queues and mappings; never open reviewer verdicts.
    a2queue=directory/'train8_quality_blind_audit.jsonl'
    a2map=directory/'train8_quality_blind_audit_mapping.json'
    a4queue=ROOT/'results/sparse_reference/train8_quality_blind_audit.jsonl'
    a4map=ROOT/'results/sparse_reference/train8_quality_blind_audit_mapping.json'
    queues=[{r['item_id']:r for r in map(json.loads,p.read_text().splitlines())} for p in [a2queue,a4queue]]
    maps=[json.loads(p.read_text())['items'] for p in [a2map,a4map]]
    previous={(r['group_index'],r['question_index']):key for key,r in maps[1].items() if r['fraction']==1.}
    links=[];new=[]
    for key,record in maps[0].items():
        if record['fraction']!=1.:
            new.append(queues[0][key]);continue
        old=previous[record['group_index'],record['question_index']]
        first,second=queues[0][key],queues[1][old]
        for field in ['document_path','document_sha256','question','gold_reference','answer']:
            assert first[field]==second[field],f'Prior audit source/answer mismatch: {field}'
        links.append({'new_item_id':key,'prior_item_id':old,'group_index':record['group_index'],
            'question_index':record['question_index'],'answer_sha256':sha_text(first['answer']),
            'full_source_question_gold_answer_identical':True})
    assert len(new)==96 and len(links)==32
    remaining=directory/'train8_quality_blind_audit_new96.jsonl'
    remaining.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new))
    reuse={'scope':'Identity-only mapping; no verdict read or assigned. Root may inherit prior judgments for identical full-budget answers, not new independent samples.',
        'prior_queue':str(a4queue),'prior_queue_sha256':sha(a4queue),'prior_map_sha256':sha(a4map),
        'current_queue_sha256':sha(a2queue),'current_map_sha256':sha(a2map),
        'new_review_queue':str(remaining),'new_review_queue_sha256':sha(remaining),'links':links}
    (directory/'fullbudget_prior_audit_reuse_mapping.json').write_text(json.dumps(reuse,indent=2)+'\n')
    print(json.dumps({'status':'passed','outputs':outputs,'support_artifacts':len(checks),'gpu_hours':hours}))


def sha_text(value):return hashlib.sha256(value.encode()).hexdigest()


if __name__=='__main__':main()
