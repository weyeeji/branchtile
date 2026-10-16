"""CPU-only, outcome-blind selection of complete natural training cohorts."""
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()


def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def save(path,value):
    assert not path.exists(),path
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    from scripts.capture_model import make_inputs
    torch.set_num_threads(4)
    manifest_path=ROOT/'data/manifests/docfinqa_training_natural_cohort16plus_v1.json'
    protocol_path=ROOT/'configs/native_natural_cohort16plus_training_v1.json'
    audit_path=ROOT/'results/natural_cohort16plus_feasibility_v1/audit.json'
    assert not any(p.exists() for p in (manifest_path,protocol_path,audit_path))
    source_path=ROOT/'data/manifests/docfinqa_public_training167_v1.json'
    source=json.loads(source_path.read_text())
    assert len(source['groups'])==167
    selected=[(i,g) for i,g in enumerate(source['groups']) if g['natural_question_count']>=16][:2]
    assert len(selected)==2
    protected=[source_path,ROOT/'data/manifests/docfinqa_public_development8_v1.json',
               ROOT/'data/manifests/docfinqa_public_evaluation100_v1.json',ROOT/'configs/native_opportunity_b8_training_v1.json']
    before={str(p.relative_to(ROOT)):file_sha(p) for p in protected}
    heldout={g['document_sha256'] for p in protected[1:3] for g in json.loads(p.read_text())['groups']}
    assert all(g['document_sha256'] not in heldout for _,g in selected)
    questions={g['document_sha256']:{} for _,g in selected}
    expected=json.loads((ROOT/'results/data_docfinqa_parquet_validation.json').read_text())
    parquet_hashes={}
    splits=['train','validation','test']
    for split in splits:
        raw=ROOT/f'data/raw/docfinqa/parquet/{split}.parquet'
        actual=file_sha(raw)
        reference=next(r for r in expected['sources'] if r['path']==str(raw.relative_to(ROOT)))
        assert actual==reference['sha256']
        parquet_hashes[str(raw.relative_to(ROOT))]=actual
        row_index=0
        for batch in pq.ParquetFile(raw).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                context_hash=sha(row['Context'])
                if context_hash in questions:
                    qh=sha(row['Question'])
                    item=questions[context_hash].setdefault(qh,dict(question=row['Question'],question_sha256=qh,source_occurrences=[]))
                    item['source_occurrences'].append(dict(split=split,row=row_index,answer=row['Answer'],program_sha256=sha(row['Program'])))
                row_index+=1
    groups=[]
    for index,original in selected:
        natural=questions[original['document_sha256']]
        assert len(natural)==original['natural_question_count']
        chosen=[natural[key] for key in sorted(natural)]
        assert [q['question_sha256'] for q in chosen[:4]]==[q['question_sha256'] for q in original['questions']]
        for q in chosen:
            primary=min(q['source_occurrences'],key=lambda x:(splits.index(x['split']),x['row']))
            variants=sorted({x['answer'] for x in q['source_occurrences']})
            q.update(answer=primary['answer'],answer_source=dict(split=primary['split'],row=primary['row']),
                     source_answer_variants=variants,conflicting_source_answers=len(variants)>1)
        assert file_sha(ROOT/original['document_path'])==original['document_sha256']
        groups.append(dict(original,questions=chosen,B=len(chosen),original_training_group_index=index))
    rule='First two entries in frozen training167 order with natural_question_count>=16. Retain ALL exact-context deduplicated natural questions sorted by question SHA and the entire original document. No length, answer, output, quality, or performance selection. Infeasible selections remain failures; never substitute later entries.'
    manifest={'schema':1,'scope':'Exploratory complete natural training cohorts; no arrival-concurrency or speedup claim.',
              'source_manifest':str(source_path.relative_to(ROOT)),'source_manifest_sha256':file_sha(source_path),
              'selection_rule':rule,'source_parquet_sha256':parquet_hashes,'heldout_overlap':False,
              'protected_source_sha256':before,'groups':groups}
    # Freeze selected documents/questions before tokenization or feasibility is observed.
    save(manifest_path,manifest)
    base=json.loads(protected[3].read_text())
    model=ROOT/base['model_path']
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    config=json.loads((model/'config.json').read_text())
    pool_path=ROOT/'results/dense_capture_b8_train1_bf16_v1/summary.json'
    pool=json.loads(pool_path.read_text())
    assert pool['status']=='completed'
    measured_pool=min(r['allocated_cache_token_capacity'] for r in pool['native_receipts'])
    records=[]
    documents=[]
    for group in groups:
        prompts,begin,end=make_inputs(tokenizer,group)
        assert len(prompts)==len(group['questions'])==group['natural_question_count']
        common=0
        for tokens in zip(*prompts):
            if len(set(tokens))!=1:
                break
            common+=1
        common=common//16*16
        lengths=list(map(len,prompts))
        item={'document_id':group['document_id'],'document_sha256':group['document_sha256'],
              'original_training_group_index':group['original_training_group_index'],'B':len(prompts),
              'prompt_lengths':lengths,'prompt_token_hashes':[sha(json.dumps(p)) for p in prompts],
              'document_token_range':[begin,end],'common_complete_page_tokens':common}
        documents.append(item)
        fits=max(lengths)+base['sampling']['max_tokens']<=base['engine']['max_model_len']
        estimates=[]
        for dtype,bytes_per_element in [('bf16',2),('fp8_e4m3',1)]:
            bytes_per_slot=config['num_hidden_layers']*(config['num_key_value_heads']//2)*config['head_dim']*2*bytes_per_element
            scenarios=[]
            for cap in (128,256,2048,8192):
                private=sum(((n-common+cap+15)//16)*16 for n in lengths)
                slots=common+private
                scenarios.append({'output_cap_each':cap,'shared_prefix_token_slots':common,
                    'private_token_slots_all_questions':private,'total_KV_token_slots':slots,
                    'payload_GiB_per_rank':slots*bytes_per_slot/2**30,
                    'fraction_of_measured_B8_pool_bytes':slots*bytes_per_slot/(measured_pool*49152),
                    'fits_measured_B8_pool_payload_bytes_only':slots*bytes_per_slot<=measured_pool*49152})
            estimates.append({'kv_dtype':dtype,'bytes_per_KV_token_per_rank':bytes_per_slot,'scenarios':scenarios})
        records.append(dict(item,feasibility='context_fits_payload_only_gpu_unverified' if fits else 'FAILED_context_limit_no_replacement',
            prompt_plus_output_cap_tokens=max(lengths)+base['sampling']['max_tokens'],
            configured_max_model_len=base['engine']['max_model_len'],model_max_position_embeddings=config['max_position_embeddings'],
            memory_estimates=estimates))
    engine=copy.deepcopy(base['engine'])
    engine['max_num_seqs']=max(g['B'] for g in groups)
    engine['compilation_config']['cudagraph_capture_sizes']=list(range(1,engine['max_num_seqs']+1))
    protocol={'schema':1,'protocol_id':'native-natural-cohort16plus-training-v1-20261017',
        'status':'CPU_prepared_not_GPU_qualified','selection_rule':rule,
        'model_path':base['model_path'],'model_revision':base['model_revision'],
        'manifest':str(manifest_path.relative_to(ROOT)),'documents':documents,'engine':engine,
        'seed':base['seed'],'sampling':base['sampling'],'source_audit':str(audit_path.relative_to(ROOT)),
        'limitations':['Retain B21/B19 natural question counts; do not crop to B16 or replicate requests.',
            'Cold anchor and natural EOS workload; B1 through actual B must be measured, including all setup and any fallbacks.',
            'CPU payload estimates exclude graph/workspace/weights/allocator/prefill transients and cannot guarantee admission.',
            'Question multiplicity does not establish real-world concurrent arrival or sufficient cohort lifetime.',
            'Any context/memory failure remains a failure of the preselected document; no replacement or document truncation.']}
    assert before=={str(p.relative_to(ROOT)):file_sha(p) for p in protected}
    assert not torch.cuda.is_initialized()
    inputs=[source_path,*protected[1:],model/'config.json',model/'tokenizer.json',model/'tokenizer_config.json',
            ROOT/'scripts/capture_model.py',Path(__file__),pool_path]
    report={'status':'completed_CPU_only','uid':os.geteuid(),'python':sys.executable,'CUDA_initialized':False,
        'selection_rule':rule,'source_sha256':{str(p.relative_to(ROOT)):file_sha(p) for p in inputs},
        'selected_manifest_sha256':file_sha(manifest_path),'protected_sources_unchanged':True,
        'natural_question_count_histogram':dict(collections.Counter(g['natural_question_count'] for g in source['groups'])),
        'number_of_training_documents_with_at_least_16_questions':sum(g['natural_question_count']>=16 for g in source['groups']),
        'measured_B8_BF16_pool_token_capacity_per_rank':measured_pool,
        'measured_B8_BF16_pool_bytes_per_rank':measured_pool*49152,
        'pool_caveat':'Prior measured B8 pool is a comparison reference, not available-memory proof for B19/B21 graphs. No GPU was initialized.',
        'records':records}
    save(protocol_path,protocol)
    save(audit_path,report)
    print(json.dumps({'manifest':str(manifest_path.relative_to(ROOT)),'protocol':str(protocol_path.relative_to(ROOT)),
        'selections':[{'id':r['document_id'],'index':r['original_training_group_index'],'B':r['B'],
            'prompt_range':[min(r['prompt_lengths']),max(r['prompt_lengths'])],'feasibility':r['feasibility'],
            'worst_BF16_payload_GiB_per_rank':r['memory_estimates'][0]['scenarios'][-1]['payload_GiB_per_rank']} for r in records]},indent=2))


if __name__=='__main__':
    main()
