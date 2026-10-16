"""Tokenize the complete eleven-question training train7 input; never truncate questions or document."""
import copy
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.prepare_natural_cohort16plus import file_sha,sha,save


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    import transformers
    from transformers import AutoTokenizer
    from scripts.capture_model import make_inputs
    torch.set_num_threads(4)
    manifest_path=ROOT/'data/manifests/docfinqa_training_train7_complete11_v1.json'
    protocol_path=ROOT/'configs/native_natural_train7_complete11_v1.json'
    audit_path=ROOT/'results/training_train7_complete11_feasibility_v1/audit.json'
    base_path=ROOT/'configs/native_opportunity_b8_training_v1.json'
    assert not protocol_path.exists() and not audit_path.exists()
    manifest=json.loads(manifest_path.read_text())
    groups=manifest['groups']
    rule=manifest['selection_rule']
    assert len(groups)==1
    source_path=ROOT/manifest['source_manifest']
    assert file_sha(source_path)==manifest['source_manifest_sha256']
    source=json.loads(source_path.read_text())
    expected=[(i,g['document_id']) for i,g in enumerate(source['groups']) if g['document_id']=='docfinqa-f1d5c3003725d29c']
    assert expected==[(g['original_training_group_index'],g['document_id']) for g in groups]
    for name,digest in manifest['protected_source_sha256'].items():
        assert file_sha(ROOT/name)==digest
    base=json.loads(base_path.read_text())
    model=ROOT/base['model_path']
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    config=json.loads((model/'config.json').read_text())
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
                    'payload_GiB_per_rank':slots*bytes_per_slot/2**30})
            estimates.append({'kv_dtype':dtype,'bytes_per_KV_token_per_rank':bytes_per_slot,'scenarios':scenarios})
        records.append(dict(item,feasibility='context_fits_payload_only_gpu_unverified' if fits else 'FAILED_context_limit_no_replacement',
            prompt_plus_output_cap_tokens=max(lengths)+base['sampling']['max_tokens'],
            configured_max_model_len=base['engine']['max_model_len'],model_max_position_embeddings=config['max_position_embeddings'],
            memory_estimates=estimates))
    engine=copy.deepcopy(base['engine'])
    engine['max_num_seqs']=max(g['B'] for g in groups)
    engine['compilation_config']['cudagraph_capture_sizes']=list(range(1,engine['max_num_seqs']+1))
    protocol={'schema':1,'protocol_id':'native-natural-training-train7-complete11-v1-20261017',
        'status':'training_CPU_prepared_not_GPU_qualified','selection_rule':rule,
        'model_path':base['model_path'],'model_revision':base['model_revision'],
        'manifest':str(manifest_path.relative_to(ROOT)),'documents':documents,'engine':engine,
        'seed':base['seed'],'sampling':base['sampling'],'source_audit':str(audit_path.relative_to(ROOT)),
        'limitations':['Retain ALL natural question counts. Do not crop, pad with duplicate requests, or replace failures.',
            'Cold anchor and natural EOS workload; B1 through actual B must be measured, including all setup and any fallbacks.',
            'CPU payload estimates exclude graph/workspace/weights/allocator/prefill transients and cannot guarantee admission.',
            'Question multiplicity does not establish real-world concurrent arrival or sufficient cohort lifetime.',
            'Any context/memory failure remains a failure of the preselected document; no replacement or document truncation.']}
    assert not torch.cuda.is_initialized()
    inputs=[source_path,manifest_path,base_path,model/'config.json',model/'tokenizer.json',model/'tokenizer_config.json',ROOT/'scripts/capture_model.py',Path(__file__)]
    report={'status':'completed_CPU_only_no_model_outputs_read','uid':os.geteuid(),'python':sys.executable,
        'transformers_version':transformers.__version__,'CUDA_initialized':False,
        'selection_rule':rule,'source_sha256':{str(p.relative_to(ROOT)):file_sha(p) for p in inputs},
        'selected_manifest_sha256':file_sha(manifest_path),'protected_sources_unchanged':True,
        'selected_documents_retained':len(records),'memory_caveat':'Payload arithmetic only; excludes weights, graph/workspace, allocator and prefill transients. No existing model outputs or measured GPU pool consulted.',
        'records':records}
    for name,digest in manifest['protected_source_sha256'].items():
        assert file_sha(ROOT/name)==digest
    save(protocol_path,protocol)
    save(audit_path,report)
    print(json.dumps([{'id':r['document_id'],'index':r['original_training_group_index'],'B':r['B'],
        'prompt_range':[min(r['prompt_lengths']),max(r['prompt_lengths'])],'feasibility':r['feasibility'],
        'worst_BF16_payload_GiB_per_rank':r['memory_estimates'][0]['scenarios'][-1]['payload_GiB_per_rank']} for r in records],indent=2))


if __name__=='__main__':
    main()
