"""CPU input audit for the separate 36-layer Instruct model protocol."""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    import transformers
    from transformers import AutoTokenizer
    from scripts.capture_model import make_inputs
    torch.set_num_threads(4)
    output=ROOT/'results/gqa4_input_protocol_cpu_audit_v1.json'
    assert not output.exists()
    path=ROOT/'configs/gqa4_raw_pair_b8_training_v1.json'
    protocol=json.loads(path.read_text())
    model=ROOT/protocol['model_path']
    config=json.loads((model/'config.json').read_text())
    assert protocol['model_revision']=='local_config_sha256_'+sha(model/'config.json')
    assert (config['num_hidden_layers'],config['num_attention_heads']//2,config['num_key_value_heads']//2,config['head_dim'])==(36,16,4,128)
    assert protocol['sampling']['ignore_eos'] is False and protocol['sampling']['max_tokens']==8192
    manifest_path=ROOT/protocol['manifest'];manifest=json.loads(manifest_path.read_text())
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    records=[]
    for item in protocol['documents']:
        group=next(g for g in manifest['groups'] if g['document_id']==item['document_id'])
        prompts,begin,end=make_inputs(tokenizer,group)
        hashes=[hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]
        assert hashes==item['prompt_token_hashes'] and len(set(hashes))==len(prompts)==8
        assert list(map(len,prompts))==item['prompt_lengths'] and [begin,end]==item['document_token_range']
        assert max(map(len,prompts))+8192<=protocol['engine']['max_model_len']
        records.append({'document_id':group['document_id'],'all_prompt_hashes_match':True,
            'questions_in_pilot':8,'natural_question_count':group['natural_question_count'],
            'pilot_contains_all_natural_questions':group['natural_question_count']==8,'full_document_preserved':True})
    assert not torch.cuda.is_initialized()
    inputs=[Path(__file__),path,manifest_path,model/'config.json',model/'tokenizer.json',model/'tokenizer_config.json',ROOT/'scripts/capture_model.py']
    report={'status':'qualified_CPU_inputs_only','python':sys.executable,'transformers_version':transformers.__version__,
        'CUDA_initialized':False,'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
        'geometry_per_TP_rank':[36,16,4,128,16],'documents':records,
        'scope':'Input/config verification only. No GPU model qualification or speed evidence; train7 remains the explicitly labeled eight-of-eleven development subset.'}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
