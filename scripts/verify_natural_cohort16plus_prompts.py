"""Independently reproduce all selected prompts in the actual engine virtualenv."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    import transformers
    from transformers import AutoTokenizer
    from scripts.capture_model import make_inputs
    torch.set_num_threads(4)
    output=ROOT/'results/natural_cohort16plus_feasibility_v1/engine_tokenizer_verification.json'
    assert not output.exists()
    protocol_path=ROOT/'configs/native_natural_cohort16plus_training_v1.json'
    protocol=json.loads(protocol_path.read_text())
    manifest_path=ROOT/protocol['manifest']
    manifest=json.loads(manifest_path.read_text())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/protocol['model_path'],local_files_only=True)
    result={'status':'qualified_CPU_only','python':sys.executable,'transformers_version':transformers.__version__,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),protocol_path,manifest_path,ROOT/'scripts/capture_model.py']},'documents':[]}
    for group,item in zip(manifest['groups'],protocol['documents'],strict=True):
        prompts,begin,end=make_inputs(tokenizer,group)
        hashes=[hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]
        assert hashes==item['prompt_token_hashes'] and len(hashes)==len(set(hashes))==group['natural_question_count']
        assert list(map(len,prompts))==item['prompt_lengths'] and [begin,end]==item['document_token_range']
        result['documents'].append({'document_id':group['document_id'],'B':len(prompts),
            'all_prompt_hashes_match':True,'all_questions_unique':True,'original_document_complete':True})
    assert not torch.cuda.is_initialized()
    result['CUDA_initialized']=False
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
