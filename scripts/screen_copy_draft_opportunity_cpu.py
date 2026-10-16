"""Causal token-copy replay screen, not speculative-sampling acceptance or speed."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def replay(prompt, output, width=4, n=4):
    history=list(prompt); index={}; next_start=0
    rounds=matched=drafted=lookups=0; pos=0; histogram={}
    while pos<len(output):
        # Only candidates already fully present in the causal history are indexed.
        while next_start+n+width<=len(history):
            index[tuple(history[next_start:next_start+n])]=next_start+n
            next_start+=1
        start=index.get(tuple(history[-n:]))
        proposal=history[start:start+width] if start is not None else []
        accepted=0
        for a,b in zip(proposal,output[pos:]):
            if a!=b:break
            accepted+=1
        advance=min(accepted+1,len(output)-pos)
        rounds+=1; matched+=accepted; drafted+=len(proposal);lookups+=bool(proposal)
        histogram[str(accepted)]=histogram.get(str(accepted),0)+1
        history.extend(output[pos:pos+advance]);pos+=advance
    return {'tokens':len(output),'replay_rounds':rounds,'matched_proposal_tokens':matched,
            'proposed_tokens':drafted,'rounds_with_history_match':lookups,'matched_length_histogram':histogram,
            'tokens_per_replay_round':len(output)/rounds}

def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
    import torch
    from transformers import AutoTokenizer
    from scripts.capture_model import make_inputs
    torch.set_num_threads(2)
    protocol=json.loads((ROOT/'configs/native_opportunity_b8_training_v1.json').read_text())
    manifest=json.loads((ROOT/protocol['manifest']).read_text())
    tokenizer=AutoTokenizer.from_pretrained(ROOT/protocol['model_path'],local_files_only=True)
    rows=[]; sources={}
    for doc in (1,7):
        path=ROOT/f'results/dense_cascade_natural_v1/train{doc}_fp8/summary.json'
        data=json.loads(path.read_text());assert data['status']=='completed'
        sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        group=next(g for g in manifest['groups'] if g['document_id']==data['args']['document_id'])
        prompts,_,_=make_inputs(tokenizer,group)
        assert [hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]==data['prompt_token_hashes']
        run=next(r for r in data['runs'] if r['phase']=='native_natural')
        for answer in run['answers']:
            qi=answer['question_index']
            rows.append({'document':doc,'question_index':qi,'finish_reason':answer['finish_reason'],
                         'prompt_plus_own_past':replay(prompts[qi],answer['tokens']),
                         'own_past_only':replay([],answer['tokens'])})
    totals={}
    for scope in ('prompt_plus_own_past','own_past_only'):
        total_t=sum(r[scope]['tokens'] for r in rows); total_r=sum(r[scope]['replay_rounds'] for r in rows)
        totals[scope]={'tokens':total_t,'rounds':total_r,'tokens_per_round':total_t/total_r}
    assert not torch.cuda.is_initialized()
    result={'status':'CPU_causal_copy_replay_screen','cuda_initialized':False,'n_gram':4,'draft_width':4,
            'rows':rows,'totals':totals,'source_sha256':sources,
            'limitations':['A single native FP8 sampling trajectory for each of 16 questions, two documents only.',
              'Strings match observed sampled tokens; this is not target/draft distribution acceptance or a correctness proof.',
              'No future answer tokens are available to the proposer. Prompt tokens and own already committed replay history only.',
              'No sibling answers are available; no cross-request leakage or future transcript reuse.',
              'No target verification timing, CPU indexing cost, batch scheduling, KV rollback, or end-to-end speedup measured.',
              'A known prompt-lookup baseline used only as a mechanism-space diagnostic; no novelty claim.']}
    out=ROOT/'results/platform/copy_draft_opportunity_cpu_20261017.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(totals))
if __name__=='__main__':main()
