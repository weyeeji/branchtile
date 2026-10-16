"""CPU-only identity and saturation plan for the frozen absolute A4 grid."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sparse_budget_plan import digest,load_full_sources,parse_budgets,resolve_budget,validate_reuse
from capture_model import make_inputs
from sparse_reference import tensor_hash
from transformers import AutoTokenizer
import torch

SOURCES=[f'results/sparse_reference/natural_instruct4b_qmsum{x}.json' for x in ['0_q01','0_q23','_g1234','_g567']]


def main():
    torch.set_num_threads(4)
    expected=json.loads(Path(SOURCES[0]).read_text())
    assert expected['core_sha256']==digest(ROOT/'src/sparse_reference.py')
    assert expected['manifest_sha256']==digest(expected['manifest'])
    records,sources=load_full_sources(SOURCES,expected)
    tokenizer=AutoTokenizer.from_pretrained(expected['model_path'],local_files_only=True)
    manifest=json.loads(Path(expected['manifest']).read_text())
    grid=parse_budgets('4k,8k,16k,32k,full','tokens');rows=[]
    for gi,group in enumerate(manifest['groups']):
        inputs,ds,de=make_inputs(tokenizer,group)
        for qi,ids in enumerate(inputs):
            q={'question_sha256':group['questions'][qi]['question_sha256'],
                'input_ids_sha256':tensor_hash(torch.tensor(ids,dtype=torch.int64)),
                'prompt_tokens':len(ids)}
            record=records[gi,qi]
            validate_reuse(record,{**group,'document_range':[ds,de]},q)
            for request in grid:
                resolved=resolve_budget(request,'tokens',de-ds)
                rows.append({'group_index':gi,'question_index':qi,'document_tokens':de-ds,
                    **resolved,'generation_origin':'reused_prior_full_budget' if resolved['saturated_to_full'] else 'planned_new_generation',
                    'source_generation_id':record['provenance']['source_generation_id'] if resolved['saturated_to_full'] else None})
    result={'status':'passed','budget_mode':'tokens','requested_budgets':grid,
        'model_revision':expected['model_revision'],'manifest_sha256':expected['manifest_sha256'],
        'generation':expected['generation'],'sources':sources,'all_input_ids_and_source_artifacts_verified':True,
        'nominal_rows':len(rows),'new_generations':sum(not r['saturated_to_full'] for r in rows),
        'reused_rows':sum(r['saturated_to_full'] for r in rows),
        'independent_prior_full_generations':len({r['source_generation_id'] for r in rows if r['source_generation_id']}),
        'scope':'Frozen train8 development; reused rows are not new samples or measurements. No factual verdict inherited by this planning script.',
        'rows':rows}
    path=ROOT/'results/sparse_reference_absolute/plan.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','nominal_rows','new_generations','reused_rows','independent_prior_full_generations']}))


if __name__=='__main__':main()
