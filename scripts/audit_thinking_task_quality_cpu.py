"""Explain the frozen 0/8 score without altering metric, prompts, or outcomes."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=ROOT/'results/gqa4_thinking_task_quality_cpu_v1.json'
    assert not output.exists()
    summary_path=ROOT/'results/gqa4_thinking_survivor_e2e_v1/train1/summary.json'
    summary=json.loads(summary_path.read_text())
    group=next(g for g in json.loads((ROOT/summary['manifest']).read_text())['groups'] if g['document_id']==summary['args']['document_id'])
    run=next(r for r in summary['runs'] if r['phase']=='natural_native_0')
    assert run['quality']['score']['exact_accuracy']==run['quality']['score']['rounded_accuracy']==0
    document=ROOT/group['document_path'];text=document.read_text()
    assert sha(document)==group['document_sha256']
    normalize=lambda s:re.sub(r'\s+',' ',s.strip().lower())
    original=[];sources={}
    for split in ('train','dev','test'):
        path=ROOT/f'data/raw/finqa/{split}.json';sources[str(path.relative_to(ROOT))]=sha(path)
        original.extend(json.loads(path.read_text()))
    notes=[
        'Wrong table column and missing FINAL: the question specifies other commercial membership, but output uses total commercial 1999800/3170800=63.06%; source gold uses1815800/2799700=64.85%.',
        'Body gives correct49.6%, but FINAL:49.6 omits percent. Strict unit_mismatch is consistent with explicit prompt requiring numerical answer and unit; this is not an arithmetic failure.',
        'Output assumes primarily cash means cash equals net assets6.0 and gives1.0. Gold1.73 uses92116/53292 from adjacent acquisition table absent in current full document. Original question says Dec20Corphealth while nearby table concerns CarePlus; do not attribute the entire failure solely to model capability.',
        'Body gives correct11.1%, but FINAL:11.1 omits percent. Strict unit_mismatch is retained.',
        'Output selects consolidated liabilities4395509; gold sums acquisition table37375+23359+5915=66649. Question has no acquisition qualifier and required table is absent from full document.',
        'Output directly cites membership3170800 from the supplied table and omits FINAL. Gold3171794.87 instead inverses rounded11.7% via371100/.117. Direct document evidence and rounded-ratio-derived target differ; frozen strict failure retained.',
        'Output cites total premiums+ASO14.261 billion from actual totals. Gold12.525 comes from50.1million/.004/1000 using rounded0.4% in another passage. This target conflict warrants data review, not retrospective replacement of the score.',
        'Output selects consolidated assets6869614; gold sums acquisition table92116+6510+21315=119941. Question lacks table qualifier and source table is absent from supplied full document.',
    ]
    rows=[]
    for i,(q,a,score,note) in enumerate(zip(group['questions'],run['answers'],run['quality']['score']['per_example'],notes,strict=True)):
        matches=[r for r in original if normalize(r.get('qa',{}).get('question',''))==normalize(q['question'])]
        assert len(matches)==1
        r=matches[0]
        rows.append({'question_index':i,'question':q['question'],'frozen_gold':q['answer'],
            'final_line':next((line for line in reversed(a['text'].splitlines()) if line.strip().startswith('FINAL:')),None),
            'output_tail':a['text'][-1100:], 'strict_status':score['status'],
            'prediction_parse_error':score['prediction_parse_error'],'gold_parse_error':score['gold_parse_error'],
            'original_finqa_id':r['id'],'original_program':r['qa'].get('program'),
            'original_table':r['table'],'original_gold_evidence':r['qa'].get('gold_inds'),
            'interpretation':note})
    literals=['92,116','6,510','21,315','37,375','23,359','5,915','53,292','66,649','119,941']
    occurrence={number:bool(re.search(r'(?<!\d)'+number.replace(',',',?')+r'(?!\d)',text)) for number in literals}
    assert not any(occurrence[n] for n in ('92,116','6,510','21,315','23,359','5,915','53,292','66,649','119,941'))
    model_paths=list((ROOT/'data/models').glob('*/config.json'))+list(Path('/home/qcjz/models').glob('*/config.json'))
    model_paths.append(Path('/home/qcjz/k3s-backup-20260921/var-lib-rancher-k3s/storage/pvc-0c5f992c-d713-4629-9f39-edb49acfe82c_turnkey-t1-org-1-tenant-1_qwen3-8b-pd-prefill-pvc-0/Qwen3-8B/config.json'))
    models=[]
    for path in model_paths:
        c=json.loads(path.read_text());t=c.get('text_config',c)
        layers=t.get('layer_types',[])
        models.append({'path':str(path.parent),'config_sha256':sha(path),
            **{k:t.get(k) for k in ('model_type','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim','max_position_embeddings','rope_scaling')},
            'linear_attention_layers':layers.count('linear_attention'),'full_attention_layers':layers.count('full_attention'),
            'safetensor_file_count':len(list(path.parent.glob('*.safetensors'))),
            'README_sha256':sha(path.parent/'README.md') if (path.parent/'README.md').exists() else None})
    report={'status':'explanatory_audit_original_scores_unchanged','source_sha256':sha(Path(__file__)),
        'summary_sha256':sha(summary_path),'document_sha256':sha(document),'original_finqa_sha256':sources,
        'frozen_exact_and_rounded_correct':0,'denominator':8,'cases':rows,
        'acquisition_table_numeric_literals_present_in_supplied_document':occurrence,
        'local_model_inventory':models,
        'model_inventory_conclusion':'No inspected local stronger all-full-attention dense GQA4/8 model with native120K+ context was found. LocalQwen3-8B/32B cards state native32768 and131072 withYaRN; config40960 is not evidence of native120K training. LocalQwen3.6-27B is hybridGQA6,D256 with16 full/48linear layers, not a drop-in GQA4/8 target. Weight completeness/inference eligibility was not GPU-tested.',
        'limits':['This is explanatory case review after outcomes, not a revised benchmark score or an outcome-blind data filter.',
            'Two correct percentage explanations omit the required FINAL unit. At least one clear wrong-column answer remains; no claim that all failures are merely parsing.',
            'Missing numeric literals and the skipped acquisition-table location support an input-data concern; this audit does not reconstruct or inject a missing table.',
            'Neither prompts, data, metric, nor formal heldout selection were modified. No model downloaded or GPU initialized.']}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(rows),'models':len(models),'missing_acquisition_literals':occurrence},indent=2))


if __name__=='__main__':main()
