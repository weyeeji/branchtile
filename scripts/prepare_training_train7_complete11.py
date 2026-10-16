"""Reconstruct all eleven natural questions of the preselected training train7 document."""
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
    manifest_path=ROOT/'data/manifests/docfinqa_training_train7_complete11_v1.json'
    assert not manifest_path.exists()
    source_path=ROOT/'data/manifests/docfinqa_public_training167_v1.json'
    source=json.loads(source_path.read_text())
    assert len(source['groups'])==167
    selected=[(i,g) for i,g in enumerate(source['groups']) if g['document_id']=='docfinqa-f1d5c3003725d29c']
    assert len(selected)==1 and selected[0][1]['natural_question_count']==11
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
    rule='The previously selected training document docfinqa-f1d5c3003725d29c (training index 7), now retaining all eleven natural questions instead of the old eight-question pilot. Retain ALL exact-context deduplicated natural questions sorted by question SHA and the entire original document. No length, answer, output, quality, or performance selection. Infeasible selections remain failures; never substitute later entries.'
    manifest={'schema':1,'scope':'Training-only complete natural cohort. All eleven unique questions from the previously selected document; not a new heldout selection.',
              'source_manifest':str(source_path.relative_to(ROOT)),'source_manifest_sha256':file_sha(source_path),
              'selection_rule':rule,'source_parquet_sha256':parquet_hashes,'heldout_overlap':False,
              'protected_source_sha256':before,'groups':groups}
    # Freeze selected documents/questions before tokenization or feasibility is observed.
    save(manifest_path,manifest)
    assert before=={str(p.relative_to(ROOT)):file_sha(p) for p in protected}
    print(json.dumps({'manifest':str(manifest_path.relative_to(ROOT)), 'sha256':file_sha(manifest_path), 'selected':[(g['original_training_group_index'],g['document_id'],g['B']) for g in groups]},indent=2))


if __name__=='__main__':
    main()
