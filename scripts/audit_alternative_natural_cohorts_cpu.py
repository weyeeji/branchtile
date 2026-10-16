"""CPU-only source/length audit. No generated output, GPU/model loading, or selection by answer."""
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/alternative_cohort_metadata_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(url, path, max_bytes):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=45) as source:
            content = source.read(max_bytes + 1)
        assert len(content) <= max_bytes, url
        path.write_bytes(content)
    assert path.stat().st_size <= max_bytes
    return {'url': url, 'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(4)
    from transformers import AutoTokenizer
    tokenizer_path = ROOT / 'data/models/Qwen3-30B-A3B-Thinking-2507-FP8'
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    out = ROOT / 'results/alternative_natural_cohort_opportunity_v1.json'
    assert not out.exists()
    tree = json.loads((RAW / 'financebench_tree.json').read_text())
    revision = tree['sha']
    rows = [json.loads(line) for line in (RAW / 'financebench_open_source.jsonl').read_text().splitlines()]
    cohorts = collections.defaultdict(list)
    for i, row in enumerate(rows):
        cohorts[row['doc_name']].append((i, row))
    # Outcome-blind rule: every original natural cohort with >=6 distinct questions, in source encounter order.
    selected = [(name, rs) for name, rs in cohorts.items() if len({r['question'] for _, r in rs}) >= 6]
    selection_path = RAW / 'financebench_cohort_selection_before_pdf.json'
    selection = {'revision': revision, 'source_sha256': sha(RAW / 'financebench_open_source.jsonl'),
        'rule': 'All naturally occurring documents with >=6 distinct questions, source encounter order, all questions retained. Freeze before PDF download/tokenization; never duplicate questions or substitute documents after feasibility.',
        'documents': [{'doc_name': name, 'row_indices': [i for i, _ in rs], 'ids': [r['financebench_id'] for _, r in rs]} for name, rs in selected]}
    if not selection_path.exists():
        selection_path.write_text(json.dumps(selection, indent=2) + '\n')
    assert json.loads(selection_path.read_text()) == selection
    receipts = []
    finance_rows = []
    for name, rs in selected:
        relative = f'pdfs/{name}.pdf'
        metadata = next(r for r in tree['tree'] if r['path'] == relative)
        pdf = RAW / f'{name}.pdf'
        receipt = fetch(f'https://raw.githubusercontent.com/patronus-ai/financebench/{revision}/{relative}', pdf, 8 * 1024 * 1024)
        assert receipt['bytes'] == metadata['size']
        assert hashlib.sha1(b'blob ' + str(pdf.stat().st_size).encode() + b'\x00' + pdf.read_bytes()).hexdigest() == metadata['sha']
        receipts.append(receipt)
        text = RAW / f'{name}.layout.txt'
        if not text.exists():
            subprocess.run(['pdftotext', '-layout', str(pdf), str(text)], check=True)
        content = text.read_text()
        pages = content.split('\f')
        lengths = []
        for _, r in rs:
            messages = [{'role': 'user', 'content': 'Read the complete financial document and answer the question. Give the answer and unit.\n\n' + content + '\n\nQuestion: ' + r['question']}]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            tokens = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
            assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
            assert len(tokens) > len(tokenizer.encode(content, add_special_tokens=False))
            lengths.append(len(tokens))
        evidence = []
        def normalize(s):
            return ''.join(c.lower() for c in s if c.isalnum())
        normalized_pages = [normalize(p) for p in pages]
        for _, row in rs:
            for item in row['evidence']:
                index = item['evidence_page_num']
                normalized = normalize(item['evidence_text'])
                evidence.append({'question_id': row['financebench_id'], 'evidence_page_zero_based': index,
                    'page_exists': 0 <= index < len(pages),
                    'full_normalized_evidence_is_substring_on_cited_page': 0 <= index < len(pages) and normalized in normalized_pages[index],
                    'full_normalized_evidence_is_substring_anywhere': any(normalized in p for p in normalized_pages),
                    'scope': 'Exact normalized substring is an extraction diagnostic, not semantic evidence validation; false may reflect PDF column order or formatting.'})
        finance_rows.append({'doc_name': name, 'B': len(rs), 'all_questions_unique': len({r['question'] for _, r in rs}) == len(rs),
            'pdf_sha256': sha(pdf), 'extracted_text_sha256': sha(text), 'pages': len(pages) - (not pages[-1].strip()),
            'document_tokens': len(tokenizer.encode(content, add_special_tokens=False)), 'prompt_lengths': lengths,
            'max_prompt_plus_8192': max(lengths) + 8192, 'context_fits_163840': max(lengths) + 8192 <= 163840,
            'document_meets_current_P65536_lower_bound': len(tokenizer.encode(content, add_special_tokens=False)) >= 65536,
            'evidence': evidence})
    qasper_url = 'https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz'
    qasper_path = RAW / 'qasper-test-and-evaluator-v0.3.tgz'
    receipts.append(fetch(qasper_url, qasper_path, 5 * 1024 * 1024))
    with tarfile.open(qasper_path, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.name.endswith('.json')]
        assert len(members) == 1, [m.name for m in members]
        papers = json.load(tar.extractfile(members[0]))
    qasper_rows = []
    for paper_id, paper in papers.items():
        content = paper['title'] + '\n' + paper['abstract'] + '\n' + '\n'.join(
            str(s.get('section_name') or '') + '\n' + '\n'.join(s['paragraphs']) for s in paper['full_text'])
        questions = [q['question'] for q in paper['qas']]
        qasper_rows.append({'paper_id': paper_id, 'natural_questions': len(questions),
            'distinct_natural_questions': len(set(questions)),
            'full_paper_tokens': len(tokenizer.encode(content, add_special_tokens=False))})
    assert not torch.cuda.is_initialized()
    result = {'status': 'CPU_source_opportunity_audit_only', 'uid': os.geteuid(), 'python': sys.executable,
        'CUDA_initialized': False, 'script_sha256': sha(Path(__file__)), 'receipts': receipts,
        'tokenizer_json_sha256': sha(tokenizer_path / 'tokenizer.json'),
        'financebench': {'official_repository_revision': revision, 'rows': len(rows), 'natural_documents': len(cohorts),
            'cohort_histogram': dict(sorted(collections.Counter(len(rs) for rs in cohorts.values()).items())),
            'source_sha256': sha(RAW / 'financebench_open_source.jsonl'), 'pre_tokenization_selection_sha256': sha(selection_path),
            'selected': finance_rows},
        'qasper_official_test': {'papers': len(papers), 'questions': sum(r['natural_questions'] for r in qasper_rows),
            'cohort_histogram': dict(sorted(collections.Counter(r['distinct_natural_questions'] for r in qasper_rows).items())),
            'max_full_paper_tokens': max(r['full_paper_tokens'] for r in qasper_rows),
            'natural_B_ge6': sum(r['distinct_natural_questions'] >= 6 for r in qasper_rows),
            'natural_B_ge6_and_P_ge65536': sum(r['distinct_natural_questions'] >= 6 and r['full_paper_tokens'] >= 65536 for r in qasper_rows),
            'papers_metadata': qasper_rows},
        'limits': 'No model weights, GPU execution, generated answers, or speed-based selection. Prompt is a length-audit sketch, not a frozen performance/quality protocol. FinanceBench public release is 150 cases, not all 10231 advertised questions. PDF formatting and answer semantic validity need separate checks before a main quality claim. QASPER counts apply only to official test split.'}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(out.relative_to(ROOT)), 'financebench': [(r['doc_name'], r['B'], r['document_tokens'], r['max_prompt_plus_8192']) for r in finance_rows], 'qasper_max_tokens': result['qasper_official_test']['max_full_paper_tokens']}))


if __name__ == '__main__':
    main()
