"""Preselect one unique candidate per development question without reading verdicts."""
import collections
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
sources = ['results/sparse_reference/train8_quality_blind_audit.jsonl',
           'results/fastkvzip_reference/train8_quality_blind_audit.jsonl',
           'results/sparse_reference_absolute/train8_quality_blind_audit_new56.jsonl']
groups = collections.defaultdict(dict)
for source in sources:
    for line in (ROOT / source).read_text().splitlines():
        row = json.loads(line)
        key = (row['document_sha256'], row['question'])
        identity = (row['gold_reference'], row['answer'])
        entry = groups[key].setdefault(identity, {'candidate': row, 'origins': []})
        entry['origins'].append({'source_queue': source, 'item_id': row['item_id']})
assert len(groups) == 32
rng = random.Random(20261016)
selected = []
for key in sorted(groups):
    options = sorted(groups[key].values(), key=lambda v: (v['candidate']['gold_reference'], v['candidate']['answer']))
    entry = options[rng.randrange(len(options))]
    selected.append((entry, len(options)))
rng.shuffle(selected)
queue, mapping = [], {}
for index, (entry, choices) in enumerate(selected, 1):
    key = f'J2-{index:04d}'
    item = dict(entry['candidate'], item_id=key)
    source = Path(item['document_path'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['document_sha256']
    queue.append(item)
    mapping[key] = {'origins': entry['origins'], 'unique_answer_choices_for_question': choices,
                    'answer_sha256': hashlib.sha256(item['answer'].encode()).hexdigest()}
output = ROOT / 'results/factual_audit_second'
output.mkdir(exist_ok=True)
queue_path = output / 'development32_blind_queue.jsonl'
queue_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in queue))
(output / 'development32_identity_mapping.json').write_text(json.dumps(mapping, indent=2) + '\n')
protocol = {'scope': 'Independent second LLM-assisted judge on one unique answer per development question; not human review or the full population.',
    'seed': 20261016, 'questions': 32, 'documents': 8,
    'sampling_rule': 'Sort questions by(source SHA,question). Deduplicate candidate text+gold within each question across the three fixed queues. Uniformly select one unique answer per question with seeded RNG, then shuffle. Not uniform across methods/budgets; no verdict file is read.',
    'source_queues_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
    'queue_sha256': hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'rubric': 'results/factual_audit/rubric_v1.json',
    'rule': 'Judge must not read first-judge verdicts, identity map, method/budget quality results or implementation. Record uncertainty and complete-source evidence. Do not adjudicate disagreements by silently replacing either version.',
    'limit': 'Both judges may share the same model family and general project context; this is assisted agreement evidence, not independent human reliability. Dataset has seven higher source clusters;32 questions are not32 independent source documents.'}
(output / 'protocol_v1.json').write_text(json.dumps(protocol, indent=2) + '\n')
print(json.dumps({'questions': len(queue), 'unique_candidates': sum(len(v) for v in groups.values()), 'output': str(queue_path)}))
