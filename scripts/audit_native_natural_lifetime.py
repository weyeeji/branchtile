"""CPU audit of actual native ready-group lifetimes, with no length normalization."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    reports = []
    for path in sorted((ROOT / args.root).glob('*/summary.json')):
        data = json.loads(path.read_text())
        assert data['status'] == 'completed'
        for run in data['runs']:
            if not run['natural_eos_enabled']:
                continue
            rank_rows = []
            graph_hashes = []
            for rank in (0, 1):
                graph_path = path.parent / f'rank{rank}' / f'{run["phase"]}_graph.json'
                graph = json.loads(graph_path.read_text())
                rows = [(r['batch'], r['questions'], r['positions']) for r in graph['replays']]
                assert all(b == len(q) == len(pos) for b, q, pos in rows)
                assert all(rows[i][0] >= rows[i+1][0] for i in range(len(rows)-1))
                rank_rows.append(rows)
                graph_hashes.append({'rank': rank, 'path': str(graph_path.relative_to(ROOT)),
                                     'sha256': hashlib.sha256(graph_path.read_bytes()).hexdigest()})
            assert rank_rows[0] == rank_rows[1]
            counts = collections.Counter(q for _, qs, _ in rank_rows[0] for q in qs)
            lengths = [len(a['tokens']) for a in run['answers']]
            assert [counts[q] for q in range(data['B'])] == lengths
            positions = collections.defaultdict(list)
            for _, qs, ps in rank_rows[0]:
                for q, pos in zip(qs, ps):
                    positions[q].append(pos)
            assert all(ps == list(range(ps[0], ps[0]+len(ps))) for ps in positions.values())
            hist = collections.Counter(b for b, _, _ in rank_rows[0])
            assert sum(hist.values()) == max(lengths)
            assert sum(b*n for b, n in hist.items()) == sum(lengths) == run['total_output_tokens']
            reports.append({'source': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'document_id': data['document_id'], 'method': data['method'], 'phase': run['phase'],
                            'natural_generate_wall_seconds': run['complete_generate_wall_seconds'],
                            'anchor_wall_seconds': run['anchor_wall_seconds'],
                            'cold_document_total_wall_seconds': run['cold_shared_document_total_wall_seconds'],
                            'output_tokens_by_question': lengths,
                            'finish_reason_by_question': [a['finish_reason'] for a in run['answers']],
                            'cap_hit_questions': [a['question_index'] for a in run['answers'] if a['finish_reason'] == 'length'],
                            'actual_model_graph_steps': len(rank_rows[0]),
                            'actual_active_batch_step_histogram': dict(sorted(hist.items(), reverse=True)),
                            'all_initial_requests_live_steps': hist[data['B']],
                            'all_initial_requests_live_fraction_of_model_steps': hist[data['B']] / len(rank_rows[0]),
                            'request_token_steps_covered_by_full_group': data['B'] * hist[data['B']],
                            'both_rank_trajectories_equal': True, 'graph_files': graph_hashes,
                            'scope': 'Observed synchronous ready group natural lifetimes; not a quality score or future guarantee.'})
    assert reports
    result = {'status': 'passed', 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runs': reports, 'limitations': ['Natural output length and active-batch mix differ across precisions; do not normalize them away.',
                'A fixed full-group graph profile cannot be multiplied by all natural steps after requests exit.',
                'Full-group live steps do not guarantee a particular candidate metadata plan stays valid for that duration.',
                'One natural sample per document/precision is exploratory, not a formal quality or latency distribution.']}
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
