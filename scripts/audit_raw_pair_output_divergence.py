"""Retain and locate failed natural-output equality without claiming a speedup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    path, output = ROOT/args.input, ROOT/args.output
    assert not output.exists()
    summary = json.loads(path.read_text())
    assert summary['status'] == 'failed'
    native, candidate = summary['runs'][:2]
    assert native['method'] == 'native' and candidate['method'] == 'cascade'
    for name, expected in summary['source_sha256'].items():
        assert sha(ROOT/name) == expected
    assert len(native['answers']) == len(candidate['answers']) == summary['B']
    differences = []
    for question, (a, b) in enumerate(zip(native['answers'], candidate['answers'], strict=True)):
        x, y = a['tokens'], b['tokens']
        if x != y:
            first = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
            differences.append({'question_index': question, 'first_different_token_index_zero_based': first,
                'native_length': len(x), 'candidate_length': len(y),
                'native_token': x[first] if first < len(x) else None,
                'candidate_token': y[first] if first < len(y) else None,
                'native_window': x[max(0, first-2):first+3],
                'candidate_window': y[max(0, first-2):first+3]})
    assert differences
    earliest = min(differences, key=lambda d: d['first_different_token_index_zero_based'])
    rank_records = []
    for rank in (0, 1):
        worker_a = next(w for w in native['workers'] if w['rank'] == rank)
        worker_b = next(w for w in candidate['workers'] if w['rank'] == rank)
        pa, pb = Path(worker_a['graph_metadata']), Path(worker_b['graph_metadata'])
        a, b = json.loads(pa.read_text())['replays'], json.loads(pb.read_text())['replays']
        first_meta = next((i for i, (x, y) in enumerate(zip(a, b))
            if any(x[k] != y[k] for k in ('batch', 'questions', 'positions'))), None)
        # Locate the pure decode row consuming the token immediately before the
        # first differing output. Prefill samples output index zero.
        target = earliest['first_different_token_index_zero_based']-1
        keys = ('batch', 'questions', 'positions', 'candidate_graph_replayed', 'native_graph_replayed')
        rank_records.append({'rank': rank, 'native_graph_sha256': sha(pa), 'candidate_graph_sha256': sha(pb),
            'first_batch_question_position_difference_graph_index': first_meta,
            'first_output_difference_nearby_graph_rows': [dict(graph_index=i,
                native={k:a[i][k] for k in keys}, candidate={k:b[i][k] for k in keys})
                for i in range(max(0,target-1),min(len(a),len(b),target+2))]})
    ends = [r for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()
        if (r:=json.loads(line)).get('event') == 'end' and '--output' in r.get('argv',[]) and
        (ROOT/r['argv'][r['argv'].index('--output')+1]).resolve() == path.parent.resolve()]
    assert len(ends) == 1, 'Wait for the failed process terminal receipt before auditing.'
    report = {'status': 'output_divergence_failure_retained_no_paired_speed_claim',
        'audit_source_sha256':sha(Path(__file__)), 'summary_sha256':sha(path),
        'runtime_sources_verified':True, 'ledger_terminal_receipt':ends[0], 'B':summary['B'],
        'error':summary.get('error'), 'divergent_questions':differences,
        'first_output_divergence':earliest, 'ranks':rank_records,
        'native_output_lengths':[len(a['tokens']) for a in native['answers']],
        'candidate_output_lengths':[len(a['tokens']) for a in candidate['answers']],
        'native_cap_hits':native['cap_hits'], 'candidate_cap_hits':candidate['cap_hits'],
        'native_preemptions':native['total_request_preemptions'],
        'candidate_preemptions':candidate['total_request_preemptions'],
        'limits':['This records unequal natural outputs, not proof of the first internal numerical divergence.',
            'Batch/question/position equality is not a bitwise hidden-state or input-token-state check.',
            'No paired speed ratio is assigned; the selected document remains a failure in formal evaluation.',
            'Native repeatability versus candidate numerical error requires a separate same-state diagnostic; neither is assumed here.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':report['status'],'divergent_questions':len(differences),'earliest':earliest},indent=2))


if __name__ == '__main__':
    main()
