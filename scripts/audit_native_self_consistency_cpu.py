"""Independent finite-queue/seed/vote audit for the frozen native utility pilot."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_application import extract_final
from src.financial_answer_metric import score_pairs
from src.financial_self_consistency import sample_seed, vote


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(path):
    summary = json.loads(path.read_text())
    assert summary['status'] == 'completed'
    protocol = summary['protocol']
    assert sha(ROOT / summary['args']['protocol']) == summary['protocol_sha256']
    for source, digest in summary['sources'].items():
        assert sha(ROOT / source) == digest, source
    base_path = ROOT / protocol['base_protocol']
    assert sha(base_path) == protocol['base_protocol_sha256']
    base = json.loads(base_path.read_text())
    assert sha(ROOT / base['manifest']) == summary['manifest_sha256']
    group = next(g for g in json.loads((ROOT / base['manifest']).read_text())['groups'] if g['document_id'] == protocol['document_id'])
    document = next(d for d in base['documents'] if d['document_id'] == protocol['document_id'])
    assert len(group['questions']) == summary['B'] == 21
    assert summary['P'] == document['common_complete_page_tokens']
    hashes = document['prompt_token_hashes']
    expected = [{'question_index': q, 'sample_index': s, 'prompt_sha256': hashes[q],
        'seed': sample_seed(protocol['document_id'], hashes[q], s, protocol['study_seed'])}
        for s in range(4) for q in range(21)]
    assert expected == summary['request_manifest'] and len({r['seed'] for r in expected}) == 84
    assert protocol['sample_counts'] == [1, 4]
    assert 'scripts/probe_native_self_consistency_v2.py' in summary['sources']
    for source, digest in summary['request_transport_sources'].items():
        assert sha(Path(source)) == digest
    supplemental = []
    for name in ['scripts/benchmark_native_raw_pair_direct_production.py', 'scripts/probe_engine_docfinqa.py']:
        if name in summary['sources']:
            continue
        historical = subprocess.run(['git', 'show', f'9787571:{name}'], cwd=ROOT,
            check=True, capture_output=True).stdout
        old_sha = hashlib.sha256(historical).hexdigest()
        assert old_sha == sha(ROOT / name)
        supplemental.append({'path': name, 'sha256': old_sha,
            'evidence': 'Current file equals prelaunch 9787571 git snapshot; omitted from runtime sources receipt, not retroactively a runtime hash.'})
    ledger = [json.loads(line) for line in (ROOT / 'results/job_ledger.jsonl').read_text().splitlines()]
    terminals = [r for r in ledger if r.get('event') == 'end' and '--output' in r.get('argv', []) and
        (ROOT / r['argv'][r['argv'].index('--output') + 1]).resolve() == path.parent.resolve()]
    assert len(terminals) == 1 and terminals[0]['exit_code'] == 0 and not terminals[0]['timed_out']
    golds = [q['answer'] for q in group['questions']]
    reports = []
    for n, run in zip((1, 4), summary['runs'], strict=True):
        assert run['samples_per_question'] == n and run['phase'] == f'native_SC{n}'
        manifest = [r for r in expected if r['sample_index'] < n]
        answers = run['answers']
        assert len(answers) == len(manifest) == 21 * n
        assert [a['sample'] for a in answers] == manifest
        assert len({a['request_id'] for a in answers}) == len(answers)
        by_id = {a['internal_request_id']: a for a in answers}
        assert len(by_id) == len(answers)
        assert run['total_output_tokens'] == sum(len(a['tokens']) for a in answers)
        assert run['cap_hits'] == [a['sample'] for a in answers if a['finish_reason'] == 'length']
        assert all(a['request_metrics'] is not None and not a['request_metrics']['is_corrupted'] for a in answers)
        assert run['preemptions'] == sum(a['request_metrics']['num_preemptions'] for a in answers)
        predictions = [[] for _ in range(21)]
        for answer in answers:
            final, _, parsed = extract_final(answer['text'], thinking=True)
            predictions[answer['sample']['question_index']].append(
                final if answer['finish_reason'] == 'stop' and parsed == 'closed_thinking' else '')
        votes = [vote(values) for values in predictions]
        assert votes == run['quality']['votes']
        scores = [score_pairs([p[s] for p in predictions], golds) for s in range(n)]
        assert scores == run['quality']['individual_sample_scores']
        assert scores[0] == run['quality']['sample0_within_this_queue']
        assert score_pairs([v['prediction'] for v in votes], golds) == run['quality']['vote_score']
        assert all(v['denominator'] == n and v['strict_majority'] == (v['count'] > n / 2) for v in votes)
        oracle = {metric: sum(any(score['per_example'][q][metric] for score in scores) for q in range(21))
            for metric in ('exact_match', 'rounded_match')}
        queue_reports = []
        signatures = []
        assert sorted(w['rank'] for w in run['workers']) == [0, 1]
        for worker in run['workers']:
            queue_path = Path(worker['queue_metadata'])
            queue = json.loads(queue_path.read_text())
            assert queue['rank'] == worker['rank'] and queue['phase'] == run['phase']
            admissions = queue['sample_admissions']
            assert len(admissions) == len(answers)
            by_admission = {r['request_id']: r for r in admissions}
            assert set(by_admission) == set(by_id)
            for rid, answer in by_id.items():
                record = by_admission[rid]
                assert record['phase'] == run['phase']
                assert record['seed'] == answer['sample']['seed']
                assert record['question_index'] == answer['sample']['question_index']
            steps = queue['all_steps']
            assert steps
            times = [s['step_started_monotonic_seconds'] for s in steps]
            assert all(a <= b for a, b in zip(times, times[1:]))
            assert run['generation_started_monotonic_seconds'] <= times[0] <= times[-1] <= run['generation_ended_monotonic_seconds']
            seen = set()
            histogram = Counter()
            pure_histogram = Counter()
            spans = Counter()
            eligible_steps = 0
            eligible_span = 0.0
            last_admission = max(a['admitted_monotonic_seconds'] for a in admissions)
            final_drain_span = 0.0
            signatures.append([(s['batch'], s['num_tokens'], s['request_ids'], s['positions']) for s in steps])
            for i, step in enumerate(steps):
                assert step['phase'] == run['phase']
                b = step['batch']
                assert 0 < b <= 21 and len(step['request_ids']) == len(step['positions']) == b
                assert len(set(step['request_ids'])) == b
                assert set(step['request_ids']) <= set(by_id)
                seen.update(step['request_ids'])
                histogram[b] += 1
                end = times[i + 1] if i + 1 < len(times) else run['generation_ended_monotonic_seconds']
                duration = end - times[i]
                spans[b] += duration
                final_drain_span += max(0.0, end - max(times[i], last_admission))
                pure = b == step['num_tokens'] and all(position >= document['prompt_lengths'][by_id[rid]['sample']['question_index']]
                    for rid, position in zip(step['request_ids'], step['positions'], strict=True))
                if pure:
                    pure_histogram[b] += 1
                eligible = pure and b >= 6 and summary['P'] >= 65536 and (max(step['positions']) + 1 - summary['P']) / summary['P'] <= .125
                if eligible:
                    eligible_steps += 1
                    eligible_span += duration
            assert seen == set(by_id)
            queue_reports.append({'rank': queue['rank'], 'queue_sha256': sha(queue_path),
                'unique_admissions': len(admissions), 'all_requests_seen_in_actual_steps': True,
                'actual_steps': len(steps), 'actual_batch_histogram': dict(histogram),
                'pure_decode_histogram_prompt_boundary_checked': dict(pure_histogram),
                'shape_policy_eligible_steps': eligible_steps,
                'shape_policy_eligible_fraction_all_steps': eligible_steps / len(steps),
                'host_between_step_span_seconds_by_batch': dict(spans),
                'shape_policy_eligible_host_span_seconds': eligible_span,
                'shape_policy_eligible_fraction_observed_host_span': eligible_span / sum(spans.values()),
                'last_admission_seconds_after_generate_start': last_admission - run['generation_started_monotonic_seconds'],
                'post_last_admission_drain_host_span_seconds': final_drain_span,
                'scope': 'Finite initially enqueued queue; no future-output selection. Host between-step spans include scheduling and are not GPU attribution. Policy-shape eligibility is prospective only, not actual candidate qualification or speed.'})
        assert signatures[0] == signatures[1]
        reports.append({'phase': run['phase'], 'requests': len(answers), 'output_tokens': run['total_output_tokens'],
            'cap_samples': run['cap_hits'], 'preemptions': run['preemptions'],
            'vote_score': run['quality']['vote_score'], 'sample0_score': scores[0],
            'oracle_any_sample_correct_count_diagnostic_only': oracle,
            'strict_majority_questions': sum(v['strict_majority'] for v in votes),
            'invalid_sample_count': sum(n - v['valid_samples'] for v in votes),
            'no_valid_sample_questions': sum(v['valid_samples'] == 0 for v in votes),
            'vote_support_histogram': dict(Counter(v['count'] for v in votes)),
            'queue_audits': queue_reports, 'rank_queue_trajectories_equal': True,
            'costs': {k: run[k] for k in ['complete_generate_wall_seconds', 'cold_anchor_wall_seconds', 'cold_document_task_wall_seconds', 'vote_wall_seconds', 'full_phase_including_control_export_and_vote_wall_seconds']}})
    return {'status': 'native_utility_queue_audit_passed', 'summary_sha256': sha(path),
        'audit_source_sha256': sha(Path(__file__)), 'runtime_reported_source_hashes_verified': True,
        'omitted_source_git_snapshot_checks': supplemental, 'ledger_terminal_receipts': terminals,
        'unique_frozen_sample_seeds': 84, 'original_question_denominator': 21, 'runs': reports,
        'common_costs': {k: summary[k] for k in ['input_preparation_wall_seconds', 'engine_initialization_wall_seconds', 'shared_sampling_warmup_wall_seconds']},
        'limits': ['Native-only development utility pilot, no candidate, no candidate acceleration or unseen generalization claim.',
            'Gold is used only to score and compute oracle diagnostics after frozen gold-free votes; oracle is not a deployable selection rule.',
            'SC1 and SC4 have different batch schedules. Compare SC4 vote to sample0 within SC4 separately, not only to independent SC1.',
            'Cold task records only anchor+generate. Vote is separately timed. Full experimental phase includes extraction, vote, gold scoring, controls, export and precision receipt; do not equate it to deployment-only inference pipeline latency.',
            'Model initialization and sampling warmup remain explicit common components, not hidden costs.',
            'Finite refill and final drain are included; duplicated questions here are explicitly independent SC samples contributing to the declared vote, not mislabeled natural question multiplicity.',
            'Policy-shape coverage is not a numerical qualification, actual candidate execution fraction, or GPU-time estimate.',
            'One training document and fixed sample counts do not establish broader task utility.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = ROOT / args.output
    assert not output.exists()
    result = audit(ROOT / args.input)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'phases': len(result['runs'])}))


if __name__ == '__main__':
    main()
