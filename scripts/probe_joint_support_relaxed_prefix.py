#!/usr/bin/env python3
"""Independent minimum top-prefix control; relaxed <=k, CPU score proxy only."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
import torch
from probe_joint_support_exchange import ROOT, protected_mask, describe, sha

OUT = ROOT / 'results/joint_support_relaxed_prefix_v1'


def total(score, ids):
    return math.fsum(score[ids].tolist())


def meets(initial, retained, epsilon):
    loss = initial - retained
    return loss <= epsilon * initial and loss / initial <= epsilon


def shortest_prefix(score, baseline, protected, epsilon):
    """Binary search with actual fsum checks; no cumulative-sum tolerance."""
    assert np.isfinite(score).all() and (score >= 0).all()
    mandatory = np.flatnonzero(protected)
    optional = np.flatnonzero(~protected)
    ranked = optional[np.lexsort((optional, -score[optional]))]
    limit = int(baseline.sum()) - len(mandatory)
    assert limit >= 0 and baseline[mandatory].all()
    expected = protected.copy()
    expected[ranked[:limit]] = True
    assert np.array_equal(expected, baseline), 'Input must be canonical protected top-k'
    initial = total(score, np.flatnonzero(baseline))
    assert initial > 0

    def value(count):
        return total(score, np.concatenate((mandatory, ranked[:count])))

    lo, hi, evaluations = 0, limit, 0
    while lo < hi:
        mid = (lo + hi) // 2
        evaluations += 1
        if meets(initial, value(mid), epsilon):
            hi = mid
        else:
            lo = mid + 1
    result = protected.copy()
    result[ranked[:lo]] = True
    actual = total(score, np.flatnonzero(result))
    assert meets(initial, actual, epsilon)
    preceding = value(lo - 1) if lo else None
    assert lo == 0 or not meets(initial, preceding, epsilon)
    return result, dict(optional_selected=lo, optional_original=limit,
                       evaluations=evaluations + 1 + int(lo > 0),
                       preceding_prefix_score=preceding,
                       preceding_prefix_is_infeasible=lo > 0,
                       mandatory_only=lo == 0)


def self_test():
    cases = [([9, 8, 7, 6], [1, 0, 0, 0], 3, 0),
             ([9, 8, 7, 6], [1, 0, 0, 0], 3, .30),
             ([9, 8, 8, 1], [0, 0, 0, 1], 3, .45),
             ([5, 0, 0], [1, 0, 0], 3, 0)]
    rows = []
    for values, fixed, k, eps in cases:
        s = np.array(values, dtype=np.float64)
        p = np.array(fixed, dtype=bool)
        ids = np.flatnonzero(~p)
        ranked = ids[np.lexsort((ids, -s[ids]))]
        b = p.copy(); b[ranked[:k-int(p.sum())]] = True
        out, audit = shortest_prefix(s, b, p, eps)
        candidates = []
        for count in range(k-int(p.sum())+1):
            mask = p.copy(); mask[ranked[:count]] = True
            if meets(total(s, np.flatnonzero(b)), total(s, np.flatnonzero(mask)), eps):
                candidates.append(count)
        assert audit['optional_selected'] == min(candidates)
        assert out[p].all() and out.sum() <= k
        rows.append(dict(scores=values, protected=fixed, k=k, epsilon=eps,
                         selected=np.flatnonzero(out).tolist(), exhaustive_prefix_check=True))
    return rows


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    checks = self_test()
    if args.self_test:
        print(json.dumps(checks)); return
    protocol_path = OUT / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert not (OUT / 'results.jsonl').exists() and not (OUT / 'summary.json').exists()
    source_path = ROOT / protocol['source_protocol']
    assert sha(source_path) == protocol['source_protocol_sha256']
    assert sha(ROOT / 'scripts/probe_joint_support_exchange.py') == protocol['helper_source_sha256']
    prior_path = ROOT / protocol['exchange_results']
    assert sha(prior_path) == protocol['exchange_results_sha256']
    source_protocol = json.loads(source_path.read_text())
    prior = [json.loads(line) for line in prior_path.read_text().splitlines()]
    rows, inputs, independent_checks = [], [], 0
    started, wall_start, cpu_start = time.time(), time.perf_counter(), time.process_time()
    for source in source_protocol['sources']:
        directory = ROOT / source['directory']
        meta = json.loads((directory / 'metadata.json').read_text())
        assert sha(directory / 'metadata.json') == source['metadata_sha256']
        assert meta['export_semantics_version'] == 2
        ds, de = meta['document_token_range']
        cs, ce = meta['actual_shared_document_token_range']
        assert cs == ds
        n, common_n = de-ds, ce-ds
        protected = protected_mask(n, source_protocol['protected_first'], source_protocol['protected_last'])
        assert protected[common_n:].all()
        loaded = []
        for item in source['selector_files']:
            path = directory / item['name']
            assert sha(path) == item['sha256']
            x = torch.load(path, map_location='cpu', weights_only=True)
            assert x['observation_is_question_suffix_only']
            assert bool((x['observation_positions'] >= de).all())
            loaded.append(x)
            inputs.append(dict(path=str(path.relative_to(ROOT)), sha256=item['sha256']))
        for budget in source_protocol['budgets']:
            for head in range(2):
                scores = np.stack([x['pooled_scores'][head].numpy().astype(np.float64) for x in loaded])
                baseline = np.stack([x['supports'][budget][head].numpy() for x in loaded])
                assert baseline.shape == scores.shape == (meta['B'], n)
                assert np.all(baseline.sum(axis=1) == budget)
                for eps in protocol['epsilon']:
                    tw, tc = time.perf_counter(), time.process_time()
                    pairs = [shortest_prefix(scores[q], baseline[q], protected, eps) for q in range(meta['B'])]
                    masks = np.stack([x[0] for x in pairs])
                    selection_wall, selection_cpu = time.perf_counter()-tw, time.process_time()-tc
                    row = dict(document=source['id'], document_sha256=meta['document_sha256'],
                               rank=0, layer=15, kv_head=head, B=meta['B'], budget=budget, epsilon=eps,
                               method='independent_minimal_top_prefix', budget_constraint='at_most_k_not_exact_k',
                               minimum_prefix_audit=[x[1] for x in pairs],
                               selection_wall_seconds=selection_wall, selection_cpu_seconds=selection_cpu,
                               original_sum_selected_entries=int(baseline.sum()), sum_selected_entries=int(masks.sum()),
                               selected_entries_reduction_fraction=1-int(masks.sum())/int(baseline.sum()),
                               **describe(scores, masks, baseline, common_n))
                    row['exchange_comparison_same_score_floor_different_cardinality_constraint'] = {}
                    for order in ['ascending', 'descending']:
                        old = [r for r in prior if r['method']=='exchange' and r['document']==source['id']
                               and r['budget']==budget and r['kv_head']==head and r['epsilon']==eps and r['request_order']==order]
                        assert len(old) == 1
                        row['exchange_comparison_same_score_floor_different_cardinality_constraint'][order] = dict(
                            union_tokens=old[0]['actual_common_source_logical']['union_tokens'],
                            sum_selected_entries=old[0]['full_document_logical_only']['selected_entries'])
                    # Independent prefix/count, score, identity and minimality audit.
                    for q, actual in enumerate(masks):
                        ids = np.flatnonzero(~protected)
                        order = ids[np.argsort(-scores[q, ids], kind='stable')]
                        count = int(actual.sum())-int(protected.sum())
                        expected = protected.copy(); expected[order[:count]] = True
                        assert np.array_equal(actual, expected) and np.all(~actual | baseline[q])
                        a = math.fsum(scores[q, baseline[q]].tolist())
                        retained = math.fsum(scores[q, actual].tolist())
                        assert a-retained <= eps*a and (a-retained)/a <= eps
                        if count:
                            expected[order[count-1]] = False
                            shorter = math.fsum(scores[q, expected].tolist())
                            assert a-shorter > eps*a or (a-shorter)/a > eps
                        assert row['per_request'][q]['selected_indices_sha256'] == hashlib.sha256(np.flatnonzero(actual).astype('<i8').tobytes()).hexdigest()
                        independent_checks += 1
                    assert int(np.any(masks[:, :common_n], axis=0).sum()) == row['actual_common_source_logical']['union_tokens']
                    assert np.array_equal(masks[:, common_n:], baseline[:, common_n:])
                    rows.append(row)
        print(json.dumps(dict(document=source['id'], completed_rows=len(rows))), flush=True)
    assert len(rows) == protocol['expected_rows'] == 48 and not torch.cuda.is_initialized()
    result_path = OUT / 'results.jsonl'
    result_path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    aggregate = []
    for eps in protocol['epsilon']:
        rs = [r for r in rows if r['epsilon']==eps]
        ur = [r['actual_common_source_logical']['union_reduction_fraction'] for r in rs]
        er = [r['selected_entries_reduction_fraction'] for r in rs]
        aggregate.append(dict(epsilon=eps, head_cells=len(rs),
            union_reduction_min_median_max=[min(ur), float(np.median(ur)), max(ur)],
            selected_entries_reduction_min_median_max=[min(er), float(np.median(er)), max(er)],
            max_request_score_loss=max(q['score_loss_fraction'] for r in rs for q in r['per_request']),
            selection_wall_seconds=sum(r['selection_wall_seconds'] for r in rs),
            lower_union_than_exact_k_exchange_cells={order:sum(r['actual_common_source_logical']['union_tokens'] <
                r['exchange_comparison_same_score_floor_different_cardinality_constraint'][order]['union_tokens'] for r in rs)
                for order in ['ascending','descending']}))
    summary = dict(scope=protocol['scope'], budget_constraint='at_most_k_not_exact_k', result_rows=len(rows),
                   protocol_sha256=sha(protocol_path), script_sha256=sha(Path(__file__)), results_sha256=sha(result_path),
                   started_unix_seconds=started, completed_unix_seconds=time.time(),
                   wall_seconds=time.perf_counter()-wall_start, cpu_seconds=time.process_time()-cpu_start,
                   uid=os.geteuid(), venv=sys.prefix, torch_version=torch.__version__, numpy_version=np.__version__,
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   input_provenance=inputs, aggregates=aggregate, self_test=checks,
                   cuda_initialized=False, gpu_hours=0, physical_capacity_released_bytes=0)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    validation = dict(status='passed', result_sha256=sha(result_path), checked_rows=len(rows),
                      checked_request_rows=independent_checks, independent_stable_prefix_reconstruction=True,
                      actual_fsum_thresholds_no_tolerance=True, all_preceding_prefixes_infeasible=True,
                      all_masks_subset_of_original=True, protected_private_tail_unchanged=True,
                      all_selected_index_hashes_verified=True, common_union_directly_counted=True,
                      exact_k_comparison_is_explicitly_different_feasible_set=True)
    (OUT / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['input_provenance','self_test']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
