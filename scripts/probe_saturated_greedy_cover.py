#!/usr/bin/env python3
"""CPU-only full-universe saturated weighted-rank greedy cover control."""
from __future__ import annotations

import argparse
import datetime
from fractions import Fraction
import hashlib
import heapq
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import torch
import probe_joint_support_exchange as prior

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/saturated_greedy_cover_v1"


class BudgetExpired(Exception):
    pass


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def integer_scores(scores):
    """Exact per-request integer representation of the stored binary scores."""
    rows, denominators = [], []
    for row in scores:
        ratios = [float(v).as_integer_ratio() for v in row]
        denominator = max(d for _, d in ratios)
        assert denominator & (denominator - 1) == 0
        rows.append([n * (denominator // d) for n, d in ratios])
        denominators.append(denominator)
    return rows, denominators


def setup(scores, baseline, protected, common_n, epsilon):
    b, n = scores.shape
    assert np.isfinite(scores).all() and (scores >= 0).all()
    assert baseline.shape == (b, n) and baseline[:, protected].all()
    assert protected[common_n:].all(), "Private document tokens must remain mandatory"
    candidates = np.flatnonzero(~protected[:common_n]).tolist()
    fixed = np.flatnonzero(protected).tolist()
    rank = [int(row.sum()) - len(fixed) for row in baseline]
    assert min(rank) >= 0 and max(rank) <= len(candidates)
    integers, denominators = integer_scores(scores)
    fraction = Fraction(str(epsilon))
    p, d = fraction.numerator, fraction.denominator
    original = [sum(integers[q][t] for t in np.flatnonzero(baseline[q])) for q in range(b)]
    constants = [sum(integers[q][t] for t in fixed) for q in range(b)]
    tau = [(d-p)*original[q] - d*constants[q] for q in range(b)]
    kmax = max(rank)
    normalizers = [v for v in tau if v > 0] + ([kmax] if kmax else [])
    lcm = math.lcm(*normalizers) if normalizers else 1
    weights = [lcm // t if t > 0 else 0 for t in tau]
    card_weight = lcm // kmax if kmax else 0
    return {"integers": integers, "denominators": denominators, "candidates": candidates,
            "fixed": fixed, "rank": rank, "tau": tau, "epsilon_denominator": d,
            "weights": weights, "card_weight": card_weight, "kmax": kmax,
            "normalization_lcm": lcm, "original_exact_integer_scores": original}


def run_cover(scores, baseline, protected, common_n, epsilon, check_budget=lambda: None,
              naive=False, trace=False):
    state = setup(scores, baseline, protected, common_n, epsilon)
    values = state["integers"]
    tau, rank = state["tau"], state["rank"]
    weights, d = state["weights"], state["epsilon_denominator"]
    b, n = scores.shape
    heaps = [[] for _ in range(b)]
    sums = [0] * b
    chosen = []
    marginal_evaluations = 0
    heap_pops = 0
    selected_gains = []

    def gain(token):
        nonlocal marginal_evaluations
        marginal_evaluations += 1
        result = state["card_weight"] if len(chosen) < state["kmax"] else 0
        for q in range(b):
            deficit = max(0, tau[q] - d*sums[q])
            if not deficit or rank[q] == 0:
                continue
            cutoff = heaps[q][0][0] if len(heaps[q]) == rank[q] else 0
            increase = max(values[q][token] - cutoff, 0)
            result += min(d*increase, deficit) * weights[q]
        return result

    def feasible():
        return len(chosen) >= state["kmax"] and all(d*sums[q] >= tau[q] for q in range(b))

    candidates = state["candidates"]
    queue = [(-gain(token), token) for token in candidates]
    heapq.heapify(queue)
    alive = set(candidates) if naive else None
    while not feasible():
        if len(chosen) % 128 == 0:
            check_budget()
        if naive:
            minus_gain, token = min((-gain(t), t) for t in alive)
            actual = -minus_gain
            alive.remove(token)
        else:
            assert queue, "The full candidate universe must contain feasible independent supports"
            while True:
                negative_upper, token = heapq.heappop(queue)
                heap_pops += 1
                if heap_pops % 4096 == 0:
                    check_budget()
                actual = gain(token)
                assert actual <= -negative_upper, "Exact submodular marginal cannot increase"
                if not queue or (-actual, token) <= queue[0]:
                    break
                heapq.heappush(queue, (-actual, token))
        assert actual > 0, "Unsaturated feasible cover must have a positive marginal"
        chosen.append(token)
        if trace:
            selected_gains.append(str(actual))
        for q in range(b):
            if rank[q] == 0:
                continue
            value = values[q][token]
            item = (value, -token)
            if len(heaps[q]) < rank[q]:
                heapq.heappush(heaps[q], item)
                sums[q] += value
            elif item > heaps[q][0]:
                removed = heapq.heapreplace(heaps[q], item)
                sums[q] += value - removed[0]
    # Refill to exact k by stable per-request top-k from U, including ties and
    # zero-score entries. Mandatory common and private tokens are fixed.
    masks = np.tile(protected, (b, 1))
    chosen_array = np.array(chosen, dtype=np.int64)
    for q in range(b):
        ordered = prior.stable_order(chosen_array, scores[q], descending=True)
        selected = ordered[:rank[q]]
        masks[q, selected] = True
        assert len(selected) == rank[q]
        assert set(selected) == {-negative_token for _, negative_token in heaps[q]}
        assert sum(values[q][t] for t in selected) == sums[q]
        original = prior.score_sum(scores[q], baseline[q])
        retained = prior.score_sum(scores[q], masks[q])
        loss = original - retained
        assert loss <= epsilon*original and loss/original <= epsilon, "Strict original fsum bound failed; never relax it"
    assert np.array_equal(masks.sum(1), baseline.sum(1)) and masks[:, protected].all()
    mask_union = int(masks[:, :common_n].any(0).sum())
    info = {"candidate_universe_tokens": len(candidates), "greedy_chosen_free_tokens": len(chosen),
            "free_union_after_refill_tokens": mask_union - int(protected[:common_n].sum()),
            "unused_greedy_tokens_removed_by_refill": len(chosen) - (mask_union-int(protected[:common_n].sum())),
            "heap_pops": heap_pops, "marginal_evaluations": marginal_evaluations,
            "exact_integer_score_denominators": [str(x) for x in state["denominators"]],
            "normalized_objective_common_denominator_bits": state["normalization_lcm"].bit_length(),
            "tau_integer_numerators": [str(x) for x in tau], "tau_score_denominator_multiplier": d,
            "greedy_chosen_order_sha256": hashlib.sha256(np.array(chosen, dtype='<i8').tobytes()).hexdigest(),
            "all_exact_cover_constraints_satisfied": True, "all_original_fsum_constraints_satisfied": True,
            "reverse_delete_postprocessing": False}
    if trace:
        info.update(chosen_order=chosen, exact_gain_numerators=selected_gains)
    return masks, info


def independent_masks(scores, protected, rank):
    result = np.tile(protected, (len(scores), 1))
    free = np.flatnonzero(~protected)
    for q in range(len(scores)):
        result[q, prior.stable_order(free, scores[q], descending=True)[:rank[q]]] = True
    return result


def enumerate_optimum(scores, baseline, protected, common_n, epsilon):
    state = setup(scores, baseline, protected, common_n, epsilon)
    v, d = state["integers"], state["epsilon_denominator"]
    for count in range(state["kmax"], len(state["candidates"])+1):
        for subset in itertools.combinations(state["candidates"], count):
            if all(d*sum(sorted([v[q][t] for t in subset], reverse=True)[:state["rank"][q]]) >= state["tau"][q]
                   for q in range(len(scores))):
                return count
    raise AssertionError("No feasible exact subset")


def self_test():
    results = []
    fixtures = [
        ("multi_owner_triangle", np.array([[102,101,100],[102,100,101],[100,102,101]], float), np.zeros(3,bool), [2,2,2], 3, .005, 2),
        ("outside_initial_union", np.array([[100,1,99],[1,100,99]], float), np.zeros(3,bool), [1,1], 3, .01, 1),
        ("fixed_common_and_private", np.array([[3,9,2,5,7],[3,2,9,5,7]], float), np.array([1,0,0,0,1],bool), [1,1], 4, .2, None),
        ("zero_score_cardinality", np.array([[5,0,0,0],[6,0,0,0]], float), np.array([1,0,0,0],bool), [2,2], 4, .01, 2),
    ]
    rng = np.random.default_rng(17019)
    for i in range(60):
        scores = rng.integers(1, 40, size=(3,6)).astype(float) / 8
        protected = np.zeros(6,bool)
        common_n = 6
        if i % 2:
            protected[[0,5]] = True
            common_n = 5
        ranks = rng.integers(1,4,size=3).tolist()
        fixtures.append((f"random_{i}",scores,protected,ranks,common_n,[.001,.005,.01,.25,.5][i%5],None))
    for name, scores, protected, ranks, common_n, epsilon, expected in fixtures:
        baseline = independent_masks(scores, protected, ranks)
        masks, info = run_cover(scores, baseline, protected, common_n, epsilon, trace=True)
        naive_masks, naive = run_cover(scores, baseline, protected, common_n, epsilon, naive=True, trace=True)
        assert np.array_equal(masks,naive_masks) and info["chosen_order"] == naive["chosen_order"]
        assert info["exact_gain_numerators"] == naive["exact_gain_numerators"]
        optimum = enumerate_optimum(scores,baseline,protected,common_n,epsilon)
        assert info["free_union_after_refill_tokens"] >= optimum
        if expected is not None:
            assert info["free_union_after_refill_tokens"] == optimum == expected
        results.append({"name":name,"optimal_free_union_by_exact_enumeration":optimum,
                        "greedy_free_union":info["free_union_after_refill_tokens"],
                        "chosen_order":info["chosen_order"],"lazy_matches_full_greedy":True})
    assert not torch.cuda.is_initialized()
    return {"status":"passed","fixtures":results,"fixture_count":len(results),
            "scope":"Exact small-instance feasibility and lazy-versus-full greedy definition, not proof of global greedy optimality."}


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    protocol_path=BASE/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    assert sha(Path(__file__)) == protocol['script_sha256']
    assert sha(ROOT/'scripts/probe_joint_support_exchange.py') == protocol['helper_sha256']
    assert not (BASE/'results.jsonl').exists()
    resource.setrlimit(resource.RLIMIT_CPU,(590,600))
    signal.signal(signal.SIGXCPU,lambda *_: (_ for _ in ()).throw(BudgetExpired('CPU soft limit')))
    def check_budget():
        if time.process_time()-started_cpu >= 575 or time.perf_counter()-started_wall >= 590:
            raise BudgetExpired('Frozen CPU/wall budget reached')
    checks=self_test()
    (BASE/'self_test.json').write_text(json.dumps(checks,indent=2)+'\n')
    rows=[]
    status='completed'
    current=None
    error=None
    provenance=[]
    try:
        with (BASE/'results.jsonl').open('w') as output:
            for source in protocol['sources']:
                check_budget()
                directory=ROOT/source['directory']
                assert sha(directory/'metadata.json') == source['metadata_sha256']
                metadata=json.loads((directory/'metadata.json').read_text())
                assert metadata['export_semantics_version']==2 and metadata['actual_tp_rank']==0
                start,end=metadata['document_token_range']; common_start,common_end=metadata['actual_shared_document_token_range']
                assert common_start==start
                n,common_n=end-start,common_end-start
                protected=prior.protected_mask(n,protocol['protected_first'],protocol['protected_last'])
                items=[]
                for q,file in enumerate(source['selector_files']):
                    path=directory/file['name']; assert sha(path)==file['sha256']
                    item=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                    assert item['observation_is_question_suffix_only']
                    positions=item['observation_positions'].numpy()
                    assert len(positions)<=32 and (positions>=end).all()
                    items.append({'scores':item['pooled_scores'].numpy().astype(np.float64),
                                  'supports':{k:item['supports'].get(k,item['supports'].get(str(k))).numpy().copy() for k in protocol['budgets']}})
                    provenance.append({'path':str(path.relative_to(ROOT)),'sha256':file['sha256'],'source_request_id':item['source_request_id'],'suffix_only':True})
                all_scores=np.stack([x['scores'] for x in items]); b,h,_=all_scores.shape
                assert h==2 and b==source['expected_B']
                for budget in protocol['budgets']:
                    all_masks=np.stack([x['supports'][budget] for x in items])
                    for head in protocol['kv_heads']:
                        scores,baseline=all_scores[:,head],all_masks[:,head]
                        for q in range(b):
                            assert baseline[q].sum()==budget and baseline[q,protected].all()
                            assert scores[q,baseline[q]&~protected].min()>=scores[q,~baseline[q]].max()
                        for epsilon in protocol['epsilon']:
                            check_budget()
                            current={'document':source['id'],'budget':budget,'kv_head':head,'epsilon':epsilon}
                            tw,tc=time.perf_counter(),time.process_time()
                            if epsilon==0:
                                masks=baseline.copy(); info={'identity_original_mask':True,'candidate_universe_tokens':int((~protected[:common_n]).sum())}
                            else:
                                masks,info=run_cover(scores,baseline,protected,common_n,epsilon,check_budget)
                            selection_wall,selection_cpu=time.perf_counter()-tw,time.process_time()-tc
                            row={**current,'method':'independent_identity' if epsilon==0 else 'full_universe_saturated_greedy_cover',
                                 'rank':0,'layer':15,'B':b,'document_sha256':metadata['document_sha256'],
                                 'document_tokens':n,'common_source_tokens':common_n,
                                 'selected_entries':int(masks.sum()),'mandatory_common_tokens':int(protected[:common_n].sum()),
                                 'private_document_mandatory_tokens_per_request':int(protected[common_n:].sum()),
                                 **prior.describe(scores,masks,baseline,common_n),'cover':info,
                                 'selection_wall_seconds':selection_wall,'selection_cpu_seconds':selection_cpu,
                                 'process_peak_rss_kib_so_far':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                 'elapsed_cpu_seconds':time.process_time()-started_cpu,'elapsed_wall_seconds':time.perf_counter()-started_wall}
                            assert all(r['score_loss_fraction']<=epsilon for r in row['per_request'])
                            output.write(json.dumps(row,allow_nan=False)+'\n'); output.flush(); rows.append(row)
                            print(json.dumps({'completed':current,'rows':len(rows),'selection_cpu_seconds':selection_cpu,
                                              'union_reduction':row['actual_common_source_logical']['union_reduction_fraction']}),flush=True)
                            current=None
    except BudgetExpired as exc:
        status='partial_budget_stop'; error=str(exc)
    except Exception as exc:
        status='failed'; error=f'{type(exc).__name__}: {exc}'
        import traceback
        traceback.print_exc()
    assert not torch.cuda.is_initialized()
    summary={'status':status,'error':error,'incomplete_cell':current,'rows':len(rows),'expected_rows':protocol['expected_rows'],
             'protocol_sha256':sha(protocol_path),'script_sha256':sha(Path(__file__)),'results_sha256':sha(BASE/'results.jsonl'),
             'started_utc':started_utc,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'cpu_seconds_post_import':time.process_time()-started_cpu,'wall_seconds_post_import':time.perf_counter()-started_wall,
             'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'uid':os.getuid(),'venv':sys.prefix,
             'numpy_version':np.__version__,'torch_version':torch.__version__,'cuda_initialized':False,'gpu_hours':0,
             'input_provenance':provenance,'scope':protocol['scope'],'self_test_cases':checks['fixture_count']}
    if status=='completed':
        assert len(rows)==protocol['expected_rows']
    (BASE/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    raise SystemExit(1 if status=='failed' else 0)


if __name__=='__main__':
    main()
