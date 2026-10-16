#!/usr/bin/env python3
"""Independent CPU audit of the frozen native physical-address experiment."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/native_physical_alias_independent_audit_v1'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    assert os.getuid() != 0 and sys.prefix != sys.base_prefix
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    checks = []
    inputs = {}

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})
        assert condition, name

    def read(relative):
        path = ROOT / relative
        inputs[str(relative)] = sha(path)
        return json.loads(path.read_text())

    protocol_path = 'configs/native_physical_alias_v1.json'
    protocol = read(protocol_path)
    jobs = read('configs/native_physical_alias_jobs_v1.json')
    for relative, expected in protocol['code_sha256'].items():
        inputs[relative] = sha(ROOT / relative)
        check('frozen_source:' + relative, inputs[relative] == expected)
    ledger = [json.loads(s) for s in (ROOT / 'results/job_ledger.jsonl').read_text().splitlines() if s.strip()]
    matched_ledger = []
    capture_data = {}
    for source in protocol['sources']:
        directory = ROOT / source['directory']
        for relative, expected in source['files_sha256'].items():
            path = directory / relative
            inputs[str(path.relative_to(ROOT))] = sha(path)
            check('frozen_capture:' + str(path.relative_to(ROOT)), inputs[str(path.relative_to(ROOT))] == expected)
        src = torch.load(directory / 'source_l15.pt', map_location='cpu', weights_only=True)
        prefix = src['prefix_k'].shape[2]
        document = src['document_k'].shape[2]
        dec = [torch.load(directory / f'decode_q{i}_l15_s64.pt', map_location='cpu', weights_only=True) for i in range(3)]
        metadata = [d['metadata'] for d in dec]
        fields = ('actual_scheduled_request_ids', 'query_start_loc', 'same_event_query_positions',
                  'sequence_lengths', 'ready_group_anchor_step', 'shared_prefix_end')
        check(source['id'] + ':same_ready_event', all(all(m[k] == metadata[0][k] for k in fields) for m in metadata))
        check(source['id'] + ':cache_validated', all(m['physical_full_prefix_pages_match'] and m['hooked_kv_matches_actual_cache'] for m in metadata))
        tails = [d['private_suffix_k'].shape[2] for d in dec]
        capture_data[source['id']] = {'prefix': prefix, 'document': document, 'tails': tails,
                                      'n': prefix + document + sum(tails), 'directory': directory}
        del src, dec

    cells = []
    for job in jobs:
        argv = job['argv']
        document = argv[argv.index('--document')+1]
        budget = int(argv[argv.index('--budget')+1])
        path = argv[argv.index('--output')+1]
        result = read(path)
        starts = [e for e in ledger if e['name'] == job['name'] and e['event'] == 'start']
        check(job['name'] + ':one_start', len(starts) == 1)
        start = starts[0]
        ends = [e for e in ledger if e['name'] == job['name'] and e['event'] == 'end' and e['pid'] == start['pid']]
        check(job['name'] + ':one_end', len(ends) == 1)
        end = ends[0]
        matched_ledger.extend((start, end))
        check(job['name'] + ':clean_terminal', result['status'] == 'completed' and end['exit_code'] == 0 and not end['timed_out'])
        check(job['name'] + ':ledger_identity', start['gpus'] == end['gpus'] == job['gpus'] and end['category'] == job['category'] and start['argv'] == end['argv'] == argv)
        check(job['name'] + ':ledger_charge', math.isclose(end['gpu_hours'], end['allocated_seconds']*len(job['gpus'])/3600, abs_tol=1e-14))
        check(job['name'] + ':protocol_identity', result['protocol_sha256'] == inputs[protocol_path] and result['document'] == document and result['budget'] == budget)
        check(job['name'] + ':ordinary_venv', result['uid'] != 0 and '/.venv-latest/' in result['python'] and result['CUDA_VISIBLE_DEVICES'] == str(job['gpus'][0]))
        check(job['name'] + ':library', result['packages']['flashinfer-python'] == '0.7.2')
        capture = capture_data[document]
        masks = torch.load(capture['directory'] / f'masks_l15_b{budget}.pt', map_location='cpu', weights_only=True)['supports']
        check(job['name'] + ':mask_shape', list(masks.shape) == [3, 2, capture['document']])
        supports = []
        tail_start = capture['prefix'] + capture['document']
        for request in range(3):
            supports.append([torch.cat((torch.arange(capture['prefix']),
                            torch.where(masks[request, head])[0] + capture['prefix'],
                            torch.arange(tail_start, tail_start + capture['tails'][request]))) for head in range(2)])
            tail_start += capture['tails'][request]
        lengths = [[len(s) for s in row] for row in supports]
        unique = sum(len(torch.unique(torch.cat([supports[r][h] for r in range(3)]))) for h in range(2))
        check(job['name'] + ':support_lengths', lengths == result['logical_access']['per_request_head_tokens'])
        check(job['name'] + ':active_payload_counts', unique == result['logical_access']['shared_unique_tokens'] and sum(map(sum, lengths)) == result['logical_access']['private_unique_tokens'])
        check(job['name'] + ':equal_total_allocation', result['allocation']['same_KV_allocation_every_arm'] and result['allocation']['replicas'] == 3 and result['allocation']['replica_KV_bytes'] == 3*2*capture['n']*128*2*2)
        quals = result['qualifications']
        check(job['name'] + ':six_qualifications', len(quals) == 6 and {q['arm'] for q in quals} == set(protocol['arms']) and all(q['status'] == 'passed' for q in quals))
        check(job['name'] + ':bitwise_O_LSE', len({q['output_sha256'] for q in quals}) == 1 and len({q['raw_lse_sha256'] for q in quals}) == 1)
        check(job['name'] + ':FP64_max_abs_sanity', all(math.isfinite(q['max_abs_error_fp64']) and q['max_abs_error_fp64'] < .006 for q in quals))
        base_plan = result['plans']['shared0']
        for arm in protocol['arms']:
            rotation = int(arm[-1])
            mapping = [(r+rotation)%3 for r in range(3)] if arm.startswith('private') else [rotation]*3
            expected_ids = torch.cat([supports[r][h] + h*capture['n'] + mapping[r]*2*capture['n'] for r in range(3) for h in range(2)]).to(torch.int32)
            digest = hashlib.sha256(expected_ids.numpy().tobytes()).hexdigest()
            qual = next(q for q in quals if q['arm'] == arm)
            plan = result['plans'][arm]
            check(job['name'] + ':indices:' + arm, qual['indices_sha256'] == digest and plan['per_request_copy'] == mapping)
            check(job['name'] + ':same_plan_buffers:' + arm, plan['pointers'] == base_plan['pointers'] and plan['plan_info'] == base_plan['plan_info'] and plan['workspace_bytes'] == base_plan['workspace_bytes'] and plan['resolved_backend'] == 'fa2')
        names = protocol['arms'] + ['shared0_AA', 'private0_AA']
        mutation_expected = {(cache, arm, replica) for cache in ('hot', 'cold') for arm in names for replica in range(3)}
        mutation = result['address_mutation_qualification']
        check(job['name'] + ':all_address_mutations', len(mutation) == 48 and {(q['cache'],q['arm'],q['replica']) for q in mutation} == mutation_expected and all(q['status'] == 'passed' for q in mutation))
        check(job['name'] + ':flush', result['flush_bytes'] > 2*result['L2_bytes'])
        check(job['name'] + ':rows', len(result['measurements']) == 120)
        rng = random.Random(protocol['random_seed'])
        regimes = []
        for cache in ('hot', 'cold'):
            rows = [r for r in result['measurements'] if r['cache'] == cache]
            keys = {(r['block'], r['direction']) for r in rows}
            check(job['name'] + ':paired_rows:' + cache, len(rows) == 60 and keys == {(b,d) for b in range(30) for d in ('random','reverse')})
            matrices = []
            for block in range(30):
                order = names.copy(); rng.shuffle(order)
                block_rows = []
                for direction, expected_order in (('random',order), ('reverse',order[::-1])):
                    row = next(r for r in rows if r['block'] == block and r['direction'] == direction)
                    check(job['name'] + ':order:' + cache + ':' + str(block) + ':' + direction, row['order'] == expected_order and set(row['us']) == set(names) and all(math.isfinite(x) and x > 0 for x in row['us'].values()))
                    block_rows.append([row['us'][name] for name in names])
                matrices.append(block_rows)
            matrix = np.asarray(matrices)
            source_summary = next(s for s in result['summary'] if s['cache'] == cache)
            check(job['name'] + ':repetitions:' + cache, source_summary['repetitions'] == (32 if cache == 'hot' else 1))
            for i,name in enumerate(names):
                check(job['name'] + ':median:' + cache + ':' + name, math.isclose(float(np.median(matrix[:,:,i])), source_summary['median_us'][name], abs_tol=1e-12))
            draws = np.random.default_rng(protocol['bootstrap_seed']).integers(0,30,size=(10000,30))
            pairs = [(f'private{i}',f'shared{i}') for i in range(3)] + [('shared0','shared0_AA'),('private0','private0_AA')]
            recomputed = {}
            for numerator, denominator in pairs:
                a,b = names.index(numerator), names.index(denominator)
                ratio = matrix[:,:,a] / matrix[:,:,b]
                ci = np.quantile(np.median(ratio[draws],axis=(1,2)), [.025,.975])
                key = numerator+'_over_'+denominator
                orig = source_summary['paired_ratios'][key]
                check(job['name'] + ':bootstrap:' + cache + ':' + key, math.isclose(float(np.median(ratio)),orig['median'],abs_tol=1e-12) and np.allclose(ci,orig['paired_block_bootstrap95'],rtol=0,atol=1e-12))
                recomputed[key] = {'median_ratio':float(np.median(ratio)), 'paired_block_bootstrap95':ci.tolist(),
                    'paired_median_delta_us':float(np.median(matrix[:,:,a]-matrix[:,:,b])),
                    'difference_of_marginal_medians_us':float(np.median(matrix[:,:,a])-np.median(matrix[:,:,b]))}
            aa = max(abs(recomputed[key][field]) for key in ('shared0_over_shared0_AA','private0_over_private0_AA') for field in ('paired_median_delta_us','difference_of_marginal_medians_us'))
            for rotation in range(3):
                rec = recomputed[f'private{rotation}_over_shared{rotation}']
                low,high = rec['paired_block_bootstrap95']
                delta = rec['paired_median_delta_us']
                directional = (low > 1 and delta > 0) or (high < 1 and delta < 0)
                rec['CI_direction'] = 'shared_faster' if low > 1 else 'private_faster' if high < 1 else 'unresolved'
                rec['cold_gate_passed'] = bool(directional and abs(delta) >= 2.048 and abs(delta) > aa) if cache == 'cold' else None
            regimes.append({'cache':cache,'rows':60,'AA_max_absolute_shift_us':aa,'comparisons':recomputed})
        cells.append({'name':job['name'],'document':document,'budget':budget,'gpu':job['gpus'][0],
                      'gpu_uuid':result['gpu_uuid'],'terminal':end,'qualifications':6,'address_mutations':48,
                      'max_abs_error_fp64':max(q['max_abs_error_fp64'] for q in quals),
                      'active_unique_payload_bytes':{key:result['logical_access'][key]*512 for key in ('shared_unique_tokens','private_unique_tokens')},
                      'allocated_replica_KV_bytes':result['allocation']['replica_KV_bytes'],
                      'regimes':regimes})
    check('all_eight_distinct_cells_and_devices', len(cells) == 8 and len({(c['document'],c['budget']) for c in cells}) == 8 and len({c['gpu_uuid'] for c in cells}) == 8)
    total = sum(c['terminal']['gpu_hours'] for c in cells)
    check('within_frozen_budget', total <= protocol['maximum_allocated_gpu_hours_including_15s_cleanup'])
    check('CUDA_not_initialized', not torch.cuda.is_initialized())
    report = {'schema':1,'status':'passed','uid':os.getuid(),'python':sys.executable,'audit_GPU_hours':0,
              'source_freeze_commit':'6c9c7da','input_sha256':inputs,'checks':checks,'check_count':len(checks),
              'cells':cells,'totals':{'jobs':8,'clean_jobs':8,'qualifications':48,'address_mutations':384,
                  'paired_rows':960,'timed_graph_samples':7680,'allocated_GPU_hours':total,
                  'ledger_category':'P1 (as predeclared; this audit does not recategorize)',
                  'cold_rotation_comparisons':24,'cold_gate_passed':sum(r['comparisons'][f'private{i}_over_shared{i}']['cold_gate_passed'] for c in cells for r in c['regimes'] if r['cache']=='cold' for i in range(3))},
              'matched_ledger_rows':matched_ledger,
              'limitations':['Single layer/rank contiguous exported BF16 KV with real Q and supports; not original 48-layer engine layout or full model.',
                  'FP64 O is checked by frozen producer assertions and max-error receipts; raw LSE only has cross-arm bitwise equivalence, not a new FP64 LSE oracle.',
                  'Same total resident replicas in every arm. Only accessed addresses vary; active footprint, address mapping, TLB and cache effects are jointly intervened.',
                  'Installation and replica creation are outside measured calls; mandatory output copy is inside. O-only timing excludes LSE.',
                  'Paired random/reverse rows are clustered by block, not 60 independent samples; rotations are not independent documents.',
                  'Different budgets use different GPUs. No causal cross-budget threshold or DRAM/L2-counter claim.',
                  'Bootstrap CIs are pointwise and quantized cold events can yield degenerate intervals. They are not extra precision or familywise inference.',
                  'No independent GPU rerun was performed by this audit.']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(OUT/'audit.json'),'checks':len(checks),'totals':report['totals']}))


if __name__ == '__main__':
    main()
