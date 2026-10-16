"""Bounded FP64 and changed-input CUDA-graph qualification of B8 row extension."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from branchtile.attention import reference
from branchtile.attention_b8 import Attention, make_plan
from branchtile.union_b8 import FixedUnion


def main():
    assert os.geteuid() != 0 and sys.prefix != sys.base_prefix
    torch.set_num_threads(4)
    torch.manual_seed(20261016)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/b8_pilot/kernel_qualification.json')
    parser.add_argument('--representative-only', action='store_true')
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(scope='Exploratory B8 M16/M32/M64 extension; no performance or natural quality claim',
        output_tolerance=dict(atol=.006, rtol=.015), lse_tolerance=dict(atol=.00005, rtol=.00005),
        rows=[], source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), *Path('src/branchtile').glob('*b8.py')]})
    owners = torch.arange(257).remainder(256).repeat(2, 1)
    owners[1] &= 127  # Includes an empty query row and every active M branch.
    owners = owners[:, torch.randperm(257)]
    q = torch.randn(2, 64, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, 257, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    configurations = [(64, 16)] if args.representative_only else [(32, 1), (32, 16), (64, 1), (64, 16)]
    for block_n, splits in configurations:
        for method in ['compact', 'union']:
            row = dict(block_n=block_n, splits=splits, method=method, status='running', checks=[])
            report['rows'].append(row)
            started = time.monotonic()
            try:
                plan = make_plan(owners, device='cuda')
                op = Attention(plan, splits=splits, block_n=block_n) if method == 'compact' else FixedUnion(plan, owners, splits=splits, block_n=block_n)
                op(q, k, v)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual, lse = op(q, k, v)
                for case in ['original', 'changed_qkv', 'changed_physical_mapping', 'all_empty']:
                    expected_owners = owners
                    if case == 'changed_qkv':
                        q.normal_(); k.normal_(); v.normal_()
                    elif case == 'changed_physical_mapping':
                        expected_owners = owners.roll(13, dims=1)
                        new_plan = make_plan(expected_owners, device='cuda')
                        assert new_plan.labels == plan.labels
                        for old, new in zip(plan.tensors, new_plan.tensors):
                            old.copy_(new)
                        if method == 'union':
                            op.owners.copy_(expected_owners)
                            op.lengths.copy_(plan.offsets[:, -1])
                    elif case == 'all_empty':
                        expected_owners = torch.zeros_like(owners)
                        plan.offsets.zero_(); plan.counts.zero_(); plan.owner_rows.zero_()
                        if method == 'union':
                            op.owners.zero_(); op.lengths.zero_()
                    graph.replay()
                    expected, expected_lse = reference(q, k, v, expected_owners, batch=8, gqa=8)
                    torch.testing.assert_close(actual.float(), expected.float(), **report['output_tolerance'])
                    torch.testing.assert_close(lse, expected_lse.float(), **report['lse_tolerance'])
                    row['checks'].append(dict(case=case, max_abs=float((actual.double()-expected).abs().max()), passed=True))
                row['status'] = 'passed'
            except Exception:
                row.update(status='failed', error=traceback.format_exc())
            row['elapsed_seconds'] = time.monotonic()-started
            path.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k != 'error'}), flush=True)
    report['status'] = 'passed' if all(r['status'] == 'passed' for r in report['rows']) else 'failed'
    path.write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(report['status'] != 'passed')


if __name__ == '__main__':
    main()
