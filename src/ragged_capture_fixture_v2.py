"""CPU load/validate real consecutive native AR captures and assemble groups."""
import hashlib
import json
from pathlib import Path

import torch


def load_ragged_fixture(paths, recovery_receipts=()):
    assert 1 <= len(paths) <= 2
    recoveries = {}
    for recovery_name in recovery_receipts:
        recovery_path = Path(recovery_name).resolve()
        rec = json.loads(recovery_path.read_text())
        assert rec['schema'] == 'ragged_capture_recovery_integrity_v1'
        assert rec['status'] == 'capture_integrity_passed_pending_GPU_numerical_qualification'
        assert rec['requires_original_AR_and_full_and_shared_FP64'] is True
        recoveries[str(Path(rec['run']).resolve())] = (rec, str(recovery_path), hashlib.sha256(recovery_path.read_bytes()).hexdigest())
    payloads, receipts = [], []
    for name in paths:
        path = Path(name).resolve()
        summary_path = path.parents[2]/'summary.json'
        summary = json.loads(summary_path.read_text())
        recovery = None
        if summary['status'] != 'completed':
            assert summary['status'] == 'failed'
            recovery = recoveries[str(summary_path.parent)]
            assert recovery[0]['summary_sha256'] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
            assert any(x['path'] == str(path) and x['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest() for x in recovery[0]['files'])
        data = torch.load(path, map_location='cpu', weights_only=True)
        assert data['schema'] == 'real_ar_multitoken_attention_v1'
        worker = next(x for x in summary['workers'] if x['rank']==data['rank'])
        record = next(x for x in worker['files'] if x['layer']==data['layer'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record['sha256']
        if recovery is None:
            native = next(x for x in summary['native_receipts'] if x['rank']==data['rank'])
            layer = next(x for x in native['layers'] if x['layer']==data['layer'])
            assert layer['stride'] == data['original_cache_stride']
            assert layer['host_scales']['_k_scale_float'] == data['k_scale']
            assert layer['host_scales']['_v_scale_float'] == data['v_scale']
        else:
            recovered = next(x for x in recovery[0]['files'] if x['path']==str(path))
            assert recovered['original_stride'] == data['original_cache_stride']
            assert recovered['k_scale']==data['k_scale'] and recovered['v_scale']==data['v_scale']
        assert torch.equal(data['positions'][:,1:], data['positions'][:,:-1]+1)
        assert data['q'].shape[1] >= 8
        payloads.append(data)
        receipts.append(dict(path=str(path),sha256=digest,summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            document_id=summary['args']['document_id'],layer=data['layer'],rank=data['rank'],
            original_run_status=summary['status'], recovery_receipt=None if recovery is None else recovery[1],
            recovery_sha256=None if recovery is None else recovery[2],
            eager_AR_subkernel='unspecified' if recovery is not None else 'see native receipt'))
    first = payloads[0]
    assert all(x['k_scale']==first['k_scale'] and x['v_scale']==first['v_scale'] and
               x['kv_semantic_dtype']==first['kv_semantic_dtype'] and
               x['original_cache_stride']==first['original_cache_stride'] for x in payloads)
    if len(paths)==2:
        assert receipts[0]['document_id'] != receipts[1]['document_id'], 'Two-group input must contain distinct real documents'
    rows_per_group = 8//len(paths)
    caches, tables, queries, outputs, positions, groups, prefixes = [], [], [], [], [], [], []
    offset = 0
    width = max(x['block_tables'].shape[1] for x in payloads)
    for g,data in enumerate(payloads):
        caches.append(data['cache'])
        table = torch.zeros((rows_per_group,width),dtype=torch.int32)
        table[:,:data['block_tables'].shape[1]] = data['block_tables'][:rows_per_group]+offset
        offset += data['cache'].shape[0]
        tables.append(table)
        queries.append(data['q'][:rows_per_group])
        outputs.append(data['native_o'][:rows_per_group])
        positions.append(data['positions'][:rows_per_group])
        groups.extend([g]*rows_per_group)
        prefixes.append(data['common_prefix_tokens'])
    return dict(cache=torch.cat(caches),block_tables=torch.cat(tables),q=torch.cat(queries),
                native_o=torch.cat(outputs),positions=torch.cat(positions),groups=groups,prefixes=prefixes,
                k_scale=first['k_scale'],v_scale=first['v_scale'],stride=first['original_cache_stride'],
                kv_semantic_dtype=first['kv_semantic_dtype'],provenance=receipts)
