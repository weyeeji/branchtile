"""Audit natural B8 readiness, logical fragmentation, and bounded replay results."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    manifest_path = Path('data/manifests/docfinqa_training_large_group_pilot_v1.json')
    manifest = read(manifest_path)
    report = dict(scope='Natural B8 exploratory rejection screen; captured dense trajectories are not sparse quality',
        manifest=str(manifest_path), manifest_sha256=hashlib.sha256((ROOT/manifest_path).read_bytes()).hexdigest(),
        captures=[], replays=[], limitations=[
            'Original proposal and frozen formal cohorts used B<=4. This explicitly explores an expanded natural training scope.',
            '256-token instrumented capture has no application performance or quality meaning.',
            'Logical tile counts ignore runtime and split; 4140 fixed partitions exhaust the B8 combinatorial tile model only.',
            'GPU replay includes independent native and fused union controls, not all 4140 partitions or all exact hierarchies.',
            'A loss to a qualified control rejects this bounded candidate; a win would require complete controls and application validation.'])
    for group in manifest['groups']:
        index = group['original_training_group_index']
        name = f'main30b_tp2_train{index}_b8_256'
        platform = read(f'results/platform/capture_{name}.json')
        assert platform['status'] == 'completed' and platform['input']['manifest_sha256'] == report['manifest_sha256']
        assert platform['input']['B'] == 8 and not platform['input']['truncated_document']
        output = f'results/captures/{name}_export' + ('_retry1' if index == 1 else '')
        item = dict(original_training_index=index, document_id=group['document_id'],
            prompt_lengths=platform['input']['prompt_lengths'], ranks=[], stage_instrumented_seconds={s['name']:s['seconds'] for s in platform['stages']})
        signatures = {}
        for rank in [0, 1]:
            raw_dir = ROOT/f'results/captures/{name}/rank{rank}'
            raw = json.loads((raw_dir/'metadata.json').read_text())
            meta = read(f'{output}/rank{rank}/metadata.json')
            summary = read(f'{output}/rank{rank}/export_summary.json')
            assert meta['B'] == 8 and meta['actual_tp_rank'] == rank
            entries = {e['file']:e for e in raw['files']}
            warm = sorted([rid for rid, r in raw['requests'].items() if r['phase'] == 'apc_warm_natural_group'],
                key=lambda rid: raw['requests'][rid]['question_index'])
            assert len(warm) == 8
            for event in summary['ready_groups']:
                event_rows = []
                for rid in warm:
                    file = f'layer{event["layer"]}_ready_anchor{event["anchor"]}_request{rid}.pt'
                    assert hashlib.sha256((raw_dir/file).read_bytes()).hexdigest() == entries[file]['sha256']
                    x = torch.load(raw_dir/file, weights_only=True, map_location='cpu')
                    m = x['metadata']
                    assert m['hooked_kv_matches_actual_cache'] and m['physical_full_prefix_pages_match']
                    assert set(m['actual_scheduled_request_ids']) == set(warm)
                    assert m['query_start_loc'] == list(range(9))
                    assert all(len(pos) == 1 for pos in m['same_event_query_positions'])
                    assert x['private_k'].shape[1] == m['query_token_position']+1-m['shared_prefix_end']
                    event_rows.append({k:m[k] for k in ['actual_scheduled_request_ids','query_start_loc','same_event_query_positions','sequence_lengths']})
                assert all(row == event_rows[0] for row in event_rows)
                key = (event['layer'], event['anchor'])
                if rank == 0:
                    signatures[key] = event_rows[0]
                else:
                    assert signatures[key] == event_rows[0]
            assert len(summary['ready_groups']) == 12
            opportunities = read(f'{output}/rank{rank}/opportunity.json')
            logical = []
            for row in opportunities:
                counts = row['owner_token_counts']
                assert len(counts) == 256
                actual_m_tiles = sum(((n+63)//64) * (1 if owner.bit_count() <= 2 else 2 if owner.bit_count() <= 4 else 4)
                    for owner,n in enumerate(counts) if owner)
                logical.append(dict(layer=row['layer'], budget=row['budget'], kv_head=row['kv_head'],
                    nonempty_owner_classes=sum(n>0 for n in counts[1:]),
                    ideal_compact_over_best_fixed=row['compact_vs_best_fixed_tile_ratio'],
                    actual_M16_M32_M64_over_best_fixed=actual_m_tiles/row['logical_best_fixed_partition_m16_n64_tiles'],
                    best_logical_fixed_partition=row['best_fixed_partition']))
            item['ranks'].append(dict(rank=rank, snapshot_checks=summary['snapshot_checks'],
                all_ready_events_cross_rank_verified=True, ready_groups=summary['ready_groups'], logical_opportunities=logical))
        report['captures'].append(item)
        for budget in [4096,16384]:
            path = Path(f'results/b8_pilot/train{index}_l15_s64_b{budget}.json')
            if (ROOT/path).exists():
                result = read(path)
                assert result['provenance']['ready_lifetime_measured'] and result['q_shape'] == [2,64,128]
                assert result['provenance']['original_active_requests'] == list(range(8))
                report['replays'].append(dict(group_index=index,budget=budget,path=str(path),
                    qualified_configurations=sum(r['status']=='ok' for r in result['tuning']),
                    failed_configurations=[r for r in result['tuning'] if r['status']!='ok'],
                    summary=result.get('summary'), status=result.get('status','completed')))
    events = [json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()]
    events = [e for e in events if 'b8' in e['name']]
    ended = {(e['name'],e['pid']):e for e in events if e['event']=='end'}
    report['budget'] = dict(ended_jobs=len(ended), active_jobs=[e for e in events if e['event']=='start' and (e['name'],e['pid']) not in ended],
        allocated_gpu_hours={c:sum(e['gpu_hours'] for e in ended.values() if e['category']==c) for c in ['P1','P2']})
    (ROOT/'results/b8_pilot/summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(captures=len(report['captures']),replays=len(report['replays']),budget=report['budget']),indent=2))


if __name__ == '__main__':
    main()
