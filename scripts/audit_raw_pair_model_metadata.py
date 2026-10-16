"""Independent metadata/graph/epoch audit of completed general model receipts.

Qualification timings include intrusive synchronization and are never treated as
production latency evidence. Numerical gates are independently checked per rank.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.vllm_native_raw_pair_general_setup import REQUIRED_QUALIFICATION_FIELDS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    audited=[]
    for name in args.inputs:
        summary=json.loads(Path(name).read_text())
        assert summary['status']=='completed', (name,summary['status'])
        for run in summary['runs']:
            for worker in run['workers']:
                path=Path(worker['raw_pair_record']);r=json.loads(path.read_text())
                checks=r['checks'];prepares=r['prepare_records'];bindings=[p for p in prepares if p['prefix_binding'] is not None]
                assert r['phase_valid'] and all(c['qualification_passed'] for c in checks)
                assert all(all(c[k] is True for k in REQUIRED_QUALIFICATION_FIELDS) for c in checks)
                assert len(bindings)<=1, 'Physical prefix check repeated within one fixed cohort phase'
                assert all(p['native_GPU_lengths_aliased'] and not p['host_length_copy'] for p in prepares)
                assert all(p['host_physical_prefix_check']==(p['prefix_binding'] is not None) for p in prepares)
                assert all(p['cohort_request_slot_epoch_subset_guard'] for p in prepares)
                assert r['plan']['workspace_allocation_sets']==r['plan']['raw_state_allocation_sets']==1
                assert r['plan']['workspace_bytes']==64*1024*1024
                assert all(all(layers==list(range(48)) for layers in by_b.values())
                           for by_b in r['captured_layers_by_batch'].values())
                assert all(hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha
                           for file,sha in r['source_sha256'].items()), 'Runtime source changed after record'
                native_sm=next(iter(r['plan']['plans'].values()))['native_SM']
                for p in prepares:
                    b=p['batch'];plan=r['plan']['plans'][str(b)]
                    expected=min(max(1,native_sm//(b*2)),(p['native_table_columns']*16+255)//256)
                    assert p['native_splits']==plan['native_splits']==expected
                    assert plan['tail_SM']//(b*2)==expected
                normal=[p['complete_prepare_host_seconds'] for p in prepares if p['prefix_binding'] is None]
                audited.append({'summary':name,'receipt':str(path),'rank':r['rank'],'phase':r['phase'],
                    'receipt_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'strict_states':len(checks),'states_by_batch':dict(Counter(c['batch'] for c in checks)),
                    'qualified_batches':r['qualified_batches'],'candidate_replays':r['candidate_replays'],
                    'prepare_records':len(prepares),'full_physical_bindings':len(bindings),
                    'first_binding_prepare_seconds':bindings[0]['complete_prepare_host_seconds'] if bindings else None,
                    'subsequent_prepare_median_seconds':statistics.median(normal) if normal else None,
                    'fallback_reasons':dict(Counter(f['reason'] for f in r['fallbacks'])),
                    'workspace_bytes':r['plan']['workspace_bytes'],'raw_state_bytes':r['plan']['raw_state_bytes'],
                    'host_length_copies':0,'native_length_aliases_and_original_B_splits_checked':True,
                    'timing_scope':'Intrusive qualification timings are not performance evidence.' if checks else 'Measured phase; host timestamps may include upstream wait, not additive causal overhead.'})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'status':'audited','records':audited},indent=2)+'\n')
    print(json.dumps({'output':str(output),'audited_receipts':len(audited)}))


if __name__=='__main__': main()
