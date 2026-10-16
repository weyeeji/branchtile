#!/usr/bin/env python3
"""CPU-only audit of all six bounded same-native E0 mechanism cells."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/joint_e0_gate_v1'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert os.getuid()!=0 and sys.prefix!=sys.base_prefix
    initial=json.loads((BASE/'jobs.json').read_text())
    jobs=json.loads((BASE/'jobs_retry1.json').read_text())
    protocol=json.loads((BASE/'protocol_retry1.json').read_text())
    rows=[];qualification=[];raw_distribution=[];result_hashes={};preparation=[]
    for job in jobs:
        path=BASE/(job['name']+'.json');result=json.loads(path.read_text())
        result_hashes[path.name]=sha(path)
        assert result['status']=='completed'
        assert result['protocol_sha256']==sha(BASE/'protocol_retry1.json')
        assert len(result['qualification'])==24 and all(q['status']=='passed' for q in result['qualification'])
        assert len(result['measurements'])==640 and len(result['summary'])==16
        assert result['flush_bytes']>2*result['L2_bytes']
        for order in protocol['batching_orders']:
            plans=[result['plans'][order+'/'+method] for method in protocol['methods']]
            assert all(p['plan_info']==plans[0]['plan_info'] for p in plans)
            assert all(p['support_lengths']==plans[0]['support_lengths'] for p in plans)
            assert all(p['resolved_backend']=='fa2' and p['kv_aliases_source'] for p in plans)
        qualifier={'document':result['document'],'budget':result['budget']}
        qualification.extend(qualifier|q for q in result['qualification'])
        preparation.append(qualifier|{'selectors':result['selector_preparation'],'plans':result['plans'],
                                    'export_query_conversion_seconds':result['export_HB_to_production_BH_query_preparation_seconds']})
        for item in result['summary']:
            subset=[m for m in result['measurements'] if all(m[k]==item[k] for k in ('batching_order','contract','call_scope','cache'))]
            assert len(subset)==40
            for block in range(20):
                pair=[m for m in subset if m['block']==block]
                assert len(pair)==2 and {m['direction'] for m in pair}=={'random','reverse'}
                random=next(m for m in pair if m['direction']=='random')
                reverse=next(m for m in pair if m['direction']=='reverse')
                assert random['order']==list(reversed(reverse['order']))
                assert all(set(m['us'])=={'independent','exchange','cover','independent_AA'} for m in pair)
                assert all(np.isfinite(value) and value>0 for m in pair for value in m['us'].values())
            for denominator in ('exchange','cover','independent_AA'):
                actual=np.median([m['us']['independent']/m['us'][denominator] for m in subset])
                assert actual==item['paired_ratios']['independent_over_'+denominator]['median']
            rows.append(qualifier|item)
            if item['cache']=='cold':
                hist={}
                for method in ('independent','exchange','cover','independent_AA'):
                    values=np.array([m['us'][method] for m in subset])
                    unique,counts=np.unique(values,return_counts=True)
                    order=np.argsort(-counts,kind='stable')
                    hist[method]={'unique_measured_us':len(unique),
                                  'top_observed_times':[{'us':float(unique[i]),'count':int(counts[i])} for i in order[:5]],
                                  'within_1e_minus5_us_of_2_048us_multiple':int(np.sum(np.abs(values/2.048-np.rint(values/2.048))<1e-5/2.048))}
                raw_distribution.append(qualifier|{k:item[k] for k in ('batching_order','contract','call_scope','cache')}|{'histogram':hist})
    names={j['name'] for j in initial+jobs}
    events=[r for r in map(json.loads,(ROOT/'results/job_ledger.jsonl').read_text().splitlines()) if r.get('name') in names]
    starts={(r['name'],r['pid']) for r in events if r['event']=='start'}
    ends=[r for r in events if r['event']=='end']
    assert len(starts)==len(ends)==12 and all((r['name'],r['pid']) in starts for r in ends)
    initial_names={j['name'] for j in initial};retry_names={j['name'] for j in jobs}
    assert all(r['exit_code']==0 and not r['timed_out'] for r in ends if r['name'] in retry_names)
    assert all(r['exit_code']==1 and not r['timed_out'] for r in ends if r['name'] in initial_names)
    accounting={'P2_GPU_hours_including_initial_failures':sum(r['gpu_hours'] for r in ends),
                'initial_failure_GPU_hours':sum(r['gpu_hours'] for r in ends if r['name'] in initial_names),
                'successful_retry_GPU_hours':sum(r['gpu_hours'] for r in ends if r['name'] in retry_names),'events':events}
    assert accounting['P2_GPU_hours_including_initial_failures']<.15
    aggregates=[]
    for order in protocol['batching_orders']:
        for contract in protocol['contracts']:
            for scope in protocol['call_scopes']:
                for cache in ('hot','cold'):
                    rr=[r for r in rows if (r['batching_order'],r['contract'],r['call_scope'],r['cache'])==(order,contract,scope,cache)]
                    assert len(rr)==6
                    for method in ('exchange','cover','independent_AA'):
                        ratios=[r['paired_ratios']['independent_over_'+method] for r in rr]
                        values=[r['median'] for r in ratios]
                        aggregates.append({'batching_order':order,'contract':contract,'call_scope':scope,'cache':cache,'method':method,
                            'cells':6,'paired_ratio_min_median_max':[min(values),float(np.median(values)),max(values)],
                            'within_run_CI_above_one':sum(r['paired_block_bootstrap95'][0]>1 for r in ratios),
                            'within_run_CI_below_one':sum(r['paired_block_bootstrap95'][1]<1 for r in ratios),
                            'within_run_CI_including_one':sum(r['paired_block_bootstrap95'][0]<=1<=r['paired_block_bootstrap95'][1] for r in ratios),
                            'within_run_CI_above_1_15':sum(r['paired_block_bootstrap95'][0]>1.15 for r in ratios)})
    summary={'status':'six_cells_completed_after_preserved_initial_import_assertion_failures',
             'result_hashes':result_hashes,'qualified_comparisons':len(qualification),'paired_measurement_rows':6*640,
             'qualification_max_output_absolute_error':max(q['max_output_abs_error_fp64'] for q in qualification),
             'qualification_max_LSE_absolute_error':max(q.get('max_lse_abs_error_fp64',0) for q in qualification),
             'same_fa2_TC_defaultsplit_plan_within_order':True,'actual_shared_payload_aliasing_verified':True,
             'all_per_request_support_lengths_equal_across_methods':True,'accounting':accounting,
             'engine_scope_limitations':['BH complete_layout returns a view only; it does not include the production engine destination output.copy_. HB includes permutation copies, not full engine integration.',
                 'This uses FlashInfer 0.7.2 with contiguous exported KV and external graph capture; production comparison uses FlashInfer 0.7.0, strided engine KV and shared-plan integration.',
                 'Preparation records are partial disjointly scoped observations, not a complete decomposition and must not be summed into total preparation cost.'],
             'preparation_scope_complete':False,
             'all_cells':rows,'aggregates_without_pooling_contracts_orders_caches':aggregates,'preparation':preparation,
             'cold_observed_timing_histograms':raw_distribution,
             'interpretation':'Conditional cold-cache signal exists, notably all three 16K production-BH/O-only cells; cannot claim no same-kernel signal. Hot gains are small and mixed. Repeated cold timing levels and sometimes degenerate bootstrap intervals limit within-run precision claims. No repeated-run/cross-GPU stability or full-engine/cache-layout claim; no additional GPU stage authorized by this summary.',
             'scope':protocol['scope']}
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','qualified_comparisons','paired_measurement_rows','qualification_max_output_absolute_error','qualification_max_LSE_absolute_error')}) )
    print(json.dumps({k:v for k,v in accounting.items() if k!='events'}))


if __name__=='__main__':main()
