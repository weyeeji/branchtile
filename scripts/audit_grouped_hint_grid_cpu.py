"""Independent replay-block audit of the synthetic grouped-hint development grid."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(blocks,numerator,denominator):
    ratios=[b[numerator]/b[denominator] for b in blocks]
    rng=random.Random(20261017)
    boot=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(2000))
    return {'paired_median_ratio':statistics.median(ratios),'bootstrap_95ci':[boot[49],boot[1949]]}


def main():
    assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    output=ROOT/'results/grouped_hint_grid_independent_audit_v1.json';assert not output.exists()
    batches=[6,7,8,9,10,14,16,19,21];prefixes=[65536,81920,120528,153552];tails=[0,8192]
    policy_path=ROOT/'results/xqa_raw_dispatch_policy_v1/policy.json';policy=json.loads(policy_path.read_text())
    assert policy['parameters']=={'min_batch':6,'min_prefix':65536,'max_private_prefix_ratio':.125}
    ledger=[json.loads(line) for line in (ROOT/'results/job_ledger.jsonl').read_text().splitlines()]
    files=[];rows=[]
    required={'paired_bits','AA_10_bits','native_FP64','candidate_FP64','changed_graph_bits','final_bits','revisit_after_other_batch_shapes_bits'}
    for prefix in prefixes:
        for tail in tails:
            path=ROOT/f'results/grouped_hint_grid_v1/p{prefix}t{tail}.json';data=json.loads(path.read_text())
            assert data['status']=='qualified_and_measured'
            assert data['args']['capture'] is None and data['layout']=='synthetic Q and KV'
            assert data['args']['prefix']==prefix and data['args']['extra_tail']==tail
            assert data['args']['blocks']==20 and [c['batch'] for c in data['cases']]==batches
            for name,expected in data['source_sha256'].items():assert sha(ROOT/name)==expected,name
            ends=[r for r in ledger if r.get('event')=='end' and '--output' in r.get('argv',[]) and
                (ROOT/r['argv'][r['argv'].index('--output')+1]).resolve()==path.resolve()]
            assert len(ends)==1 and ends[0]['exit_code']==0 and not ends[0]['timed_out']
            files.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'ledger_terminal_receipt':ends[0],
                'source_sha256':data['source_sha256'],'build_identity':data['build_identity']})
            for c in data['cases']:
                b=c['batch'];assert c['prefix']==prefix and len(c['lengths'])==len(c['rows'])==b
                assert len(set(c['rows']))==b and required<=set(c['checks'])
                assert all(value is True for value in c['checks'].values())
                assert c['S']==max(1,170//(b*2)) and c['paired_b']==b
                assert c['prefix_SM_parameter']==c['S']*((b+1)//2)*2
                assert c['tail_SM_parameter']==c['S']*b*2
                if b%2:assert c['checks']['dummy_Q_stays_zero']
                hot=c['hot'];blocks=hot['blocks_ms'];assert len(blocks)==len(hot['orders'])==20
                for block,order in zip(blocks,hot['orders'],strict=True):
                    assert set(block)==set(order)=={'native','native_AA','candidate'}
                    assert all(math.isfinite(v) and v>0 for v in block.values())
                assert hot['median_ms']=={k:statistics.median(r[k] for r in blocks) for k in blocks[0]}
                ratio,aa=paired(blocks,'native','candidate'),paired(blocks,'native','native_AA')
                assert ratio==hot['native_over_candidate'] and aa==hot['AA']
                private=max(c['lengths'])-prefix
                admitted=b>=6 and prefix>=65536 and private>0 and private*8<=prefix
                rows.append({'prefix':prefix,'extra_tail':tail,'max_actual_private_tokens':private,'batch':b,
                    'checks':c['checks'],'old_policy_admits':admitted,'native_over_candidate':ratio,'native_AA':aa,
                    'native_median_us':hot['median_ms']['native']*1000,'candidate_median_us':hot['median_ms']['candidate']*1000,
                    'setup_wall_seconds':c['setup_wall_seconds'],'capture_wall_seconds':c['capture_wall_seconds']})
    assert len(rows)==72 and len({(r['prefix'],r['extra_tail'],r['batch']) for r in rows})==72
    admitted=[r for r in rows if r['old_policy_admits']]
    summary=lambda items:{'cells':len(items),
        'median_ratio_range':[min(r['native_over_candidate']['paired_median_ratio'] for r in items),max(r['native_over_candidate']['paired_median_ratio'] for r in items)],
        'minimum_paired_bootstrap_lower95':min(r['native_over_candidate']['bootstrap_95ci'][0] for r in items),
        'lower95_at_most_one':[r for r in items if r['native_over_candidate']['bootstrap_95ci'][0]<=1],
        'lower95_below_previous_1_05_development_margin':[r for r in items if r['native_over_candidate']['bootstrap_95ci'][0]<1.05]}
    report={'status':'qualified_measured_grid_independently_audited','source_sha256':sha(Path(__file__)),
        'old_policy_sha256':sha(policy_path),'files':files,'cells':rows,'all_cells':summary(rows),'old_policy_admitted_cells':summary(admitted),
        'old_policy_rejected_cells':summary([r for r in rows if not r['old_policy_admits']]),
        'limits':['Synthetic Q/KV grid; no unseen document output or whole-model latency is measured.',
            'All timings are hot complete O-only CUDA graph calls with Q padding/copies. Setup/capture are separate; no cold-cache or full engine claim.',
            '20 randomized paired blocks with10 replays estimate this fixed-input timing distribution; bootstrap intervals are not document-level or implementation-correctness guarantees.',
            'Old policy is evaluated unchanged, not refitted here. Its previous minimum1.05 margin need not survive a new grouped-hint implementation.',
            'Only nine batch sizes and two tail endpoints per prefix are covered; missing batches/intermediate or longer tails are not qualified by interpolation.',
            'Passing72 synthetic cells does not prove the bitwise contract for arbitrary states; the known eval2 regression and full new-validation trajectories remain mandatory.']}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'all_cells':report['all_cells'],'old_policy_admitted_cells':report['old_policy_admitted_cells']},indent=2))


if __name__=='__main__':main()
