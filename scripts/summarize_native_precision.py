"""Descriptive native KV precision pilot; preserve all scheduled arms and failures."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default='results/native_precision_natural_development_v1');a=parser.parse_args()
    folder=ROOT/a.directory
    jobs_path=ROOT/'configs/native_precision_natural_jobs_v1.json'
    jobs=json.loads(jobs_path.read_text())
    ledger_path=ROOT/'results/job_ledger.jsonl'
    ledger=[json.loads(line) for line in ledger_path.read_text().splitlines()]
    rows=[]
    for job in jobs:
        name=Path(job['argv'][-1]).name
        directory=folder/name
        path=directory/'summary.json'
        events=[event for event in ledger if event['name']==job['name']]
        starts=[e for e in events if e['event']=='start'];ends=[e for e in events if e['event']=='end']
        assert len(starts)<=1 and len(ends)<=1
        row={'arm':name,'job_name':job['name'],'gpus':job['gpus'],'ledger_events':events,
             'source':str(path.relative_to(ROOT)),'source_sha256':sha(path) if path.exists() else None,
             'status':'pending','all_scheduled_questions_in_denominator':3}
        rows.append(row)
        if not path.exists():continue
        data=json.loads(path.read_text())
        row.update(document_id=data['document_id'],method=data['methods'][0],driver_status=data['status'],
                   cold_AA=data.get('cold_AA'),runs_completed=len(data['runs']),
                   initialization_wall_seconds=data.get('initialization_wall_seconds'),
                   input_preparation_wall_seconds=data.get('generator_input_preparation_wall_seconds'),
                   warmup_wall_seconds=data.get('stochastic_sampler_warmup_wall_seconds'))
        row['status']='completed' if data['status']=='completed' and ends and ends[0]['exit_code']==0 and not ends[0]['timed_out'] else ('failed' if ends else 'running')
        if ends:row['allocated_GPU_hours']=ends[0]['gpu_hours']
        if not data['runs']:continue
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_vllm_sparse_natural.py'),'--directory',str(directory)],check=True,cwd=ROOT,stdout=subprocess.DEVNULL)
        scored_path=directory/'natural_execution_summary.json'
        scored=json.loads(scored_path.read_text());row['scored_sha256']=sha(scored_path)
        row['runs']=[{'repeat':r['repeat'],'phase':r['phase'],'exact':r['score']['exact_accuracy'],
                      'rounded':r['score']['rounded_accuracy'],'all_natural_stop':r['all_natural_stop'],
                      'cap_hit':r['any_cap_hit'],'output_tokens':r['total_output_tokens'],
                      'generate_seconds':r['complete_generate_wall_seconds'],
                      'cold_anchor_plus_generate_seconds':r['anchor_plus_generate_wall_seconds'],
                      'anchor_seconds':r['dense_anchor_wall_seconds'],
                      'APC_hits':[ans['num_cached_tokens'] for ans in r['answers']],
                      'answer_tokens':[ans['output_tokens'] for ans in r['answers']],
                      'answer_eligible':[ans['completion_eligible'] for ans in r['answers']],
                      'answer_exact':[ans['score']['exact_match'] for ans in r['answers']],
                      'answer_rounded':[ans['score']['rounded_match'] for ans in r['answers']]}
                     for r in scored['rows']]
        if row['status']=='completed' and data.get('cold_AA',{}).get('all_equal'):
            row['median_same_trajectory_generate_seconds']=statistics.median(r['generate_seconds'] for r in row['runs'])
            row['median_same_trajectory_cold_anchor_plus_generate_seconds']=statistics.median(r['cold_anchor_plus_generate_seconds'] for r in row['runs'])
        if 'precision_after_generation' in data:
            row['memory']=[{key:worker[key] for key in ('rank','cache_unique_storage_bytes','allocated_cache_token_capacity','allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes')}
                           for worker in data['precision_after_generation']]
            row['CPU_identity_wall_seconds_by_rank']=[sum(r['CPU_identity_wall_seconds'] for r in worker['admission_identity_records']) for worker in data['precision_after_generation']]
    comparisons=[]
    for prefix in ('original','new'):
        bf=next(row for row in rows if row['arm']==prefix+'_bf16')
        fp=next(row for row in rows if row['arm']==prefix+'_fp8')
        comp={'document':prefix,'BF16_status':bf['status'],'FP8_status':fp['status']}
        if bf['status']==fp['status']=='completed':
            comp['cross_precision_APC_hits_equal']=[a['APC_hits']==b['APC_hits'] for a,b in zip(bf['runs'],fp['runs'])]
            if 'median_same_trajectory_generate_seconds' in bf and 'median_same_trajectory_generate_seconds' in fp:
                comp['descriptive_BF16_over_FP8_generation_ratio']=bf['median_same_trajectory_generate_seconds']/fp['median_same_trajectory_generate_seconds']
                comp['descriptive_BF16_over_FP8_anchor_plus_generation_ratio']=bf['median_same_trajectory_cold_anchor_plus_generate_seconds']/fp['median_same_trajectory_cold_anchor_plus_generate_seconds']
            comp['scope']='Different precision, output trajectories and GPU pairs; descriptive joint quality/cost comparison, not an isolated executor causal effect or quality noninferiority.'
        comparisons.append(comp)
    result={'status':'completed' if all(r['status']=='completed' for r in rows) else 'incomplete_or_failed',
            'jobs_manifest_sha256':sha(jobs_path),'ledger_snapshot_sha256':sha(ledger_path),
            'independent_documents':2,'questions':6,'seeds':1,'AA_repeats_per_arm':2,
            'AA_repeats_do_not_increase_quality_sample_size':True,'rows':rows,'comparisons':comparisons,
            'total_terminal_allocated_GPU_hours':sum(r.get('allocated_GPU_hours',0) for r in rows),
            'limitations':['Known development documents, not a formal evaluation; no noninferiority claim.',
                'Official uncalibrated unit-scale FP8 KV baseline, not a new method.',
                'Pair/precision confounding within document; no cross-device causal precision effect.',
                'Cold means reset APC after warm engine; anchor plus group is a sum of two measured phases.',
                'Initialization/JIT/warmup, CPU preparation and ledger allocation remain separate.',
                'Minimal CPU metadata/hash observations remain in timings. No GPU prompt readback or Q/O/logit snapshots.',
                'A/A repetitions describe reproducibility, not two independent quality samples.']}
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'rows':[{k:r.get(k) for k in ('arm','status','runs_completed','cold_AA','runs')} for r in rows]},indent=2))


if __name__=='__main__':main()
