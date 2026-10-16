#!/usr/bin/env python3
"""Summarize every frozen alias cell, retaining quantization and A/A controls."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/native_physical_alias_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    protocol=json.loads((ROOT/'configs/native_physical_alias_v1.json').read_text())
    jobs=json.loads((ROOT/'configs/native_physical_alias_jobs_v1.json').read_text())
    ledger=list(map(json.loads,(ROOT/'results/job_ledger.jsonl').read_text().splitlines()))
    results=[];cells=[];cost=0.;files={}
    for job in jobs:
        path=ROOT/job['argv'][-1]
        r=json.loads(path.read_text());files[str(path.relative_to(ROOT))]=sha(path)
        assert r['status']=='completed'
        assert r['protocol_sha256']==sha(ROOT/'configs/native_physical_alias_v1.json')
        assert len(r['qualifications'])==6 and len(r['address_mutation_qualification'])==48
        assert len({x['output_sha256'] for x in r['qualifications']})==1
        assert len({x['raw_lse_sha256'] for x in r['qualifications']})==1
        assert all(x['status']=='passed' for x in r['address_mutation_qualification'])
        ends=[e for e in ledger if e.get('event')=='end' and e.get('name')==job['name']]
        assert len(ends)==1 and ends[0]['exit_code']==0 and not ends[0]['timed_out']
        cost+=ends[0]['gpu_hours']
        log=BASE/'job_logs'/f"{job['name']}.txt";log.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'logs'/f"{job['name']}.log",log)
        files[str(log.relative_to(ROOT))]=sha(log)
        access=r['logical_access'];size=access['payload_bytes_per_token']/2**20
        cells.append(dict(document=r['document'],budget=r['budget'],gpu=job['gpus'][0],
            source_sha256=files[str(path.relative_to(ROOT))],
            shared_unique_payload_MiB=access['shared_unique_tokens']*size,
            private_unique_payload_MiB=access['private_unique_tokens']*size,
            total_replica_allocation_MiB=r['allocation']['replica_KV_bytes']/2**20))
        for cache in ['hot','cold']:
            rows=[x for x in r['measurements'] if x['cache']==cache]
            assert len(rows)==60 and {(x['block'],x['direction']) for x in rows}=={(i,d) for i in range(30) for d in ['random','reverse']}
            rows.sort(key=lambda x:(x['block'],x['direction']))
            aa={kind:statistics.median(x['us'][kind+'0']-x['us'][kind+'0_AA'] for x in rows) for kind in ['shared','private']}
            for rotation in range(3):
                a,b=f'private{rotation}',f'shared{rotation}'
                ratio=np.array([x['us'][a]/x['us'][b] for x in rows]).reshape(30,2)
                diff=np.array([x['us'][a]-x['us'][b] for x in rows]).reshape(30,2)
                draws=np.random.default_rng(protocol['bootstrap_seed']).integers(0,30,(10000,30))
                ci=np.quantile(np.median(ratio[draws],axis=(1,2)),[.025,.975]).tolist()
                med=float(np.median(diff))
                resolution=2.048 if cache=='cold' else 2.048/protocol['hot_repetitions']
                directional=(ci[0]>1 or ci[1]<1) and abs(med)>=resolution and abs(med)>max(map(abs,aa.values()))
                results.append(dict(document=r['document'],budget=r['budget'],cache=cache,rotation=rotation,
                    private_median_us=statistics.median(x['us'][a] for x in rows),
                    shared_median_us=statistics.median(x['us'][b] for x in rows),
                    private_over_shared=float(np.median(ratio)),paired_block_bootstrap95=ci,
                    median_paired_private_minus_shared_us=med,AA_paired_median_difference_us=aa,
                    conservative_event_resolution_us=resolution,
                    passes_directional_resolution_and_AA_gate=directional))
    out=dict(protocol_sha256=sha(ROOT/'configs/native_physical_alias_v1.json'),source_files_sha256=files,
        allocated_gpu_hours=cost,cell_count=len(cells),numeric_qualifications=48,address_mutation_qualifications=384,
        timed_graph_windows=8*120*8,timed_attention_invocations=8*60*8*33,paired_statistic_rows=results,cells=cells,
        limitations=['One real layer/step, two documents; eight devices are eight different cells, not independent replication of every cell.',
            'Hot repeats one working set. Cold is one call after flush; event timings have prominent 2.048us quantization. Bootstrap intervals omit systematic/quantization uncertainty.',
            'Different active footprints and address mappings may affect TLB, partitioning and multiple cache levels. No DRAM bytes or L2 hit rate were measured.',
            'Index installation is outside timing; full source replicas stay allocated in every arm. Not a capacity, energy, end-to-end, or new-method gain.'])
    (BASE/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# Native E0 physical KV alias control', '',
        'All eight frozen cells completed cleanly. Numerical O/raw-LSE bitwise checks: **48/48**; graph address-sensitive V mutations: **384/384**. GPU allocation (including imports/initialization/qualification): **%.6f GPUh**.'%cost,'',
        'Each pair uses one FlashInfer 0.7.2 native page-one sparse E0 wrapper, plan, workspace, query, output/copy destination and index buffer. This is not the vLLM 0.31 native dense page16 XQA configuration. All B physical replicas remain allocated. Only prepared index values change outside the timing interval. Private mappings and shared-copy base addresses both rotate; A/A covers shared0 and private0. The output copy is timed.','',
        '| Document | Budget | Shared/private unique payload MiB | Hot private/shared ratio (3 rotations) | Cold ratio (3 rotations) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for c in cells:
        subsets={cache:[r['private_over_shared'] for r in results if r['document']==c['document'] and r['budget']==c['budget'] and r['cache']==cache] for cache in ['hot','cold']}
        lines.append('| %s | %dK | %.2f / %.2f | %.3f–%.3f | %.3f–%.3f |'%(c['document'],c['budget']//1024,c['shared_unique_payload_MiB'],c['private_unique_payload_MiB'],min(subsets['hot']),max(subsets['hot']),min(subsets['cold']),max(subsets['cold'])))
    lines+=['','The cold result shows that the existing native E0 already benefits from shared physical addresses. At 32K the same is evident in hot repeated calls. Below that hot point, small effects change sign across address rotations; these are not a robust new speedup. The logical private working set at 32K is about 96.4 MiB versus about 40 MiB shared, near this GPU’s 96 MiB L2 capacity, but that coincidence does **not** identify L2 hit rates or isolate cache capacity from other address effects.','',
        'This explains why a comparison against private-copy execution can exaggerate the incremental opportunity for a new sharing kernel. It does not prove that all possible fusion gains are exhausted. The earlier dynamic/joint stop decisions remain unchanged.','',
        '## Limits','']+['- '+x for x in out['limitations']]+['',
        'Raw per-call measurements, source hashes, rotations, A/A, and all confidence intervals are in `analysis.json` and the eight cell files. `job_logs/*.txt` preserves every terminal log. Protocol/code were committed before execution as `6c9c7da`. No source outputs, supports, tolerances or frozen evaluation data were changed.','']
    (BASE/'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:out[k] for k in ['allocated_gpu_hours','cell_count','numeric_qualifications','address_mutation_qualifications']}))


if __name__=='__main__':main()
