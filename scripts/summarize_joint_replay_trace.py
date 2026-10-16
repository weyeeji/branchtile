"""CPU-only attribution of actual production E0 model traces.

Output copies are identified by adjacency after a native FI merge on the same
device/context/stream AND correlation/graph/node identity, not by summing every
memcpy. Kernel-duration sums and full-model graph event durations are separate
denominators; neither is uninstrumented application JCT.
"""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def union_us(events):
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in events)
    total=0.;start=stop=None
    for left,right in intervals:
        if start is None:start,stop=left,right
        elif left<=stop:stop=max(stop,right)
        else:total+=stop-start;start,stop=left,right
    return total+(stop-start if start is not None else 0.)


def summarize_rank(snapshot_path):
    snapshot=json.loads(snapshot_path.read_text())
    path=snapshot_path.parent/'replay_actual_model_trace.json.gz'
    if not path.exists():path=path.with_suffix('')
    encoded=path.read_bytes();raw=gzip.decompress(encoded) if path.suffix=='.gz' else encoded
    events=json.loads(raw)['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel']
    streams=collections.defaultdict(list)
    for event in kernels:
        args=event.get('args',{})
        streams[tuple(args.get(k) for k in ('device','context','stream'))].append(event)
    for stream in streams.values():stream.sort(key=lambda e:e['ts'])
    primary=[];merge=[];copies=[];unmatched=[]
    launches=collections.defaultdict(lambda:collections.defaultdict(list))
    for stream_key,stream in streams.items():
        for index,event in enumerate(stream):
            args=event.get('args',{});name=event['name']
            launch_key=(stream_key,args.get('correlation'),args.get('graph id'))
            if 'flashinfer::BatchPrefillWithPagedKVCacheKernel<' in name:
                primary.append(event);launches[launch_key]['primary'].append(event)
            elif 'flashinfer::PersistentVariableLengthMergeStatesKernel<' in name:
                merge.append(event);launches[launch_key]['merge'].append(event)
                after=stream[index+1] if index+1<len(stream) else None
                aa=after.get('args',{}) if after else {}
                matched=(after is not None and after['name']=='memcpy32_post'
                    and aa.get('correlation')==args.get('correlation')
                    and aa.get('graph id')==args.get('graph id')
                    and isinstance(args.get('graph node id'),int)
                    and aa.get('graph node id')==args['graph node id']+1)
                if matched:
                    copies.append(after);launches[launch_key]['matched_output_copy'].append(after)
                else:
                    unmatched.append({'merge_args':args,'next_name':after['name'] if after else None,
                        'next_args':aa})
    model_events=[e for e in snapshot['model_graph_events'] if e['profiled']]
    expected=48*len(model_events)
    counts={'native_primary':len(primary),'native_merge':len(merge),'matched_required_output_copy':len(copies)}
    assert len(model_events)==4 and all(count==expected for count in counts.values()) and not unmatched,(snapshot_path,counts,unmatched)
    assert len({id(e) for e in copies})==len(copies)
    assert len(launches)==4
    assert all(len(events)==48 for launch in launches.values() for events in launch.values())
    attention=primary+merge+copies
    model_kernels=[]
    for event in kernels:
        args=event.get('args',{})
        key=(tuple(args.get(k) for k in ('device','context','stream')),
            args.get('correlation'),args.get('graph id'))
        if key in launches:model_kernels.append(event)
    durations={'native_primary':sum(e['dur'] for e in primary),'native_merge':sum(e['dur'] for e in merge),
        'matched_required_output_copy':sum(e['dur'] for e in copies)}
    attention_us=sum(durations.values());kernel_sum_us=sum(e['dur'] for e in kernels)
    model_event_us=sum(e['model_graph_cuda_us'] for e in model_events)
    all_memcpy=[e for e in kernels if 'memcpy' in e['name'].lower()]
    matched_ids={id(e) for e in copies}
    unrelated_memcpy=[e for e in all_memcpy if id(e) not in matched_ids]
    launch_rows=[]
    for (stream_key,correlation,graph_id),parts in sorted(launches.items()):
        launch_rows.append({'device_context_stream':stream_key,'correlation':correlation,'graph_id':graph_id,
            'counts':{name:len(values) for name,values in parts.items()},
            'duration_us':{name:sum(e['dur'] for e in values) for name,values in parts.items()},
            'copy_graph_node_ids':[e['args']['graph node id'] for e in parts['matched_output_copy']]})
    return {'snapshot':str(snapshot_path),'trace':str(path),
        'trace_file_sha256':hashlib.sha256(encoded).hexdigest(),'trace_uncompressed_sha256':hashlib.sha256(raw).hexdigest(),
        'rank':snapshot['rank'],'device':snapshot['device'],'profiled_B3_steps':len(model_events),
        'positions_by_step':[e['positions'] for e in model_events],
        'qualification':'All four actual model graph launches have exactly 48 primary, merge and identity-matched output-copy nodes.',
        'counts':counts,'duration_us':durations,'attention_with_required_copy_us':attention_us,
        'attention_with_required_copy_per_model_step_us':attention_us/len(model_events),
        'kernel_sum_us':kernel_sum_us,'kernel_count':len(kernels),'kernel_union_busy_us':union_us(kernels),
        'matched_four_model_graph_kernel_sum_us':sum(e['dur'] for e in model_kernels),
        'matched_four_model_graph_kernel_count':len(model_kernels),
        'attention_fraction_of_matched_model_graph_kernel_sum':attention_us/sum(e['dur'] for e in model_kernels),
        'kernel_first_to_last_span_us':max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels),
        'profiled_model_graph_cuda_events_us':model_event_us,
        'profiled_model_graph_cuda_event_us_by_step':[e['model_graph_cuda_us'] for e in model_events],
        'attention_fraction_of_kernel_sum':attention_us/kernel_sum_us,
        'attention_kernel_sum_over_profiled_model_graph_events':attention_us/model_event_us,
        'all_memcpy_named_kernel_count':len(all_memcpy),'unrelated_memcpy_named_kernel_count':len(unrelated_memcpy),
        'unrelated_memcpy_named_kernel_us':sum(e['dur'] for e in unrelated_memcpy),
        'unmatched_merge_nodes':unmatched,'graph_launches':launch_rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='results/joint_real48_replay_v1')
    parser.add_argument('--output',default='results/joint_real48_replay_v1/trace_attribution.json')
    args=parser.parse_args();root=Path(args.root)
    rows=[summarize_rank(p) for p in sorted(root.glob('joint_real48_*/rank*/replay_snapshot.json'))]
    assert len(rows)==8
    report={'schema':1,'status':'completed','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'CPU attribution of four actual no-shadow E0 B3 model graph steps per rank, instrumented on the original 48-layer vLLM caches.',
        'matching_rule':'Immediate next kernel after FI PersistentVariableLengthMergeStatesKernel on identical device/context/stream, named memcpy32_post, same correlation and graph id, graph node id exactly merge+1.',
        'ranks':rows,'all_1536_required_output_copies_matched':sum(r['counts']['matched_required_output_copy'] for r in rows)==1536,
        'limitations':['Kernel sum measures accumulated kernel durations; it is not a critical-path latency and may overlap in general.',
            'All-trace kernel sum includes between-graph prepare/sampling. Matched-four-model-graph sums use the same stream/correlation/graph identities as the four native attention groups.',
            'The full-model CUDA-event denominator is a separately measured graph interval including model computation/TP. It excludes external prepare and sampling; do not label it full engine step/JCT.',
            'Trace first-to-last span also contains between-graph prepare/sampling/host gaps. The first prepare precedes profiler activation.',
            'Instrumentation and event recording add overhead; these are attribution figures, not uninstrumented application speedups.',
            'Only independent 32K E0 is profiled here; changed-mask timing comes from separate frozen-Q attention replay, not a measured full-model acceleration.',
            'B1/B2 replay subsets have no real whole-model profile here; do not extrapolate B3 attention fractions to shrinking groups.']}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'ranks':len(rows),'attention_kernel_sum_fractions':[r['attention_fraction_of_kernel_sum'] for r in rows],
        'attention_model_event_ratios':[r['attention_kernel_sum_over_profiled_model_graph_events'] for r in rows]},indent=2))


if __name__=='__main__':main()
