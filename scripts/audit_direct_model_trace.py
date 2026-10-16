"""CPU audit of short whole-model traces; durations are never performance evidence."""
import argparse,collections,hashlib,json,pathlib,re
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 summary=pathlib.Path(a.summary);r=json.loads(summary.read_text());assert r['status']=='completed_trace_collection_pending_CUPTI_audit'
 for key in ['qualified_execution_sources','instrumentation_and_driver_sources']:
  for name,h in r[key].items():assert sha(R/name)==h
 report={'status':'passed','summary':str(summary.resolve()),'summary_sha256':sha(summary),'runs':[]}
 direct_pattern=re.compile(r'void\s*\*,\s*float\s*\*,\s*unsigned int,\s*unsigned int,\s*unsigned int\)$')
 for run in r['runs']:
  workers={w['rank']:w for w in run['workers']}
  for trace in run['traces']:
   rank=trace['rank'];w=workers[rank];path=pathlib.Path(trace['trace'])
   events=json.loads(path.read_text())['traceEvents'];ks=[e for e in events if e.get('cat')=='kernel']
   launches={e['args']['correlation']:e for e in events if e.get('cat')=='cuda_runtime' and e['name']=='cudaGraphLaunch'}
   direct=[e for e in ks if 'kernel_mha(' in e['name'] and direct_pattern.search(e['name'])]
   native=[e for e in ks if 'kernel_mha(' in e['name'] and e not in direct]
   assert len(direct)==96*w['candidate_replays'],(rank,len(direct),w['candidate_replays'])
   groups=collections.defaultdict(list)
   for e in direct:
    assert e['args']['graph id']>0 and e['args']['graph node id']>0
    assert e['args']['correlation'] in launches
    groups[e['args']['correlation']].append(e)
   assert len(groups)==w['candidate_replays'] and all(len(x)==96 for x in groups.values())
   native_graph=collections.defaultdict(list)
   for e in native:
    if e['args'].get('graph id',0)>0:
     assert e['args']['correlation'] in launches
     native_graph[e['args']['correlation']].append(e)
   assert len(native_graph)==w['native_replays'] and all(len(x)==48 for x in native_graph.values()),(rank,w['native_replays'],{k:len(x) for k,x in native_graph.items()})
   assert not(set(groups)&set(native_graph)), 'One model graph unexpectedly mixes native and candidate attention'
   classified={e['args']['correlation'] for e in direct}|set(native_graph)
   copies=collections.Counter(e['name'] for e in ks if e['args'].get('correlation') in classified and any(t in e['name'].lower() for t in ['copy','memcpy','kv_cache','reshape_and_cache']))
   report['runs'].append({'method':run['method'],'rank':rank,'trace':str(path.resolve()),'sha256':sha(path),
       'candidate_graphs':len(groups),'direct_XQA_kernels':len(direct),'direct_XQA_per_candidate_model_graph':96 if groups else None,
       'native_graphs':len(native_graph),'native_XQA_kernels_in_model_graphs':sum(map(len,native_graph.values())),
       'native_XQA_outside_model_graphs':len(native)-sum(map(len,native_graph.values())),
       'framework_copy_or_cache_named_kernels_in_model_graphs':dict(copies),
       'all_direct_XQA_are_graph_nodes':True})
 report['limitations']=['Trace is a fixed short control; its profiled walltime and kernel durations must not be used for E2E speed claims.',
     '96 direct XQA nodes per candidate graph establishes two launches per48-layer forward; CPU bank/source review establishes no candidate Q/O/metadata copy calls.',
     'Normal framework cache updates, output staging and other copies remain. Kernel names alone cannot prove arbitrary copy ownership.',
     'Capture-address records observe arguments during capture; graph node parameter memory is not independently decoded.',
     'This observer/profiler subclass inherits qualified execution without overriding attention arithmetic or dispatch; instrumentation sources are separately hashed.']
 out=pathlib.Path(a.output);assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','traces':len(report['runs'])}))
if __name__=='__main__':main()
