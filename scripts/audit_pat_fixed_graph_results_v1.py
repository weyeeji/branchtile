"""CPU-only independent provenance, numerical, paired-timing and graph-coverage audit."""
import collections,hashlib,json,pathlib,random,statistics
R=pathlib.Path(__file__).resolve().parents[1]
ARMS=('native_pdl','native_pdl_AA','PAT_compat_fixed_scheduler','raw_pair_grouped_hint_complete')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ratio(rows,a,b):
 values=[x[a]/x[b] for x in rows];rng=random.Random(20261017)
 bs=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(2000))
 return {'paired_median':statistics.median(values),'bootstrap_95ci':[bs[49],bs[1949]]}
def trace_audit(path):
 events=json.loads(path.read_text())['traceEvents'];ranges=[e for e in events if e.get('cat')=='user_annotation'];grouped=collections.defaultdict(list)
 launches={e['args']['correlation']:e for e in events if e.get('cat')=='cuda_runtime' and e['name']=='cudaGraphLaunch'}
 assert len(launches)==4
 for arm in ARMS:
  u,=[e for e in ranges if e['name']==arm]
  runtime=[e for e in events if e.get('cat')=='cuda_runtime' and u['ts']<=e['ts']<u['ts']+u['dur']]
  assert sum(e['name']=='cudaGraphLaunch' for e in runtime)==1
  assert not any('LaunchKernel' in e['name'] or 'Memcpy' in e['name'] or 'Memset' in e['name'] for e in runtime),runtime
 for kernel in (e for e in events if e.get('cat')=='kernel'):
  launch=launches[kernel['args']['correlation']]
  parents=[u['name'] for u in ranges if u['ts']<=launch['ts']<u['ts']+u['dur']];assert len(parents)==1
  assert kernel['args']['graph id']>0 and kernel['args']['graph node id']>0
  grouped[parents[0]].append(kernel)
 assert set(grouped)==set(ARMS)
 assert len(grouped[ARMS[0]])==len(grouped[ARMS[1]])==1
 raw=grouped[ARMS[3]];assert len(raw)==5 and sum('kernel_mha' in k['name'] for k in raw)==2
 pat=grouped[ARMS[2]];forward=[k for k in pat if 'pat_fwd_splitkv_kernel' in k['name']];gather=[k for k in pat if 'gather_kernel' in k['name']]
 assert len(gather)==1 and len(forward)>=1 and len(pat)==len(forward)+1
 assert gather[0]['ts']+.1>=max(k['ts']+k['dur'] for k in forward)
 return {arm:{'kernel_count':len(ks),'streams':sorted({k['args']['stream'] for k in ks}),'graph_ids':sorted({k['args']['graph id'] for k in ks}),'all_kernels_correlate_to_graph_launch':True,'no_eager_launch_or_memcpy_inside_arm':True,'kernels':[{'name':k['name'],'grid':k['args']['grid'],'block':k['args']['block']} for k in ks]} for arm,ks in grouped.items()}
report={'status':'passed','runs':[],'numerical_checkpoints':0,'performance':[]}
for kind,count in [('qualification',2),('performance',4)]:
 paths=sorted(p for p in (R/f'results/pat_fixed_graph_{kind}_v1').glob('*.json') if not p.name.endswith('.trace.json'));assert len(paths)==count
 for p in paths:
  r=json.loads(p.read_text());assert r['status']==('completed_qualified_graph_only' if kind=='performance' else 'qualified_fixed_scheduler_graph_artifact')
  assert r['args']['caller_stream']=='nondefault' and r['all_arms_use_graph_replay']
  assert set(r['graph_capture'])==set(ARMS) and all(x['full_call'] for x in r['graph_capture'].values())
  assert len({x['graph_python_identity'] for x in r['graph_capture'].values()})==4
  for name,h in r['source_sha256'].items():assert sha(R/name)==h,name
  assert sha(R/r['capture']['path'])==r['capture']['sha256'];assert sha(R/r['build_receipt']['path'])==r['build_receipt']['sha256']
  build=json.loads((R/r['build_receipt']['path']).read_text());assert sha(build['module'])==build['module_sha256']
  for name,h in build['generated_sha256'].items():assert sha(R/'build/pat_fixed_graph_v1'/name)==h
  assert r['raw_build_receipt']['identity']=='5e5940e0894a8c8a'
  for name,h in r['raw_build_receipt']['generated_sha256'].items():assert sha(R/'build/xqa_raw_grouped_hint'/r['raw_build_receipt']['identity']/name)==h
  assert len(r['repeat_qualification'])==30 and r['all_repeated_numerical_checks_passed']
  checks=list(r['checks'].values())+[x['checks'] for x in r['repeat_qualification']]
  assert len(r['changed_input_checks'])==3
  for x in r['changed_input_checks']:
   assert x['same_graphs_same_pointers'] and set(x['graph_equals_own_eager_bits'])==set(ARMS) and all(x['graph_equals_own_eager_bits'].values())
   checks.append(x['FP64_and_native_checks'])
  changed=r['changed_input_checks'];assert changed[0]['Q_differs_from_capture'] and not changed[1]['Q_differs_from_capture'] and not changed[2]['Q_differs_from_capture']
  original=changed[0]['lengths'];short=changed[1]['lengths'];assert changed[2]['lengths']==original
  assert sum(a-b for a,b in zip(original,short))==1 and sum(a!=b for a,b in zip(original,short))==1
  assert changed[2]['FP64_and_native_checks']['captured_native_O']['native_bit_equal']
  if kind=='performance':
   for regime in ('hot','cold'):checks+=r['timing'][regime]['outside_timing_block_checks']
  for c in checks:
   for arm in ARMS:
    assert c[arm]['finite'] and c[arm]['FP64']['passed']
    if arm!=ARMS[2]:assert c[arm]['native_bit_mismatch_count']==0
  report['numerical_checkpoints']+=len(checks)
  report['runs'].append({'path':str(p.relative_to(R)),'sha256':sha(p),'checkpoints':len(checks),'DISABLE_STREAM':r['environment']['DISABLE_STREAM']})
  if kind=='performance':
   out={'case':p.stem,'prefix':r['shape']['prefix'],'timing':{},'setup':r['setup'],'graph_capture':r['graph_capture']}
   for regime in ('hot','cold'):
    t=r['timing'][regime];assert len(t['orders'])==len(t['outside_timing_block_checks'])==30
    assert all(sorted(order)==sorted(ARMS) for order in t['orders'])
    assert t['replays_per_sample']==(5 if regime=='hot' else 1)
    out['timing'][regime]={}
    for metric in ('caller_stream_event','synchronized_wall'):
     data=t[metric];rows=data['blocks_ms'];assert len(rows)==30
     for arm,val in data['median_ms'].items():assert statistics.median(x[arm] for x in rows)==val
     for key,a,b in [('AA',ARMS[0],ARMS[1]),('native_over_PAT',ARMS[0],ARMS[2]),('native_over_raw',ARMS[0],ARMS[3]),('PAT_over_raw',ARMS[2],ARMS[3])]:assert ratio(rows,a,b)==data[key]
     out['timing'][regime][metric]={k:v for k,v in data.items() if k!='blocks_ms'}
   trace=R/r['profile_trace'];out['CUPTI']={'path':str(trace.relative_to(R)),'sha256':sha(trace),'audit':trace_audit(trace)}
   assert (len(out['CUPTI']['audit'][ARMS[2]]['streams'])>1)==(r['environment']['DISABLE_STREAM']=='0')
   report['performance'].append(out)
report['limitations']=['Two actual B8 states, not all16 capture cells and not model E2E.','PAT includes local scheduler and stream compatibility repairs; CUDA arithmetic and partition/tile choices unchanged.','Prepared topology/metadata; one-token truncation stays inside an assigned private CTA. Arbitrary growth/replanning is untested.','Changed-input qualification calls each own eager reference outside measured graph blocks; performance loops use only graph.replay.','CUPTI proves complete graph coverage in the sampled trace, not every timing block. No profiler occupancy/duration performance inference.','Raw candidate here is grouped-hint with original three copy kernels, not the newer direct-input two-launch variant.','No graph-specific memcheck run in this artifact; prior eager fixed-scheduler memchecks passed.']
out=R/'results/pat_fixed_graph_independent_audit_v1.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checkpoints':report['numerical_checkpoints'],'runs':len(report['runs'])}))
