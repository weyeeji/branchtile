"""Independent CPU source, numerical, paired-timing and CUPTI audit of repaired PAT comparisons."""
import collections,hashlib,json,math,pathlib,random,statistics
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ratio(rows,a,b):
 values=[x[a]/x[b] for x in rows];rng=random.Random(20261017)
 bs=sorted(statistics.median(rng.choices(values,k=len(values))) for _ in range(2000))
 return {'paired_median':statistics.median(values),'bootstrap_95ci':[bs[49],bs[1949]]}
def trace_audit(path):
 events=json.loads(path.read_text())['traceEvents'];ranges=[e for e in events if e.get('cat')=='user_annotation']
 launches={e['args']['correlation']:e for e in events if e.get('cat')=='cuda_runtime' and 'LaunchKernel' in e['name']}
 grouped=collections.defaultdict(list)
 for kernel in (e for e in events if e.get('cat')=='kernel'):
  launch=launches[kernel['args']['correlation']]
  parents=[u['name'] for u in ranges if u['ts']<=launch['ts']<u['ts']+u['dur']];assert len(parents)==1
  assert kernel['args'].get('graph id',0)==0
  grouped[parents[0]].append(kernel)
 result={}
 for arm,ks in grouped.items():
  result[arm]={'kernel_count':len(ks),'streams':sorted({k['args']['stream'] for k in ks}),'kernels':[{'name':k['name'],'stream':k['args']['stream'],'grid':k['args']['grid'],'block':k['args']['block']} for k in ks]}
 assert len(grouped['native_pdl'])==len(grouped['native_pdl_AA'])==1
 raw=grouped['raw_pair_grouped_hint_complete'];assert len(raw)==5 and sum('kernel_mha' in k['name'] for k in raw)==2
 pat=grouped['PAT_compat_fixed_scheduler'];forward=[k for k in pat if 'pat_fwd_splitkv_kernel' in k['name']];gather=[k for k in pat if 'gather_kernel' in k['name']]
 assert len(gather)==1 and len(forward)>=1
 assert gather[0]['ts']+.1>=max(k['ts']+k['dur'] for k in forward)
 result['PAT_gather_after_all_forward']=True;result['scope']='Coverage/stream audit only; profiler durations and occupancy estimates are not timing evidence.'
 return result
output=R/'results/pat_fixed_perf_independent_audit_v1.json';assert not output.exists()
report={'status':'passed','scope':__doc__,'runs':[],'numerical_checkpoints':0,'performance':[],'memchecks':[]}
for kind,count in [('qualification',6),('eager',4),('memcheck',3)]:
 paths=sorted(p for p in (R/f'results/pat_fixed_scheduler_{kind}_v1').glob('*.json') if not p.name.endswith('.trace.json'));assert len(paths)==count
 for p in paths:
  r=json.loads(p.read_text());assert r['status']==('completed_qualified_eager_only' if kind=='eager' else 'qualified_fixed_scheduler_perf_artifact')
  for name,h in r['source_sha256'].items():assert sha(R/name)==h,name
  assert sha(R/r['capture']['path'])==r['capture']['sha256'];assert sha(R/r['build_receipt']['path'])==r['build_receipt']['sha256']
  build=json.loads((R/r['build_receipt']['path']).read_text());assert sha(build['module'])==build['module_sha256']
  for name,h in build['generated_sha256'].items():assert sha(R/'build/pat_fixed_scheduler_perf_v1'/name)==h
  assert r['raw_build_receipt']['identity']=='5e5940e0894a8c8a' and r['raw_build_receipt']['native_group']==8 and r['raw_build_receipt']['fold']==2
  for name,h in r['raw_build_receipt']['generated_sha256'].items():assert sha(R/'build/xqa_raw_grouped_hint'/r['raw_build_receipt']['identity']/name)==h
  assert len(r['repeat_qualification'])==30 and r['all_repeated_numerical_checks_passed']
  allchecks=list(r['checks'].values())+[x['checks'] for x in r['repeat_qualification']]
  if kind=='eager':
   for regime in ('hot','cold'):allchecks+=r['timing'][regime]['outside_timing_block_checks']
  for checks in allchecks:
   for arm in ('native_pdl','native_pdl_AA','PAT_compat_fixed_scheduler','raw_pair_grouped_hint_complete'):
    c=checks[arm];assert c['finite'] and c['FP64']['passed']
    if arm!='PAT_compat_fixed_scheduler':assert c['native_bit_mismatch_count']==0
  report['numerical_checkpoints']+=len(allchecks)
  report['runs'].append({'kind':kind,'path':str(p.relative_to(R)),'sha256':sha(p),'numerical_checkpoints':len(allchecks),'stream':r['args']['caller_stream'],'DISABLE_STREAM':r['environment']['DISABLE_STREAM']})
  if kind=='eager':
   summary={'case':p.stem,'prefix_tokens':r['shape']['prefix'],'timing':{},'setup':r['setup']}
   for regime in ('hot','cold'):
    t=r['timing'][regime];assert len(t['orders'])==30 and len(t['outside_timing_block_checks'])==30
    assert t['replays_per_sample']==(5 if regime=='hot' else 1)
    summary['timing'][regime]={}
    for metric in ('caller_stream_event','synchronized_wall'):
     data=t[metric];rows=data['blocks_ms'];assert len(rows)==30
     for arm,val in data['median_ms'].items():assert statistics.median(x[arm] for x in rows)==val
     for key,a,b in [('AA','native_pdl','native_pdl_AA'),('native_over_PAT','native_pdl','PAT_compat_fixed_scheduler'),('native_over_raw','native_pdl','raw_pair_grouped_hint_complete'),('PAT_over_raw','PAT_compat_fixed_scheduler','raw_pair_grouped_hint_complete')]:assert ratio(rows,a,b)==data[key]
     summary['timing'][regime][metric]={k:v for k,v in data.items() if k!='blocks_ms'}
   trace=R/r['profile_trace'];summary['CUPTI']={'path':str(trace.relative_to(R)),'sha256':sha(trace),'audit':trace_audit(trace)}
   streams=summary['CUPTI']['audit']['PAT_compat_fixed_scheduler']['streams']
   assert (len(streams)>1)==(r['environment']['DISABLE_STREAM']=='0')
   report['performance'].append(summary)
  if kind=='memcheck':
   logfile=R/f'logs/pat_fixed_perf_memcheck_{p.stem}_v1.memcheck.log';text=logfile.read_text();assert 'ERROR SUMMARY: 0 errors' in text
   report['memchecks'].append({'path':str(logfile.relative_to(R)),'sha256':sha(logfile),'errors':0})
assert report['numerical_checkpoints']==647
report['limitations']=['Two B8 states on one GPU generation; no natural end-to-end or general-prefix claim.','PAT is locally repaired, not unmodified upstream; arithmetic and boxes/tile partitions preserved, split numbering and stream dependencies changed.','Prepared metadata excludes online table readback/planning/upload from operator time; setup separately recorded.','Cold single-call native-AA intervals are wide, so precise cold native advantage is uncertain.','PAT need not equal native bits; all four arms use the same FP64 tolerance while raw and AA additionally require native bit equality.','No per-input best-of modes; multistream and single-stream are reported separately.']
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'runs':len(report['runs']),'numerical_checkpoints':report['numerical_checkpoints'],'memchecks':len(report['memchecks'])}))
