"""Independent CPU audit of the complete direct-input actual-capture grid."""
import hashlib,json,pathlib,random,statistics
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ratio(rows,a,b):
 vals=[x[a]/x[b] for x in rows];rng=random.Random(20261017)
 boot=sorted(statistics.median(rng.choices(vals,k=len(vals))) for _ in range(2000))
 return {'paired_median_ratio':statistics.median(vals),'bootstrap_95ci':[boot[49],boot[1949]]}
report={'status':'passed','runs':[],'cells':[]}
paths=sorted((R/'results/xqa_raw_direct_input_actual_v1').glob('*.json'));assert len(paths)==16
observed=set()
for p in paths:
 r=json.loads(p.read_text());assert r['status']=='qualified_and_measured'
 assert r['build_identity']=='c28391a037f29141' and r['copied_control_build_identity']=='5e5940e0894a8c8a'
 assert r['args']['extra_tail']==0 and r['layout']=='native original physical IDs and stride'
 for name,h in r['source_sha256'].items():assert sha(R/name)==h,name
 cap=r['capture'];assert sha(R/cap['path'])==cap['sha256']
 doc=1 if 'train1_' in cap['path'] else 7;assert f'train{doc}_' in cap['path']
 cell=(doc,cap['rank'],cap['layer']);assert cell not in observed;observed.add(cell)
 assert [c['batch'] for c in r['cases']]==[8,7,6]
 for c in r['cases']:
  b=c['batch'];assert len(c['rows'])==len(set(c['rows']))==b and all(0<=x<8 for x in c['rows'])
  assert c['S']==170//(b*2) and c['prefix_SM_parameter']==c['S']*((b+1)//2)*2 and c['tail_SM_parameter']==c['S']*b*2
  expected={'paired_bits','copied_control_bits','AA_10_bits','native_FP64','candidate_FP64','changed_graph_bits','copied_changed_graph_bits','final_bits','revisit_after_other_batch_shapes_bits'}
  assert set(c['checks'])==expected and all(c['checks'].values())
  hot=c['hot'];rows=hot['blocks_ms'];assert len(rows)==len(hot['orders'])==30
  names={'native','native_AA','candidate','copied_control'}
  assert all(set(row)==set(order)==names for row,order in zip(rows,hot['orders']))
  assert hot['median_ms']=={k:statistics.median(x[k] for x in rows) for k in names}
  for name,a,z in [('native_over_candidate','native','candidate'),('AA','native','native_AA'),('copied_over_direct','copied_control','candidate')]:assert hot[name]==ratio(rows,a,z)
  report['cells'].append({'document':doc,'rank':cap['rank'],'layer':cap['layer'],'batch':b,'checks':c['checks'],'median_ms':hot['median_ms'],**{k:hot[k] for k in ['native_over_candidate','copied_over_direct','AA']}})
 report['runs'].append({'path':str(p.relative_to(R)),'sha256':sha(p)})
assert observed=={(d,r,l) for d in [1,7] for r in [0,1] for l in [0,15,31,47]}
for ident,subdir in [('c28391a037f29141','xqa_raw_direct_input'),('5e5940e0894a8c8a','xqa_raw_grouped_hint')]:
 directory=R/'build'/subdir/ident;receipt=json.loads((directory/'receipt.json').read_text())
 for path,h in receipt['generated_sha256'].items():assert sha(directory/path)==h
report['ranges']={k:{'min':min(c[k]['paired_median_ratio'] for c in report['cells']),'max':max(c[k]['paired_median_ratio'] for c in report['cells']),'minimum_lower95':min(c[k]['bootstrap_95ci'][0] for c in report['cells'])} for k in ['native_over_candidate','copied_over_direct','AA']}
report['limitations']=['B6/B7 are subsets of captured B8 queries, not later natural survivor queries.','Only hot CUDAgraph event timings; each sample ten replays with same-arm warmup. No wall/cold/wholemodel estimate.','Checks occur before, after changed graph inputs, at end and shared-workspace revisit, not after each timing block.','Source receipts omit the imported bootstrap helper; this audit independently recomputes all ratios and intervals without importing it.','Generated sources verified; this report does not claim an independent binary disassembly proof or arbitrary-stride Q support.']
out=R/'results/xqa_direct_input_actual_independent_audit_v1.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','cells':len(report['cells']),'ranges':report['ranges']}))
