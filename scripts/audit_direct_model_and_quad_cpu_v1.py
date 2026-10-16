"""Independent CPU source/provenance review and GQA4 direct-quad grid audit."""
import ast,hashlib,json,pathlib,random,statistics,sys
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.vllm_native_raw_pair_general_setup import validate_offline_records

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ratio(rows,a,b):
 vals=[x[a]/x[b] for x in rows];rng=random.Random(20261017)
 boot=sorted(statistics.median(rng.choices(vals,k=len(vals))) for _ in range(2000))
 return {'paired_median_ratio':statistics.median(vals),'bootstrap_95ci':[boot[49],boot[1949]]}
worker=R/'scripts/vllm_native_raw_pair_direct_worker.py'
tree=ast.parse(worker.read_text())
source_files=next(ast.literal_eval(x.value) for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in x.targets))
required={'src/vllm_native_raw_pair_direct_plan.py','src/xqa_raw_direct_input_runtime.py','src/xqa_raw_direct_input_build.py','src/xqa_raw_grouped_hint_build.py','src/xqa_raw_pair_build.py','src/xqa_raw_continuation_build.py'}
assert required.issubset(source_files)
current_sources={name:sha(R/name) for name in source_files}
oldpaths=[R/f'results/grouped_hint_model_qual_v1/train26_bf16/rank{rank}/qualification_raw_pair.json' for rank in [0,1]]
old=[json.loads(p.read_text()) for p in oldpaths]
try:validate_offline_records(old,current_sources,old[0]['execution_fingerprint'],[],2)
except ValueError as exc:assert str(exc)=='Offline source fingerprint mismatch';rejection=str(exc)
else:raise AssertionError('Old copied-input receipts accepted by direct model worker')
bank=ast.parse((R/'src/vllm_native_raw_pair_direct_plan.py').read_text())
cls=next(x for x in bank.body if isinstance(x,ast.ClassDef));run=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='run_layer')
assert not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='copy_' for x in ast.walk(run))
report={'status':'passed_CPU_review_and_quad_grid_audit','direct_model':{'source_sha256':current_sources,'old_receipts':[{'path':str(p.relative_to(R)),'sha256':sha(p)} for p in oldpaths],'old_receipt_rejection':rejection,'run_layer_contains_no_copy_call':True,'scope':'CPU source review and old-receipt rejection only; GPU whole-model alias qualification is separate.'},'quad_cells':[]}
seen=set()
for p in sorted((R/'results/xqa_raw_direct_quad_grid_v1').glob('*.json')):
 r=json.loads(p.read_text());assert r['status']=='qualified_and_measured';b=r['batch'];tail=r['args']['extra_tail'];seen.add((b,tail))
 assert r['args']['capture'] is None and r['prefix']==120528
 assert r['build_identity']=='bc26510388b35577' and r['copied_build_identity']=='28ecb045c76babdb'
 for name,h in r['source_sha256'].items():assert sha(R/name)==h,name
 assert r['S']==170//(b*4)
 assert all(v is True for v in r['checks'].values())
 for key in ['candidate_bits','copied_quad_bits','AA_bits','unshared_checkpoint_bits','dummy_one_capture_replay_bits','changed_real_capture_replay_bits','swapped_pair_membership_bits','final_candidate_bits']:assert r['checks'][key]
 assert all(r['checks'][f'{which}_FP64_row{row}'] for which in ['native','candidate'] for row in {0,b-1})
 out={'batch':b,'synthetic_extra_tail':tail,'path':str(p.relative_to(R)),'sha256':sha(p),'checks':r['checks'],'timing':{}}
 for regime in ['hot','cold']:
  t=r[regime];rows=t['blocks_ms'];assert len(rows)==len(t['orders'])==30
  names={'native','AA','native_PDL_off','unshared_checkpoint','candidate','copied_quad'}|({'pair'} if b>=6 else set())
  assert all(set(row)==set(order)==names for row,order in zip(rows,t['orders']))
  for name,value in t['median_ms'].items():assert value==statistics.median(x[name] for x in rows)
  ratios=[('native_over_candidate','native','candidate'),('AA','native','AA'),('native_over_unshared','native','unshared_checkpoint'),('copied_over_direct','copied_quad','candidate')]
  if b>=6:ratios.append(('pair_over_quad','pair','candidate'))
  for name,a,z in ratios:assert t[name]==ratio(rows,a,z)
  out['timing'][regime]={key:value for key,value in t.items() if key not in ['blocks_ms','orders']}
 report['quad_cells'].append(out)
assert seen=={(b,t) for b in range(1,9) for t in [0,8192]}
d=R/'build/xqa_raw_direct_quad/bc26510388b35577';receipt=json.loads((d/'receipt.json').read_text())
for name,h in receipt['generated_sha256'].items():assert sha(d/name)==h
report['quad_scope']='All16 cells are remapped synthetic data at nominalP120528; initial tail is17+31*row tokens, long-tail adds8192. No actual model capture or natural survivor claim.'
report['limitations']=['FP64 compares first and last request; all rows compare native bits.','Hot graph samples average10 replays; cold1 after256MiB flush. Only event timings, not whole-model walltime.','B1..3 are deliberate standalone extension tests; every hot cell loses to native despite savings over copied quad.','B4..8 positives do not authorize enlarging any frozen model dispatch policy without separate actual qualification.','Model run_layer CPU no-copy finding is not an audited CUDAgraph trace count; model-level copies elsewhere are expected.']
out=R/'results/direct_model_quad_independent_audit_v1.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'old_receipt_rejection':rejection,'quad_cells':len(report['quad_cells'])}))
