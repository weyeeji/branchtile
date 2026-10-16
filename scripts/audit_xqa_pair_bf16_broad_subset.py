"""CPU-only full 16-capture BF16 matrix and eight survivor-subset audit."""
import json
from collections import Counter
from pathlib import Path
from audit_xqa_pair_native_capture_v2 import paired, sha

ROOT=Path(__file__).resolve().parents[1]

def audit(path):
 d=json.loads(path.read_text()); assert d['status']=='bitwise_qualified_and_benchmarked_prefix_only'
 assert d['args']['dtype']=='bf16'
 assert all(sha(p)==h for p,h in d['sources'].items())
 assert sha(ROOT/d['args']['capture'])==d['capture_sha256']
 assert d['capture_ABI']['runtime_table_dtype']=='torch.int32'
 assert d['capture_ABI']['page_roundtrip_bytes_exact'] and d['production_entry_bitwise_match']
 for method in ('native_AA','pair_fixed_split'):
  checks=[r for r in d['checks'] if r['method']==method]
  assert len(checks)==6 and all(r['bitwise'] and r['different_elements']==0 and r['finite'] for r in checks)
 assert all(not r['bitwise'] for r in d['checks'] if r['method']=='pair_auto_split')
 assert all(s['errors']['native']==s['errors']['pair_fixed_split'] for s in d['FP64_reference']['samples'])
 batch=d['shape']['batch']; split=d['shape']['native_splits']; assert split==min(170//(batch*2),(d['shape']['tokens']+255)//256)
 for method in ('native','pair_fixed_split','pair_auto_split'):
  s=split if method!='pair_auto_split' else d['shape']['paired_auto_splits']
  expected=[s,2,batch if method=='native' else batch//2]
  trace=json.loads(Path(d['launch_profiles'][method]['trace']).read_text())
  launches=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and 'kernel_mha' in e.get('name','')]
  assert launches and all(e['args']['grid']==expected for e in launches)
 if 'subset' in d:
  subset=d['subset']; assert sha(subset['trace'])==subset['trace_sha256']
  r=json.loads(Path(subset['trace']).read_text())['replays'][subset['trace_replay_index']]
  assert r['questions']==subset['source_rows'] and r['batch']==batch
 return {'result':str(path.relative_to(ROOT)),'result_sha256':sha(path),'batch':batch,'tokens':d['shape']['tokens'],'split':split,
         'median_us':d['median_us'],'native_over_fixed':paired(d['timing_blocks'],'native','pair_fixed_split'),
         'AA_over_fixed':paired(d['timing_blocks'],'native_AA','pair_fixed_split'),
         'subset':d.get('subset'),'all_hash_ABI_grid_bitwise_checks_passed':True,
         'FP64_max_relative_L2':max(s['errors']['native']['relative_l2'] for s in d['FP64_reference']['samples'])}

def main():
 full={}; subsets={}; membership={}
 for doc in (1,7):
  for rank in (0,1):
   for layer in (0,15,31,47):
    name=f'train{doc}_r{rank}_l{layer}'
    p=ROOT/'results/xqa_pair_bf16_broad_v2'/f'{name}.json'
    if (doc,rank,layer) in ((1,0,15),(7,0,47)):
     p=ROOT/'results/xqa_pair_padding_native_capture_v2'/f'train{doc}_bf16_r{rank}_l{layer}.json'
    full[name]=audit(p)
  for batch in (2,4,6,8):
   name=f'train{doc}_b{batch}'; subsets[name]=audit(ROOT/'results/xqa_pair_padding_natural_subset_v3'/f'{name}.json')
  trace=ROOT/f'results/dense_cascade_device_natural_v1/train{doc}_bf16/rank0/native_natural_graph.json'
  rows=json.loads(trace.read_text())['replays']; counts=Counter(r['batch'] for r in rows)
  membership[f'train{doc}']={'trace_sha256':sha(trace),'decode_step_count_by_active_batch':dict(sorted(counts.items())),
   'total_steps':len(rows),'B6_or_more_step_fraction':sum(v for b,v in counts.items() if b>=6)/len(rows),
   'B2_or_less_step_fraction':sum(v for b,v in counts.items() if b<=2)/len(rows),
   'scope':'Step counts only; neither time-weighted nor an end-to-end speed estimate.'}
 report={'status':'audited','full_B8_matrix':full,'subsets':subsets,'natural_membership':membership,
  'counts':{'full_B8_capture_cells':16,'subset_cells':8,'fixed_split_bitwise_checks':144,'native_AA_bitwise_checks':144,'auto_split_changed_checks':144},
  'limitations':['Prefix-only; private tails and model integration pending.','Subset Q is from initial B8 capture, not later natural Q.','Existing-kernel diagnostic; not a novelty or end-to-end result.','All timing blocks retained; bootstrap is within-run only.']}
 output=ROOT/'results/xqa_pair_native_capture_audit_v2/broad_subset_audit.json'; output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'full_speed_range':[min(v['native_over_fixed']['median_paired_ratio'] for v in full.values()),max(v['native_over_fixed']['median_paired_ratio'] for v in full.values())],
  'subsets':{k:v['native_over_fixed'] for k,v in subsets.items()},'membership':membership},indent=2))
if __name__=='__main__':main()
