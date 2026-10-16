"""CPU independent audit of the NaN-initialized local PAT scheduler repair."""
import hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
output=R/'results/pat_fixed_scheduler_independent_audit_v1.json';assert not output.exists()
report={'status':'passed','scope':__doc__,'cases':[]}
for mode in ('multi','single'):
 p=R/f'results/pat_fixed_scheduler_qual_v1/{mode}.json';r=json.loads(p.read_text())
 assert r['status']=='qualified_fixed_scheduler' and r['PAT_numerical_pass'] and r['all_full_gather_slots_finite']
 for name,h in r['source_sha256'].items():assert sha(R/name)==h
 assert sha(R/r['build_receipt']['path'])==r['build_receipt']['sha256']
 assert sha(R/r['capture']['path'])==r['capture']['sha256']
 for key in ('initial','final'):
  for arm in ('native_pdl','native_pdl_AA','PAT_default','raw_pair_complete'):
   check=r['checks'][key][arm];assert check['finite'] and check['FP64']['passed']
   if arm!='PAT_default':assert check['native_bit_mismatch_count']==0
 for x in r['iterations']:assert x['FP64']['passed'] and x['finite']
 for check in [r['scratch_initial']]+[x['scratch'] for x in r['iterations']]:
  assert len(check['slots'])==253
  assert all(s['O_all_finite'] and s['LSE_all_finite'] and s['O_nan']==s['LSE_nan']==0 for s in check['slots'])
 for case in r['isolated_groups']:
  g=r['schedule'][case['group']];writers={(row,rank) for qs,n,rank in zip(g['queries'],g['nseq'],g['rank']) for row in qs[:n]}
  for s in case['scratch']['slots']:
   expected=(s['row'],s['split']) in writers
   assert (s['O_all_finite'] and s['LSE_all_finite']) if expected else (s['O_nan']==2048 and s['LSE_nan']==16)
 report['cases'].append({'mode':mode,'path':str(p.relative_to(R)),'sha256':sha(p),'repeats':len(r['iterations']),'gather_slots':253,'isolated_groups':len(r['isolated_groups']),'PAT_initial_native_bit_mismatch':r['checks']['initial']['PAT_default']['native_bit_mismatch_count'],'all_FP64_pass':True})
report['limitation']='One captured B8 state. Debug NaN initialization/retained temporary tensors must be removed and numerical qualification repeated before timing.'
output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
