"""Independent CPU gate of known-input late-window grouped-hint regression."""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='results/raw_pair_eval2_grouped_hint_model_diag_v1');p.add_argument('--output',required=True);a=p.parse_args()
 root=Path(a.input);out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 s=json.loads((root/'summary.json').read_text());assert s['status']=='regression_qualified' and s['grouped_hint_regression_passed']
 for path,digest in s['source_sha256'].items():assert sha(path)==digest,path
 assert s['native_reference_prefix_all_equal'] and len(s['native_reference_prefix_checks'])==14
 assert all(x['equal_observed_prefix'] and x['first_difference'] is None for x in s['native_reference_prefix_checks'])
 ranks=[]
 for f in sorted(root.glob('rank*/same_state_diagnostic.json')):
  d=json.loads(f.read_text());rs=d['records'];assert len(rs)==175 and [x['graph_index'] for x in rs]==list(range(1786,1961))
  assert not d['snapshots'] and d['window']==[1760,1960]
  assert d['source_sha256']==sha('scripts/vllm_raw_pair_same_state_grouped_hint_worker.py')
  for r in rs:
   assert r['batch']==9 and r['S']==9 and r['prefix']==113280
   assert r['full_model_qualification_passed'] and r['native_model_AA_bits_equal']
   assert [l['layer'] for l in r['layers']]==list(range(48))
   for l in r['layers']:
    assert all(v for k,v in l.items() if k.endswith('bits_equal'))
    assert l['candidate_bit_mismatch_count']==0 and l['candidate_max_abs']==0.
  ranks.append({'rank':d['rank'],'path':str(f),'sha256':sha(f),'checked_states':len(rs),'checked_layers':48*len(rs),'all_bit_checks_passed':True})
 assert {r['rank'] for r in ranks}=={0,1}
 r={'status':'known_regression_audited','input':a.input,'summary_sha256':sha(root/'summary.json'),'ranks':ranks,'same_Q_and_fullmodel_layer_states':sum(x['checked_layers'] for x in ranks),'original_native_prefix_all_14_questions_equal':True,'limits':['Known counterexample regression, not a new heldout set.','Diagnostic cap1970 is not a natural-EOS performance or quality result.','Only B9 window1786..1960 is exhaustively checked; no universal exactness claim.']}
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'status':r['status'],'layer_states':r['same_Q_and_fullmodel_layer_states']}))
if __name__=='__main__':main()
