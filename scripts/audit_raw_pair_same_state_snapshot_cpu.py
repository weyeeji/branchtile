"""CPU-only raw checkpoint audit of preserved late-state counterexamples."""
import argparse,hashlib,json
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='results/raw_pair_eval2_late_same_state_v1');p.add_argument('--output',required=True);a=p.parse_args()
 torch.set_num_threads(2);out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 r={'scope':__doc__,'input':a.input,'snapshots':[]}
 def comp(a,b):
  m=a.view(torch.int16)!=b.view(torch.int16);inds=m.nonzero()
  return {'bits_equal':not bool(m.any()),'mismatch_count':int(m.sum()),'max_abs':float((a.float()-b.float()).abs().max()),'coordinates':inds[:10].tolist()}
 for f in sorted(Path(a.input).glob('rank*/same_state_first_failure*.pt')):
  x=torch.load(f,map_location='cpu',weights_only=True);b=len(x['q']);S=x['S'];ps=x['paired_prefix_raw_state'].reshape((b+1)//2,2,S,4,32,35);ns=x['native_prefix_raw_state'].reshape(b,2,S,4,32,35)
  rec={'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'rank':x['rank'],'layer':x['layer'],'graph_index':x['graph_index'],'original_checks':x['checks'],'same_Q':comp(x['q'],x['model_candidate_q']),'output_checks':{},'raw_prefix_differences':[]}
  for key in ['native_aa_o','native_pdl_off_o','witness_o','candidate_o','candidate_poison_o','model_candidate_o','unshared_checkpoint_o']:rec['output_checks'][key]=comp(x['native_o'],x[key])
  for row in range(b):
   for field,lo,hi in [('U',0,32),('m',32,33),('z',33,34),('hint',34,35)]:
    if field=='U':
     slots=[i for i in range(32) if i%4<2];src=ps[row//2,...,[i+2*(row%2) for i in slots]];dst=ns[row,...,slots]
    else:src=ps[row//2,...,8*(row%2):8*(row%2)+8,lo:hi];dst=ns[row,...,:8,lo:hi]
    mask=src.view(torch.int32)!=dst.view(torch.int32)
    if mask.any():rec['raw_prefix_differences'].append({'row':row,'question':x['questions'][row],'field':field,'count':int(mask.sum()),'max_abs':float((src-dst).abs().max()),'coordinates':mask.nonzero()[:10].tolist()})
  r['snapshots'].append(rec)
 assert len(r['snapshots'])>0 and not torch.cuda.is_initialized();r['cuda_initialized']=False;r['status']='audited'
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'snapshots':len(r['snapshots']),'output':str(out)}))
if __name__=='__main__':main()
