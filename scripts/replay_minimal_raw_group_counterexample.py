"""S1 two-query reduction of the real split5 cross-request hint counterexample.

The artifact retains post-RoPE Q/K/V values, but extracts one native split's
ordered token tiles and one KV head. This is a regression artifact, not a task.
"""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',default='results/raw_pair_minimal_counterexample_v1/split5_two_queries.pt');p.add_argument('--output',required=True);a=p.parse_args()
 assert os.geteuid()!=0 and sys.prefix!=sys.base_prefix
 out=Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 r={'status':'initializing','scope':__doc__,'args':vars(a),'artifact_sha256':hashlib.sha256(Path(a.artifact).read_bytes()).hexdigest(),'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['scripts/replay_minimal_raw_group_counterexample.py','src/xqa_raw_pair_build.py','src/xqa_raw_grouped_hint_build.py']}}
 def save():out.write_text(json.dumps(r,indent=2)+'\n')
 save()
 try:
  import torch
  from flashinfer.xqa import xqa
  from src.xqa_raw_pair_build import jit_specs as old_specs
  from src.xqa_raw_grouped_hint_build import jit_specs as fixed_specs
  torch.set_num_threads(2)
  x=torch.load(a.artifact,map_location='cpu',weights_only=True)
  q=x['q'].cuda();L=len(x['kv_tokens']);assert L%256==0 and q.shape==(2,8,128)
  cache=x['kv_tokens'].cuda().reshape(L//16,16,1,256).permute(0,2,1,3)
  k,v=cache[...,:128],cache[...,128:];wk,wv=k.transpose(1,2),v.transpose(1,2)
  table=torch.arange(L//16,device='cuda',dtype=torch.int32).repeat(2,1)
  lengths=torch.full((2,1),L,device='cuda',dtype=torch.uint32)
  packed=q.reshape(1,1,16,128).contiguous();unused=torch.empty_like(packed)
  workspace=torch.zeros(64*1024*1024,device='cuda',dtype=torch.uint8);sem=workspace[:8*1024*1024];scratch=workspace[8*1024*1024:]
  native=torch.empty_like(q);aa=torch.empty_like(q)
  def native_run(o):xqa(q.unsqueeze(1),k,v,table,lengths,o.unsqueeze(1),scratch,sem,1,16,q_scale=1.,kv_scale=1.,kv_layout='HND',sm_count=2,enable_pdl=True)
  def difference(a,b):
   mask=a.contiguous().view(torch.uint8)!=b.contiguous().view(torch.uint8)
   return {'bits_equal':not bool(mask.any()),'mismatched_bytes':int(mask.sum()),'max_abs':float((a.float()-b.float()).abs().max())}
  native_run(native);native_run(aa);r['native_AA']=difference(native,aa);assert r['native_AA']['bits_equal']
  for name,build in [('old',old_specs),('fixed',fixed_specs)]:
   specs,receipt=build();mods={g:s.build_and_load() for g,s in specs.items()}
   raw=torch.empty(2*4*32*35,device='cuda',dtype=torch.float32);pair=torch.empty(4*32*35,device='cuda',dtype=torch.float32)
   output=torch.empty_like(q);single=torch.empty_like(q)
   def call(g,phase,cut,state):
    pair_prefix=g==16;b=1 if pair_prefix else 2
    mods[g].xqa_wrapper(False,b,1,0,1.,None,unused if pair_prefix else output.unsqueeze(1),1.,packed if pair_prefix else q.unsqueeze(1),None,wk,wv,None,None,table[:1] if pair_prefix else table,L,lengths[:1] if pair_prefix else lengths,b,1.,None,1,None,None,sem,scratch,False,state,cut,phase)
   valid=torch.zeros((2,4,32,35),device='cuda',dtype=torch.bool)
   for reg in range(32):
    if reg%4<2:valid[...,reg]=True
   valid[...,:8,32:]=True
   mismatches=[]
   for cut in range(1,L//256+1):
    call(8,1,cut,raw);call(16,1,cut,pair)
    src=pair.reshape(4,32,35);mapped=torch.zeros_like(raw).reshape(2,4,32,35)
    for row in range(2):
     for reg in range(32):
      if reg%4<2:mapped[row,...,reg]=src[...,reg+2*row]
     mapped[row,:,:8,32:]=src[:,row*8:row*8+8,32:]
    mask=(mapped.view(torch.int32)!=raw.reshape_as(mapped).view(torch.int32))&valid
    if bool(mask.any()):mismatches.append({'cut_tiles':cut,'fields':{label:int(mask[...,lo:hi].sum()) for label,lo,hi in [('U',0,32),('m',32,33),('z',33,34),('hint',34,35)]},'first_coordinates':mask.nonzero()[:8].cpu().tolist()})
   call(16,1,L//256,pair);call(8,3,L//256,pair);pair_output=output.clone()
   call(8,1,L//256,raw);call(8,2,L//256,raw)
   r[name]={'identity':receipt['identity'],'scanned_cuts':L//256,'raw_mismatches':mismatches,'output_vs_native':difference(pair_output,native),'unshared_vs_native':difference(output,native)};save()
   assert r[name]['unshared_vs_native']['bits_equal']
  assert r['old']['raw_mismatches'], 'Extraction did not retain counterexample'
  assert not r['fixed']['raw_mismatches'] and r['fixed']['output_vs_native']['bits_equal']
  r['status']='minimal_counterexample_reproduced_and_fixed'
 except Exception:r['status']='failed';r['error']=traceback.format_exc();save();raise
 save();print(json.dumps({'status':r['status'],'output':str(out)}))
if __name__=='__main__':main()
