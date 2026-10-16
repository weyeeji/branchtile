"""CPU scope, offline coverage, actual packing and shared-arena adapter tests."""
import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.gqa4_quad_scope import in_scope,receipt
from src.vllm_gqa4_raw_pair_setup import REQUIRED_QUALIFICATION_FIELDS,validate_offline_records
from src.vllm_gqa4_quad_plan import LayerSharedGQA4QuadBank

class FakeModule:
 def __init__(self,group):self.group=group;self.calls=[]
 def xqa_wrapper(self,*args):
  self.calls.append(args)
  if self.group==4:args[6].copy_(args[8])

def main():
 checks={};torch.set_num_threads(2)
 for b in range(4,9):
  checks[f'B{b}_scope']=in_scope(b,120528,[120529]*(b-1)+[128720]) and not in_scope(b,120528,[128721]*b)
 checks['B1to3_native']=all(not in_scope(b,120528,[120600]*b) for b in (1,2,3))
 checks['wrongP_native']=not in_scope(8,120320,[120600]*8)
 mods={g:FakeModule(g) for g in (4,16)}
 caches=[torch.empty((1,4,16,256),dtype=torch.bfloat16) for _ in range(36)]
 table=torch.zeros((8,10240),dtype=torch.int32);lengths=torch.ones(8,dtype=torch.int32)
 with patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(multi_processor_count=170)):
  bank=LayerSharedGQA4QuadBank(mods,caches,table,lengths,120528,[4,5,6,7,8])
  for b in range(4,9):
   p=bank.plans[b];q=torch.arange(b*16*128).reshape(b,16,128).to(torch.bfloat16);o=torch.empty_like(q)
   bank.validate_aliases(table[:b],lengths[:b],b);bank.run_layer(35,q,o)
   assert torch.equal(q,o) and p.lengths.data_ptr()==lengths.data_ptr()
   assert p.workspace.data_ptr()==bank.workspace.data_ptr() and p.state.data_ptr()==bank.raw_state.data_ptr()
   packed=mods[16].calls[-1][8].view(p.groups,4,4,4,128)
   for row in range(p.groups*4):
    actual=packed[row//4,:,row%4].reshape(16,128)
    assert torch.equal(actual,q[row] if row<b else torch.zeros_like(q[0]))
   assert mods[4].calls[-1][1]==p.splits*b*4 and mods[16].calls[-1][1]==p.splits*p.groups*4
   assert mods[4].calls[-1][10].data_ptr()==caches[35].data_ptr()
   checks[f'B{b}_nativeS_pack_dummy_alias_layer35']=True
 assert not torch.cuda.is_initialized()
 records=[];fp={'shape':[36,16,4,128,16],'scope':receipt()};sources={'build':'new_quad_grouped_hint'}
 for rank in (0,1):
  steps=[]
  for b in range(4,9):
   c={k:True for k in REQUIRED_QUALIFICATION_FIELDS};c.update(batch=b,qualification_passed=True,layer_checks=[{'layer':l,'Q_bits_equal':True,'O_bits_equal':True} for l in range(36)],modes=[{'mode':m,'hidden_bits_equal':True} for m in ['checked','production']]);steps.append(c)
  records.append({'rank':rank,'source_sha256':sources,'execution_fingerprint':fp,'setup_mode':'strict','qualification_passed':True,'checks':steps,'qualification_steps':5,'captured_layers_by_batch':{m:{str(b):list(range(36)) for b in range(4,9)} for m in ['witness','checked','production']}})
 validate_offline_records(records,sources,fp,list(range(4,9)),2);checks['bothrank_all5B_offline']=True
 for name,modify in [('missingB4',lambda r:r[1]['checks'].pop(0)),('oldsource',lambda r:r[0].update(source_sha256={'build':'old_pair'}))]:
  rs=copy.deepcopy(records);modify(rs)
  try:validate_offline_records(rs,sources,fp,list(range(4,9)),2)
  except ValueError:checks[name+'_rejected']=True
  else:checks[name+'_rejected']=False
 assert all(checks.values());print(json.dumps({'status':'passed','cuda_initialized':torch.cuda.is_initialized(),'checks':checks,'arena_receipt':bank.receipt()},indent=2))
if __name__=='__main__':main()
