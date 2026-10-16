"""CPU native-address, state-arena and stale-receipt checks for direct GQA4 quad."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.vllm_gqa4_direct_quad_plan import LayerSharedGQA4DirectQuadBank
from src.gqa4_quad_scope import in_scope
from scripts.vllm_gqa4_direct_quad_worker import source_hashes
from src.vllm_gqa4_raw_pair_setup import validate_offline_records

class Module:
 def __init__(self,group):self.group=group;self.calls=[]
 def xqa_wrapper(self,*args):
  self.calls.append(args)
  if self.group==4:args[6].copy_(args[8])

def main():
 torch.set_num_threads(2);mods={g:Module(g) for g in (4,16)}
 caches=[torch.empty((1,4,16,256),dtype=torch.bfloat16) for _ in range(36)]
 table=torch.zeros((8,10240),dtype=torch.int32);lengths=torch.ones(8,dtype=torch.int32);checks={};count=0
 with patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(multi_processor_count=170)):
  bank=LayerSharedGQA4DirectQuadBank(mods,caches,table,lengths,120528,[4,5,6,7,8])
  for b in range(4,9):
   assert in_scope(b,120528,[128720]*b) and not in_scope(b,120528,[128721]*b)
   for layer in range(36):
    q=torch.arange(b*16*128).reshape(b,16,128).to(torch.bfloat16);o=torch.empty_like(q)
    result=bank.run_layer(layer,q,o);p=bank.plans[b]
    assert result.data_ptr()==o.data_ptr() and torch.equal(q,o)
    a,z=mods[16].calls[-1],mods[4].calls[-1]
    for call in (a,z):
     assert call[8].data_ptr()==q.data_ptr() and call[14].data_ptr()==table.data_ptr()
     assert call[16].data_ptr()==lengths.data_ptr() and call[10].data_ptr()==caches[layer].data_ptr()
     assert call[26].data_ptr()==bank.raw_state.data_ptr() and call[23].data_ptr()==bank.workspace.data_ptr()
    assert z[6].data_ptr()==o.data_ptr() and z[17]==b and a[17]==(b+3)//4
    assert a[1]==p.splits*p.groups*4 and z[1]==p.splits*b*4
    bank.validate_aliases(table[:b],lengths[:b],b);count+=1
   q=torch.empty((b,16,256),dtype=torch.bfloat16)[...,::2];o=torch.empty((b,16,128),dtype=torch.bfloat16)
   try:bank.run_layer(0,q,o)
   except AssertionError:pass
   else:raise AssertionError('Noncontiguous native Q accepted')
   try:bank.run_layer(0,o,q)
   except AssertionError:pass
   else:raise AssertionError('Noncontiguous native O accepted')
  assert len(mods[4].calls)==len(mods[16].calls)==180
  checks.update(all180_layer_batch_native_pointer_selections=True,two_candidate_module_calls_per_layer=True,
                original_B_S_shared_arena_preserved=True,noncontiguous_Q_O_rejected=True)
 old=Path('results/gqa4_thinking_quad_model_qual_v1/train1/rank0/qualification_raw_pair.json')
 if old.exists():
  record=json.loads(old.read_text())
  try:validate_offline_records([record],source_hashes(),record['execution_fingerprint'],[4,5,6,7,8],2)
  except ValueError as exc:assert 'source fingerprint' in str(exc);checks['actual_old_quad_receipt_rejected']=True
  else:raise AssertionError('Old copied-input quad receipt accepted')
 else:raise FileNotFoundError(old)
 assert all(not in_scope(b,120528,[120600]*b) for b in (1,2,3));checks['B1to3_stay_native']=True
 assert not torch.cuda.is_initialized()
 print(json.dumps({'status':'passed','cuda_initialized':False,'checks':checks,'actual_layer_batch_cases':count,
                   'scope':'CPU pointer selection and guards only; GPU numerical qualification remains required.'},indent=2))
if __name__=='__main__':main()
