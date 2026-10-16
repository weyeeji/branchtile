"""CPU graph-bank ownership, alias, switching and energy-null contracts."""
import json,sys,copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.vllm_raw_pair_direct_multidoc_plan import LayerSharedRawPairDirectMultiDocumentBank
from scripts.vllm_raw_pair_direct_multidoc_worker import RawPairDirectMultiDocumentWorker
from src.phase_energy_counter import interval

class Module:
 def __init__(self,g):self.g=g;self.calls=[]
 def xqa_wrapper(self,*args):
  self.calls.append(args)
  if self.g==8:args[6].copy_(args[8])

def main():
 torch.set_num_threads(2);checks={};mods={g:Module(g) for g in (8,16)}
 caches=[torch.empty((1,2,16,256),dtype=torch.bfloat16) for _ in range(48)]
 table=torch.zeros((21,10240),dtype=torch.int32);lengths=torch.arange(1,22,dtype=torch.int32);before=lengths.clone()
 with patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(multi_processor_count=170)),patch('torch.cuda.is_current_stream_capturing',return_value=False),patch('torch.cuda.synchronize') as sync:
  bank=LayerSharedRawPairDirectMultiDocumentBank(mods,caches,table,lengths,135376,list(range(6,22)))
  ptrs={b:(p.table.data_ptr(),p.lengths.data_ptr(),p.device_cut.data_ptr(),p.state.data_ptr()) for b,p in bank.plans.items()}
  old=bank.device_cut.clone();bank.activate_prefix(123808,False)
  checks['native_switch_no_sync_upload']=sync.call_count==0 and torch.equal(old,bank.device_cut)
  for prefix in (135376,123808,135376):
   bank.activate_prefix(prefix,True)
   assert int(bank.device_cut)==prefix//256 and torch.equal(before,lengths)
   for b,p in bank.plans.items():
    assert ptrs[b]==(p.table.data_ptr(),p.lengths.data_ptr(),p.device_cut.data_ptr(),p.state.data_ptr())
    q=torch.arange(b*16*128).reshape(b,16,128).to(torch.bfloat16);o=torch.empty_like(q);bank.run_layer(47,q,o)
    assert torch.equal(q,o) and mods[8].calls[-1][-2].data_ptr()==bank.device_cut.data_ptr()
    assert mods[16].calls[-1][-2].data_ptr()==bank.device_cut.data_ptr()
    assert mods[8].calls[-1][1]==p.splits*b*2
    for call in (mods[8].calls[-1],mods[16].calls[-1]):
     assert call[8].data_ptr()==q.data_ptr() and call[14].data_ptr()==table.data_ptr() and call[16].data_ptr()==lengths.data_ptr()
    assert mods[8].calls[-1][6].data_ptr()==o.data_ptr()
    if prefix==135376:
     for layer in range(48):
      bank.run_layer(layer,q,o)
      assert mods[8].calls[-1][10].data_ptr()==caches[layer].data_ptr() and mods[16].calls[-1][10].data_ptr()==caches[layer].data_ptr()
  checks['all16_shapes_stable_aliases_originalS_no_length_mutation']=True
  checks['all48_layers_native_Q_O_metadata_aliases']=True
  badq=torch.empty((8,16,256),dtype=torch.bfloat16)[...,::2];goodq=torch.empty((8,16,128),dtype=torch.bfloat16)
  for q,o in ((badq,goodq),(goodq,badq)):
   try:bank.run_layer(0,q,o)
   except AssertionError:pass
   else:raise AssertionError('Noncontiguous Q/O accepted')
  checks['noncontiguous_Q_O_rejected']=True
  saved=(bank.prefix_tokens,bank.device_cut.clone())
  for invalid in (0,123809,163840):
   try:bank.activate_prefix(invalid,True)
   except AssertionError:pass
   else:raise AssertionError('Invalid cut accepted')
   assert bank.prefix_tokens==saved[0] and torch.equal(bank.device_cut,saved[1])
  checks['bad_cut_atomic_reject']=True
  worker=RawPairDirectMultiDocumentWorker();cfg={'common_prefix_tokens':0,'prompt_token_hashes':[]}
  docs={'A':{'document_id':'A','B':21,'common_complete_page_tokens':135376,'prompt_token_hashes':['A']*21},'B':{'document_id':'B','B':19,'common_complete_page_tokens':123808,'prompt_token_hashes':['B']*19}}
  worker.raw_pair={'mode':None,'method':'native','armed':False,'documents':docs,'plan':bank,'config':cfg,'document_generation':0,'qualified_by_document':{'A':{6,8},'B':{7}},'document_activations':[]}
  worker.native_probe={'requests':{'stale':1},'rank':0}
  worker.activate_raw_pair_document('A',True);assert worker.raw_pair['qualified_batches']=={6,8}
  worker.raw_pair.update(cohort={'stale':1},epochs={3:99});worker.native_probe['requests']['stale']=1
  worker.activate_raw_pair_document('B',False)
  assert worker.raw_pair['cohort'] is None and worker.raw_pair['epochs']=={} and worker.native_probe['requests']=={}
  assert worker.raw_pair['qualified_batches']=={7} and not worker.raw_pair['candidate_cut_ready'] and cfg['prompt_token_hashes']==['B']*19
  checks['document_ownership_and_qualification_are_isolated']=True
  table[5,0]=99
  try:bank.bind_prefix(table[:6])
  except ValueError:checks['new_physical_prefix_mismatch_rejected']=True
  else:raise AssertionError('Mixed physical ownership accepted')
 a={'monotonic_seconds':1.,'devices':[{'physical_selector':'0','total_energy_mJ':1000},{'physical_selector':'1','total_energy_mJ':None}]}
 z={'monotonic_seconds':2.,'devices':[{'physical_selector':'0','total_energy_mJ':3000},{'physical_selector':'1','total_energy_mJ':None}]}
 result=interval(a,z);assert result['devices'][0]['joules']==2. and result['total_GPU_joules'] is None
 z['devices'][0]['total_energy_mJ']=500;assert interval(a,z)['devices'][0]['joules'] is None
 checks['unsupported_and_decreasing_energy_counters_not_estimated']=True
 a['devices']=a['devices'][:1];z['devices']=z['devices'][:1];z['devices'][0]['total_energy_mJ']=3000
 assert interval(a,z)['total_GPU_joules'] is None
 for sample in (a,z):sample['devices'][0].update(identity_verified=True,UUID='GPU-test')
 assert interval(a,z)['total_GPU_joules']==2.
 checks['unverified_energy_device_identity_cannot_claim_task_joules']=True
 from src.raw_pair_direct_multidoc_offline import load_completed_qualification
 from scripts.vllm_raw_pair_direct_multidoc_worker import source_hashes
 protocol=json.loads((ROOT/'configs/raw_pair_multidoc_training_v1.json').read_text())
 try:load_completed_qualification(ROOT/'results/raw_pair_multidoc_model_qual_v1/summary.json',protocol,ROOT,source_hashes())
 except ValueError:checks['actual_old_multidoc_qualification_rejected']=True
 else:raise AssertionError('Old non-direct multidoc receipt accepted')
 assert all(checks.values()) and not torch.cuda.is_initialized()
 print(json.dumps({'status':'passed','cuda_initialized':False,'checks':checks,'shared_cut_ptr_count':len({p.device_cut.data_ptr() for p in bank.plans.values()})},indent=2))
if __name__=='__main__':main()
